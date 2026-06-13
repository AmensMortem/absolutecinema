import os
import sys
import ast
import warnings
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup

from tqdm import tqdm


tqdm.pandas()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#############################################
# КОНФИГУРАЦИЯ
#############################################

DATASET_DIR   = "IMDb Movie Genre Classification/"
OVERVIEW_PATH = "IMDb Movie Genre Classification/movies_overview.csv"
GENRES_PATH   = "IMDb Movie Genre Classification/movies_genres.csv"
print('HALLO', OVERVIEW_PATH)
TEXT_COLUMN  = "overview"
GENRE_COLUMN = "genre_names"

RANDOM_STATE  = 42
BATCH_SIZE    = 32
EPOCHS = 40          # было 20 — модель ещё росла
LEARNING_RATE = 2e-5        # стандартный LR для fine-tuning BERT
WEIGHT_DECAY  = 1e-2
THRESHOLD = 0.5
PATIENCE = 7           # было 5 — даём больше времени на плато
MAX_LEN = 256         # описания короткие — 128 токенов хватает

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство для обучения: {device}")

#############################################
# ЗАГРУЗКА ДАННЫХ (TMDB формат)
#############################################

# for path in (OVERVIEW_PATH, GENRES_PATH):
#     if not os.path.exists(path):
#         print(f"\n[!] Файл не найден: {os.path.abspath(path)}\n")
#         sys.exit(1)
#
# print("Downloading...")

# movies_genres.csv  →  dict {id: name}
genres_df  = pd.read_csv(GENRES_PATH)
id_to_name = dict(zip(genres_df["id"], genres_df["name"]))

# movies_overview.csv  →  overview, genre_ids
overview_df = pd.read_csv(OVERVIEW_PATH)
overview_df = overview_df[["overview", "genre_ids"]].dropna()

# genre_ids — str: "[18, 80]", converting id → name
def parse_genre_ids(x):
    try:
        ids = ast.literal_eval(x) if isinstance(x, str) else x
        return [id_to_name[i] for i in ids if i in id_to_name]
    except Exception:
        return []

overview_df[GENRE_COLUMN] = overview_df["genre_ids"].apply(parse_genre_ids)

# Убираем строки без жанров
overview_df = overview_df[overview_df[GENRE_COLUMN].map(len) > 0]
df = overview_df[[TEXT_COLUMN, GENRE_COLUMN]].reset_index(drop=True)
print(f"Загружено фильмов: {len(df)}")

#############################################
# РАЗБИЕНИЕ ДАННЫХ
#############################################

X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    df[TEXT_COLUMN], df[GENRE_COLUMN], test_size=0.3, random_state=RANDOM_STATE
)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
)

# Бинарная матрица жанров
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train_raw)
y_val_bin   = mlb.transform(y_val_raw)
y_test_bin  = mlb.transform(y_test_raw)

print(f"Обучающая выборка: {len(X_train_raw)} примеров")
print(f"Тестовая выборка:  {len(X_test_raw)} примеров")
print(f"Кол-во жанров:     {len(mlb.classes_)}")

#############################################
# ТОКЕНИЗАЦИЯ — DistilBERT
# DistilBERT токенизирует текст в последовательность ID токенов
# с attention mask — указывает какие токены реальные, а какие padding
#############################################

print("Загружаем токенизатор DistilBERT...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
print("Токенизатор загружен.")

class MovieBertDataset(Dataset):
    def __init__(self, texts, labels):
        # Токенизируем все тексты сразу — быстрее чем по одному
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

print("Токенизация датасетов...")
train_dataset = MovieBertDataset(X_train_raw, y_train_bin)
val_dataset   = MovieBertDataset(X_val_raw,   y_val_bin)
test_dataset  = MovieBertDataset(X_test_raw,  y_test_bin)

# num_workers=0 на Windows
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

#############################################
# МОДЕЛЬ — DistilBERT + классификатор
# DistilBERT — облегчённый BERT (40% меньше параметров, 60% быстрее)
# Fine-tuning: обучаем все веса BERT + добавляем свой классификатор сверху
# [CLS] токен — первый токен, представляет весь текст
#############################################

class DistilBertGenreClassifier(nn.Module):
    """
    DistilBERT для мультилейбл-классификации жанров.
    Берём [CLS] токен как представление всего текста,
    пропускаем через классификатор.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # Загружаем предобученный DistilBERT
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        hidden_size = self.bert.config.hidden_size  # 768

        # Классификатор поверх [CLS] токена
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # DistilBERT возвращает hidden states для каждого токена
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] токен — первый токен, представляет весь текст
        cls_output = outputs.last_hidden_state[:, 0, :]

        return self.classifier(cls_output)

NUM_CLASSES = y_train_bin.shape[1]
model = DistilBertGenreClassifier(NUM_CLASSES).to(device)
print(f"\nМодель: {sum(p.numel() for p in model.parameters()):,} параметров")

#############################################
# ФУНКЦИЯ ПОТЕРЬ И ОПТИМИЗАТОР
#############################################

pos_counts = y_train_bin.sum(axis=0)
neg_counts = y_train_bin.shape[0] - pos_counts
pos_weight = neg_counts / (pos_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

# BCEWithLogitsLoss = Sigmoid + Binary Cross-Entropy
# Подходит для мультилейбл-задач (каждый жанр — независимый бинарный классификатор)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AdamW — стандартный оптимизатор для fine-tuning трансформеров
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Линейный warmup + decay — стандартная стратегия для BERT
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,   # 10% шагов — warmup
    num_training_steps=total_steps
)

# AMP — автоматическая смешанная точность: экономит VRAM вдвое
use_amp = (device.type == "cuda")
scaler  = GradScaler("cuda", enabled=use_amp)

#############################################
# Training
#############################################
print("\nTraining...")

train_losses     = []
train_f1_scores  = []
train_accuracies = []

best_val_f1      = 0.0
best_model_state = None
no_improve_epochs = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS}", leave=False):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_ids, attention_mask)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping — стандарт для трансформеров
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Валидация
    model.eval()
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids, attention_mask)
            preds = (torch.sigmoid(outputs) > THRESHOLD).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch["labels"].numpy())

    micro    = f1_score(all_true, all_preds, average="micro", zero_division=0)
    macro    = f1_score(all_true, all_preds, average="macro", zero_division=0)
    accuracy = np.mean(np.all(np.array(all_preds) == np.array(all_true), axis=1))

    train_f1_scores.append(micro)
    train_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | "
          f"Val Micro F1: {micro:.4f} | Val Macro F1: {macro:.4f} | "
          f"Val Accuracy: {accuracy:.4f} (exact match)")

    # Early stopping — сохраняем лучшую модель
    if micro > best_val_f1:
        best_val_f1      = micro
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= PATIENCE:
            print(f"\nEarly stopping на эпохе {epoch+1} — val F1 не улучшался {PATIENCE} эпох.")
            break

# Загружаем лучшие веса
if best_model_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    print(f"Загружены лучшие веса (val Micro F1 = {best_val_f1:.4f})")

metrics_df = pd.DataFrame({
    "epoch":    range(1, len(train_losses) + 1),
    "loss":     train_losses,
    "micro_f1": train_f1_scores,
    "accuracy": train_accuracies
})
metrics_df.to_csv("training_history.csv", index=False)
print("История обучения сохранена в training_history.csv")

#############################################
# ПОИСК ОПТИМАЛЬНОГО ПОРОГА НА ВАЛИДАЦИИ
# Делаем ДО тест-оценки, чтобы применить best_t на тесте
#############################################

val_probs = []
val_true  = []

model.eval()
with torch.no_grad():
    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_ids, attention_mask)
        val_probs.extend(torch.sigmoid(outputs).cpu().numpy())
        val_true.extend(batch["labels"].numpy())

val_probs = np.array(val_probs)
val_true  = np.array(val_true)

# Порог для Micro F1
best_t  = 0.5
best_f1 = 0.0
for t in np.arange(0.1, 0.9, 0.02):
    f1 = f1_score(val_true, (val_probs > t), average="micro", zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_t  = t

# Порог для Exact Match
best_t_acc = 0.5
best_acc   = 0.0
for t in np.arange(0.1, 0.95, 0.02):
    acc = np.mean(np.all((val_probs > t) == val_true, axis=1))
    if acc > best_acc:
        best_acc   = acc
        best_t_acc = t

# Порог для High Precision (recall >= 35%)
best_t_prec = 0.5
best_prec   = 0.0
for t in np.arange(0.3, 0.95, 0.02):
    rec = recall_score(val_true, (val_probs > t), average="micro", zero_division=0)
    if rec < 0.35:
        break
    prec = precision_score(val_true, (val_probs > t), average="micro", zero_division=0)
    if prec > best_prec:
        best_prec   = prec
        best_t_prec = t

print(f"\n BEST THRESHOLD (Micro F1):      {round(best_t, 2)}  → F1={round(best_f1, 4)}")
print(f" BEST THRESHOLD (High Precision): {round(best_t_prec, 2)}  → Precision={round(best_prec, 4)}")
print(f" BEST THRESHOLD (Exact Match):    {round(best_t_acc, 2)}  → Accuracy={round(best_acc*100, 1)}%")

#############################################
# ОЦЕНКА КАЧЕСТВА НА ТЕСТЕ — три режима
#############################################

def evaluate(threshold, label):
    preds_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids, attention_mask)
            preds_list.extend((torch.sigmoid(outputs) > threshold).cpu().numpy())
    p     = np.array(preds_list)
    micro = f1_score(all_true_list, p, average="micro", zero_division=0)
    prec  = precision_score(all_true_list, p, average="micro", zero_division=0)
    rec   = recall_score(all_true_list, p, average="micro", zero_division=0)
    acc   = np.mean(np.all(p == np.array(all_true_list), axis=1))
    print(f"\n=== {label} (порог={round(threshold,2)}) ===")
    print(f"Precision:            {prec:.4f}")
    print(f"Recall:               {rec:.4f}")
    print(f"Micro F1:             {micro:.4f}")
    print(f"Exact Match Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    return p

all_true_list = []
with torch.no_grad():
    for batch in test_loader:
        all_true_list.extend(batch["labels"].numpy())

all_preds_f1   = evaluate(best_t,      "Режим Micro F1")
all_preds_prec = evaluate(best_t_prec, "Режим High Precision")
all_preds_acc  = evaluate(best_t_acc,  "Режим Exact Match")

print("\n=== Детальный отчёт (High Precision режим) ===")
print(classification_report(
    all_true_list, all_preds_prec,
    target_names=mlb.classes_,
    zero_division=0
))

#############################################
# ИНФЕРЕНС — предсказание для нового текста
#############################################

def predict_genres(text: str, mode: str = "precision") -> tuple:
    """
    Принимает сырой текст → возвращает кортеж предсказанных жанров.
    mode="f1"        — баланс precision/recall
    mode="precision" — высокий precision, меньше лишних жанров (по умолчанию)
    mode="accuracy"  — максимальный exact match
    """
    if mode == "f1":
        threshold = best_t
    elif mode == "accuracy":
        threshold = best_t_acc
    else:
        threshold = best_t_prec

    model.eval()
    with torch.no_grad():
        # DistilBERT токенизирует текст целиком — понимает контекст фразы
        encoding = tokenizer(
            str(text),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        preds   = (torch.sigmoid(outputs) > threshold).int().cpu().numpy()
        return mlb.inverse_transform(preds)[0]

#############################################
# ДЕМОНСТРАЦИЯ
#############################################

'''example = """
A group of astronauts travel through space to save humanity from a dying Earth.
They encounter strange anomalies, dangerous black holes, and distant unknown planets.
"""'''

example = "When a menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman, James Gordon and Harvey Dent must work together to put an end to the madness."

print("\n=== Демо-предсказание ===")
print(f"Describtion: {example.strip()}")
print(f"F1 mode       (больше жанров):  {predict_genres(example, mode='f1')}")
print(f"Precision mode (меньше мусора): {predict_genres(example, mode='precision')}")
print(f"Accuracy mode  (только точные): {predict_genres(example, mode='accuracy')}")
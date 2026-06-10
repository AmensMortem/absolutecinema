import os
import sys
import ast
import copy
import re
import warnings
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score

from tqdm import tqdm
tqdm.pandas()

# Глушим предупреждения sklearn о пустых классах
warnings.filterwarnings("ignore", category=UserWarning)

#############################################
# КОНФИГУРАЦИЯ
#############################################

# Папка с датасетом TMDB — положи рядом со скриптом
DATASET_DIR   = "./IMDb Movie Genre Classification"
OVERVIEW_PATH = os.path.join(DATASET_DIR, "movies_overview.csv")
GENRES_PATH   = os.path.join(DATASET_DIR, "movies_genres.csv")

TEXT_COLUMN  = "overview"
GENRE_COLUMN = "genre_names"

RANDOM_STATE  = 42
BATCH_SIZE    = 64        # датасет маленький — меньше батч, точнее градиенты
EPOCHS        = 30        # маленький датасет сходится медленнее
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4      # L2-регуляризация — снижает переобучение
THRESHOLD     = 0.5
PATIENCE      = 7         # Early stopping: остановиться если val F1 не растёт N эпох
FASTTEXT_PATH = "cc.en.300.bin"  # путь к файлу FastText рядом со скриптом
FASTTEXT_DIM  = 300              # размерность векторов FastText

# GPU → MPS (Apple Silicon) → CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Устройство для обучения: {device}")

#############################################
# ЗАГРУЗКА И МЁРЖ ДАННЫХ (TMDB формат)
#############################################

for path in (OVERVIEW_PATH, GENRES_PATH):
    if not os.path.exists(path):
        print(
            f"\n[!] Файл не найден: {os.path.abspath(path)}\n"
            f"Положи папку 'IMDb Movie Genre Classification' рядом со скриптом.\n"
        )
        sys.exit(1)

print("Downloading...")

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
# ПРЕДОБРАБОТКА ТЕКСТА
# Для FastText важно убирать пунктуацию —
# каждое слово ищется в словаре модели
#############################################

def clean_text(text: str) -> str:
    """Чистка: нижний регистр, убираем пунктуацию и лишние пробелы."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Чистка текстов...")
df[TEXT_COLUMN] = df[TEXT_COLUMN].progress_apply(clean_text)

#############################################
# РАЗБИЕНИЕ ДАННЫХ
#############################################

X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    df[TEXT_COLUMN],
    df[GENRE_COLUMN],
    test_size=0.3,
    random_state=RANDOM_STATE
)

X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE
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
# ВЕКТОРИЗАЦИЯ — FastText с TF-IDF взвешиванием
# Строим TF-IDF только для весов слов (IDF) —
# сами векторы берём из FastText
# Важные редкие слова ("heist", "dystopian") получают больший вес
#############################################

import fasttext

if not os.path.exists(FASTTEXT_PATH):
    print(f"\n[!] Файл FastText не найден: {os.path.abspath(FASTTEXT_PATH)}")
    print("Скачай cc.en.300.bin с https://fasttext.cc/docs/en/crawl-vectors.html")
    sys.exit(1)

print(f"Загружаем FastText модель ({FASTTEXT_PATH})... (может занять минуту)")
ft_model = fasttext.load_model(FASTTEXT_PATH)
print("FastText загружен.")

# TF-IDF только для IDF-весов слов, не как признаки
print("Строим IDF-веса...")
tfidf_for_weights = TfidfVectorizer(
    max_features=50000,
    min_df=2,
    sublinear_tf=True
)
tfidf_for_weights.fit(X_train_raw)
vocab = tfidf_for_weights.vocabulary_
idf   = tfidf_for_weights.idf_

def text_to_weighted_vector(text: str) -> np.ndarray:
    """
    Взвешенное усреднение FastText векторов по TF-IDF весам слов + L2-норм.
    Важные слова (высокий IDF) вносят больший вклад в итоговый вектор.
    """
    words = text.split()
    if not words:
        return np.zeros(FASTTEXT_DIM, dtype=np.float32)

    vecs    = []
    weights = []
    for w in words:
        vec    = ft_model.get_word_vector(w)
        weight = idf[vocab[w]] if w in vocab else 1.0
        vecs.append(vec)
        weights.append(weight)

    vecs    = np.array(vecs,    dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    weighted = np.average(vecs, axis=0, weights=weights)

    # L2-нормализация — приводим к единичной длине
    norm = np.linalg.norm(weighted)
    if norm > 0:
        weighted /= norm

    return weighted

print("Векторизация текстов через FastText (взвешенная)...")
X_train = np.vstack(X_train_raw.progress_apply(text_to_weighted_vector).values)
X_val   = np.vstack(X_val_raw.progress_apply(text_to_weighted_vector).values)
X_test  = np.vstack(X_test_raw.progress_apply(text_to_weighted_vector).values)

print(f"Вектор: {X_train.shape[1]}D")

#############################################
# PYTORCH DATASET
# FastText даёт плотные матрицы (10k × 300) —
# конвертируем в тензор сразу
#############################################

class MovieDataset(Dataset):
    def __init__(self, X_dense, y_bin):
        self.X = torch.tensor(X_dense, dtype=torch.float32)
        self.y = torch.tensor(y_bin,   dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = MovieDataset(X_train, y_train_bin)
val_dataset   = MovieDataset(X_val,   y_val_bin)
test_dataset  = MovieDataset(X_test,  y_test_bin)

# num_workers=0 на Windows (>0 вызывает проблемы с multiprocessing)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

#############################################
# Model Architecture — MLP с residual connection
# Для FastText 300D входа используем глубокую сеть —
# residual connection позволяет градиентам течь напрямую
#############################################

class MovieGenreClassifier(nn.Module):
    """MLP с residual connection для мультилейбл-классификации жанров."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        # Проекция входа до размера скрытых слоёв
        self.input_proj = nn.Linear(input_dim, 512)

        self.block1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),    # BatchNorm ускоряет сходимость
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.block1(x)    # residual connection: выход + вход блока
        x = self.block2(x)
        return self.classifier(x)

INPUT_DIM   = FASTTEXT_DIM
NUM_CLASSES = y_train_bin.shape[1]

model = MovieGenreClassifier(INPUT_DIM, NUM_CLASSES).to(device)
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

# AdamW лучше Adam — встроенный weight decay работает правильно
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Снижаем LR в 2 раза, если val F1 не улучшается 2 эпохи подряд
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5, verbose=True
)

# AMP — автоматическая смешанная точность: экономит VRAM вдвое, ускоряет обучение на GPU
use_amp = (device.type == "cuda")
scaler  = torch.amp.GradScaler(enabled=use_amp)

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

    for X_batch, y_batch in tqdm(
            train_loader,
            desc=f"Эпоха {epoch + 1}/{EPOCHS}",
            leave=False):

        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Валидация
    model.eval()
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(X_batch)
            preds   = (torch.sigmoid(outputs) > THRESHOLD).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    micro    = f1_score(all_true, all_preds, average="micro", zero_division=0)
    macro    = f1_score(all_true, all_preds, average="macro", zero_division=0)
    all_preds_np = np.array(all_preds)
    all_true_np  = np.array(all_true)
    accuracy = np.mean(np.all(all_preds_np == all_true_np, axis=1))

    train_f1_scores.append(micro)
    train_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | "
          f"Val Micro F1: {micro:.4f} | Val Macro F1: {macro:.4f} | "
          f"Val Accuracy: {accuracy:.4f} (exact match)")

    scheduler.step(micro)

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
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(X_batch)
        probs   = torch.sigmoid(outputs).cpu().numpy()
        val_probs.extend(probs)
        val_true.extend(y_batch.numpy())

val_probs = np.array(val_probs)
val_true  = np.array(val_true)

best_t  = 0.5
best_f1 = 0.0

for t in np.arange(0.1, 0.95, 0.02):
    preds = (val_probs > t)
    f1    = f1_score(val_true, preds, average="micro", zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_t  = t

print(f"\n BEST THRESHOLD (Micro F1): {round(best_t, 2)}")
print(f" BEST MICRO F1 (val): {round(best_f1, 4)}")

# Отдельно ищем порог для максимального exact match accuracy
best_t_acc  = 0.5
best_acc    = 0.0

for t in np.arange(0.1, 0.95, 0.02):
    preds    = (val_probs > t)
    acc      = np.mean(np.all(preds == val_true, axis=1))
    if acc > best_acc:
        best_acc   = acc
        best_t_acc = t

# Ищем порог для максимального precision при recall >= 0.4
# Это баланс: не теряем слишком много recall, но precision высокий
from sklearn.metrics import precision_score, recall_score

best_t_prec  = 0.5
best_prec    = 0.0

for t in np.arange(0.3, 0.95, 0.02):
    preds    = (val_probs > t)
    rec      = recall_score(val_true, preds, average="micro", zero_division=0)
    if rec < 0.35:          # не даём recall упасть ниже 35%
        break
    prec     = precision_score(val_true, preds, average="micro", zero_division=0)
    if prec > best_prec:
        best_prec   = prec
        best_t_prec = t

print(f"\n BEST THRESHOLD (High Precision): {round(best_t_prec, 2)}")
print(f" BEST PRECISION (val): {round(best_prec, 4)}")
print(f"\n BEST THRESHOLD (Exact Match): {round(best_t_acc, 2)}")
print(f" BEST EXACT MATCH ACCURACY (val): {round(best_acc, 4)} ({round(best_acc*100, 1)}%)")

# THRESHOLD используется только при обучении (0.5 — математически правильный для BCE).
# На инференсе и тесте используем best_t_prec — высокий precision.

#############################################
# ОЦЕНКА КАЧЕСТВА НА ТЕСТЕ — три режима
#############################################

def evaluate(threshold, label):
    preds_list = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(X_batch)
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

# Сначала собираем all_true_list
all_true_list = []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        all_true_list.extend(y_batch.numpy())

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
    mode="f1"        — баланс precision/recall (больше жанров)
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
        cleaned   = clean_text(text)
        # FastText взвешенный: тот же пайплайн что при обучении
        vec       = text_to_weighted_vector(cleaned)
        tensor_in = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
        outputs   = model(tensor_in)
        preds     = (torch.sigmoid(outputs) > threshold).int().cpu().numpy()
        return mlb.inverse_transform(preds)[0]

#############################################
# ДЕМОНСТРАЦИЯ
#############################################

example = """
A group of astronauts travel through space to save humanity from a dying Earth.
They encounter strange anomalies, dangerous black holes, and distant unknown planets.
"""

print("\n=== Демо-предсказание ===")
print(f"Describtion: {example.strip()}")
print(f"F1:  {predict_genres(example, mode='f1')}")
print(f"Precision mode: {predict_genres(example, mode='precision')}")
print(f"Accuracy mode: {predict_genres(example, mode='accuracy')}")
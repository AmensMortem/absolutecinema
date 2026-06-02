import os
import ast
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from tqdm import tqdm
tqdm.pandas()

#############################################
# КОНФИГУРАЦИЯ
#############################################

# Папка с датасетом — положи рядом со скриптом
DATASET_DIR     = "./IMDb Movie Genre Classification"
OVERVIEW_PATH   = os.path.join(DATASET_DIR, "movies_overview.csv")
GENRES_PATH     = os.path.join(DATASET_DIR, "movies_genres.csv")

TEXT_COLUMN  = "overview"
GENRE_COLUMN = "genre_names"   # создадим сами при мерже

TEST_SIZE      = 0.2
RANDOM_STATE   = 42
BATCH_SIZE     = 64
EPOCHS         = 5
LEARNING_RATE  = 0.001

# GPU → MPS (Apple Silicon) → CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Устройство для обучения: {device}")

#############################################
# ЗАГРУЗКА И МЕРЖ ДАННЫХ
#############################################

import sys

for path in (OVERVIEW_PATH, GENRES_PATH):
    if not os.path.exists(path):
        print(
            f"\n[!] Файл не найден: {os.path.abspath(path)}\n"
            f"Положи папку 'IMDb Movie Genre Classification' рядом со скриптом.\n"
        )
        sys.exit(1)

print("Загружаю данные...")

# movies_genres.csv  →  словарь {id: name}
genres_df  = pd.read_csv(GENRES_PATH)
id_to_name = dict(zip(genres_df["id"], genres_df["name"]))

# movies_overview.csv  →  title, overview, genre_ids
overview_df = pd.read_csv(OVERVIEW_PATH)
overview_df = overview_df[["overview", "genre_ids"]].dropna()

# genre_ids — строка вида "[18, 80]", парсим и конвертируем id → названия
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
# ПРЕДОБРАБОТКА ТЕКСТА (только базовая, без NLTK)
# TF-IDF сам справляется с токенизацией и нормализацией
#############################################

def clean_text(text: str) -> str:
    """Минимальная чистка: нижний регистр + убираем лишние пробелы.
    TF-IDF обрабатывает пунктуацию и токенизацию внутри себя."""
    return str(text).lower().strip()

print("Чистка текстов...")
df[TEXT_COLUMN] = df[TEXT_COLUMN].progress_apply(clean_text)

#############################################
# РАЗБИЕНИЕ ДАННЫХ И ВЕКТОРИЗАЦИЯ
#############################################

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    df[TEXT_COLUMN],
    df[GENRE_COLUMN],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# Уни- и биграммы, топ-20 000 токенов
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)

# Оставляем разреженные матрицы — экономим оперативную память
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf  = tfidf.transform(X_test_raw)

# Бинарная матрица жанров
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train_raw)
y_test_bin  = mlb.transform(y_test_raw)

print(f"Обучающая выборка: {X_train_tfidf.shape[0]} примеров")
print(f"Тестовая выборка:  {X_test_tfidf.shape[0]} примеров")
print(f"Кол-во жанров:     {len(mlb.classes_)}")

#############################################
# PYTORCH DATASET — конвертирует разреженную
# матрицу в плотный тензор только для нужного батча
#############################################

class MovieDataset(Dataset):
    def __init__(self, X_sparse, y_bin):
        self.X = X_sparse
        self.y = torch.tensor(y_bin, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_dense = self.X[idx].toarray().squeeze()
        return torch.tensor(x_dense, dtype=torch.float32), self.y[idx]

train_dataset = MovieDataset(X_train_tfidf, y_train_bin)
test_dataset  = MovieDataset(X_test_tfidf,  y_test_bin)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

#############################################
# АРХИТЕКТУРА МОДЕЛИ — MLP (3 скрытых слоя)
#############################################

class MovieGenreClassifier(nn.Module):
    """Многослойный перцептрон для мультилейбл-классификации жанров."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

INPUT_DIM   = X_train_tfidf.shape[1]   # 20 000
NUM_CLASSES = y_train_bin.shape[1]

model = MovieGenreClassifier(INPUT_DIM, NUM_CLASSES).to(device)
print(f"\nМодель: {sum(p.numel() for p in model.parameters()):,} параметров")

#############################################
# ФУНКЦИЯ ПОТЕРЬ И ОПТИМИЗАТОР
#############################################

# BCEWithLogitsLoss = Sigmoid + Binary Cross-Entropy
# Подходит для мультилейбл-задач (каждый жанр — независимый бинарный классификатор)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Снижаем LR в 2 раза, если лосс не улучшается 2 эпохи подряд
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

#############################################
# ЦИКЛ ОБУЧЕНИЯ
#############################################

print("\nНачинаю обучение...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{EPOCHS}", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    scheduler.step(epoch_loss)
    print(f"Эпоха [{epoch + 1}/{EPOCHS}]  Loss: {epoch_loss:.4f}")

#############################################
# ОЦЕНКА КАЧЕСТВА НА ТЕСТЕ
#############################################

model.eval()
all_preds = []
all_true  = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)

        # Порог 0.5: если sigmoid(logit) > 0.5 — жанр присвоен
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

print("\n=== Отчёт классификации (тестовая выборка) ===")
print(classification_report(
    all_true, all_preds,
    target_names=mlb.classes_,
    zero_division=0
))

#############################################
# ИНФЕРЕНС — предсказание для нового текста
#############################################

def predict_genres(text: str) -> tuple:
    """Принимает сырой текст → возвращает кортеж предсказанных жанров."""
    model.eval()
    with torch.no_grad():
        cleaned   = clean_text(text)
        vectorized = tfidf.transform([cleaned]).toarray().squeeze()

        tensor_in = torch.tensor(vectorized, dtype=torch.float32).unsqueeze(0).to(device)
        outputs   = model(tensor_in)
        preds     = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()

        return mlb.inverse_transform(preds)[0]

#############################################
# ДЕМОНСТРАЦИЯ
#############################################

example = """
A group of astronauts travel through space to save humanity from a dying Earth.
They encounter strange anomalies, dangerous black holes, and distant unknown planets.
"""

print("\n=== Демо-предсказание ===")
print(f"Описание: {example.strip()}")
print(f"Жанры:    {predict_genres(example)}")
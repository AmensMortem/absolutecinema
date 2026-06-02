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

# Импортируем tqdm для отслеживания прогресса
from tqdm import tqdm

# Импортируем инструменты NLTK для обработки текста
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Скачиваем необходимые пакеты для NLTK
nltk.download('punkt')  # Для разбиения текста на слова (токенизация)
nltk.download('stopwords')  # Список стоп-слов (the, an, a...)
nltk.download('wordnet')  # База данных слов для лемматизации
nltk.download('averaged_perceptron_tagger')  # Для определения частей речи (POS-tags)

# Включаем поддержку tqdm для pandas методов (прогресс-бар для .progress_apply)
tqdm.pandas()

#############################################
# CONFIGURATION (КОНФИГУРАЦИЯ)
#############################################

# НАСТРОЙКА ОКРУЖЕНИЯ:
# Поставь True, если запускаешь код внутри Kaggle Notebook.
# Поставь False, если запускаешь локально на компьютере.
IS_KAGGLE_NOTEBOOK = False

# Ссылка на датасет Kaggle (нужна, если IS_KAGGLE_NOTEBOOK = False)
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/adilshamim8/nlp-task"

# Пути к файлам (исправлено имя датасета согласно URL)
if IS_KAGGLE_NOTEBOOK:
    DATA_PATH = "/kaggle/input/nlp-task/movies.csv"
else:
    DATA_PATH = "./nlp-task/movies.csv"

# Названия целевых колонок в таблице
TEXT_COLUMN = "plot"
GENRE_COLUMN = "genres"

# Параметры разделения данных и воспроизводимости
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Гиперпараметры нейросети
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# Выбираем устройство для вычислений: GPU (CUDA/MPS), если доступны, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Используемое устройство для обучения: {device}")

#############################################
# DATA DOWNLOAD & LOADING (ЗАГРУЗКА ДАННЫХ)
#############################################

# Если работаем локально и папки с данными еще нет — скачиваем ее
if not IS_KAGGLE_NOTEBOOK and not os.path.exists(DATA_PATH):
    import opendatasets as od

    print("Запускаю скачивание датасета с Kaggle...")
    od.download(KAGGLE_DATASET_URL)

# Читаем CSV файл в DataFrame
print("Загрузка данных в Pandas...")
df = pd.read_csv(DATA_PATH)

# Избавляемся от лишних колонок и удаляем строки с пропусками (NaN)
df = df[[TEXT_COLUMN, GENRE_COLUMN]].dropna()


#############################################
# GENRE PARSER (ПАРСЕР ЖАНРОВ)
#############################################

def parse_genres(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return [g.strip() for g in str(x).split(",")]


# Применяем парсер к колонке жанров
df[GENRE_COLUMN] = df[GENRE_COLUMN].apply(parse_genres)

#############################################
# NLTK TEXT PREPROCESSING (ПРЕДОБРАБОТКА ТЕКСТА)
#############################################

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def get_wordnet_pos(treebank_tag):
    """Вспомогательная функция для конвертации тегов частей речи NLTK в теги WordNet"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # По умолчанию существительное


def clean_text_with_nltk(text):
    text = str(text).lower()
    tokens = word_tokenize(text)

    # Определяем части речи для всего списка токенов сразу (так быстрее)
    pos_tags = nltk.pos_tag(tokens)

    cleaned_tokens = []
    for token, tag in pos_tags:
        if token.isalpha():
            if token not in stop_words:
                # Передаем правильную часть речи в лемматизатор
                wordnet_pos = get_wordnet_pos(tag)
                lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
                cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


print("Запуск глубокой очистки текста через NLTK (лемматизация с POS-тегами)...")
# Использование progress_apply покажет красивый прогресс-бар в консоли/ноутбуке
df[TEXT_COLUMN] = df[TEXT_COLUMN].progress_apply(clean_text_with_nltk)

#############################################
# DATA SPLIT & VECTORIZATION (ПОДГОТОВКА)
#############################################

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    df[TEXT_COLUMN],
    df[GENRE_COLUMN],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# Ограничиваем словарь до 20000 самых частых униграм и биграм
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)

# ВАЖНО: Оставляем матрицы разреженными (убрали .toarray()), чтобы не забить RAM
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)

# Переводим списки жанров в бинарную матрицу
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train_raw)
y_test_bin = mlb.transform(y_test_raw)


#############################################
# PYTORCH DATASET & DATALOADER
#############################################

class MovieDataset(Dataset):
    """Кастомный класс, конвертирующий разреженную матрицу Scipy в плотные тензоры НА ЛЕТУ"""

    def __init__(self, X_sparse, y_bin):
        self.X = X_sparse
        self.y = torch.tensor(y_bin, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Достаем одну строку из разреженной матрицы и превращаем её в плотный массив только в момент запроса батча
        x_dense = self.X[idx].toarray().squeeze()
        return torch.tensor(x_dense, dtype=torch.float32), self.y[idx]


# Создаем объекты датасетов
train_dataset = MovieDataset(X_train_tfidf, y_train_bin)
test_dataset = MovieDataset(X_test_tfidf, y_test_bin)

# Нарезка на батчи
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


#############################################
# NEURAL NETWORK ARCHITECTURE (МОДЕЛЬ)
#############################################

class MovieGenreClassifier(nn.Module):
    """Двухслойная полносвязная нейросеть (MLP)"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


INPUT_DIM = X_train_tfidf.shape[1]
NUM_CLASSES = y_train_bin.shape[1]

model = MovieGenreClassifier(INPUT_DIM, NUM_CLASSES).to(device)

#############################################
# LOSS & OPTIMIZER (ОШИБКА И ОПТИМИЗАТОР)
#############################################

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#############################################
# TRAINING LOOP (ЦИКЛ ОБУЧЕНИЯ)
#############################################

print("Старт обучения нейросети...")
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Эпоха [{epoch + 1}/{EPOCHS}] -> Ошибка (Loss): {epoch_loss:.4f}")

#############################################
# EVALUATION (ОЦЕНКА КАЧЕСТВА)
#############################################

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

print("\n=== Отчет классификации на тестовой выборке ===")
print(classification_report(all_true, all_preds, target_names=mlb.classes_, zero_division=0))


#############################################
# PREDICTION FUNCTION (ИНФЕРЕНС ДЛЯ НОВЫХ ТЕКСТОВ)
#############################################

def predict_genres(text):
    """Функция принимает сырой текст и возвращает список предсказанных жанров"""
    model.eval()
    with torch.no_grad():
        cleaned_text = clean_text_with_nltk(text)
        # Получаем разреженную строку, затем безопасно переводим в плотный вид для одной строки
        vectorized = tfidf.transform([cleaned_text]).toarray().squeeze()

        tensor_input = torch.tensor(vectorized, dtype=torch.float32).unsqueeze(0).to(device)

        outputs = model(tensor_input)
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()

        labels = mlb.inverse_transform(preds)
        return labels[0]


#############################################
# DEMO RUN (ДЕМОНСТРАЦИЯ)
#############################################

example_plot = """
A group of astronauts travel through space to save humanity from a dying Earth. 
They encounter strange anomalies, dangerous black holes, and distant unknown planets.
"""

print("\n=== Демонстрация предсказания ===")
print(f"Описание сюжета: {example_plot.strip()}")
print(f"Предсказанные жанры: {predict_genres(example_plot)}")

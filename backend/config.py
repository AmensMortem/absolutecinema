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

# CONFIGURATION
# Folder with the dataset - put it next to the script
DATASET_DIR = "IMDb Movie Genre Classification"
OVERVIEW_PATH = os.path.join(DATASET_DIR, "movies_overview.csv")
GENRES_PATH = os.path.join(DATASET_DIR, "movies_genres.csv")

TEXT_COLUMN = "overview"
GENRE_COLUMN = "genre_names"  # we will create it ourselves during the merge

TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# GPU -> MPS (Apple Silicon) -> CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
print(f"Device for training: {device}")

# LOADING AND MERGING DATA


import sys

for path in (OVERVIEW_PATH, GENRES_PATH):
    if not os.path.exists(path):
        print(
            f"\n[!] File not found: {os.path.abspath(path)}\n"
            f"Put the 'IMDb Movie Genre Classification' folder next to the script.\n"
        )
        sys.exit(1)

print("Loading data...")

# movies_genres.csv -> dictionary {id: name}
genres_df = pd.read_csv(GENRES_PATH)
id_to_name = dict(zip(genres_df["id"], genres_df["name"]))

# movies_overview.csv ->-> title, overview, genre_ids
overview_df = pd.read_csv(OVERVIEW_PATH)
overview_df = overview_df[["overview", "genre_ids"]].dropna()


# genre_ids - a string like "[18, 80]", parse and convert id -> names
def parse_genre_ids(x):
    try:
        ids = ast.literal_eval(x) if isinstance(x, str) else x
        return [id_to_name[i] for i in ids if i in id_to_name]
    except Exception:
        return []


overview_df[GENRE_COLUMN] = overview_df["genre_ids"].apply(parse_genre_ids)

# Remove lines without genres
overview_df = overview_df[overview_df[GENRE_COLUMN].map(len) > 0]

df = overview_df[[TEXT_COLUMN, GENRE_COLUMN]].reset_index(drop=True)
print(f"Movies loaded: {len(df)}")


# TEXT PREPROCESSING (basic only, without NLTK)
# TF-IDF handles tokenization and normalization itself


def clean_text(text: str) -> str:
    """Minimal cleaning: lowercase + remove extra spaces.
    TF-IDF handles punctuation and tokenization internally."""
    return str(text).lower().strip()


print("Cleaning texts...")
df[TEXT_COLUMN] = df[TEXT_COLUMN].progress_apply(clean_text)

# DATA PARTITIONING AND VECTORIZATION


X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    df[TEXT_COLUMN],
    df[GENRE_COLUMN],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# Uni- and bigrams, top 20,000 tokens
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)

# Leave sparse matrices - save RAM
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)

# Binary matrix of genres
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train_raw)
y_test_bin = mlb.transform(y_test_raw)

print(f"Training set: {X_train_tfidf.shape[0]} examples")
print(f"Test sample: {X_test_tfidf.shape[0]} examples")
print(f"Number of genres: {len(mlb.classes_)}")


# PYTORCH DATASET - converts sparse
# matrix into a dense tensor only for the required batch


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
test_dataset = MovieDataset(X_test_tfidf, y_test_bin)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# MODEL ARCHITECTURE - MLP (3 hidden layers)


class MovieGenreClassifier(nn.Module):
    """Multilayer perceptron for multi-label genre classification."""

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


INPUT_DIM = X_train_tfidf.shape[1]  # 20,000
NUM_CLASSES = y_train_bin.shape[1]

model = MovieGenreClassifier(INPUT_DIM, NUM_CLASSES).to(device)
print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

# LOSS FUNCTION AND OPTIMIZER


# BCEWithLogitsLoss = Sigmoid + Binary Cross-Entropy
# Suitable for multi-label tasks (each genre is an independent binary classifier)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Reduce LR by 2 times if the loss does not improve for 2 epochs in a row
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# LEARNING CYCLE


print("\nStarting training...")
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    scheduler.step(epoch_loss)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

metrics_df = pd.DataFrame({
    "epoch": range(1, len(train_losses) + 1),
    "loss": train_losses})
metrics_df.to_csv("training_history_fasttext.csv", index=False)
print("Training history saved in training_history.csv")

# QUALITY ASSESSMENT ON THE TEST


model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)

        # Threshold 0.5: if sigmoid(logit) > 0.5 - genre is assigned
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

print("\n=== Classification report (test set) ===")
print(classification_report(
    all_true, all_preds,
    target_names=mlb.classes_,
    zero_division=0
))


# INFERENCE - prediction for a new text


def predict_genres(text: str) -> tuple:
    """Accepts raw text -> returns a tuple of predicted genres."""
    model.eval()
    with torch.no_grad():
        cleaned = clean_text(text)
        vectorized = tfidf.transform([cleaned]).toarray().squeeze()

        tensor_in = torch.tensor(vectorized, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(tensor_in)
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()

        return mlb.inverse_transform(preds)[0]


# DEMONSTRATION


example = """
A group of astronauts travel through space to save humanity from a dying Earth.
They encounter strange anomalies, dangerous black holes, and distant unknown planets.
"""

print("\n=== Demo prediction ===")
print(f"Description: {example.strip()}")
print(f"Genres: {predict_genres(example)}")

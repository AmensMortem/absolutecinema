import os
import sys
import ast
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score  # <-- оставляем только здесь

from tqdm import tqdm

tqdm.pandas()

# CONFIGURATION
DATASET_DIR = "IMDb Movie Genre Classification"
OVERVIEW_PATH = os.path.join(DATASET_DIR, "movies_overview.csv")
GENRES_PATH = os.path.join(DATASET_DIR, "movies_genres.csv")

TEXT_COLUMN = "overview"
GENRE_COLUMN = "genre_names"  # we will create it ourselves during the merge

TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
THRESHOLD = 0.5

# GPU -> MPS (Apple Silicon) -> CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
print(f"Device for training: {device}")

# Download and data merge
for path in (OVERVIEW_PATH, GENRES_PATH):
    if not os.path.exists(path):
        print(
            f"\n[!] File not found: {os.path.abspath(path)}\n"
            f"Put the 'IMDb Movie Genre Classification' folder next to the script.\n"
        )
        sys.exit(1)

print("Downloading...")

genres_df = pd.read_csv(GENRES_PATH)  # movies_genres.csv -> dict {id: name}
id_to_name = dict(zip(genres_df["id"], genres_df["name"]))

overview_df = pd.read_csv(OVERVIEW_PATH)  # movies_overview.csv -> title, overview, genre_ids
overview_df = overview_df[["overview", "genre_ids"]].dropna()


# genre_ids — str: "[18, 80]", converting id -> name
def parse_genre_ids(x):
    try:
        ids = ast.literal_eval(x) if isinstance(x, str) else x
        return [id_to_name[i] for i in ids if i in id_to_name]
    except Exception:
        return []


overview_df[GENRE_COLUMN] = overview_df["genre_ids"].apply(parse_genre_ids)
overview_df = overview_df[overview_df[GENRE_COLUMN].map(len) > 0]  # Remove lines without genres
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
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    df[TEXT_COLUMN],
    df[GENRE_COLUMN],
    test_size=0.3,
    random_state=RANDOM_STATE)

X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE)

# Uni- and bigrams, top 20,000 tokens
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=50000,
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

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

val_dataset = MovieDataset(
    tfidf.transform(X_val_raw),
    mlb.transform(y_val_raw))

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


class MovieGenreClassifier(nn.Module):  # Model Architecture - MLP (3 hidden layers)
    """Multilayer perceptron for multi-label genre classification."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


INPUT_DIM = X_train_tfidf.shape[1]
NUM_CLASSES = y_train_bin.shape[1]

model = MovieGenreClassifier(INPUT_DIM, NUM_CLASSES).to(device)
print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

# LOSS FUNCTION AND OPTIMIZER
pos_counts = y_train_bin.sum(axis=0)
neg_counts = y_train_bin.shape[0] - pos_counts
pos_weight = neg_counts / (pos_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

# BCEWithLogitsLoss = Sigmoid + Binary Cross-Entropy
# Suitable for multi-label tasks (each genre is an independent binary classifier)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Reduce LR by 2 times if the loss does not improve for 2 epochs in a row
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

print("\nTraining...")

train_losses = []
train_f1_scores = []
train_accuracies = []  # Exact match accuracy (all genres are accurately guessed)

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for X_batch, y_batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        preds = (torch.sigmoid(outputs) > THRESHOLD).float()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            preds = (torch.sigmoid(outputs) > THRESHOLD).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    micro = f1_score(all_true, all_preds, average="micro")
    macro = f1_score(all_true, all_preds, average="macro")

    # Exact match accuracy: the string is considered correct only if all genres are correctly guessed
    all_preds_np = np.array(all_preds)
    all_true_np = np.array(all_true)
    accuracy = np.mean(np.all(all_preds_np == all_true_np, axis=1))

    train_f1_scores.append(micro)
    train_accuracies.append(accuracy)

    print(f"Val Micro F1: {micro:.4f}")
    print(f"Val Macro F1: {macro:.4f}")
    print(f"Val Accuracy: {accuracy:.4f} (exact match)")

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
    )
    scheduler.step(epoch_loss)

metrics_df = pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "loss": train_losses,
    "micro_f1": train_f1_scores,
    "accuracy": train_accuracies})

metrics_df.to_csv("training_history_TF_IDF.csv", index=False)

# SEARCHING FOR THE OPTIMUM THRESHOLD FOR VALIDATION
# Do it BEFORE the test evaluation to apply best_t on the test


val_probs = []
val_true = []

model.eval()
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)

        probs = torch.sigmoid(outputs).cpu().numpy()

        val_probs.extend(probs)
        val_true.extend(y_batch.numpy())

val_probs = np.array(val_probs)
val_true = np.array(val_true)

best_t = 0.5
best_f1 = 0

for t in np.arange(0.1, 0.6, 0.02):
    preds = (val_probs > t)
    f1 = f1_score(val_true, preds, average="micro")
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\n BEST THRESHOLD:", round(best_t, 2))
print(" BEST MICRO F1 (val):", round(best_f1, 4))

# THRESHOLD is used for training only (0.5 is mathematically correct for BCE).
# For inference and testing we always use the best_t found in validation.
# QUALITY ASSESSMENT ON THE TEST (with best_t)
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)

        preds = (torch.sigmoid(outputs) > best_t).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

print("\n=== Classification report (test sample, threshold=best_t) ===")
print(classification_report(
    all_true, all_preds,
    target_names=mlb.classes_,
    zero_division=0
))

micro = f1_score(all_true, all_preds, average="micro")
macro = f1_score(all_true, all_preds, average="macro")

print("Micro F1:", micro)
print("Macro F1:", macro)


def predict_genres(text: str) -> tuple:
    """Accepts raw text -> returns a tuple of predicted genres."""
    model.eval()
    with torch.no_grad():
        cleaned = clean_text(text)
        vectorized = tfidf.transform([cleaned]).toarray().squeeze()

        tensor_in = torch.tensor(vectorized, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(tensor_in)
        preds = (torch.sigmoid(outputs) > best_t).int().cpu().numpy()  # use best_t

        return mlb.inverse_transform(preds)[0]


example = """
A group of astronauts travel through space to save humanity from a dying Earth.
They encounter strange anomalies, dangerous black holes, and distant unknown planets.
"""

print("\n=== Demo prediction ===")
print(f"Describtion: {example.strip()}")
print(f"Genre: {predict_genres(example)}")

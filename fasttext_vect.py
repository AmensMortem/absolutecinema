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

# Suppressing sklearn warnings about empty classes
warnings.filterwarnings("ignore", category=UserWarning)

##############################################
# CONFIGURATION
##############################################

# Folder with the TMDB dataset - place it next to the script
DATASET_DIR = "./IMDb Movie Genre Classification"
OVERVIEW_PATH = os.path.join(DATASET_DIR, "movies_overview.csv")
GENRES_PATH = os.path.join(DATASET_DIR, "movies_genres.csv")

TEXT_COLUMN = "overview"
GENRE_COLUMN = "genre_names"

RANDOM_STATE = 42
BATCH_SIZE = 64  # dataset is small - smaller batches, more precisely gradients
EPOCHS = 60  # small dataset converges slower
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4  # L2 regularization - reduces overfitting
THRESHOLD = 0.5
PATIENCE = 10  # Early stopping: stop if val F1 does not increase N epochs
FASTTEXT_PATH = "cc.en.300.bin"  # path to the FastText file next to the script
FASTTEXT_DIM = 300  # dimension of FastText vectors

# GPU → MPS (Apple Silicon) → CPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device for training: {device}")

##############################################
# LOADING AND MERGING DATA (TMDB format)
##############################################

for path in (OVERVIEW_PATH, GENRES_PATH):
    if not os.path.exists(path):
        print(
            f"\n[!] File not found: {os.path.abspath(path)}\n"
            f"Put the 'IMDb Movie Genre Classification' folder next to the script.\n"
        )
        sys.exit(1)

print("Downloading...")

# movies_genres.csv → dict {id: name}
genres_df = pd.read_csv(GENRES_PATH)
id_to_name = dict(zip(genres_df["id"], genres_df["name"]))

# movies_overview.csv → overview, genre_ids
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

# Remove lines without genres
overview_df = overview_df[overview_df[GENRE_COLUMN].map(len) > 0]

df = overview_df[[TEXT_COLUMN, GENRE_COLUMN]].reset_index(drop=True)
print(f"Movies loaded: {len(df)}")


# TEXT PREPROCESSING
# For FastText it is important to remove punctuation -
# each word is looked up in the model dictionary

def clean_text(text: str) -> str:
    """Cleanup: lowercase, remove punctuation and extra spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("Cleaning texts...")
df[TEXT_COLUMN] = df[TEXT_COLUMN].progress_apply(clean_text)

##############################################
# DATA PARTITION
##############################################

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

# Binary matrix of genres
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train_raw)
y_val_bin = mlb.transform(y_val_raw)
y_test_bin = mlb.transform(y_test_raw)

print(f"Training set: {len(X_train_raw)} examples")
print(f"Test sample: {len(X_test_raw)} examples")
print(f"Number of genres: {len(mlb.classes_)}")

##############################################
# VECTORIZATION - FastText with TF-IDF weighting
# We build TF-IDF only for word weights (IDF) -
# we take the vectors themselves from FastText
# Important rare words ("heist", "dystopian") get more weight
##############################################

import fasttext

if not os.path.exists(FASTTEXT_PATH):
    print(f"\n[!] FastText file not found: {os.path.abspath(FASTTEXT_PATH)}")
    print("Download cc.en.300.bin from https://fasttext.cc/docs/en/crawl-vectors.html")
    sys.exit(1)

print(f"Loading FastText model ({FASTTEXT_PATH})... (may take a minute)")
ft_model = fasttext.load_model(FASTTEXT_PATH)
print("FastText loaded.")

# TF-IDF is only for IDF word weights, not as features
print("Building IDF weights...")
tfidf_for_weights = TfidfVectorizer(
    max_features=50000,
    min_df=2,
    sublinear_tf=True
)
tfidf_for_weights.fit(X_train_raw)
vocab = tfidf_for_weights.vocabulary_
idf = tfidf_for_weights.idf_


def text_to_weighted_vector(text: str) -> np.ndarray:
    """
    Weighted averaging of FastText vectors by TF-IDF word weights + L2-norm.
    Important words (high IDF) contribute more to the final vector.
    """
    words = text.split()
    if not words:
        return np.zeros(FASTTEXT_DIM, dtype=np.float32)

    vecs = []
    weights = []
    for w in words:
        vec = ft_model.get_word_vector(w)
        weight = idf[vocab[w]] if w in vocab else 1.0
        vecs.append(vec)
        weights.append(weight)

    vecs = np.array(vecs, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    weighted = np.average(vecs, axis=0, weights=weights)

    # L2-normalization - reduce to unit length
    norm = np.linalg.norm(weighted)
    if norm > 0:
        weighted /= norm

    return weighted


print("Vectorization of texts using FastText (weighted)...")
X_train = np.vstack(X_train_raw.progress_apply(text_to_weighted_vector).values)
X_val = np.vstack(X_val_raw.progress_apply(text_to_weighted_vector).values)
X_test = np.vstack(X_test_raw.progress_apply(text_to_weighted_vector).values)

print(f"Vector: {X_train.shape[1]}D")


##############################################
# PYTORCH DATASET
# FastText gives dense matrices (10k × 300) -
# convert to tensor immediately
##############################################

class MovieDataset(Dataset):
    def __init__(self, X_dense, y_bin):
        self.X = torch.tensor(X_dense, dtype=torch.float32)
        self.y = torch.tensor(y_bin, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = MovieDataset(X_train, y_train_bin)
val_dataset = MovieDataset(X_val, y_val_bin)
test_dataset = MovieDataset(X_test, y_test_bin)

# num_workers=0 on Windows (>0 causes problems with multiprocessing)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


##############################################
# Model Architecture - MLP with residual connection
# For FastText 300D input we use a deep network -
# residual connection allows gradients to flow directly
##############################################

class MovieGenreClassifier(nn.Module):
    """MLP with residual connection for multi-label genre classification."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        # Project the input to the size of the hidden layers
        self.input_proj = nn.Linear(input_dim, 512)

        self.block1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # BatchNorm speeds up convergence
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
        x = x + self.block1(x)  # residual connection: block output + input
        x = self.block2(x)
        return self.classifier(x)


INPUT_DIM = FASTTEXT_DIM
NUM_CLASSES = y_train_bin.shape[1]

model = MovieGenreClassifier(INPUT_DIM, NUM_CLASSES).to(device)
print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

##############################################
# LOSS FUNCTION AND OPTIMIZER
##############################################

pos_counts = y_train_bin.sum(axis=0)
neg_counts = y_train_bin.shape[0] - pos_counts
pos_weight = neg_counts / (pos_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

# BCEWithLogitsLoss = Sigmoid + Binary Cross-Entropy
# Suitable for multi-label tasks (each genre is an independent binary classifier)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AdamW is better than Adam - built-in weight decay works correctly
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Reduce LR by 2 times if val F1 does not improve for 2 epochs in a row
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5, verbose=True
)

# AMP - automatic mixed precision: saves VRAM by half, speeds up training on GPU
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler(enabled=use_amp)

##############################################
# Training
##############################################
print("\nTraining...")

train_losses = []
train_f1_scores = []
train_accuracies = []

best_val_f1 = 0.0
best_model_state = None
no_improve_epochs = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=False):
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(X_batch)
            preds = (torch.sigmoid(outputs) > THRESHOLD).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    micro = f1_score(all_true, all_preds, average="micro", zero_division=0)
    macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
    all_preds_np = np.array(all_preds)
    all_true_np = np.array(all_true)
    accuracy = np.mean(np.all(all_preds_np == all_true_np, axis=1))

    train_f1_scores.append(micro)
    train_accuracies.append(accuracy)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {epoch_loss:.4f} | "
          f"Val Micro F1: {micro:.4f} | Val Macro F1: {macro:.4f} | "
          f"Val Accuracy: {accuracy:.4f} (exact match)")

    scheduler.step(micro)

    # Early stopping - save the best model
    if micro > best_val_f1:
        best_val_f1 = micro
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1} - val F1 did not improve {PATIENCE} epochs.")
            break

# Load the best weights
if best_model_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    print(f"Best weights loaded (val Micro F1 = {best_val_f1:.4f})")

metrics_df = pd.DataFrame({
    "epoch": range(1, len(train_losses) + 1),
    "loss": train_losses,
    "micro_f1": train_f1_scores,
    "accuracy": train_accuracies
})
metrics_df.to_csv("training_history.csv", index=False)
print("Training history saved in training_history.csv")

##############################################
# SEARCHING FOR THE OPTIMUM THRESHOLD FOR VALIDATION
# Do it BEFORE the test evaluation to apply best_t on the test
##############################################

val_probs = []
val_true = []

model.eval()
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(X_batch)
        probs = torch.sigmoid(outputs).cpu().numpy()
        val_probs.extend(probs)
        val_true.extend(y_batch.numpy())

val_probs = np.array(val_probs)
val_true = np.array(val_true)

best_t = 0.5
best_f1 = 0.0

for t in np.arange(0.1, 0.6, 0.02):
    preds = (val_probs > t)
    f1 = f1_score(val_true, preds, average="micro", zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"\n BEST THRESHOLD: {round(best_t, 2)}")
print(f" BEST MICRO F1 (val): {round(best_f1, 4)}")

# THRESHOLD is used for training only (0.5 is mathematically correct for BCE).
# For inference and testing we always use the best_t found in validation.

##############################################
# QUALITY ASSESSMENT ON THE TEST (with best_t)
##############################################

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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

micro = f1_score(all_true, all_preds, average="micro", zero_division=0)
macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
print("Micro F1:", micro)
print("Macro F1:", macro)


##############################################
# INFERENCE - prediction for a new text
##############################################

def predict_genres(text: str) -> tuple:
    """Accepts raw text → returns a tuple of predicted genres."""
    model.eval()
    with torch.no_grad():
        cleaned = clean_text(text)
        # FastText weighted: the same pipeline as for training
        vec = text_to_weighted_vector(cleaned)
        tensor_in = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(tensor_in)
        preds = (torch.sigmoid(outputs) > best_t).int().cpu().numpy()
        return mlb.inverse_transform(preds)[0]


##############################################
# DEMONSTRATION
##############################################

example = """
A group of astronauts travel through space to save humanity from a dying Earth.
They encounter strange anomalies, dangerous black holes, and distant unknown planets.
"""

print("\n=== Demo prediction ===")
print(f"Describtion: {example.strip()}")
print(f"Genre: {predict_genres(example)}")

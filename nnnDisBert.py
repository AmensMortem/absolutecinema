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
# Configuration
#############################################

DATASET_DIR   = "IMDb Movie Genre Classification/"
OVERVIEW_PATH = "IMDb Movie Genre Classification/movies_overview.csv"
GENRES_PATH   = "IMDb Movie Genre Classification/movies_genres.csv"
TEXT_COLUMN  = "overview"
GENRE_COLUMN = "genre_names"

RANDOM_STATE  = 42
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
THRESHOLD = 0.5
PATIENCE = 7
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

#############################################
# Data downloading
#############################################

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

# deleting rows without genres
overview_df = overview_df[overview_df[GENRE_COLUMN].map(len) > 0]
df = overview_df[[TEXT_COLUMN, GENRE_COLUMN]].reset_index(drop=True)
print(f"Movies are downloaded: {len(df)}")

#############################################
# Splitting data
#############################################

X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    df[TEXT_COLUMN], df[GENRE_COLUMN], test_size=0.3, random_state=RANDOM_STATE
)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
)

# Genre Binary Matrix
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train_raw)
y_val_bin   = mlb.transform(y_val_raw)
y_test_bin  = mlb.transform(y_test_raw)

print(f"Train: {len(X_train_raw)} samples")
print(f"Test: {len(X_test_raw)} samoles")
print(f"Number of genres: {len(mlb.classes_)}")

#############################################
# DistilBERT Tokenization
#############################################

print("DistilBERT downloading")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
print("Success.")

class MovieBertDataset(Dataset):
    def __init__(self, texts, labels):
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

train_dataset = MovieBertDataset(X_train_raw, y_train_bin)
val_dataset = MovieBertDataset(X_val_raw, y_val_bin)
test_dataset = MovieBertDataset(X_test_raw, y_test_bin)

# num_workers=0 on Windows
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

#############################################
# Model
#############################################

class DistilBertGenreClassifier(nn.Module):
    """
    DistilBERT for multi-label genre classification.
    We use the [CLS] token as a representation of the entire text and pass it through a classifier.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # DistilBERT returns hidden states for each token
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token - first token, represents whole text
        cls_output = outputs.last_hidden_state[:, 0, :]

        return self.classifier(cls_output)

NUM_CLASSES = y_train_bin.shape[1]
model = DistilBertGenreClassifier(NUM_CLASSES).to(device)
print(f"\nМодель: {sum(p.numel() for p in model.parameters()):,} параметров")

#############################################
# Loss Function
#############################################

pos_counts = y_train_bin.sum(axis=0)
neg_counts = y_train_bin.shape[0] - pos_counts
pos_weight = neg_counts / (pos_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

# BCEWithLogitsLoss = Sigmoid + Binary Cross-Entropy
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AdamW
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Linear warmup + decay
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# AMP
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
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids, attention_mask)
            preds = (torch.sigmoid(outputs) > THRESHOLD).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch["labels"].numpy())

    micro = f1_score(all_true, all_preds, average="micro", zero_division=0)
    macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
    accuracy = np.mean(np.all(np.array(all_preds) == np.array(all_true), axis=1))

    train_f1_scores.append(micro)
    train_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | "
          f"Val Micro F1: {micro:.4f} | Val Macro F1: {macro:.4f} | "
          f"Val Accuracy: {accuracy:.4f} (exact match)")

    # Early stopping
    if micro > best_val_f1:
        best_val_f1      = micro
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= PATIENCE:
            print(f"\nEarly stopping на эпохе {epoch+1} — val F1 не улучшался {PATIENCE} эпох.")
            break

if best_model_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    print(f"Best weights are downloaded (val Micro F1 = {best_val_f1:.4f})")

metrics_df = pd.DataFrame({
    "epoch": range(1, len(train_losses) + 1),
    "loss": train_losses,
    "micro_f1": train_f1_scores,
    "accuracy": train_accuracies
})
metrics_df.to_csv("training_history.csv", index=False)

#############################################
# Search for optimal threshold on validation
#############################################

val_probs = []
val_true = []

model.eval()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_ids, attention_mask)
        val_probs.extend(torch.sigmoid(outputs).cpu().numpy())
        val_true.extend(batch["labels"].numpy())

val_probs = np.array(val_probs)
val_true = np.array(val_true)

# Threshold for Micro F1
best_t = 0.5
best_f1 = 0.0
for t in np.arange(0.1, 0.9, 0.02):
    f1 = f1_score(val_true, (val_probs > t), average="micro", zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

# Threshold Exact Match
best_t_acc = 0.5
best_acc = 0.0
for t in np.arange(0.1, 0.95, 0.02):
    acc = np.mean(np.all((val_probs > t) == val_true, axis=1))
    if acc > best_acc:
        best_acc = acc
        best_t_acc = t

# Threshold High Precision (recall >= 35%)
best_t_prec = 0.5
best_prec = 0.0
for t in np.arange(0.3, 0.95, 0.02):
    rec = recall_score(val_true, (val_probs > t), average="micro", zero_division=0)
    if rec < 0.35:
        break
    prec = precision_score(val_true, (val_probs > t), average="micro", zero_division=0)
    if prec > best_prec:
        best_prec = prec
        best_t_prec = t

print(f"\n BEST THRESHOLD (Micro F1): {round(best_t, 2)}  → F1={round(best_f1, 4)}")
print(f" BEST THRESHOLD (High Precision): {round(best_t_prec, 2)}  → Precision={round(best_prec, 4)}")
print(f" BEST THRESHOLD (Exact Match): {round(best_t_acc, 2)}  → Accuracy={round(best_acc*100, 1)}%")

#############################################
# Test
#############################################

def evaluate(threshold, label):
    preds_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids, attention_mask)
            preds_list.extend((torch.sigmoid(outputs) > threshold).cpu().numpy())
    p = np.array(preds_list)
    micro = f1_score(all_true_list, p, average="micro", zero_division=0)
    prec = precision_score(all_true_list, p, average="micro", zero_division=0)
    rec = recall_score(all_true_list, p, average="micro", zero_division=0)
    acc = np.mean(np.all(p == np.array(all_true_list), axis=1))
    print(f"\n=== {label} (порог={round(threshold,2)}) ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Micro F1: {micro:.4f}")
    print(f"Exact Match Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    return p

all_true_list = []
with torch.no_grad():
    for batch in test_loader:
        all_true_list.extend(batch["labels"].numpy())

all_preds_f1 = evaluate(best_t, "Режим Micro F1")
all_preds_prec = evaluate(best_t_prec, "Режим High Precision")
all_preds_acc = evaluate(best_t_acc, "Режим Exact Match")

print("\n=== Report (High Precision режим) ===")
print(classification_report(
    all_true_list, all_preds_prec,
    target_names=mlb.classes_,
    zero_division=0
))

#############################################
# Interface
#############################################

def predict_genres(text: str, mode: str = "precision") -> tuple:
    if mode == "f1":
        threshold = best_t
    elif mode == "accuracy":
        threshold = best_t_acc
    else:
        threshold = best_t_prec

    model.eval()
    with torch.no_grad():
        # DistilBERT
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
# Demonstration
#############################################

'''example = """
A group of astronauts travel through space to save humanity from a dying Earth.
They encounter strange anomalies, dangerous black holes, and distant unknown planets.
"""'''

example = "When a menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman, James Gordon and Harvey Dent must work together to put an end to the madness."

print("\n=== Demo-prediction ===")
print(f"Describtion: {example.strip()}")
print(f"F1 mode:  {predict_genres(example, mode='f1')}")
print(f"Precision mode: {predict_genres(example, mode='precision')}")
print(f"Accuracy mode: {predict_genres(example, mode='accuracy')}")

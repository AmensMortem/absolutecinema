"""
loads saved model and predicts genres for a given text
call it from front
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizerFast, DistilBertModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# arcitecture must be thae same with what we had before (need to be adopted to the model we'll use in the end)
class DistilBertGenreClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def mean_pool(self, last_hidden_state, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)


def load_model(load_dir: str):
    """
    Loads model, tokenizer, mlb and thresholds from load_dir.
    Returns: (model, tokenizer, mlb, thresholds, max_len)
    """
    required = ["model_weights.pt", "mlb_classes.npy", "thresholds.json", "config.json"]
    missing  = [f for f in required if not os.path.exists(os.path.join(load_dir, f))]
    if missing:
        print(f"[!] Missing files in {load_dir}: {missing}")
        sys.exit(1)

    with open(os.path.join(load_dir, "config.json")) as f:
        cfg = json.load(f)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    model = DistilBertGenreClassifier(cfg["num_classes"]).to(device)
    model.load_state_dict(torch.load(
        os.path.join(load_dir, "model_weights.pt"),
        map_location=device,
        weights_only=True
    ))
    model.eval()

    # return genres classes
    mlb = MultiLabelBinarizer()
    mlb.classes_ = np.load(
        os.path.join(load_dir, "mlb_classes.npy"), allow_pickle=True
    )

    with open(os.path.join(load_dir, "thresholds.json")) as f:
        thresholds = json.load(f)

    print(f"Model loaded from: {os.path.abspath(load_dir)}")
    print(f"  Genres: {cfg['num_classes']}  |  max_len: {cfg['max_len']}")
    print(f"  Thresholds: f1={thresholds['f1']}, "
          f"precision={thresholds['precision']}, accuracy={thresholds['accuracy']}")

    return model, tokenizer, mlb, thresholds, cfg["max_len"]


def predict(
    text:       str,
    model,
    tokenizer,
    mlb,
    thresholds: dict,
    max_len:    int,
    mode:       str = "precision",
) -> tuple:
    """
    Predicts genres for a single text description.

    mode:
        "precision" — high precision, fewer genres (default)
        "f1"        — balanced precision/recall
        "accuracy"  — only the most confident predictions
    """
    threshold = thresholds.get(mode, thresholds["precision"])

    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            str(text),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device)
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(int)

    return mlb.inverse_transform(preds)[0]
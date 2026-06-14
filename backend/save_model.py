"""
saves trained model, classes and thresholds to disk
"""

import os
import json
import numpy as np
import torch


def save_model(
        save_dir: str,
        model,
        mlb,
        best_t_prec: float,
        num_classes: int,
        max_len: int,
):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pt"))  # model weights
    np.save(os.path.join(save_dir, "mlb_classes.npy"), mlb.classes_)  # genre classes

    with open(os.path.join(save_dir, "thresholds.json"), "w") as f:
        json.dump({
            "precision": round(float(best_t_prec), 4),
        }, f, indent=2)
    with open(os.path.join(save_dir, "config.json"), "w") as f:  # architecture config
        json.dump({"num_classes": num_classes, "max_len": max_len}, f, indent=2)

    size_mb = os.path.getsize(os.path.join(save_dir, "model_weights.pt")) / 1024 / 1024
    print(f"Model saved to: {os.path.abspath(save_dir)}/")
    print(f"model_weights.pt  ({size_mb:.1f} MB)")
    print(f"mlb_classes.npy   ({num_classes} genres)")
    print(f"config.json")

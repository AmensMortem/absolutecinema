import pandas as pd
import os
import matplotlib.pyplot as plt

def draw(name):
    df = pd.read_csv(f"training_statistic/{name}")

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["loss"], marker="o")
    plt.title(f"Training Loss, {name.split('_')[2]}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["accuracy"], marker="o")
    plt.title(f"Training Accuracy, {name.split('_')[2]}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for file in os.listdir("./training_statistic"):
    print()
    draw(file)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_history.csv")

# Loss
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["loss"], marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["accuracy"], marker="o")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

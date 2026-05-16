#!/usr/bin/env python3
"""
Linear Regression on Diabetes dataset using PyTorch + AdamW.

Implements the pytorch_task_v1 protocol in a single self-contained file.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_metadata():
    return {
        "task_name": "linreg_lvl5_diabetes_adamw",
        "series": "Linear Regression",
        "task_type": "regression",
        "dataset": "sklearn.load_diabetes",
        "input_dim": 10,
        "output_dim": 1,
        "optimizer": "AdamW",
    }


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _standardize(train_x, val_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    return (train_x - mean) / std, (val_x - mean) / std


def _standardize_target(train_y, val_y):
    mean = train_y.mean(axis=0, keepdims=True)
    std = train_y.std(axis=0, keepdims=True) + 1e-8
    return (train_y - mean) / std, (val_y - mean) / std


def make_dataloaders(batch_size=32, train_ratio=0.8, seed=42):
    set_seed(seed)
    data = load_diabetes()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)

    indices = np.random.permutation(len(x))
    split = int(len(x) * train_ratio)
    train_idx, val_idx = indices[:split], indices[split:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_train, x_val = _standardize(x_train, x_val)
    y_train, y_val = _standardize_target(y_train, y_val)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, x_train, y_train, x_val, y_val


class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def build_model(input_dim, device):
    return LinearModel(input_dim).to(device)


def train(model, train_loader, val_loader, device, epochs=600, lr=0.005, weight_decay=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["mse"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | train_mse={train_loss:.3f} | val_mse={val_loss:.3f}")

    return history


def evaluate(model, data_loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            targets.append(yb.numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(targets)

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model(x_t).cpu().numpy()
    
def _get_series(history, keys):
    for k in keys:
        v = history.get(k)
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            return np.asarray(v, dtype=np.float32), k
    return None, None


def plot_training_curves(history, out_dir, prefix="training"):
    os.makedirs(out_dir, exist_ok=True)

    # Loss curve
    train_loss, train_loss_name = _get_series(history, ["train_loss", "loss_train"])
    val_loss, val_loss_name = _get_series(history, ["val_loss", "valid_loss", "loss_val"])

    if train_loss is not None or val_loss is not None:
        n = len(train_loss) if train_loss is not None else len(val_loss)
        epochs = np.arange(1, n + 1)
        plt.figure(figsize=(8, 5))
        if train_loss is not None:
            plt.plot(epochs, train_loss, label=train_loss_name)
        if val_loss is not None:
            plt.plot(epochs, val_loss, label=val_loss_name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training/Validation Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), dpi=150)
        plt.close()

    # Accuracy curve (if available)
    train_acc, train_acc_name = _get_series(history, ["train_acc", "train_accuracy"])
    val_acc, val_acc_name = _get_series(history, ["val_acc", "val_accuracy", "valid_accuracy"])

    if train_acc is not None or val_acc is not None:
        n = len(train_acc) if train_acc is not None else len(val_acc)
        epochs = np.arange(1, n + 1)
        plt.figure(figsize=(8, 5))
        if train_acc is not None:
            plt.plot(epochs, train_acc, label=train_acc_name)
        if val_acc is not None:
            plt.plot(epochs, val_acc, label=val_acc_name)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training/Validation Accuracy")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_accuracy.png"), dpi=150)
        plt.close()


def _to_serializable_history(history):
    serializable = {}
    for k, v in history.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.astype(float).tolist()
        elif isinstance(v, (list, tuple)):
            serializable[k] = [
                float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v
            ]
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        else:
            serializable[k] = v
    return serializable


def save_artifacts(model, history, train_metrics, val_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(_to_serializable_history(history), f, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_metrics, "validation": val_metrics}, f, indent=2)

    plot_training_curves(history, out_dir, prefix="training")


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    metadata = get_task_metadata()
    train_loader, val_loader, _, _, _, _ = make_dataloaders(batch_size=32, train_ratio=0.8, seed=42)
    model = build_model(input_dim=metadata["input_dim"], device=device)

    history = train(model, train_loader, val_loader, device, epochs=600, lr=0.005, weight_decay=1e-4)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("Train metrics:", train_metrics)
    print("Validation metrics:", val_metrics)

    quality_checks = [
        val_metrics["r2"] > 0.30,
        val_metrics["mse"] < 0.70,
        abs(train_metrics["r2"] - val_metrics["r2"]) < 0.20,
        history["train_loss"][-1] < history["train_loss"][0],
    ]

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    save_artifacts(model, history, train_metrics, val_metrics, out_dir)

    passed = all(quality_checks)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

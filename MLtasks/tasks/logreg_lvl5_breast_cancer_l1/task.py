#!/usr/bin/env python3
"""
Binary Logistic Regression on Breast Cancer dataset with L1 regularization.

Implements the pytorch_task_v1 protocol in one file.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_metadata():
    return {
        "task_name": "logreg_lvl5_breast_cancer_l1",
        "series": "Logistic Regression",
        "task_type": "binary_classification",
        "dataset": "sklearn.load_breast_cancer",
        "input_dim": 30,
        "output_dim": 1,
        "features": ["L1_regularization", "BCEWithLogitsLoss"],
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


def make_dataloaders(batch_size=32, train_ratio=0.8, seed=42):
    set_seed(seed)
    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)

    idx = np.random.permutation(len(x))
    split = int(len(x) * train_ratio)
    tr_idx, va_idx = idx[:split], idx[split:]

    x_train, y_train = x[tr_idx], y[tr_idx]
    x_val, y_val = x[va_idx], y[va_idx]

    x_train, x_val = _standardize(x_train, x_val)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, x_train, y_train, x_val, y_val


class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def build_model(input_dim, device):
    return LogisticModel(input_dim).to(device)


def train(model, train_loader, val_loader, device, epochs=220, lr=0.01, l1_lambda=2e-4):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            bce = criterion(logits, yb)
            l1_penalty = torch.abs(model.linear.weight).sum()
            loss = bce + l1_lambda * l1_penalty
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["bce_loss"])

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} "
                f"| val_bce={val_metrics['bce_loss']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
            )

    return history


def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    logits_all, y_all = [], []
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            logits_all.append(logits.cpu().numpy())
            y_all.append(yb.cpu().numpy())

    logits_np = np.vstack(logits_all)
    y_true = np.vstack(y_all)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
    y_pred = (probs >= 0.5).astype(np.float32)

    mse = float(mean_squared_error(y_true, probs))
    r2 = float(r2_score(y_true, probs))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, probs))

    return {
        "mse": mse,
        "r2": r2,
        "bce_loss": float(total_loss / len(data_loader)),
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        logits = model(x_t)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()


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
    model = build_model(metadata["input_dim"], device)

    history = train(model, train_loader, val_loader, device, epochs=220, lr=0.01, l1_lambda=2e-4)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("Train metrics:", train_metrics)
    print("Validation metrics:", val_metrics)

    checks = [
        val_metrics["accuracy"] > 0.93,
        val_metrics["f1"] > 0.93,
        val_metrics["auc"] > 0.97,
        abs(train_metrics["accuracy"] - val_metrics["accuracy"]) < 0.08,
    ]

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    save_artifacts(model, history, train_metrics, val_metrics, out_dir)

    passed = all(checks)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Multiclass Logistic Regression on Iris with label smoothing and scheduler.

Implements the pytorch_task_v1 protocol in one file.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_metadata():
    return {
        "task_name": "logreg_lvl6_iris_label_smoothing",
        "series": "Logistic Regression",
        "task_type": "multiclass_classification",
        "dataset": "sklearn.load_iris",
        "input_dim": 4,
        "num_classes": 3,
        "features": ["label_smoothing", "cosine_scheduler"],
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


def make_dataloaders(batch_size=16, train_ratio=0.8, seed=42):
    set_seed(seed)
    data = load_iris()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

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


class SoftmaxLogReg(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def build_model(input_dim, num_classes, device):
    return SoftmaxLogReg(input_dim, num_classes).to(device)


def train(model, train_loader, val_loader, device, epochs=180, lr=0.04):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        train_loss = running / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["ce_loss"])

        if (epoch + 1) % 40 == 0:
            print(
                f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} "
                f"| val_ce={val_metrics['ce_loss']:.4f} | val_f1={val_metrics['f1_macro']:.4f}"
            )

    return history


def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    logits_all, y_all = [], []
    total_ce = 0.0

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_ce += criterion(logits, yb).item()
            logits_all.append(logits.cpu().numpy())
            y_all.append(yb.cpu().numpy())

    logits_np = np.vstack(logits_all)
    y_true = np.concatenate(y_all)
    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    y_pred = probs.argmax(axis=1)

    y_true_oh = np.eye(probs.shape[1], dtype=np.float32)[y_true]
    mse = float(mean_squared_error(y_true_oh, probs))
    r2 = float(r2_score(y_true_oh.reshape(-1), probs.reshape(-1)))

    return {
        "mse": mse,
        "r2": r2,
        "ce_loss": float(total_ce / len(data_loader)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        probs = torch.softmax(model(x_t), dim=1)
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
    train_loader, val_loader, _, _, _, _ = make_dataloaders(batch_size=16, train_ratio=0.8, seed=42)
    model = build_model(metadata["input_dim"], metadata["num_classes"], device)

    history = train(model, train_loader, val_loader, device, epochs=180, lr=0.04)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("Train metrics:", train_metrics)
    print("Validation metrics:", val_metrics)

    checks = [
        val_metrics["accuracy"] > 0.90,
        val_metrics["f1_macro"] > 0.90,
        abs(train_metrics["accuracy"] - val_metrics["accuracy"]) < 0.12,
        history["train_loss"][-1] < history["train_loss"][0],
    ]

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    save_artifacts(model, history, train_metrics, val_metrics, out_dir)

    passed = all(checks)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

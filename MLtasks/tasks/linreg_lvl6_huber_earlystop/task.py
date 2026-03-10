#!/usr/bin/env python3
"""
Robust Linear Regression with outliers using PyTorch Huber loss + early stopping.

Implements the pytorch_task_v1 protocol in one file.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_metadata():
    return {
        "task_name": "linreg_lvl6_huber_earlystop",
        "series": "Linear Regression",
        "task_type": "regression",
        "dataset": "synthetic_outlier_regression",
        "input_dim": 6,
        "output_dim": 1,
        "loss": "HuberLoss",
        "features": ["early_stopping", "gradient_clipping"],
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


def make_dataloaders(n_samples=1200, train_ratio=0.8, batch_size=64, seed=42):
    set_seed(seed)

    x = np.random.randn(n_samples, 6).astype(np.float32)
    true_w = np.array([1.8, -2.2, 0.7, 1.1, -0.4, 2.5], dtype=np.float32)
    y = x @ true_w + 1.2 + np.random.randn(n_samples).astype(np.float32) * 0.5

    outlier_mask = np.random.rand(n_samples) < 0.1
    y[outlier_mask] += np.random.randn(outlier_mask.sum()).astype(np.float32) * 15.0

    y = y.reshape(-1, 1)

    idx = np.random.permutation(n_samples)
    split = int(n_samples * train_ratio)
    tr_idx, va_idx = idx[:split], idx[split:]

    x_train, y_train = x[tr_idx], y[tr_idx]
    x_val, y_val = x[va_idx], y[va_idx]

    x_train, x_val = _standardize(x_train, x_val)

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


def train(model, train_loader, val_loader, device, epochs=300, lr=0.02, patience=35):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_running += loss.item()

        train_loss = train_running / len(train_loader)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_running += criterion(model(xb), yb).item()
        val_loss = val_running / len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | train_huber={train_loss:.3f} | val_huber={val_loss:.3f}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_huber"] = best_val
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
    train_loader, val_loader, _, _, _, _ = make_dataloaders(batch_size=64, train_ratio=0.8, seed=42)
    model = build_model(metadata["input_dim"], device)

    history = train(model, train_loader, val_loader, device, epochs=300, lr=0.02, patience=35)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("Train metrics:", train_metrics)
    print("Validation metrics:", val_metrics)

    checks = [
        val_metrics["r2"] > 0.45,
        val_metrics["mse"] < 30.0,
        abs(train_metrics["r2"] - val_metrics["r2"]) < 0.25,
        history["train_loss"][-1] < history["train_loss"][0],
    ]

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    save_artifacts(model, history, train_metrics, val_metrics, out_dir)

    passed = all(checks)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

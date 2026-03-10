"""
Temporal Convolutional Network (TCN) for Sequence Classification

Mathematical Formulation:
- Dilated causal convolution: y[t] = sum_{k=0}^{K-1} w[k] * x[t - d*k]
  where d = 2^layer is the dilation factor and K is the kernel size.
- Residual block: out = Activation(Conv1d_dilated(x)) + downsample(x)
- Classification head: GlobalAvgPool -> Linear -> Softmax

Dataset: Synthetic 4-class time series. Each class is a sinusoid at a
distinct frequency with additive Gaussian noise.
  class 0: f=1 Hz,  class 1: f=2 Hz,  class 2: f=4 Hz,  class 3: f=8 Hz
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, Tuple, List

torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": "tcn_lvl1_sequence_cls",
        "series": "Temporal Convolutional Networks",
        "level": 1,
        "description": "TCN with dilated causal convolutions for 4-class synthetic time series classification",
        "input_type": "time_series",
        "output_type": "class_label",
        "metrics": ["accuracy", "loss"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_time_series(
    n_per_class: int = 500,
    seq_len: int = 128,
    n_classes: int = 4,
    noise_std: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sinusoidal time series with 4 frequency classes."""
    rng = np.random.default_rng(seed)
    freqs = [1.0, 2.0, 4.0, 8.0]
    t = np.linspace(0, 1, seq_len, dtype=np.float32)

    X_list, y_list = [], []
    for cls, freq in enumerate(freqs[:n_classes]):
        phase = rng.uniform(0, 2 * np.pi, size=(n_per_class,))
        signals = np.sin(2 * np.pi * freq * t[None, :] + phase[:, None])
        signals += rng.normal(0, noise_std, size=signals.shape).astype(np.float32)
        X_list.append(signals[:, None, :])
        y_list.append(np.full(n_per_class, cls, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    perm = rng.permutation(len(y))
    return X[perm].astype(np.float32), y[perm]


def make_dataloaders(
    n_per_class: int = 500,
    seq_len: int = 128,
    batch_size: int = 64,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Returns:
        train_loader, val_loader, n_channels (1), n_classes (4)
    """
    X, y = _generate_time_series(n_per_class=n_per_class, seq_len=seq_len)

    n_train = int(len(X) * train_ratio)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, 1, 4


class CausalConv1d(nn.Module):
    """Causal convolution: pads only on the left to avoid future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualTCNBlock(nn.Module):
    def __init__(self, n_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.BatchNorm1d(n_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.BatchNorm1d(n_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x)


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilations."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 4,
        n_filters: int = 32,
        kernel_size: int = 3,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, n_filters, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualTCNBlock(n_filters, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])
        self.classifier = nn.Linear(n_filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1)
        return self.classifier(x)


def build_model(
    in_channels: int = 1,
    n_classes: int = 4,
    device: torch.device = None,
) -> TCN:
    if device is None:
        device = get_device()
    model = TCN(in_channels=in_channels, n_classes=n_classes, n_filters=32, kernel_size=3, n_layers=4)
    return model.to(device)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / n_batches

        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"Train Loss: {avg_train_loss:.4f}  "
                  f"Val Loss: {val_metrics['loss']:.4f}  "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")

    return {"history": history}


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y_batch.cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    return {
        "loss": total_loss / len(data_loader),
        "accuracy": float(acc),
        "predictions": np.array(all_preds),
        "targets": np.array(all_targets),
    }


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Return predicted class labels for input x of shape (N, 1, T)."""
    model.eval()
    x_t = torch.FloatTensor(x).to(device)
    with torch.no_grad():
        logits = model(x_t)
    return logits.argmax(dim=1).cpu().numpy()


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    save_dir: str = "output",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "tcn_model.pth"))
    metrics_clean = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in metrics.items()
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"Artifacts saved to {save_dir}")


def main() -> int:
    print("=" * 60)
    print("TCN Sequence Classification on Synthetic Time Series")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, n_ch, n_cls = make_dataloaders(
        n_per_class=500, seq_len=128, batch_size=64
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Classes: {n_cls},  Input channels: {n_ch}")

    model = build_model(in_channels=n_ch, n_classes=n_cls, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(model)

    train_result = train(model, train_loader, val_loader, device, epochs=50, lr=1e-3)

    print("\nEvaluating on train set...")
    train_metrics = evaluate(model, train_loader, device)
    print("\nEvaluating on val set...")
    val_metrics = evaluate(model, val_loader, device)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Val Accuracy:   {val_metrics['accuracy']:.4f}")
    print(f"  Train Loss:     {train_metrics['loss']:.4f}")
    print(f"  Val Loss:       {val_metrics['loss']:.4f}")

    print("\nClassification Report (val):")
    print(classification_report(val_metrics["targets"], val_metrics["predictions"],
                                 target_names=[f"class_{i}" for i in range(n_cls)]))

    save_dir = "output/tcn_lvl1_sequence_cls"
    history = train_result["history"]
    all_metrics = {
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "train_loss_history": history["train_loss"],
        "val_accuracy_history": history["val_accuracy"],
    }
    save_artifacts(model, all_metrics, save_dir)

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    check1 = val_metrics["accuracy"] > 0.85
    print(f"  {'PASS' if check1 else 'FAIL'} Val accuracy > 0.85: {val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check1

    h = history["train_loss"]
    check2 = h[-1] < h[0]
    print(f"  {'PASS' if check2 else 'FAIL'} Train loss decreased: {h[0]:.4f} -> {h[-1]:.4f}")
    checks_passed = checks_passed and check2

    check3 = val_metrics["loss"] < 0.5
    print(f"  {'PASS' if check3 else 'FAIL'} Val loss < 0.5: {val_metrics['loss']:.4f}")
    checks_passed = checks_passed and check3

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())

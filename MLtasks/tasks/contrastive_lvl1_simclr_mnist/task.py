"""
SimCLR Contrastive Learning on MNIST

Mathematical Formulation:
- NT-Xent Loss: L = -log(exp(sim(z_i, z_j)/tau) / sum_{k != i}(exp(sim(z_i, z_k)/tau)))
- Cosine similarity: sim(u, v) = u^T v / (||u|| ||v||)
- Encoder f: x -> h; Projection head g: h -> z; Contrastive loss on z

Self-supervised: labels used only for kNN probe evaluation, not training.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from typing import Dict, Any, Tuple

torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": "contrastive_lvl1_simclr_mnist",
        "series": "Contrastive Learning",
        "level": 1,
        "description": "SimCLR self-supervised contrastive learning on MNIST with NT-Xent loss",
        "input_type": "image",
        "output_type": "embedding",
        "metrics": ["nt_xent_loss", "knn_probe_accuracy"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoViewTransform:
    """Produces two randomly augmented views of the same image."""

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SingleViewTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def __call__(self, x):
        return self.transform(x)


def make_dataloaders(
    batch_size: int = 256,
    num_workers: int = 2,
    n_train: int = 10000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns:
        contrastive_loader: two-view augmented loader for SimCLR training (no labels used)
        probe_train_loader: single-view labeled subset for kNN probe fitting
        probe_val_loader:   single-view labeled test set for kNN probe evaluation
    """
    contrastive_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=TwoViewTransform()
    )
    indices = list(range(n_train))
    contrastive_subset = torch.utils.data.Subset(contrastive_dataset, indices)
    contrastive_loader = DataLoader(
        contrastive_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )

    single_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=SingleViewTransform()
    )
    probe_train_loader = DataLoader(
        torch.utils.data.Subset(single_train, indices),
        batch_size=256, shuffle=False, num_workers=num_workers
    )

    single_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=SingleViewTransform()
    )
    probe_val_loader = DataLoader(
        single_test, batch_size=256, shuffle=False, num_workers=num_workers
    )

    return contrastive_loader, probe_train_loader, probe_val_loader


class Encoder(nn.Module):
    """Small CNN encoder for 28x28 grayscale images."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        h = self.backbone(x).squeeze(-1).squeeze(-1)
        z = self.projector(h)
        return h, z


def build_model(embed_dim: int = 128, device: torch.device = None) -> Encoder:
    if device is None:
        device = get_device()
    model = Encoder(embed_dim=embed_dim).to(device)
    return model


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent loss for a batch of paired embeddings."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature

    # Mask out self-similarities on diagonal
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def train(
    model: nn.Module,
    contrastive_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    temperature: float = 0.5,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (x1, x2), _ in contrastive_loader:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent_loss(z1, z2, temperature)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  NT-Xent Loss: {avg_loss:.4f}  LR: {optimizer.param_groups[0]['lr']:.6f}")

    return {"history": history, "final_loss": history["loss"][-1]}


def _extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            h, _ = model(x.to(device))
            feats.append(h.cpu().numpy())
            labels.append(y.numpy())
    return np.vstack(feats), np.concatenate(labels)


def evaluate(
    model: nn.Module,
    probe_train_loader: DataLoader,
    probe_val_loader: DataLoader,
    device: torch.device,
    history: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Fit a kNN probe on frozen encoder features; report accuracy."""
    train_feats, train_labels = _extract_features(model, probe_train_loader, device)
    val_feats, val_labels = _extract_features(model, probe_val_loader, device)

    train_feats = normalize(train_feats)
    val_feats = normalize(val_feats)

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
    knn.fit(train_feats, train_labels)
    knn_acc = knn.score(val_feats, val_labels)

    final_loss = history["loss"][-1] if history else None
    initial_loss = history["loss"][0] if history else None

    return {
        "knn_probe_accuracy": float(knn_acc),
        "final_nt_xent_loss": float(final_loss) if final_loss is not None else None,
        "initial_nt_xent_loss": float(initial_loss) if initial_loss is not None else None,
        "loss_decreased": bool(final_loss < initial_loss) if (final_loss and initial_loss) else None,
    }


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Return encoder embeddings for input array x (N, 1, 28, 28)."""
    model.eval()
    x_t = torch.FloatTensor(x).to(device)
    with torch.no_grad():
        h, _ = model(x_t)
    return h.cpu().numpy()


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    save_dir: str = "output",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "encoder.pth"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {save_dir}")


def main() -> int:
    print("=" * 60)
    print("SimCLR Contrastive Learning on MNIST")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    contrastive_loader, probe_train_loader, probe_val_loader = make_dataloaders(
        batch_size=256, n_train=10000
    )
    print(f"Contrastive batches: {len(contrastive_loader)}")

    model = build_model(embed_dim=128, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    train_result = train(
        model, contrastive_loader, device,
        epochs=20, lr=3e-4, temperature=0.5
    )

    metrics = evaluate(
        model, probe_train_loader, probe_val_loader, device,
        history=train_result["history"]
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  NT-Xent Loss (initial): {metrics['initial_nt_xent_loss']:.4f}")
    print(f"  NT-Xent Loss (final):   {metrics['final_nt_xent_loss']:.4f}")
    print(f"  kNN Probe Accuracy:     {metrics['knn_probe_accuracy']:.4f}")

    save_dir = "output/contrastive_lvl1_simclr_mnist"
    all_metrics = {**metrics, "loss_history": train_result["history"]["loss"]}
    save_artifacts(model, all_metrics, save_dir)

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    check1 = metrics["loss_decreased"]
    print(f"  {'PASS' if check1 else 'FAIL'} NT-Xent loss decreased: "
          f"{metrics['initial_nt_xent_loss']:.4f} -> {metrics['final_nt_xent_loss']:.4f}")
    checks_passed = checks_passed and check1

    check2 = metrics["knn_probe_accuracy"] > 0.85
    print(f"  {'PASS' if check2 else 'FAIL'} kNN probe accuracy > 0.85: {metrics['knn_probe_accuracy']:.4f}")
    checks_passed = checks_passed and check2

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())

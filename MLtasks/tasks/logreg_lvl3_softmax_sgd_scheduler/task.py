"""
Softmax Regression (Multiclass) using PyTorch
Implements multiclass classification with CrossEntropyLoss
New optimization: SGD with momentum + StepLR scheduler
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata():
    return {
        "task_name": "softmax_regression_multiclass_sgd_scheduler",
        "task_type": "classification",
        "num_classes": 3,
        "input_dim": 2,
        "description": "Multiclass softmax regression using torch.nn.Module + CrossEntropyLoss + SGD with momentum + StepLR"
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(n_samples=1000, n_features=2, n_classes=3,
                     train_ratio=0.8, batch_size=32, random_state=42):
    """Create dataloaders for the multiclass classification task."""
    set_seed(random_state)

    # Generate 3-class blob data with some overlap
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        cluster_std=1.4,
        n_features=n_features,
        random_state=random_state
    )

    #Split into training and validations sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1-train_ratio, random_state=random_state, stratify=y
    )

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


class SoftmaxRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def build_model(input_dim, num_classes, device):
    """Build SoftmaxRegressionModel"""
    return SoftmaxRegressionModel(input_dim, num_classes).to(device)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100, verbose=True):
    """Train the model."""
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {history['lr'][-1]:.5f}"
            )

    return history


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model and return dict of loss, accuracy, and F1 metrics."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro')

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro
    }


def save_artifacts(model, history, metrics, output_dir="output", filename_prefix="softmax_lvl3_sgd_scheduler"):
    """Save model artifacts and visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, f"{filename_prefix}_model.pt"))

    with open(os.path.join(output_dir, f"{filename_prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_accuracy.png"), dpi=150)
    plt.close()

    print(f"Artifacts saved to {output_dir}")


def main():
    print("=" * 60)
    print("Softmax Regression (Multiclass) - SGD + Momentum + Scheduler")
    print("=" * 60)

    device = get_device()
    print(f"\nUsing device: {device}")

    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Input dimension: {metadata['input_dim']}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=1000,
        n_features=2,
        n_classes=3,
        train_ratio=0.8,
        batch_size=32,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    print("\nBuilding model...")
    model = build_model(metadata["input_dim"], metadata["num_classes"], device)
    print(f"Model architecture: {model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print("\nTraining model...")
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100, verbose=True)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, criterion, device)
    print(train_metrics)

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion, device)
    print(val_metrics)

    print("\nSaving artifacts...")
    all_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "metadata": metadata
    }
    save_artifacts(model, history, all_metrics, output_dir="output")

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    quality_passed = True

    check1 = train_metrics["accuracy"] > 0.80
    print(f"{'✓' if check1 else '✗'} Train Accuracy > 0.80: {train_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check1

    check2 = val_metrics["accuracy"] > 0.78
    print(f"{'✓' if check2 else '✗'} Val Accuracy > 0.78: {val_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check2

    check3 = val_metrics["f1_macro"] > 0.78
    print(f"{'✓' if check3 else '✗'} Val F1 Macro > 0.78: {val_metrics['f1_macro']:.4f}")
    quality_passed = quality_passed and check3

    check4 = history["train_loss"][-1] < history["train_loss"][0]
    print(f"{'✓' if check4 else '✗'} Train loss decreased: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    quality_passed = quality_passed and check4

    gap = abs(train_metrics["accuracy"] - val_metrics["accuracy"])
    check5 = gap < 0.15
    print(f"{'✓' if check5 else '✗'} Accuracy gap < 0.15: {gap:.4f}")
    quality_passed = quality_passed and check5

    print("\n" + "=" * 60)
    print("PASS: All quality checks passed!" if quality_passed else "FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if quality_passed else 1


if __name__ == "__main__":
    sys.exit(main())
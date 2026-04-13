"""
CNN Task: RGB Image Classification
Implements a PyTorch CNN for classifying synthetic RGB images.

New task based off cnn_lvl1_from_scratch_conv. Implements new dataset (synthetic rgb images) and new method.
Key changes include new dataset, updated model with 3 channel inputs instead of 1 for rgb images and simplified metrics.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, List

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = "/Developer/AIserver/output/tasks/cnn_rgb_image_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "cnn_rgb_image_classification",
        "description": "CNN classification task using synthetic RGB images",
        "input_type": "image_rgb",
        "output_type": "classification",
        "model_type": "CNN",
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get computation device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_rgb_image_dataset(
    num_samples: int, img_size: int = 32, num_classes: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic RGB image dataset with learnable class patterns.

    Class meanings:
    0 -> Red-dominant images
    1 -> Green-dominant images
    2 -> Blue-dominant images
    3 -> Bright mixed-color images

    Returns:
        X: Tensor of shape (num_samples, 3, img_size, img_size)
        y: Tensor of shape (num_samples,)
    """
    X = torch.zeros(num_samples, 3, img_size, img_size, dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    for i in range(num_samples):
        label = y[i].item()

        # Small random background noise
        image = torch.rand(3, img_size, img_size) * 0.15

        if label == 0:
            # Red dominant
            image[0] += 0.75
            image[1] += 0.10
            image[2] += 0.10
        elif label == 1:
            # Green dominant
            image[0] += 0.10
            image[1] += 0.75
            image[2] += 0.10
        elif label == 2:
            # Blue dominant
            image[0] += 0.10
            image[1] += 0.10
            image[2] += 0.75
        else:
            # Bright mixed-color
            image += 0.55

        # Add a small random square patch to introduce spatial variety
        patch_size = 6
        x_start = np.random.randint(0, img_size - patch_size + 1)
        y_start = np.random.randint(0, img_size - patch_size + 1)

        if label == 0:
            image[
                0, y_start : y_start + patch_size, x_start : x_start + patch_size
            ] += 0.15
        elif label == 1:
            image[
                1, y_start : y_start + patch_size, x_start : x_start + patch_size
            ] += 0.15
        elif label == 2:
            image[
                2, y_start : y_start + patch_size, x_start : x_start + patch_size
            ] += 0.15
        else:
            image[
                :, y_start : y_start + patch_size, x_start : x_start + patch_size
            ] += 0.10

        image = torch.clamp(image, 0.0, 1.0)
        X[i] = image

    return X, y


def make_dataloaders(
    batch_size: int = 64,
    num_samples_train: int = 800,
    num_samples_val: int = 200,
    num_classes: int = 4,
    img_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    # Generate synthetic image data
    X_train, y_train = generate_rgb_image_dataset(
        num_samples=num_samples_train, img_size=img_size, num_classes=num_classes
    )

    X_val, y_val = generate_rgb_image_dataset(
        num_samples=num_samples_val, img_size=img_size, num_classes=num_classes
    )

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class RGBImageCNN(nn.Module):
    """
    Simple CNN for RGB image classification.
    Accepts 3-channel input images.
    """

    def __init__(self, num_classes: int = 4):
        super(RGBImageCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(device: torch.device) -> nn.Module:
    """Build and return the RGB CNN model."""
    model = RGBImageCNN(num_classes=4)
    model = model.to(device)
    return model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 6,
    lr: float = 0.001,
) -> List[float]:
    """Train the model and return loss history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for data, target in train_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_history


def evaluate(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Calculate metrics
            total_loss += loss.item()
            _, predicted = torch.max(output, dim=1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    return metrics


def predict(model: nn.Module, data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Make predictions on input data."""
    model.eval()

    with torch.no_grad():
        # Ensure data is on the correct device
        data = data.to(device)
        output = model(data)
        _, predicted = torch.max(output, dim=1)

    return predicted


def save_artifacts(
    model: nn.Module,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    loss_history: List[float],
) -> None:
    """Save model artifacts and evaluation results."""
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Task Metadata:\n")
        metadata = get_task_metadata()
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTraining Metrics:\n")
        for key, value in train_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nValidation Metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value}\n")

    # Save loss history
    loss_path = os.path.join(OUTPUT_DIR, "loss_history.txt")
    with open(loss_path, "w") as f:
        f.write("Loss History:\n")
        for i, loss in enumerate(loss_history):
            f.write(f"Epoch {i + 1}: {loss}\n")

    # Save model architecture info
    arch_path = os.path.join(OUTPUT_DIR, "architecture.txt")
    with open(arch_path, "w") as f:
        f.write("Model Architecture:\n")
        f.write(str(model))
        f.write("\n\n")
        f.write("This model accepts RGB images with shape (3, 32, 32).\n")


def main() -> int:
    """Main function to run the RGB image classification task."""
    print("=" * 60)
    print("CNN Task: RGB Image Classification")
    print("=" * 60)

    set_seed(42)

    metadata = get_task_metadata()
    print("Task Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    device = get_device()
    print(f"\nUsing device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("\nBuilding model...")
    model = build_model(device)
    print(model)

    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    loss_history = train(model, train_loader, device, epochs=6, lr=0.001)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Training Loss: {train_metrics['loss']:.4f}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")

    print("\nRunning sample prediction test...")
    sample_batch, sample_labels = next(iter(val_loader))
    sample_preds = predict(model, sample_batch[:8], device)
    print(f"Sample Predictions: {sample_preds.cpu().tolist()}")
    print(f"Ground Truth:       {sample_labels[:8].cpu().tolist()}")

    print("\n" + "=" * 60)
    print("Quality Thresholds Check")
    print("=" * 60)

    min_train_accuracy = 0.95
    min_val_accuracy = 0.90
    max_val_loss = 0.40

    train_pass = train_metrics["accuracy"] >= min_train_accuracy
    val_pass = val_metrics["accuracy"] >= min_val_accuracy
    loss_pass = val_metrics["loss"] <= max_val_loss

    print(
        f"Training accuracy >= {min_train_accuracy}: "
        f"{'PASS' if train_pass else 'FAIL'} ({train_metrics['accuracy']:.4f})"
    )
    print(
        f"Validation accuracy >= {min_val_accuracy}: "
        f"{'PASS' if val_pass else 'FAIL'} ({val_metrics['accuracy']:.4f})"
    )
    print(
        f"Validation loss <= {max_val_loss}: "
        f"{'PASS' if loss_pass else 'FAIL'} ({val_metrics['loss']:.4f})"
    )

    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    save_artifacts(model, train_metrics, val_metrics, loss_history)
    print(f"Artifacts saved to {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_pass = train_pass and val_pass and loss_pass

    if all_pass:
        print("✓ ALL TESTS PASSED")
        print(f"  - Training accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  - Validation accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  - Validation loss: {val_metrics['loss']:.4f}")
        print("  - Model artifacts saved")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        if not train_pass:
            print("  - Training accuracy below threshold")
        if not val_pass:
            print("  - Validation accuracy below threshold")
        if not loss_pass:
            print("  - Validation loss above threshold")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

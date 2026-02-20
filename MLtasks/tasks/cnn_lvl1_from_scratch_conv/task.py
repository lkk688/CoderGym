"""
CNN Task: Manual Conv2D Implementation (Educational)
Implements minimal Conv2D forward pass to understand convolution
and compares to torch.nn.Conv2d output.
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
OUTPUT_DIR = "/Developer/AIserver/output/tasks/cnn_lvl1_from_scratch_conv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "cnn_lvl1_from_scratch_conv",
        "description": "Manual Conv2D implementation for educational purposes",
        "input_type": "image",
        "output_type": "classification",
        "model_type": "CNN"
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

def make_dataloaders(
    batch_size: int = 64,
    num_samples_train: int = 500,
    num_samples_val: int = 100,
    num_classes: int = 10,
    img_size: int = 28,
    num_channels: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    # Generate synthetic image data
    X_train = torch.randn(num_samples_train, num_channels, img_size, img_size)
    y_train = torch.randint(0, num_classes, (num_samples_train,))
    
    X_val = torch.randn(num_samples_val, num_channels, img_size, img_size)
    y_val = torch.randint(0, num_classes, (num_samples_val,))
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class ManualConv2d(nn.Module):
    """
    Manual implementation of Conv2D using direct convolution.
    This is for educational purposes to understand how convolution works.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super(ManualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights using Kaiming initialization
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) * 
            np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform 2D convolution on input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = x.shape
        
        # Apply padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = nn.functional.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Initialize output tensor
        output = torch.zeros(batch_size, self.out_channels, out_height, out_width, device=x.device)
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Calculate input region
                        ih_start = oh * self.stride[0]
                        iw_start = ow * self.stride[1]
                        ih_end = ih_start + self.kernel_size[0]
                        iw_end = iw_start + self.kernel_size[1]
                        
                        # Extract region
                        region = x[b, :, ih_start:ih_end, iw_start:iw_end]
                        
                        # Compute dot product with weights
                        conv_value = (region * self.weight[oc]).sum()
                        
                        # Add bias if present
                        if self.bias is not None:
                            conv_value += self.bias[oc]
                        
                        output[b, oc, oh, ow] = conv_value
        
        return output

class SimpleCNN(nn.Module):
    """
    Simple CNN using manual Conv2D implementation.
    """
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Manual Conv2D layers
        self.conv1 = ManualConv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = ManualConv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TorchCNN(nn.Module):
    """
    CNN using PyTorch's built-in Conv2d for comparison.
    """
    
    def __init__(self, num_classes: int = 10):
        super(TorchCNN, self).__init__()
        
        # PyTorch Conv2D layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def build_model(device: torch.device, use_manual: bool = True) -> nn.Module:
    """Build and return the model."""
    if use_manual:
        model = SimpleCNN(num_classes=10)
    else:
        model = TorchCNN(num_classes=10)
    
    model = model.to(device)
    return model

def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 0.001
) -> List[float]:
    """Train the model and return loss history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return loss_history

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Calculate metrics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Store predictions and targets for R2 calculation
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Calculate R2 score (pseudo R2 for classification)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # For classification, we can use accuracy as a proxy for R2
    # Calculate pseudo R2 based on correct predictions
    r2_score = accuracy  # Simplified R2 for classification
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "r2_score": r2_score,
        "correct": correct,
        "total": total
    }
    
    return metrics

def predict(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Make predictions on input data."""
    model.eval()
    
    with torch.no_grad():
        # Ensure data is on the correct device
        data = data.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
    
    return predicted

def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, float],
    loss_history: List[float],
    use_manual: bool = True
) -> None:
    """Save model artifacts and evaluation results."""
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Evaluation Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Save loss history
    loss_path = os.path.join(OUTPUT_DIR, "loss_history.txt")
    with open(loss_path, "w") as f:
        f.write("Loss History:\n")
        for i, loss in enumerate(loss_history):
            f.write(f"Epoch {i+1}: {loss}\n")
    
    # Save model architecture info
    arch_path = os.path.join(OUTPUT_DIR, "architecture.txt")
    with open(arch_path, "w") as f:
        f.write(f"Model Type: {'Manual Conv2D' if use_manual else 'PyTorch Conv2D'}\n")
        f.write(f"Model Architecture:\n{str(model)}\n")

def compare_conv_layers(
    manual_model: SimpleCNN,
    torch_model: TorchCNN,
    device: torch.device,
    test_input: torch.Tensor
) -> float:
    """
    Compare manual Conv2D with PyTorch's Conv2d.
    Returns the maximum absolute difference between outputs.
    """
    # Copy weights from torch model to manual model
    manual_model.conv1.weight.data = torch_model.conv1.weight.data.clone()
    manual_model.conv1.bias.data = torch_model.conv1.bias.data.clone()
    manual_model.conv2.weight.data = torch_model.conv2.weight.data.clone()
    manual_model.conv2.bias.data = torch_model.conv2.bias.data.clone()
    
    # Set both models to eval mode
    manual_model.eval()
    torch_model.eval()
    
    # Move test input to device
    test_input = test_input.to(device)
    
    # Get outputs
    with torch.no_grad():
        manual_output = manual_model(test_input)
        torch_output = torch_model(test_input)
    
    # Calculate maximum absolute difference
    max_diff = torch.max(torch.abs(manual_output - torch_output)).item()
    
    return max_diff

def main():
    """Main function to run the CNN task."""
    print("=" * 60)
    print("CNN Task: Manual Conv2D Implementation")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build models
    print("\nBuilding models...")
    manual_model = build_model(device, use_manual=True)
    torch_model = build_model(device, use_manual=False)
    print("Manual CNN model created.")
    print("PyTorch CNN model created.")
    
    # Compare convolution layers first
    print("\n" + "=" * 60)
    print("Comparing Manual Conv2D with PyTorch Conv2d")
    print("=" * 60)
    
    # Create test input
    test_input = torch.randn(4, 1, 28, 28).to(device)
    
    # Copy initial weights for fair comparison
    torch_model.conv1.weight.data = manual_model.conv1.weight.data.clone()
    torch_model.conv1.bias.data = manual_model.conv1.bias.data.clone()
    torch_model.conv2.weight.data = manual_model.conv2.weight.data.clone()
    torch_model.conv2.bias.data = manual_model.conv2.bias.data.clone()
    
    # Compare outputs
    max_diff = compare_conv_layers(manual_model, torch_model, device, test_input)
    print(f"Maximum absolute difference between manual and PyTorch Conv2D: {max_diff:.2e}")
    
    # Check if difference is within tolerance
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"✓ Conv2D implementation verified (diff < {tolerance})")
    else:
        print(f"✗ Conv2D implementation error (diff >= {tolerance})")
    
    # Train manual model
    print("\n" + "=" * 60)
    print("Training Manual CNN Model")
    print("=" * 60)
    
    loss_history = train(manual_model, train_loader, device, epochs=5, lr=0.001)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(manual_model, train_loader, device)
    print(f"Training Loss: {train_metrics['loss']:.4f}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Training R2 Score: {train_metrics['r2_score']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(manual_model, val_loader, device)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation R2 Score: {val_metrics['r2_score']:.4f}")
    
    # Quality thresholds
    print("\n" + "=" * 60)
    print("Quality Thresholds Check")
    print("=" * 60)
    
    # Define thresholds
    min_accuracy = 0.85
    max_loss = 0.5
    
    # Check thresholds
    train_pass = train_metrics['accuracy'] >= min_accuracy
    val_pass = val_metrics['accuracy'] >= min_accuracy
    loss_pass = train_metrics['loss'] <= max_loss and val_metrics['loss'] <= max_loss
    
    print(f"Training accuracy >= {min_accuracy}: {'PASS' if train_pass else 'FAIL'} ({train_metrics['accuracy']:.4f})")
    print(f"Validation accuracy >= {min_accuracy}: {'PASS' if val_pass else 'FAIL'} ({val_metrics['accuracy']:.4f})")
    print(f"Loss <= {max_loss}: {'PASS' if loss_pass else 'FAIL'} (Train: {train_metrics['loss']:.4f}, Val: {val_metrics['loss']:.4f})")
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    
    save_artifacts(manual_model, val_metrics, loss_history, use_manual=True)
    print(f"Artifacts saved to {OUTPUT_DIR}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    all_pass = train_pass and val_pass and loss_pass and (max_diff < tolerance)
    
    if all_pass:
        print("✓ ALL TESTS PASSED")
        print(f"  - Conv2D implementation verified (max diff: {max_diff:.2e})")
        print(f"  - Training accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  - Validation accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  - Model artifacts saved")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        if not (max_diff < tolerance):
            print(f"  - Conv2D implementation error (diff: {max_diff:.2e})")
        if not train_pass:
            print(f"  - Training accuracy below threshold")
        if not val_pass:
            print(f"  - Validation accuracy below threshold")
        if not loss_pass:
            print(f"  - Loss above threshold")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
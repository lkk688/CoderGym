"""
CNN with Squeeze-Excitation (SE) Attention Modules

Mathematical Formulation:
- SE Module: x_out = x * sigmoid(FC2(ReLU(FC1(GlobalAvgPool(x)))))
- Channel Attention: Learn per-channel scaling weights
- Global Average Pooling: Squeezes spatial dimensions to 1x1
- Two FC layers (compression-expansion): FC(C) -> FC(C/r) -> FC(C)
  where r is reduction ratio (typically 16)

This task demonstrates:
1. Implementing SE-Net attention from scratch
2. Integrating attention into CNN architectures
3. Comparing baseline vs attention-enhanced performance
4. Measuring computational overhead vs accuracy gain
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import json
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory for artifacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'cnn_lvl5_attention_mechanisms')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'cnn_attention_squeeze_excitation',
        'description': 'ResNet-18 style CNN with SE-Net attention modules',
        'architecture': 'ResNet-18 with SE blocks',
        'attention_type': 'Squeeze-Excitation (Channel Attention)',
        'reduction_ratio': 16,
        'task_type': 'image_classification',
        'dataset': 'MNIST'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=32, download=True):
    """
    Create MNIST dataloaders for train and validation.
    
    Args:
        batch_size: Batch size
        download: Whether to download MNIST
    
    Returns:
        train_loader, val_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST, expand to 3 channels for CNN
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=download, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=download, transform=transform
    )
    
    # Split training into 80/20 train/val
    n_train = int(0.8 * len(mnist_train))
    indices = torch.randperm(len(mnist_train))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = torch.utils.data.Subset(mnist_train, train_indices)
    val_subset = torch.utils.data.Subset(mnist_train, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class SEBlock(nn.Module):
    """
    Squeeze-Excitation Block: Channel Attention
    
    SE(x) = x ⊗ σ(FC2(ReLU(FC1(GlobalAvgPool(x)))))
    where ⊗ is element-wise multiplication, σ is sigmoid
    """
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global average pooling is implicit in forward
        # FC1: channels -> channels/reduction (compress)
        self.fc1 = nn.Linear(channels, channels // reduction)
        # FC2: channels/reduction -> channels (expand)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Global average pooling: (B, C, H, W) -> (B, C)
        squeeze = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        squeeze = squeeze.view(squeeze.size(0), -1)  # (B, C)
        
        # Excitation: FC-ReLU-FC-Sigmoid
        excitation = self.fc1(squeeze)  # (B, C/r)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)  # (B, C)
        excitation = self.sigmoid(excitation)  # (B, C)
        
        # Scale: (B, C) -> (B, C, 1, 1) -> broadcast to (B, C, H, W)
        excitation = excitation.view(excitation.size(0), excitation.size(1), 1, 1)
        
        return x * excitation


class BasicBlock(nn.Module):
    """Basic residual block with optional SE attention."""
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, use_se=False, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # SE block (optional)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels, reduction=reduction)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet-18 style architecture with optional SE attention."""
    
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10, use_se=False, reduction=16):
        super(ResNet, self).__init__()
        self.use_se = use_se
        self.in_channels = 64
        
        # Initial convolution layer for MNIST (1 channel)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride=stride,
                                use_se=self.use_se))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1,
                                    use_se=self.use_se))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def build_model(use_se=False, **kwargs):
    """Build ResNet model with or without SE attention."""
    return ResNet(use_se=use_se, **kwargs)


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, data_loader, device, return_dict=True):
    """Evaluate model on data loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    if not return_dict:
        return accuracy
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def train(train_loader, val_loader, device, use_se=False, epochs=20, lr=0.001):
    """Train a model (with or without SE attention)."""
    model = build_model(use_se=use_se)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device, return_dict=True)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        if (epoch + 1) % 5 == 0:
            se_type = "with SE" if use_se else "without SE"
            print(f"Epoch [{epoch+1}/{epochs}] {se_type} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    return model, history


def predict(model, x, device):
    """Make predictions on input."""
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x)
    x = x.to(device)
    
    with torch.no_grad():
        outputs = model(x)
        _, predictions = torch.max(outputs, 1)
    
    return predictions.cpu().numpy()


def evaluate(model, data_loader, device, return_dict=True):
    """Evaluate wrapper for protocol."""
    return evaluate_model(model, data_loader, device, return_dict=return_dict)


def save_artifacts(baseline_model, attention_model, histories, metrics, output_dir=OUTPUT_DIR):
    """Save models and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    baseline_path = os.path.join(output_dir, 'baseline_model.pt')
    torch.save(baseline_model.state_dict(), baseline_path)
    print(f"Saved baseline model to {baseline_path}")
    
    attention_path = os.path.join(output_dir, 'attention_model.pt')
    torch.save(attention_model.state_dict(), attention_path)
    print(f"Saved attention model to {attention_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save histories
    history_path = os.path.join(output_dir, 'histories.json')
    with open(history_path, 'w') as f:
        json.dump(histories, f, indent=2)
    print(f"Saved histories to {history_path}")


if __name__ == '__main__':
    """
    Main pipeline:
    1. Load MNIST data
    2. Train baseline CNN (no SE attention)
    3. Train CNN with SE attention
    4. Compare performance and parameters
    5. Assert quality thresholds
    """
    
    try:
        device = get_device()
        print(f"Using device: {device}")
        
        # Load data
        print("\nLoading MNIST data...")
        train_loader, val_loader = make_dataloaders(batch_size=64, download=True)
        
        # Train baseline model
        print("\n" + "="*60)
        print("Training Baseline Model (No SE Attention)")
        print("="*60)
        baseline_model, baseline_history = train(
            train_loader, val_loader, device, use_se=False, epochs=15, lr=0.001
        )
        
        # Evaluate baseline
        baseline_train_metrics = evaluate_model(
            baseline_model, train_loader, device, return_dict=True
        )
        baseline_val_metrics = evaluate_model(
            baseline_model, val_loader, device, return_dict=True
        )
        
        print(f"\nBaseline - Train Acc: {baseline_train_metrics['accuracy']:.4f}, "
              f"Val Acc: {baseline_val_metrics['accuracy']:.4f}")
        
        # Train attention model
        print("\n" + "="*60)
        print("Training Model with SE Attention")
        print("="*60)
        attention_model, attention_history = train(
            train_loader, val_loader, device, use_se=True, epochs=15, lr=0.001
        )
        
        # Evaluate attention model
        attention_train_metrics = evaluate_model(
            attention_model, train_loader, device, return_dict=True
        )
        attention_val_metrics = evaluate_model(
            attention_model, val_loader, device, return_dict=True
        )
        
        print(f"\nAttention - Train Acc: {attention_train_metrics['accuracy']:.4f}, "
              f"Val Acc: {attention_val_metrics['accuracy']:.4f}")
        
        # Compare architectures
        print("\n" + "="*60)
        print("Architecture Comparison")
        print("="*60)
        
        baseline_params = count_parameters(baseline_model)
        attention_params = count_parameters(attention_model)
        param_increase = (attention_params - baseline_params) / baseline_params * 100
        
        print(f"Baseline parameters: {baseline_params}")
        print(f"Attention parameters: {attention_params}")
        print(f"Parameter increase: {param_increase:.2f}%")
        
        accuracy_gain = attention_val_metrics['accuracy'] - baseline_val_metrics['accuracy']
        print(f"\nAccuracy gain (val): {accuracy_gain:.4f}")
        print(f"Baseline val accuracy: {baseline_val_metrics['accuracy']:.4f}")
        print(f"Attention val accuracy: {attention_val_metrics['accuracy']:.4f}")
        
        # Collect metrics
        all_metrics = {
            'baseline': {
                'train': baseline_train_metrics,
                'val': baseline_val_metrics,
                'params': baseline_params
            },
            'attention': {
                'train': attention_train_metrics,
                'val': attention_val_metrics,
                'params': attention_params
            },
            'comparison': {
                'param_increase_percent': param_increase,
                'accuracy_gain': accuracy_gain,
                'efficiency_ratio': param_increase / (accuracy_gain * 100) if accuracy_gain > 0 else float('inf')
            }
        }
        
        all_histories = {
            'baseline': baseline_history,
            'attention': attention_history
        }
        
        save_artifacts(baseline_model, attention_model, all_histories, all_metrics)
        
        # Assertions for quality
        print("\n" + "="*60)
        print("Quality Assertions")
        print("="*60)
        
        assert baseline_val_metrics['accuracy'] > 0.95, \
            f"Baseline accuracy {baseline_val_metrics['accuracy']:.4f} must be > 0.95"
        print("✓ Baseline validation accuracy > 0.95")
        
        assert attention_val_metrics['accuracy'] > baseline_val_metrics['accuracy'], \
            f"Attention model should match or exceed baseline"
        print("✓ Attention model >= baseline accuracy")
        
        assert attention_train_metrics['accuracy'] > 0.97, \
            f"Attention train accuracy {attention_train_metrics['accuracy']:.4f} must be > 0.97"
        print("✓ Attention model train accuracy > 0.97")
        
        assert param_increase < 15, \
            f"Attention parameter overhead {param_increase:.2f}% should be < 15%"
        print(f"✓ Attention overhead reasonable ({param_increase:.2f}%)")
        
        print("\n" + "="*60)
        print("SUCCESS: All assertions passed!")
        print("="*60)
        sys.exit(0)
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

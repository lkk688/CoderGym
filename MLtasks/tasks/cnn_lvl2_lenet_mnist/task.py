"""
LeNet on MNIST - CNN Image Classification Task
Implements LeNet-style CNN with data loaders, augmentation, and evaluation.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'lenet_mnist',
        'task_type': 'classification',
        'num_classes': 10,
        'input_shape': [1, 28, 28],
        'description': 'LeNet-style CNN for MNIST digit classification'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the device for training."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    """LeNet-5 architecture for MNIST classification."""
    
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Block 1: Conv -> ReLU -> Pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Block 2: Conv -> ReLU -> Pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def make_dataloaders(batch_size=64, val_ratio=0.15, num_workers=2):
    """Create data loaders for MNIST with augmentation."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full training dataset
    full_train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # Split into train and validation
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # For validation, we need to reapply the val_transform
    # since random_split preserves the original transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Also create a train eval loader without augmentation
    train_eval_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=val_transform
    )
    # Use same samples as train_dataset
    train_eval_dataset = torch.utils.data.Subset(
        train_eval_dataset, train_dataset.indices
    )
    train_eval_loader = DataLoader(
        train_eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Test dataset
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=val_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, train_eval_loader, test_loader


def build_model(num_classes=10):
    """Build the LeNet model."""
    model = LeNet(num_classes=num_classes)
    model = model.to(device)
    return model


def train(model, train_loader, val_loader, epochs=20, lr=0.001, weight_decay=1e-4):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Training model...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
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
            
            running_loss += loss.item()
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, return_predictions=False)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


def evaluate(model, data_loader, return_predictions=True):
    """Evaluate the model on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy
    }
    
    if return_predictions:
        metrics['predictions'] = np.array(all_preds)
        metrics['targets'] = np.array(all_targets)
    
    return metrics


def predict(model, data_loader):
    """Get predictions for a dataset."""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            
            # Get predictions and probabilities
            _, predicted = torch.max(output, 1)
            probs = torch.softmax(output, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)


def save_artifacts(model, metrics, output_dir='./output'):
    """Save model artifacts and evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix plot
    if 'val_predictions' in metrics and 'val_targets' in metrics:
        cm = confusion_matrix(metrics['val_targets'], metrics['val_predictions'])
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - MNIST')
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, range(10))
        plt.yticks(tick_marks, range(10))
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Artifacts saved to {output_dir}")


def main():
    """Main function to run the LeNet training and evaluation."""
    print("=" * 60)
    print("LeNet on MNIST - Image Classification Task")
    print("=" * 60)
    
    # Get metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Type: {metadata['task_type']}")
    print(f"Classes: {metadata['num_classes']}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, train_eval_loader, test_loader = make_dataloaders(
        batch_size=64, val_ratio=0.15
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(num_classes=metadata['num_classes'])
    print(f"Model architecture:\n{model}")
    
    # Train model
    print("\n" + "-" * 60)
    train_history = train(
        model, train_loader, val_loader,
        epochs=20, lr=0.001, weight_decay=1e-4
    )
    
    # Evaluate on training set
    print("\n" + "-" * 60)
    print("Evaluating on training set...")
    train_metrics = evaluate(model, train_eval_loader)
    print(f"Train Loss: {train_metrics['loss']:.4f}")
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, return_predictions=True)
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    all_metrics = {
        'train_loss': train_metrics['loss'],
        'train_accuracy': train_metrics['accuracy'],
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
        'train_losses': train_history['train_losses'],
        'val_losses': train_history['val_losses'],
        'val_accuracies': train_history['val_accuracies']
    }
    
    # Add predictions for confusion matrix
    if 'predictions' in val_metrics:
        all_metrics['val_predictions'] = val_metrics['predictions'].tolist()
        all_metrics['val_targets'] = val_metrics['targets'].tolist()
    
    save_artifacts(model, all_metrics, output_dir='./output')
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"Train Loss:      {train_metrics['loss']:.4f}")
    print(f"Val Loss:        {val_metrics['loss']:.4f}")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)
    
    checks_passed = True
    
    # Check 1: Train accuracy > 0.98
    check1 = train_metrics['accuracy'] > 0.98
    status1 = "✓" if check1 else "✗"
    print(f"{status1} Train Accuracy > 0.98: {train_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check1
    
    # Check 2: Val accuracy > 0.98
    check2 = val_metrics['accuracy'] > 0.98
    status2 = "✓" if check2 else "✗"
    print(f"{status2} Val Accuracy > 0.98: {val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check2
    
    # Check 3: Val loss < 0.05
    check3 = val_metrics['loss'] < 0.05
    status3 = "✓" if check3 else "✗"
    print(f"{status3} Val Loss < 0.05: {val_metrics['loss']:.4f}")
    checks_passed = checks_passed and check3
    
    # Check 4: No significant overfitting (accuracy gap < 0.02)
    accuracy_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
    check4 = accuracy_gap < 0.02
    status4 = "✓" if check4 else "✗"
    print(f"{status4} Accuracy gap < 0.02: {accuracy_gap:.4f}")
    checks_passed = checks_passed and check4
    
    # Final summary
    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

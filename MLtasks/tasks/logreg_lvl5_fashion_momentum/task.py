"""
Logistic Regression with SGD + Momentum on Fashion-MNIST

Mathematical Formulation:
- Softmax: P(y=k|x) = exp(W_k @ x) / sum_j(exp(W_j @ x))
- Cross-Entropy Loss: L = -sum_i sum_k y_ik * log(P(y=k|x_i))
- SGD with Momentum: v_t = beta * v_{t-1} + (1-beta) * grad
                      theta_t = theta_{t-1} - lr * v_t
- Nesterov Momentum: Look-ahead gradient evaluation for faster convergence

This implementation compares vanilla SGD, momentum SGD, and Nesterov momentum
on the Fashion-MNIST dataset (10-class image classification).
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict

# Output directory for artifacts
OUTPUT_DIR = './output/tasks/logreg_lvl5_fashion_momentum'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'logistic_regression_fashion_mnist_momentum',
        'description': 'Multiclass Logistic Regression with Momentum on Fashion-MNIST',
        'input_dim': 784,
        'output_dim': 10,
        'model_type': 'multiclass_logistic_regression',
        'loss_type': 'cross_entropy',
        'optimization': 'sgd_with_momentum',
        'dataset': 'fashion_mnist'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=128):
    """
    Create Fashion-MNIST dataloaders.
    
    Fashion-MNIST: 60k train + 10k test images of 10 clothing categories
    Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
             Sneaker, Bag, Ankle boot
    
    Args:
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    try:
        from torchvision import datasets, transforms
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        # Download Fashion-MNIST
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Split train into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
    except:
        # Create synthetic data if Fashion-MNIST unavailable
        print("  Creating synthetic Fashion-MNIST-like data...")
        
        def create_synthetic_data(n_samples, input_dim=784, n_classes=10):
            X = torch.randn(n_samples, input_dim) * 0.5
            y = torch.randint(0, n_classes, (n_samples,))
            # Add class-specific patterns
            for c in range(n_classes):
                mask = y == c
                X[mask] += torch.randn(1, input_dim) * 0.3
            return torch.utils.data.TensorDataset(X, y)
        
        train_dataset = create_synthetic_data(48000)
        val_dataset = create_synthetic_data(12000)
        test_dataset = create_synthetic_data(10000)
        
        class_names = [f'Class_{i}' for i in range(10)]
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader, class_names


class LogisticRegressionMomentum(nn.Module):
    """
    Multiclass Logistic Regression with custom momentum optimizer.
    
    Implements three optimization variants:
    1. Vanilla SGD
    2. SGD with Momentum
    3. SGD with Nesterov Momentum
    """
    
    def __init__(self, input_dim=784, num_classes=10, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        
        # Linear layer: y = Wx + b
        self.linear = nn.Linear(input_dim, num_classes)
        
        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        # Momentum buffers
        self.velocity_weight = torch.zeros_like(self.linear.weight.data)
        self.velocity_bias = torch.zeros_like(self.linear.bias.data)
        
        self.to(self.device)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input of shape (N, 784) or (N, 1, 28, 28)
        
        Returns:
            Logits of shape (N, 10)
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.linear(x)
    
    def update_with_momentum(self, lr, momentum=0.9, use_nesterov=False):
        """
        Manual parameter update with momentum.
        
        Standard Momentum:
            v_t = beta * v_{t-1} + (1-beta) * grad
            theta_t = theta_{t-1} - lr * v_t
        
        Nesterov Momentum:
            v_t = beta * v_{t-1} + grad
            theta_t = theta_{t-1} - lr * (grad + beta * v_t)
        
        Args:
            lr: Learning rate
            momentum: Momentum coefficient (beta)
            use_nesterov: Whether to use Nesterov momentum
        """
        with torch.no_grad():
            if self.linear.weight.grad is not None:
                # Update velocity for weights
                self.velocity_weight = momentum * self.velocity_weight + self.linear.weight.grad
                
                if use_nesterov:
                    # Nesterov: look-ahead gradient
                    self.linear.weight -= lr * (self.linear.weight.grad + momentum * self.velocity_weight)
                else:
                    # Standard momentum
                    self.linear.weight -= lr * self.velocity_weight
                
                # Zero gradient
                self.linear.weight.grad.zero_()
            
            if self.linear.bias.grad is not None:
                # Update velocity for bias
                self.velocity_bias = momentum * self.velocity_bias + self.linear.bias.grad
                
                if use_nesterov:
                    self.linear.bias -= lr * (self.linear.bias.grad + momentum * self.velocity_bias)
                else:
                    self.linear.bias -= lr * self.velocity_bias
                
                self.linear.bias.grad.zero_()
    
    def reset_momentum(self):
        """Reset momentum buffers."""
        self.velocity_weight.zero_()
        self.velocity_bias.zero_()


def build_model(input_dim=784, num_classes=10, device=None):
    """Build Logistic Regression model."""
    return LogisticRegressionMomentum(input_dim, num_classes, device)


def train(model, train_loader, val_loader, epochs=10, lr=0.1, momentum=0.9, 
          optimizer_type='momentum', verbose=True):
    """
    Train Logistic Regression model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        momentum: Momentum coefficient
        optimizer_type: 'vanilla', 'momentum', or 'nesterov'
        verbose: Print progress
    
    Returns:
        Training history dictionary
    """
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    use_momentum = optimizer_type in ['momentum', 'nesterov']
    use_nesterov = optimizer_type == 'nesterov'
    
    print(f"\nTraining with {optimizer_type.upper()} optimizer...")
    print(f"  LR: {lr}, Momentum: {momentum if use_momentum else 0.0}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Manual update
            if use_momentum:
                model.update_with_momentum(lr, momentum, use_nesterov)
            else:
                # Vanilla SGD
                with torch.no_grad():
                    model.linear.weight -= lr * model.linear.weight.grad
                    model.linear.bias -= lr * model.linear.bias.grad
                    model.linear.weight.grad.zero_()
                    model.linear.bias.grad.zero_()
            
            # Track metrics
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        # Epoch metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        val_metrics = evaluate(model, val_loader, split_name='Val', verbose=False)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        if verbose and (epoch + 1) % 2 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    return history


def evaluate(model, data_loader, split_name='Test', verbose=True):
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with metrics: loss, accuracy, per-class accuracy, confusion matrix
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall metrics
    n_samples = len(all_labels)
    loss = total_loss / n_samples
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class accuracy
    n_classes = len(np.unique(all_labels))
    per_class_acc = []
    for c in range(n_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc.append((all_preds[mask] == all_labels[mask]).mean())
        else:
            per_class_acc.append(0.0)
    
    # Confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        conf_matrix[true, pred] += 1
    
    # Macro F1 (average of per-class F1 scores)
    f1_scores = []
    for c in range(n_classes):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': conf_matrix,
        'split': split_name
    }
    
    if verbose:
        print(f"\n{split_name} Metrics:")
        print(f"  Loss:         {loss:.6f}")
        print(f"  Accuracy:     {accuracy:.6f}")
        print(f"  Macro F1:     {macro_f1:.6f}")
        print(f"  Mean Per-Class Acc: {np.mean(per_class_acc):.6f}")
    
    return metrics


def predict(model, X):
    """Make predictions on new data."""
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X = X.to(model.device)
    with torch.no_grad():
        logits = model(X)
        _, predicted = torch.max(logits, 1)
    return predicted


def save_artifacts(histories, test_metrics_dict, class_names):
    """Save training curves and metrics visualizations."""
    
    # Plot training curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    optimizer_types = list(histories.keys())
    colors = {'vanilla': 'blue', 'momentum': 'orange', 'nesterov': 'green'}
    
    # Train loss
    for opt_type in optimizer_types:
        axes[0, 0].plot(histories[opt_type]['train_loss'], 
                        label=opt_type.capitalize(), color=colors.get(opt_type, 'gray'))
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Val loss
    for opt_type in optimizer_types:
        axes[0, 1].plot(histories[opt_type]['val_loss'],
                        label=opt_type.capitalize(), color=colors.get(opt_type, 'gray'))
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Validation Loss Comparison', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train accuracy
    for opt_type in optimizer_types:
        axes[1, 0].plot(histories[opt_type]['train_acc'],
                        label=opt_type.capitalize(), color=colors.get(opt_type, 'gray'))
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy', fontsize=11)
    axes[1, 0].set_title('Training Accuracy Comparison', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Val accuracy
    for opt_type in optimizer_types:
        axes[1, 1].plot(histories[opt_type]['val_acc'],
                        label=opt_type.capitalize(), color=colors.get(opt_type, 'gray'))
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy', fontsize=11)
    axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimizer_comparison.png'), dpi=150)
    plt.close()
    
    # Plot confusion matrix for best model (Nesterov)
    best_metrics = test_metrics_dict['nesterov']
    conf_matrix = best_metrics['confusion_matrix']
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Nesterov Momentum)', fontsize=13)
    
    # Add text annotations
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), 
                    ha='center', va='center', color='red' if i == j else 'black',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    print(f"\nArtifacts saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    print("=" * 70)
    print("Task: Logistic Regression with SGD + Momentum on Fashion-MNIST")
    print("=" * 70)
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Get metadata
    metadata = get_task_metadata()
    print(f"\nTask Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading Fashion-MNIST dataset...")
    train_loader, val_loader, test_loader, class_names = make_dataloaders(batch_size=128)
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Classes: {len(class_names)}")
    
    # Train with different optimizers
    histories = {}
    test_metrics_dict = {}
    
    for opt_type in ['vanilla', 'momentum', 'nesterov']:
        print(f"\n{'=' * 70}")
        print(f"Training with {opt_type.upper()} optimizer")
        print(f"{'=' * 70}")
        
        model = build_model(input_dim=784, num_classes=10, device=device)
        
        history = train(
            model, train_loader, val_loader,
            epochs=10, lr=0.1, momentum=0.9,
            optimizer_type=opt_type, verbose=True
        )
        
        histories[opt_type] = history
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, split_name=f'Test ({opt_type})')
        test_metrics_dict[opt_type] = test_metrics
    
    # Save artifacts
    save_artifacts(histories, test_metrics_dict, class_names)
    
    # Validation checks
    print(f"\n{'=' * 70}")
    print("VALIDATION CHECKS")
    print(f"{'=' * 70}")
    
    # Check 1: Nesterov should achieve > 0.80 accuracy
    nesterov_acc = test_metrics_dict['nesterov']['accuracy']
    acc_threshold = 0.80
    acc_pass = nesterov_acc > acc_threshold
    print(f"✓ Nesterov Test Accuracy > {acc_threshold}: {nesterov_acc:.6f} - {'PASS' if acc_pass else 'FAIL'}")
    
    # Check 2: Nesterov should have > 0.75 Macro F1
    nesterov_f1 = test_metrics_dict['nesterov']['macro_f1']
    f1_threshold = 0.75
    f1_pass = nesterov_f1 > f1_threshold
    print(f"✓ Nesterov Macro F1 > {f1_threshold}: {nesterov_f1:.6f} - {'PASS' if f1_pass else 'FAIL'}")
    
    # Check 3: Momentum methods should converge faster than vanilla
    vanilla_final_loss = histories['vanilla']['val_loss'][-1]
    momentum_final_loss = histories['momentum']['val_loss'][-1]
    nesterov_final_loss = histories['nesterov']['val_loss'][-1]
    
    faster_convergence = (momentum_final_loss <= vanilla_final_loss) or (nesterov_final_loss <= vanilla_final_loss)
    print(f"✓ Momentum methods converge better: Vanilla={vanilla_final_loss:.4f}, "
          f"Momentum={momentum_final_loss:.4f}, Nesterov={nesterov_final_loss:.4f} - "
          f"{'PASS' if faster_convergence else 'FAIL'}")
    
    # Check 4: Per-class accuracy reasonable (mean > 0.75)
    mean_per_class = np.mean(test_metrics_dict['nesterov']['per_class_accuracy'])
    per_class_threshold = 0.75
    per_class_pass = mean_per_class > per_class_threshold
    print(f"✓ Mean per-class accuracy > {per_class_threshold}: {mean_per_class:.6f} - "
          f"{'PASS' if per_class_pass else 'FAIL'}")
    
    # Final verdict
    all_checks_pass = acc_pass and f1_pass and faster_convergence and per_class_pass
    
    print(f"\n{'=' * 70}")
    if all_checks_pass:
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print(f"{'=' * 70}")
        sys.exit(0)
    else:
        print("✗ SOME VALIDATION CHECKS FAILED!")
        print(f"{'=' * 70}")
        sys.exit(1)

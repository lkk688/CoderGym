"""
MLP Classifier with Dropout and BatchNorm Options
Implements a general MLP classifier using PyTorch autograd.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "mlp_classifier",
        "task_type": "classification",
        "input_type": "tabular",
        "output_type": "multiclass",
        "description": "MLP classifier with dropout and batchnorm options",
        "metrics": ["accuracy", "loss", "precision", "recall", "f1_score"],
        "default_epochs": 50,
        "default_batch_size": 64,
        "default_lr": 0.001
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_samples: int = 1000,
    num_features: int = 20,
    num_classes: int = 3,
    noise: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        batch_size: Batch size for training
        val_ratio: Ratio of validation data
        num_samples: Total number of samples
        num_features: Number of features
        num_classes: Number of classes
        noise: Noise level for synthetic data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Generate synthetic classification data
    X = np.random.randn(num_samples, num_features)
    
    # Create non-linear decision boundaries
    y = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        # Multiple decision boundaries for non-linear classification
        score = (
            0.5 * X[i, 0] + 
            0.3 * X[i, 1] ** 2 - 
            0.4 * X[i, 2] * X[i, 3] + 
            0.2 * np.sin(X[i, 4]) +
            np.random.randn() * noise
        )
        if score < -1:
            y[i] = 0
        elif score < 1:
            y[i] = 1
        else:
            y[i] = 2
    
    # Add some noise to features
    X += np.random.randn(*X.shape) * 0.1
    
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    # Split into train and validation
    split_idx = int(num_samples * (1 - val_ratio))
    indices = np.random.permutation(num_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    # Create tensors
    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.LongTensor(y[train_idx])
    X_val = torch.FloatTensor(X[val_idx])
    y_val = torch.LongTensor(y[val_idx])
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Print class distribution
    train_classes = np.bincount(y_train.numpy(), minlength=num_classes)
    val_classes = np.bincount(y_val.numpy(), minlength=num_classes)
    print(f"Class distribution - Train: {dict(enumerate(train_classes))}")
    print(f"Class distribution - Val: {dict(enumerate(val_classes))}")
    
    return train_loader, val_loader


class MLPClassifier(nn.Module):
    """MLP Classifier with dropout and batchnorm options."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU(inplace=True))
            
            # Dropout (except for last hidden layer)
            if i < len(hidden_dims) - 1 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


def build_model(
    input_dim: int = 20,
    hidden_dims: List[int] = [128, 64, 32],
    num_classes: int = 3,
    dropout: float = 0.5,
    use_batchnorm: bool = True,
    lr: float = 0.001
) -> Tuple[nn.Module, optim.Optimizer]:
    """
    Build MLP model and optimizer.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of classes
        dropout: Dropout rate
        use_batchnorm: Whether to use batch normalization
        lr: Learning rate
        
    Returns:
        Tuple of (model, optimizer)
    """
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
        use_batchnorm=use_batchnorm
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Num classes: {num_classes}")
    print(f"  Dropout: {dropout}")
    print(f"  Use batchnorm: {use_batchnorm}")
    
    return model, optimizer


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 50,
    print_every: int = 10
) -> List[float]:
    """
    Train the model.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        epochs: Number of epochs
        print_every: Print frequency
        
    Returns:
        List of training losses per epoch
    """
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(model.get_device())
            batch_y = batch_y.to(model.get_device())
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return losses


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module
) -> Dict[str, float]:
    """
    Evaluate the model.
    
    Args:
        model: Neural network model
        data_loader: Data loader (train or validation)
        criterion: Loss function
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Move data to device
            batch_X = batch_X.to(model.get_device())
            batch_y = batch_y.to(model.get_device())
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Store for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    accuracy = correct / total
    
    # Calculate precision, recall, F1 for each class
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


def predict(
    model: nn.Module,
    X: np.ndarray
) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        model: Neural network model
        X: Input features (numpy array)
        
    Returns:
        Predicted class labels
    """
    model.eval()
    
    # Convert to tensor and move to device
    X_tensor = torch.FloatTensor(X).to(model.get_device())
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.cpu().numpy()


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, float],
    output_dir: str = "output"
) -> None:
    """
    Save model artifacts and metrics.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save model architecture
    arch_path = output_path / "architecture.txt"
    with open(arch_path, 'w') as f:
        f.write(str(model))
    print(f"Architecture saved to {arch_path}")


def main():
    """Main function to run the MLP classification task."""
    print("=" * 60)
    print("MLP Classifier with Dropout and BatchNorm")
    print("=" * 60)
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Type: {metadata['task_type']}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Data parameters
    num_samples = 1000
    num_features = 20
    num_classes = 3
    val_ratio = 0.2
    batch_size = 64
    
    # Model parameters
    hidden_dims = [128, 64, 32]
    dropout = 0.5
    use_batchnorm = True
    lr = 0.001
    epochs = 50
    
    # Create dataloaders
    print("\n" + "-" * 40)
    print("Creating dataloaders...")
    print("-" * 40)
    train_loader, val_loader = make_dataloaders(
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_samples=num_samples,
        num_features=num_features,
        num_classes=num_classes
    )
    
    # Build model
    print("\n" + "-" * 40)
    print("Building model...")
    print("-" * 40)
    model, optimizer = build_model(
        input_dim=num_features,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        lr=lr
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print("\n" + "-" * 40)
    print("Training model...")
    print("-" * 40)
    train_losses = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        print_every=10
    )
    
    # Evaluate on training set
    print("\n" + "-" * 40)
    print("Evaluating on training set...")
    print("-" * 40)
    train_metrics = evaluate(model, train_loader, criterion)
    print("Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation set
    print("\n" + "-" * 40)
    print("Evaluating on validation set...")
    print("-" * 40)
    val_metrics = evaluate(model, val_loader, criterion)
    print("Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"\nTraining Metrics:")
    print(f"  Loss:  {train_metrics['loss']:.4f}")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1_score']:.4f}")
    
    print(f"\nValidation Metrics:")
    print(f"  Loss:  {val_metrics['loss']:.4f}")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1_score']:.4f}")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("Quality Checks")
    print("=" * 60)
    
    quality_passed = True
    
    # Check 1: Training accuracy > 0.8
    train_acc_check = train_metrics['accuracy'] > 0.8
    print(f"{'✓' if train_acc_check else '✗'} Train Accuracy > 0.8: {train_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and train_acc_check
    
    # Check 2: Validation accuracy > 0.7
    val_acc_check = val_metrics['accuracy'] > 0.7
    print(f"{'✓' if val_acc_check else '✗'} Val Accuracy > 0.7: {val_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and val_acc_check
    
    # Check 3: Validation accuracy >= training accuracy - 0.1 (no severe overfitting)
    overfit_check = val_metrics['accuracy'] >= train_metrics['accuracy'] - 0.1
    print(f"{'✓' if overfit_check else '✗'} Val Accuracy >= Train Accuracy - 0.1: {val_metrics['accuracy']:.4f} >= {train_metrics['accuracy'] - 0.1:.4f}")
    quality_passed = quality_passed and overfit_check
    
    # Check 4: Training F1 > 0.7
    train_f1_check = train_metrics['f1_score'] > 0.7
    print(f"{'✓' if train_f1_check else '✗'} Train F1 > 0.7: {train_metrics['f1_score']:.4f}")
    quality_passed = quality_passed and train_f1_check
    
    # Check 5: Validation F1 > 0.7
    val_f1_check = val_metrics['f1_score'] > 0.7
    print(f"{'✓' if val_f1_check else '✗'} Val F1 > 0.7: {val_metrics['f1_score']:.4f}")
    quality_passed = quality_passed and val_f1_check    # Check 6: Loss decreased during training
    loss_decrease_check = len(train_losses) > 1 and train_losses[-1] < train_losses[0]
    print(f"{'✓' if loss_decrease_check else '✗'} Loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    quality_passed = quality_passed and loss_decrease_check
    
    # Final quality check result
    print(f"\n{'✓' if quality_passed else '✗'} All quality checks passed: {quality_passed}")
    
    # Save artifacts
    print("\n" + "-" * 40)
    print("Saving artifacts...")
    print("-" * 40)
    save_artifacts(model, val_metrics)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'quality_passed': quality_passed
    }


if __name__ == "__main__":
    main()

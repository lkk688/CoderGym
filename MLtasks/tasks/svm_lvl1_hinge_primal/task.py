"""
Linear SVM (Primal, Hinge Loss) Implementation using PyTorch Autograd

Mathematical formulation:
- Hinge loss: L(y, f(x)) = max(0, 1 - y * f(x))
- Objective: min_w (1/2) * ||w||^2 + C * sum_i max(0, 1 - y_i * (w^T x_i))
- With L2 regularization: min_w (1/2) * ||w||^2 + C * sum_i max(0, 1 - y_i * (w^T x_i))

Where:
- w is the weight vector
- C is the regularization parameter
- y ∈ {-1, 1} are the labels
- f(x) = w^T x is the decision function
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    mean_squared_error, 
    r2_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

# Set paths
OUTPUT_DIR = "/Developer/AIserver/output/tasks/svm_lvl1_hinge_primal"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_task_metadata() -> dict:
    """Return metadata about the task."""
    return {
        "task_name": "linear_svm_hinge_primal",
        "description": "Linear SVM with hinge loss + L2 regularization using autograd",
        "model_type": "linear_classifier",
        "loss_function": "hinge_loss",
        "regularization": "l2",
        "input_type": "continuous",
        "output_type": "binary",
        "framework": "pytorch"
    }


def make_dataloaders(
    n_samples: int = 1000,
    n_features: int = 20,
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42
) -> tuple:
    """
    Create synthetic binary classification dataset and dataloaders.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        test_size: Proportion of data for validation
        batch_size: Batch size for training
        random_state: Random seed
        
    Returns:
        train_loader, val_loader, X_train, X_val, y_train, y_val
    """
    set_seed(random_state)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=2,
        class_sep=1.5,
        random_state=random_state,
        flip_y=0.05
    )
    
    # Convert labels to {-1, 1} for hinge loss
    y_hinge = np.where(y == 0, -1, 1)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_hinge, test_size=test_size, random_state=random_state, stratify=y_hinge
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val


class LinearSVM(nn.Module):
    """
    Linear SVM with hinge loss and L2 regularization.
    
    The model implements the primal form of SVM:
    - Decision function: f(x) = w^T x + b
    - Hinge loss: max(0, 1 - y * f(x))
    - L2 regularization: (1/2) * ||w||^2
    """
    
    def __init__(self, n_features: int, C: float = 1.0):
        """
        Initialize the Linear SVM model.
        
        Args:
            n_features: Number of input features
            C: Regularization parameter (inverse of regularization strength)
        """
        super(LinearSVM, self).__init__()
        self.n_features = n_features
        self.C = C
        
        # Linear layer for decision function
        self.linear = nn.Linear(n_features, 1, bias=True)
        
        # Initialize weights with small values
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the decision function values."""
        return self.linear(x)
    
    def hinge_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute hinge loss with L2 regularization.
        
        Hinge loss: max(0, 1 - y * f(x))
        L2 regularization: (1/2) * ||w||^2
        
        Args:
            y_pred: Predicted values (decision function)
            y_true: True labels in {-1, 1}
            
        Returns:
            Total loss (hinge loss + L2 regularization)
        """
        # Hinge loss: max(0, 1 - y * f(x))
        margin = y_true * y_pred
        hinge = torch.clamp(1 - margin, min=0)
        
        # Mean hinge loss
        mean_hinge_loss = torch.mean(hinge)
        
        # L2 regularization on weights (excluding bias)
        l2_reg = 0.5 * torch.sum(self.linear.weight ** 2)
        
        # Total loss: hinge loss + C * L2 regularization
        total_loss = mean_hinge_loss + self.C * l2_reg
        
        return total_loss
    
    def predict_labels(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels (0 or 1)."""
        with torch.no_grad():
            decision = self.forward(X)
            return (decision >= 0).float()


def build_model(n_features: int, C: float = 1.0) -> LinearSVM:
    """
    Build and return the Linear SVM model.
    
    Args:
        n_features: Number of input features
        C: Regularization parameter
        
    Returns:
        Initialized LinearSVM model
    """
    device = get_device()
    model = LinearSVM(n_features=n_features, C=C).to(device)
    return model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 0.01,
    epochs: int = 100,
    patience: int = 10,
    device: torch.device = None
) -> dict:
    """
    Train the Linear SVM model.
    
    Args:
        model: The LinearSVM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        lr: Learning rate
        epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Computation device
        
    Returns:
        Training history dictionary
    """
    if device is None:
        device = get_device()
    
    model.to(device)
    
    # Use SGD optimizer (common for SVM)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_model_state': None,
        'patience_counter': 0
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = model.hinge_loss(y_pred, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * batch_X.size(0)
            
            # Calculate accuracy
            predicted = (y_pred >= 0).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Average training metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        val_results = evaluate(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        
        # Learning rate scheduling
        scheduler.step(val_results['loss'])
        
        # Early stopping check
        if val_results['accuracy'] > history['best_val_acc']:
            history['best_val_acc'] = val_results['accuracy']
            history['best_model_state'] = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            history['patience_counter'] = 0
        else:
            history['patience_counter'] += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']:.4f}")
        
        # Early stopping
        if history['patience_counter'] >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if history['best_model_state'] is not None:
        model.load_state_dict(history['best_model_state'])
    
    return history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None
) -> dict:
    """
    Evaluate the model on given data.
    
    Args:
        model: The LinearSVM model
        data_loader: Data loader for evaluation
        device: Computation device
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = get_device()
    
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            y_pred = model(batch_X)
            
            # Compute loss
            loss = model.hinge_loss(y_pred, batch_y)
            
            # Accumulate loss
            total_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)
            
            # Get predictions
            predictions = (y_pred >= 0).float()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Classification report
    # Convert back to {0, 1} format for sklearn metrics
    y_true_binary = (all_targets > 0).astype(int)
    y_pred_binary = (all_predictions > 0).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    return {
        'loss': total_loss / total_samples,
        'mse': mse,
        'r2': r2,
        'accuracy': accuracy,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'f1': 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))) 
              if (tp + fp) > 0 and (tp + fn) > 0 else 0.0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def predict(model: nn.Module, X: np.ndarray, device: torch.device = None) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        model: The LinearSVM model
        X: Input features as numpy array
        device: Computation device
        
    Returns:
        Predicted labels as numpy array
    """
    if device is None:
        device = get_device()
    
    model.eval()
    model.to(device)
    
    # Convert to tensor and move to device
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        # Get predictions
        predictions = model.predict_labels(X_tensor)
    
    return predictions.cpu().numpy()


def save_artifacts(
    model: nn.Module,
    history: dict,
    metrics: dict,
    metadata: dict
) -> None:
    """
    Save model artifacts, metrics, and history.
    
    Args:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
        metadata: Task metadata
    """
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "svm_model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "task_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run the SVM training and evaluation."""
    print("=" * 60)
    print("Linear SVM (Primal, Hinge Loss) Training")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")
    
    # Create dataloaders
    print("\nCreating datasets...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=1000,
        n_features=20,
        test_size=0.2,
        batch_size=32
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(n_features=20, C=1.0)
    print(f"Model architecture: {model}")
    
    # Train model
"""
kNN Classifier (Brute Force) - Pure PyTorch Implementation
Implements k-Nearest Neighbors with vectorized distance computations.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'knn_lvl1_bruteforce',
        'task_type': 'classification',
        'description': 'k-Nearest Neighbors classifier with pure tensor distance computations',
        'input_type': 'float32',
        'output_type': 'int64'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the device for computation."""
    return device


def make_dataloaders(n_train=800, n_val=200, n_features=5, n_classes=3, batch_size=32):
    """
    Create synthetic classification dataset and dataloaders.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        n_features: Number of features
        n_classes: Number of classes
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, input_dim, output_dim
    """
    # Generate synthetic data with clear class separation
    X_train = []
    y_train = []
    
    # Create class centers with separation
    class_centers = np.random.randn(n_classes, n_features) * 2
    
    for i in range(n_train):
        class_idx = i % n_classes
        center = class_centers[class_idx]
        # Add noise to create clusters
        sample = center + np.random.randn(n_features) * 0.5
        X_train.append(sample)
        y_train.append(class_idx)
    
    X_val = []
    y_val = []
    
    for i in range(n_val):
        class_idx = i % n_classes
        center = class_centers[class_idx]
        sample = center + np.random.randn(n_features) * 0.5
        X_val.append(sample)
        y_val.append(class_idx)
    
    X_train = torch.FloatTensor(np.array(X_train))
    y_train = torch.LongTensor(np.array(y_train))
    X_val = torch.FloatTensor(np.array(X_val))
    y_val = torch.LongTensor(np.array(y_val))
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, n_features, n_classes


class kNNModel(nn.Module):
    """
    k-Nearest Neighbors model (lazy learner).
    Stores training data and performs prediction via distance computation.
    """
    
    def __init__(self, n_neighbors=5):
        super(kNNModel, self).__init__()
        self.n_neighbors = n_neighbors
        self.training_data = None
        self.training_labels = None
    
    def forward(self, x):
        """Forward pass computes predictions via kNN."""
        return self.predict(x)
    
    def fit(self, X, y):
        """
        Store training data (lazy learning).
        
        Args:
            X: Training features [N, D]
            y: Training labels [N]
        """
        self.training_data = X.clone().detach().to(device)
        self.training_labels = y.clone().detach().to(device)
    
    def compute_distances(self, X):
        """
        Compute L2 distances between test and training points (vectorized).
        
        Args:
            X: Test features [N_test, D]
        
        Returns:
            distances: L2 distances [N_test, N_train]
        """
        # Vectorized L2 distance computation
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        X_sq = torch.sum(X**2, dim=1, keepdim=True)  # [N_test, 1]
        train_sq = torch.sum(self.training_data**2, dim=1, keepdim=True)  # [N_train, 1]
        cross_term = torch.matmul(X, self.training_data.t())  # [N_test, N_train]
        
        distances = torch.sqrt(torch.clamp(X_sq + train_sq.t() - 2 * cross_term, min=1e-12))
        return distances
    
    def predict(self, X):
        """
        Predict labels for input samples using kNN.
        
        Args:
            X: Input features [N_test, D]
        
        Returns:
            predictions: Predicted labels [N_test]
        """
        X = X.to(device)
        
        with torch.no_grad():
            # Compute distances
            distances = self.compute_distances(X)  # [N_test, N_train]
            
            # Find k nearest neighbors
            _, indices = torch.topk(distances, k=self.n_neighbors, dim=1, largest=False)
            
            # Get labels of neighbors
            neighbor_labels = self.training_labels[indices]  # [N_test, k]
            
            # Majority vote - count occurrences of each class
            predictions = torch.mode(neighbor_labels, dim=1)[0]
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities using kNN.
        
        Args:
            X: Input features [N_test, D]
        
        Returns:
            probabilities: Class probabilities [N_test, n_classes]
        """
        X = X.to(device)
        
        with torch.no_grad():
            # Compute distances
            distances = self.compute_distances(X)
            
            # Find k nearest neighbors
            _, indices = torch.topk(distances, k=self.n_neighbors, dim=1, largest=False)
            
            # Get labels of neighbors
            neighbor_labels = self.training_labels[indices]
            
            # Count class occurrences
            n_classes = len(torch.unique(self.training_labels))
            batch_size = X.shape[0]
            probabilities = torch.zeros(batch_size, n_classes, device=device)
            
            for i in range(batch_size):
                for j in range(self.n_neighbors):
                    probabilities[i, neighbor_labels[i, j]] += 1
            
            # Normalize to probabilities
            probabilities /= self.n_neighbors
        
        return probabilities


def build_model(n_neighbors=5):
    """
    Build the kNN model.
    
    Args:
        n_neighbors: Number of neighbors to use
    
    Returns:
        model: kNNModel instance
    """
    model = kNNModel(n_neighbors=n_neighbors)
    return model


def train(model, train_loader, val_loader=None, epochs=1, verbose=True):
    """
    Train the kNN model (stores training data).
    
    Args:
        model: kNNModel instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of epochs (kNN is lazy, so this is mostly for interface)
        verbose: Print training progress
    
    Returns:
        model: Trained model
    """
    # kNN is a lazy learner - just store the training data
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        model.fit(X_batch, y_batch)
    
    if verbose:
        print(f"Training completed. Stored {len(model.training_data)} training samples.")
    
    return model


def evaluate(model, data_loader, verbose=True):
    """
    Evaluate the kNN model.
    
    Args:
        model: kNNModel instance
        data_loader: Data loader for evaluation
        verbose: Print evaluation metrics
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Get predictions
            predictions = model.predict(X_batch)
            
            # Compute accuracy (as a proxy for "loss" - 1 - accuracy)
            correct = (predictions == y_batch).sum().item()
            total = y_batch.shape[0]
            accuracy = correct / total
            
            all_predictions.append(predictions.cpu())
            all_labels.append(y_batch.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Compute overall metrics
    correct = (all_predictions == all_labels).sum().item()
    total = len(all_labels)
    accuracy = correct / total
    
    metrics = {
        'accuracy': accuracy,
        'loss': 1.0 - accuracy,  # Cross-entropy-like loss proxy
        'correct': correct,
        'total': total
    }
    
    if verbose:
        print(f"Evaluation Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Correct: {correct}/{total}")
    
    return metrics


def predict(model, X):
    """
    Make predictions on input data.
    
    Args:
        model: kNNModel instance
        X: Input features [N, D]
    
    Returns:
        predictions: Predicted labels [N]
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        predictions = model.predict(X_tensor)
    
    return predictions.cpu().numpy()


def save_artifacts(model, metrics, output_dir='output'):
    """
    Save model artifacts and metrics.
    
    Args:
        model: kNNModel instance
        metrics: Dictionary of metrics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state (training data)
    state = {
        'training_data': model.training_data.cpu() if model.training_data is not None else None,
        'training_labels': model.training_labels.cpu() if model.training_labels is not None else None,
        'n_neighbors': model.n_neighbors
    }
    torch.save(state, os.path.join(output_dir, 'model_state.pt'))
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")


def main():
    """Main function to run the kNN classification task."""
    print("=" * 60)
    print("kNN Classifier (Brute Force) - PyTorch Implementation")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, n_features, n_classes = make_dataloaders(
        n_train=800, n_val=200, n_features=5, n_classes=3, batch_size=32
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Features: {n_features}, Classes: {n_classes}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(n_neighbors=5)
    print(f"Model: kNN with k=5 neighbors")
    
    # Train model
    print("\nTraining model...")
    model = train(model, train_loader, val_loader, epochs=1, verbose=True)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, verbose=True)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, verbose=True)
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, {'train': train_metrics, 'val': val_metrics}, output_dir='output')
    
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
    
    # Check 1: Train accuracy > 0.85
    check1 = train_metrics['accuracy'] > 0.85
    print(f"{'✓' if check1 else '✗'} Train Accuracy > 0.85: {train_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check1
    
    # Check 2: Val accuracy > 0.80
    check2 = val_metrics['accuracy'] > 0.80
    print(f"{'✓' if check2 else '✗'} Val Accuracy > 0.80: {val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check2
    
    # Check 3: Val accuracy > Train accuracy - 0.15 (no significant overfitting)
    check3 = (train_metrics['accuracy'] - val_metrics['accuracy']) < 0.15
    print(f"{'✓' if check3 else '✗'} Accuracy gap < 0.15: {train_metrics['accuracy'] - val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check3
    
    # Check 4: Loss decreased
    check4 = train_metrics['loss'] < 1.0
    print(f"{'✓' if check4 else '✗'} Train Loss < 1.0: {train_metrics['loss']:.4f}")
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
    sys.exit(main())

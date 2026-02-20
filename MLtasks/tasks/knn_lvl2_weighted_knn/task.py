"""
kNN (Distance-Weighted + Regression) Implementation
Implements k-Nearest Neighbors with distance-weighted voting for classification and regression.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
OUTPUT_DIR = '/Developer/AIserver/output/tasks/knn_lvl2_weighted_knn'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'name': 'knn_weighted',
        'description': 'k-Nearest Neighbors with distance-weighted voting',
        'supported_modes': ['classification', 'regression'],
        'default_mode': 'regression'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(cfg):
    """
    Create dataloaders for training and validation.
    
    Args:
        cfg: Configuration dictionary containing data parameters
        
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    # Generate synthetic data based on mode
    mode = cfg.get('model', {}).get('mode', 'regression')
    n_samples = cfg.get('data', {}).get('n_samples', 500)
    n_features = cfg.get('data', {}).get('n_features', 5)
    noise = cfg.get('data', {}).get('noise', 0.1)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    if mode == 'regression':
        # Regression: continuous target
        y = np.sum(X, axis=1) + noise * np.random.randn(n_samples)
        y = y.astype(np.float32)
    else:  # classification
        # Classification: binary target based on sum of features
        y = (np.sum(X, axis=1) > 0).astype(np.int64)
    
    # Split into train and validation
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) if mode == 'regression' else torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1) if mode == 'regression' else torch.LongTensor(y_val)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    batch_size = cfg.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class DistanceWeightedKNN(nn.Module):
    """
    Distance-Weighted k-Nearest Neighbors model.
    
    For classification: Uses inverse distance weighting
    For regression: Uses inverse distance weighting for predictions
    """
    
    def __init__(self, cfg):
        super(DistanceWeightedKNN, self).__init__()
        self.k = cfg.get('model', {}).get('k', 5)
        self.mode = cfg.get('model', {}).get('mode', 'regression')
        self.device = get_device()
        
    def forward(self, x):
        """Forward pass - computes predictions using distance-weighted kNN."""
        return self
    
    def fit(self, X_train, y_train):
        """Store training data for kNN."""
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def _compute_distances(self, x):
        """Compute Euclidean distances between x and training points."""
        # x shape: (N_test, n_features)
        # X_train shape: (N_train, n_features)
        # Output shape: (N_test, N_train)
        
        # Using broadcasting: (N_test, 1, n_features) - (1, N_train, n_features)
        diff = x.unsqueeze(1) - self.X_train.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
        return distances
    
    def predict(self, x):
        """Make predictions using distance-weighted kNN."""
        with torch.no_grad():
            distances = self._compute_distances(x)
            
            # Get k nearest neighbors
            k_distances, k_indices = torch.topk(distances, k=min(self.k, len(self.X_train)), 
                                               dim=1, largest=False)
            
            # Get corresponding labels
            if self.mode == 'regression':
                # For regression, y_train should be (N_train, 1) or (N_train,)
                if self.y_train.dim() == 1:
                    k_labels = self.y_train[k_indices]
                else:
                    k_labels = self.y_train[k_indices].squeeze(-1)
                
                # Compute inverse distances as weights (with small epsilon to avoid division by zero)
                epsilon = 1e-8
                weights = 1.0 / (k_distances + epsilon)
                
                # Normalize weights
                weights_sum = torch.sum(weights, dim=1, keepdim=True)
                weights = weights / weights_sum
                
                # Weighted average
                predictions = torch.sum(weights * k_labels, dim=1, keepdim=True)
                return predictions
            else:  # classification
                # For classification, get the class labels
                k_labels = self.y_train[k_indices]
                
                # Compute inverse distances as weights
                epsilon = 1e-8
                weights = 1.0 / (k_distances + epsilon)
                
                # Get unique classes
                unique_classes = torch.unique(self.y_train)
                predictions = torch.zeros(x.size(0), len(unique_classes), device=x.device)
                
                # Accumulate weighted votes for each class
                for i, cls in enumerate(unique_classes):
                    mask = (k_labels == cls).float()
                    class_weights = mask * weights
                    predictions[:, i] = torch.sum(class_weights, dim=1)
                
                # Return class with highest weighted vote
                _, predicted = torch.max(predictions, 1)
                return predicted.unsqueeze(1)


def build_model(cfg):
    """
    Build the kNN model.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        model: Built model instance
    """
    model = DistanceWeightedKNN(cfg)
    
    # Get data loaders to fit the model
    train_loader, _ = make_dataloaders(cfg)
    
    # Extract training data and fit the model
    X_train, y_train = next(iter(train_loader))
    
    # Fit the model with training data
    model.fit(X_train, y_train)
    
    return model


def train(model, train_loader, cfg):
    """
    Train the model (for kNN, this is just fitting with training data).
    
    Args:
        model: Model to train
        train_loader: Training data loader
        cfg: Configuration dictionary
        
    Returns:
        model: Trained model
    """
    # For kNN, "training" means storing the training data
    # We already did this in build_model, but we can refit if needed
    all_X, all_y = [], []
    for X, y in train_loader:
        all_X.append(X)
        all_y.append(y)
    
    X_train = torch.cat(all_X, dim=0)
    y_train = torch.cat(all_y, dim=0)
    
    model.fit(X_train, y_train)
    
    return model


def evaluate(model, val_loader, cfg):
    """
    Evaluate the model on validation data.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        cfg: Configuration dictionary
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    mode = cfg.get('model', {}).get('mode', 'regression')
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(model.device)
            y = y.to(model.device)
            
            # Make predictions
            if mode == 'regression':
                preds = model.predict(X)
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
            else:
                preds = model.predict(X)
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics based on mode
    metrics = {}
    
    if mode == 'regression':
        # For regression: MSE, RMSE, R2
        mse = torch.mean((all_preds - all_targets) ** 2).item()
        rmse = np.sqrt(mse)
        
        # R2 score
        ss_res = torch.sum((all_targets - all_preds) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot).item() if ss_tot.item() != 0 else 0.0
        
        metrics['mse'] = mse
        metrics['rmse'] = rmse
        metrics['r2'] = r2
        metrics['mae'] = torch.mean(torch.abs(all_preds - all_targets)).item()
    else:  # classification
        # For classification: accuracy, precision, recall, F1
        correct = (all_preds.squeeze() == all_targets).sum().item()
        total = len(all_targets)
        accuracy = correct / total
        
        metrics['accuracy'] = accuracy
        metrics['error_rate'] = 1 - accuracy
        
        # Per-class metrics (simplified)
        unique_classes = torch.unique(all_targets)
        for cls in unique_classes:
            cls_mask = (all_targets == cls)
            tp = ((all_preds.squeeze() == cls) & cls_mask).sum().item()
            fp = ((all_preds.squeeze() == cls) & ~cls_mask).sum().item()
            fn = (~((all_preds.squeeze() == cls) | cls_mask)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f'precision_class_{cls.item()}'] = precision
            metrics[f'recall_class_{cls.item()}'] = recall
            metrics[f'f1_class_{cls.item()}'] = f1
    
    return metrics


def predict(model, X):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        X: Input features (numpy array or tensor)
        
    Returns:
        predictions: Model predictions
    """
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X).to(model.device)
    
    model.eval()
    with torch.no_grad():
        predictions = model.predict(X)
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    return predictions


def save_artifacts(model, metrics, cfg, epoch=None):
    """
    Save model artifacts and evaluation results.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics dictionary
        cfg: Configuration dictionary
        epoch: Optional epoch number for checkpointing
    """
    # Save model state (for kNN, save the training data)
    model_path = os.path.join(OUTPUT_DIR, 'knn_model.pth')
    torch.save({
        'X_train': model.X_train if hasattr(model, 'X_train') else None,
        'y_train': model.y_train if hasattr(model, 'y_train') else None,
        'k': model.k,
        'mode': model.mode,
        'state_dict': model.state_dict()
    }, model_path)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save configuration
    config_path = os.path.join(OUTPUT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    
    # Create visualization if regression
    mode = cfg.get('model', {}).get('mode', 'regression')
    if mode == 'regression' and hasattr(model, 'X_train'):
        # Create a simple visualization of predictions vs targets
        plt.figure(figsize=(10, 6))
        
        # Get predictions on training data for visualization
        with torch.no_grad():
            train_preds = model.predict(model.X_train).cpu().numpy()
            train_targets = model.y_train.cpu().numpy()
        
        # Scatter plot
        plt.scatter(train_targets, train_preds, alpha=0.5, label='Predictions')
        plt.plot([train_targets.min(), train_targets.max()], 
                [train_targets.min(), train_targets.max()], 'r--', lw=2, label='Perfect fit')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'kNN Regression: Predictions vs True Values (k={model.k})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f"MSE: {metrics.get('mse', 0):.4f}\nRMSE: {metrics.get('rmse', 0):.4f}\nR²: {metrics.get('r2', 0):.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plot_path = os.path.join(OUTPUT_DIR, 'predictions.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run the kNN task."""
    print("=" * 60)
    print("kNN (Distance-Weighted + Regression) Task")
    print("=" * 60)
    
    # Configuration
    cfg = {
        'model': {
            'mode': 'regression',  # 'regression' or 'classification'
            'k': 5,
        },
        'data': {
            'n_samples': 500,
            'n_features': 5,
            'noise': 0.1
        },
        'training': {
            'batch_size': 32,
            'epochs': 10
        }
    }
    
    # Parse command line arguments for mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode in ['classification', 'regression']:
            cfg['model']['mode'] = mode
            print(f"Using mode: {mode}")
    
    print(f"\nConfiguration:")
    print(json.dumps(cfg, indent=2))
    
    # Set device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(cfg)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)
    print(f"Model: k={model.k}, mode={model.mode}")
    
    # Train model
    print("\nTraining model...")
    model = train(model, train_loader, cfg)
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_metrics = evaluate(model, train_loader, cfg)
    print("Training Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation data
    print("\nEvaluating on validation data...")
    val_metrics = evaluate(model, val_loader, cfg)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Quality assertions
    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)
    
    mode = cfg['model']['mode']
    passed = True
    
    if mode == 'regression':
        # Regression quality checks
        r2_threshold = 0.85
        mse_threshold = 0.5
        
        r2 = val_metrics.get('r2', 0)
        mse = val_metrics.get('mse', float('inf'))
        rmse = val_metrics.get('rmse', float('inf'))
        
        print(f"\nR² Score: {r2:.4f} (threshold: > {r2_threshold})")
        if r2 > r2_threshold:
            print("  ✓ PASS: R² score meets threshold")
        else:
            print(f"  ✗ FAIL: R² score {r2:.4f} below threshold {r2_threshold}")
            passed = False
        
        print(f"MSE: {mse:.4f} (threshold: < {mse_threshold})")
        if mse < mse_threshold:
            print("  ✓ PASS: MSE meets threshold")
        else:
            print(f"  ✗ FAIL: MSE {mse:.4f} above threshold {mse_threshold}")
            passed = False
        
        print(f"RMSE: {rmse:.4f}")
        
    else:  # classification
        # Classification quality checks
        accuracy_threshold = 0.80
        
        accuracy = val_metrics.get('accuracy', 0)
        error_rate = val_metrics.get('error_rate', 1)
        
        print(f"\nAccuracy: {accuracy:.4f} (threshold: > {accuracy_threshold})")
        if accuracy > accuracy_threshold:
            print("  ✓ PASS: Accuracy meets threshold")
        else:
            print(f"  ✗ FAIL: Accuracy {accuracy:.4f} below threshold {accuracy_threshold}")
            passed = False
        
        print(f"Error Rate: {error_rate:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics, cfg)
    
    # Final summary
    print("\n" + "=" * 60)
    if passed:
        print("PASS: All quality thresholds met!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality thresholds not met")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
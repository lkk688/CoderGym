"""
SVM Multiclass (One-vs-Rest) Implementation
Implements OVR wrapper and reports macro metrics.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Get the appropriate device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata():
    """Return task metadata."""
    return {
        'task_type': 'multiclass_classification',
        'num_classes': 3,
        'input_type': 'tabular',
        'metrics': ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'mse', 'r2']
    }

def make_dataloaders(n_train=800, n_val=200, n_features=2, n_classes=3, batch_size=32):
    """Create train and validation dataloaders."""
    # Generate synthetic data with 3 distinct clusters
    # Use same random_state for both to ensure consistent cluster positions
    X_train, y_train = make_blobs(
        n_samples=n_train,
        centers=n_classes,
        cluster_std=1.5,
        random_state=42
    )
    
    X_val, y_val = make_blobs(
        n_samples=n_val,
        centers=n_classes,
        cluster_std=1.5,
        random_state=42  # Same random state for consistent clusters
    )
    
    # Standardize features - fit on train, transform both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Store scalers for later use
    data_info = {
        'scaler': scaler,
        'n_features': n_features,
        'n_classes': n_classes,
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'train_class_dist': {int(i): int(np.sum(y_train == i)) for i in range(n_classes)},
        'val_class_dist': {int(i): int(np.sum(y_val == i)) for i in range(n_classes)}
    }
    
    return train_loader, val_loader, data_info

class SVMModel(nn.Module):
    """Wrapper for SVM model to work with PyTorch interface."""
    
    def __init__(self, n_features, n_classes):
        super(SVMModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.device = get_device()
        
        # Initialize SVM with RBF kernel
        base_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        self.model = OneVsRestClassifier(base_svm)
        self.trained = False
    
    def forward(self, x):
        """Forward pass - returns predictions."""
        if not self.trained:
            raise RuntimeError("Model must be trained before inference")
        
        # Handle both tensor and numpy inputs
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        predictions = self.model.predict(x)
        return torch.LongTensor(predictions)
    
    def predict_proba(self, x):
        """Predict class probabilities."""
        if not self.trained:
            raise RuntimeError("Model must be trained before inference")
        
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        return self.model.predict_proba(x)

def build_model(n_features, n_classes):
    """Build the SVM model."""
    model = SVMModel(n_features, n_classes)
    return model

def train(model, train_loader, val_loader, epochs=10, device=None):
    """Train the SVM model."""
    if device is None:
        device = get_device()
    
    # Extract all training data
    X_train = []
    y_train = []
    
    for batch_X, batch_y in train_loader:
        X_train.append(batch_X)
        y_train.append(batch_y)
    
    X_train = torch.cat(X_train).numpy()
    y_train = torch.cat(y_train).numpy()
    
    print(f"Training samples: {len(y_train)}")
    print(f"Class distribution - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Train the SVM model
    print("Training SVM model...")
    model.model.fit(X_train, y_train)
    model.trained = True
    
    print("Training completed!")
    return model

def evaluate(model, data_loader, data_info, split_name='Validation'):
    """Evaluate the model and return metrics."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Move to device
            batch_X = batch_X.to(model.device)
            
            # Get predictions
            preds = model(batch_X)
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_y.numpy())
            
            # Get probabilities if available
            if hasattr(model.model, 'predict_proba'):
                probs = model.predict_proba(batch_X)
                all_probs.append(probs)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1_macro': f1_score(all_targets, all_preds, average='macro'),
        'precision_macro': precision_score(all_targets, all_preds, average='macro'),
        'recall_macro': recall_score(all_targets, all_preds, average='macro'),
        'mse': mean_squared_error(all_targets, all_preds),
        'r2': r2_score(all_targets, all_preds)
    }
    
    print(f"\n{split_name} Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics

def predict(model, X):
    """Make predictions on input data."""
    model.eval()
    
    if isinstance(X, torch.Tensor):
        X = X.to(model.device)
    else:
        X = torch.FloatTensor(X).to(model.device)
    
    with torch.no_grad():
        predictions = model(X)
    
    return predictions.numpy()

def save_artifacts(model, metrics, data_info, output_dir='output'):
    """Save model artifacts and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(output_dir, 'svm_model.pkl')
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model.model, f)
    
    # Save data info - exclude scaler for JSON serialization
    info_path = os.path.join(output_dir, 'data_info.json')
    data_info_for_json = {
        'n_features': data_info['n_features'],
        'n_classes': data_info['n_classes'],
        'train_samples': data_info['train_samples'],
        'val_samples': data_info['val_samples'],
        'train_class_dist': data_info['train_class_dist'],
        'val_class_dist': data_info['val_class_dist']
    }
    with open(info_path, 'w') as f:
        json.dump(data_info_for_json, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nArtifacts saved to {output_dir}")

def main():
    """Main function to run the SVM multiclass OVR task."""
    print("=" * 60)
    print("SVM Multiclass (One-vs-Rest) - Task Execution")
    print("=" * 60)
    
    # Set seeds
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task type: {metadata['task_type']}")
    print(f"Number of classes: {metadata['num_classes']}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, data_info = make_dataloaders(
        n_train=800,
        n_val=200,
        n_features=2,
        n_classes=3,
        batch_size=32
    )
    
    print(f"Training samples: {data_info['train_samples']}")
    print(f"Validation samples: {data_info['val_samples']}")
    print(f"Class distribution - Train: {data_info['train_class_dist']}")
    print(f"Class distribution - Val: {data_info['val_class_dist']}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(
        n_features=data_info['n_features'],
        n_classes=data_info['n_classes']
    )
    
    # Train model
    print("\nTraining model...")
    model = train(model, train_loader, val_loader, epochs=10, device=device)
    
    # Evaluate on training set
    print("\n" + "=" * 60)
    print("Evaluating on training set...")
    print("=" * 60)
    train_metrics = evaluate(model, train_loader, data_info, split_name='Train')
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Evaluating on validation set...")
    print("=" * 60)
    val_metrics = evaluate(model, val_loader, data_info, split_name='Validation')
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics, data_info, output_dir='output')
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nTrain Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"Train F1 Macro:  {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:    {val_metrics['f1_macro']:.4f}")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)
    
    quality_checks = []
    
    # Check 1: Train accuracy > 0.85
    check1 = train_metrics['accuracy'] > 0.85
    quality_checks.append(('Train Accuracy > 0.85', check1, train_metrics['accuracy']))
    
    # Check 2: Val accuracy > 0.80
    check2 = val_metrics['accuracy'] > 0.80
    quality_checks.append(('Val Accuracy > 0.80', check2, val_metrics['accuracy']))
    
    # Check 3: Val F1 macro > 0.85
    check3 = val_metrics['f1_macro'] > 0.85
    quality_checks.append(('Val F1 Macro > 0.85', check3, val_metrics['f1_macro']))
    
    # Check 4: Val MSE < 1.0
    check4 = val_metrics['mse'] < 1.0
    quality_checks.append(('Val MSE < 1.0', check4, val_metrics['mse']))
    
    # Check 5: Accuracy gap < 0.15
    accuracy_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
    check5 = accuracy_gap < 0.15
    quality_checks.append(('Accuracy gap < 0.15', check5, accuracy_gap))
    
    # Print check results
    all_passed = True
    for check_name, passed, value in quality_checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}: {value:.4f}")
        if not passed:
            all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
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

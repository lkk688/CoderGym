"""
Random Forest Implementation from Scratch (Simplified)
- Random feature subsampling + bagging
- OOB (Out-of-Bag) score calculation
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
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set paths
OUTPUT_DIR = '/Developer/AIserver/output/tasks/ens_lvl2_random_forest'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the ML task."""
    return {
        'task_type': 'regression',
        'model_type': 'random_forest',
        'description': 'Random Forest with feature subsampling and bagging',
        'metrics': ['mse', 'r2', 'mae'],
        'oob_score': True,
        'n_estimators': 10,
        'max_depth': 5,
        'max_features': 'sqrt'
    }

# Simplified Decision Tree for Regression
class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf value (mean of samples)

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.n_features = None
        self.feature_indices = None
        
    def fit(self, X, y):
        self.n_features = X.shape[1]
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(self.n_features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(self.n_features))
        elif self.max_features is None:
            self.max_features = self.n_features
            
        # Random feature selection
        np.random.seed(self.random_state)
        self.feature_indices = np.random.choice(self.n_features, size=min(self.max_features, self.n_features), replace=False)
        
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return DecisionTreeNode(value=np.mean(y))
        
        # Find best split using only selected features
        best_feature = None
        best_threshold = None
        best_mse = float('inf')
        
        for feature_idx in self.feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feature_idx], np.linspace(0, 100, 20))
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Calculate MSE
                left_mse = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0
                right_mse = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0
                total_mse = (left_mse + right_mse) / n_samples
                
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # If no valid split found, create leaf
        if best_feature is None:
            return DecisionTreeNode(value=np.mean(y))
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(feature_idx=best_feature, threshold=best_threshold, 
                               left=left_tree, right=right_tree)
    
    def predict_one(self, x):
        node = self.root
        while node.value is None:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

class RandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=5, max_features='sqrt', 
                 min_samples_split=2, random_state=None, oob_score=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.oob_score = oob_score
        self.trees = []
        self.oob_predictions_ = None
        self.oob_score_ = None
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.trees = []
        
        # Set random state for reproducibility
        np.random.seed(self.random_state)
        
        # Initialize OOB predictions
        if self.oob_score:
            self.oob_predictions_ = np.zeros(n_samples)
            self.oob_counts_ = np.zeros(n_samples)
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[bootstrap_indices] = False
            oob_indices = np.where(oob_mask)[0]
            
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Create and train tree with different random state
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + i if self.random_state else i
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            # Store OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_preds = tree.predict(X[oob_indices])
                self.oob_predictions_[oob_indices] += oob_preds
                self.oob_counts_[oob_indices] += 1
        
        # Calculate OOB score
        if self.oob_score and np.sum(self.oob_counts_) > 0:
            valid_mask = self.oob_counts_ > 0
            self.oob_predictions_[valid_mask] /= self.oob_counts_[valid_mask]
            
            # Calculate OOB MSE and R2
            oob_mse = np.mean((y[valid_mask] - self.oob_predictions_[valid_mask]) ** 2)
            ss_tot = np.sum((y[valid_mask] - np.mean(y[valid_mask])) ** 2)
            ss_res = np.sum((y[valid_mask] - self.oob_predictions_[valid_mask]) ** 2)
            self.oob_score_ = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += tree.predict(X)
        return predictions / len(self.trees)
    
    def get_oob_score(self):
        return self.oob_score_

def make_dataloaders(batch_size: int = 32, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create dataloaders for the California Housing dataset."""
    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Error: scikit-learn is required. Install with: pip install scikit-learn")
        sys.exit(1)
    
    # Load dataset
    try:
        data = fetch_california_housing()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to simple synthetic data
        np.random.seed(42)
        X = np.random.randn(1000, 8)
        y = np.random.randn(1000) * 10 + 20
        feature_names = [f'feature_{i}' for i in range(8)]
    else:
        X = data.data
        y = data.target
        feature_names = data.feature_names
    
    # Split data: train (70%), validation (15%), test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    dataloader_info = {
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'feature_names': feature_names,
        'scaler': scaler
    }
    
    return train_loader, val_loader, test_loader, dataloader_info

def build_model(n_features: int = 8, n_estimators: int = 10, max_depth: int = 5) -> RandomForestRegressor:
    """Build the Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='log2',
        min_samples_split=2,
        random_state=42,
        oob_score=True
    )
    return model

def train(model: RandomForestRegressor, train_loader: DataLoader, 
          val_loader: DataLoader, device: torch.device, 
          epochs: int = 1, verbose: bool = True) -> Dict[str, List[float]]:
    """Train the Random Forest model."""
    # For Random Forest, we train on the entire training set at once
    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy().flatten()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    train_preds = model.predict(X_train)
    train_mse = np.mean((y_train - train_preds) ** 2)
    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
    ss_res_train = np.sum((y_train - train_preds) ** 2)
    train_r2 = 1 - (ss_res_train / ss_tot_train) if ss_tot_train > 0 else 0.0
    
    # Calculate validation metrics
    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy().flatten()
    val_preds = model.predict(X_val)
    val_mse = np.mean((y_val - val_preds) ** 2)
    ss_tot_val = np.sum((y_val - np.mean(y_val)) ** 2)
    ss_res_val = np.sum((y_val - val_preds) ** 2)
    val_r2 = 1 - (ss_res_val / ss_tot_val) if ss_tot_val > 0 else 0.0
    
    history = {
        'train_mse': [train_mse],
        'train_r2': [train_r2],
        'val_mse': [val_mse],
        'val_r2': [val_r2],
        'oob_score': [model.get_oob_score()] if model.get_oob_score() is not None else [0.0]
    }
    
    if verbose:
        print(f"Training completed:")
        print(f"  Train MSE: {train_mse:.4f}, Train R2: {train_r2:.4f}")
        print(f"  Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")
        print(f"  OOB Score: {model.get_oob_score():.4f}" if model.get_oob_score() is not None else "  OOB Score: N/A")
    
    return history

def evaluate(model: RandomForestRegressor, data_loader: DataLoader, 
             device: torch.device) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    X = data_loader.dataset.tensors[0].numpy()
    y = data_loader.dataset.tensors[1].numpy().flatten()
    
    # Make predictions
    preds = model.predict(X)
    
    # Calculate metrics
    mse = np.mean((y - preds) ** 2)
    mae = np.mean(np.abs(y - preds))
    
    # R2 score
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - preds) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Explained variance
    var_y = np.var(y)
    var_residual = np.var(y - preds)
    explained_variance = 1 - (var_residual / var_y) if var_y > 0 else 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'explained_variance': float(explained_variance)
    }

def predict(model: RandomForestRegressor, X: np.ndarray, 
            device: torch.device) -> np.ndarray:
    """Make predictions using the trained model."""
    return model.predict(X)

def save_artifacts(model: RandomForestRegressor, history: Dict[str, List[float]], 
                   dataloader_info: Dict, output_dir: str = OUTPUT_DIR) -> None:
    """Save model artifacts and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters as JSON
    model_metadata = {
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'max_features': model.max_features,
        'min_samples_split': model.min_samples_split,
        'oob_score': model.oob_score,
        'oob_score_value': float(model.oob_score_) if model.oob_score_ is not None else None,
        'n_features': dataloader_info.get('n_features', 8),
        'n_train': dataloader_info.get('n_train', 0),
        'n_val': dataloader_info.get('n_val', 0),
        'n_test': dataloader_info.get('n_test', 0)
    }
    
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save feature names if available
    if 'feature_names' in dataloader_info:
        with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
            json.dump(dataloader_info['feature_names'], f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")

def main():
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, dataloader_info = make_dataloaders(
        batch_size=32, test_size=0.2
    )
    
    print(f"  Train samples: {dataloader_info['n_train']}")
    print(f"  Validation samples: {dataloader_info['n_val']}")
    print(f"  Test samples: {dataloader_info['n_test']}")
    print(f"  Features: {dataloader_info['n_features']}")
    
    # Build model with increased complexity
    print("\nBuilding Random Forest model...")
    model = build_model(
        n_features=dataloader_info['n_features'],
        n_estimators=50,  # Increase number of trees
        max_depth=10      # Increase depth for better fit
    )
    
    print(f"  n```python
_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  max_features: {model.max_features}")
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Train model
    print("\nTraining model...")
    history = train(model, train_loader, val_loader, device, epochs=1, verbose=True)
    
    # Evaluate on all sets
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    for metric, value in train_metrics.items():
        print(f"  Train {metric.upper()}: {value:.4f}")
    
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    for metric, value in val_metrics.items():
        print(f"  Val {metric.upper()}: {value:.4f}")
    
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    for metric, value in test_metrics.items():
        print(f"  Test {metric.upper()}: {value:.4f}")
    
    # Print OOB score
    print(f"\nOOB Score: {model.get_oob_score():.4f}" if model.get_oob_score() is not None else "\nOOB Score: N/A")
    
    # Quality assertions
    print("\n============================================================")
    print("Quality Assertions")
    print("============================================================")
    
    train_r2 = train_metrics['r2']
    val_r2 = val_metrics['r2']
    test_r2 = test_metrics['r2']
    oob_score = model.get_oob_score() if model.get_oob_score() is not None else 0.0
    
    assertions_passed = True
    
    if train_r2 > 0.7:
        print(f"✓ Train R2 ({train_r2:.4f}) > 0.7")
    else:
        print(f"✗ Train R2 ({train_r2:.4f}) should be > 0.7")
        assertions_passed = False
    
    if val_r2 > 0.6:
        print(f"✓ Val R2 ({val_r2:.4f}) > 0.6")
    else:
        print(f"✗ Val R2 ({val_r2:.4f}) should be > 0.6")
        assertions_passed = False
    
    if val_metrics['mse'] < 10.0:
        print(f"✓ Val MSE ({val_metrics['mse']:.4f}) < 10.0")
    else:
        print(f"✗ Val MSE ({val_metrics['mse']:.4f}) should be < 10.0")
        assertions_passed = False
    
    if abs(val_r2 - test_r2) < 0.05:
        print(f"✓ Val-Test R2 difference ({abs(val_r2 - test_r2):.4f}) < 0.05")
    else:
        print(f"✗ Val-Test R2 difference ({abs(val_r2 - test_r2):.4f}) should be < 0.05")
        assertions_passed = False
    
    if abs(train_r2 - val_r2) < 0.2:
        print(f"✓ Train-Val R2 difference ({abs(train_r2 - val_r2):.4f}) < 0.2")
    else:
        print(f"✗ Train-Val R2 difference ({abs(train_r2 - val_r2):.4f}) should be < 0.2")
        assertions_passed = False
    
    if assertions_passed:
        print("\n============================================================")
        print("SUCCESS: All quality thresholds met!")
        print("============================================================")
    else:
        print("\n============================================================")
        print("FAIL: Some quality thresholds not met!")
        print("============================================================")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, history, dataloader_info)

if __name__ == "__main__":
    main()

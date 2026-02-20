"""
Boosting (Engineering & Benchmarking)
XGBoost-style gradient boosting with shrinkage, subsampling, and depth tuning.
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
import matplotlib.pyplot as plt
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
def get_device() -> torch.device:
    """Get computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============== Data Functions ==============

def make_dataloaders(
    n_samples: int = 1000,
    n_features: int = 20,
    noise: float = 0.1,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create synthetic regression dataset and dataloaders.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level in target
        train_ratio: Training data ratio
        batch_size: Batch size
        device: Computation device
        
    Returns:
        Tuple of (train_loader, val_loader, input_dim)
    """
    if device is None:
        device = get_device()
    
    # Generate synthetic data with non-linear relationships
    X = torch.randn(n_samples, n_features).to(device)
    
    # Create non-linear target with feature interactions
    y = (
        2.0 * X[:, 0] + 
        1.5 * X[:, 1] ** 2 + 
        0.8 * X[:, 2] * X[:, 3] +
        0.5 * torch.sin(X[:, 4] * 3) +
        0.3 * X[:, 5] * X[:, 6] +
        torch.randn(n_samples).to(device) * noise
    )
    
    # Split data
    n_train = int(n_samples * train_ratio)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = TensorDataset(X[train_indices], y[train_indices].unsqueeze(1))
    val_dataset = TensorDataset(X[val_indices], y[val_indices].unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, n_features

# ============== Model Components ==============

class DecisionTreeNode:
    """Node in a decision tree."""
    def __init__(self, feature_idx: int = None, threshold: float = None,
                 left: 'DecisionTreeNode' = None, right: 'DecisionTreeNode' = None,
                 value: float = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf value

class DecisionTree:
    """Simple decision tree for regression."""
    
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> 'DecisionTree':
        """Fit the decision tree to data."""
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X: torch.Tensor, y: torch.Tensor, depth: int) -> DecisionTreeNode:
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_samples < self.min_samples_leaf):
            return DecisionTreeNode(value=y.mean().item())
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return DecisionTreeNode(value=y.mean().item())
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return DecisionTreeNode(value=y.mean().item())
        
        # Recursively build subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _find_best_split(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[int, float]:
        """Find the best split for a node."""
        n_samples, n_features = X.shape
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        current_mse = self._mse(y)
        
        for feature_idx in range(n_features):
            thresholds = torch.unique(X[:, feature_idx])
            
            # Sample thresholds if too many unique values
            if len(thresholds) > 20:
                thresholds = torch.quantile(X[:, feature_idx], 
                                           torch.linspace(0, 1, 20, device=X.device))
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Calculate weighted MSE reduction
                left_mse = self._mse(left_y)
                right_mse = self._mse(right_y)
                
                n_left = left_y.shape[0]
                n_right = right_y.shape[0]
                n_total = n_samples
                
                weighted_mse = (n_left * left_mse + n_right * right_mse) / n_total
                gain = current_mse - weighted_mse
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold.item()
        
        return best_feature, best_threshold
    
    def _mse(self, y: torch.Tensor) -> float:
        """Calculate mean squared error."""
        if len(y) == 0:
            return 0.0
        return ((y - y.mean()) ** 2).mean().item()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict for all samples."""
        return torch.tensor([self._predict_single(x) for x in X], 
                           device=X.device, dtype=torch.float32)
    
    def _predict_single(self, x: torch.Tensor) -> float:
        """Predict for a single sample."""
        node = self.root
        while node.value is None:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def get_depth(self) -> int:
        """Get the maximum depth of the tree."""
        return self._get_depth_recursive(self.root)
    
    def _get_depth_recursive(self, node: DecisionTreeNode) -> int:
        """Recursively get tree depth."""
        if node is None or node.value is not None:
            return 0
        return 1 + max(self._get_depth_recursive(node.left), 
                      self._get_depth_recursive(node.right))

class GradientBoostingRegressor:
    """Gradient Boosting Regressor with shrinkage and subsampling."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 device: Optional[torch.device] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.device = device if device else get_device()
        
        self.trees: List[DecisionTree] = []
        self.initial_prediction: float = 0.0
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> 'GradientBoostingRegressor':
        """Fit the gradient boosting model."""
        n_samples = X.shape[0]
        
        # Initialize with mean
        self.initial_prediction = y.mean().item()
        predictions = torch.full((n_samples,), self.initial_prediction, 
                                device=self.device, dtype=torch.float32)
        
        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient for squared error)
            residuals = y.squeeze() - predictions
            
            # Subsampling (bagging)
            if self.subsample < 1.0:
                indices = torch.randperm(n_samples)[:int(n_samples * self.subsample)]
                X_sample = X[indices]
                residuals_sample = residuals[indices]
            else:
                X_sample = X
                residuals_sample = residuals
            
            # Fit tree to residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sample, residuals_sample.unsqueeze(1))
            self.trees.append(tree)
            
            # Update predictions with shrinkage
            tree_pred = tree.predict(X)
            predictions = predictions + self.learning_rate * tree_pred
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict for new data."""
        predictions = torch.full((X.shape[0],), self.initial_prediction,
                                device=self.device, dtype=torch.float32)
        
        for tree in self.trees:
            predictions = predictions + self.learning_rate * tree.predict(X)
        
        return predictions.unsqueeze(1)
    
    def get_complexity(self) -> Dict[str, Any]:
        """Get model complexity information."""
        depths = [tree.get_depth() for tree in self.trees]
        return {
            'n_estimators': len(self.trees),
            'max_depth': max(depths) if depths else 0,
            'avg_depth': sum(depths) / len(depths) if depths else 0,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample
        }

# ============== Training Functions ==============

def build_model(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    device: Optional[torch.device] = None
) -> GradientBoostingRegressor:
    """
    Build a gradient boosting model with specified hyperparameters.
    
    Args:
        n_estimators: Number of boosting stages
        learning_rate: Shrinkage factor
        max_depth: Maximum depth of trees
        subsample: Fraction of samples for bagging
        device: Computation device
        
    Returns:
        GradientBoostingRegressor model
    """
    if device is None:
        device = get_device()
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        device=device
    )
    return model

def train(
    model: GradientBoostingRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train the gradient boosting model.
    
    Args:
        model: GradientBoostingRegressor model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Computation device
        verbose: Print training progress
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = get_device()
    
    # Collect all training data
    X_train_list, y_train_list = [], []
    for X_batch, y_batch in train_loader:
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)
    
    X_train = torch.cat(X_train_list, dim=0).to(device)
    y_train = torch.cat(y_train_list, dim=0).to(device)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, train_pred)
    
    # Calculate validation metrics
    X_val_list, y_val_list = [], []
    for X_batch, y_batch in val_loader:
        X_val_list.append(X_batch)
        y_val_list.append(y_batch)
    
    X_val = torch.cat(X_val_list, dim=0).to(device)
    y_val = torch.cat(y_val_list, dim=0).to(device)
    
    val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_pred)
    
    history = {
        'train_mse': [train_metrics['mse']],
        'train_r2': [train_metrics['r2']],
        'val_mse': [val_metrics['mse']],
        'val_r2': [val_metrics['r2']]
    }
    
    if verbose:
        print(f"Training complete!")
        print(f"Train MSE: {train_metrics['mse']:.4f}, Train R2: {train_metrics['r2']:.4f}")
        print(f"Val MSE: {val_metrics['mse']:.4f}, Val R2: {val_metrics['r2']:.4f}")
    
    return history

def evaluate(
    model: GradientBoostingRegressor,
    data_loader: DataLoader,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate the model on data.
    
    Args:
        model: GradientBoostingRegressor model
        data_loader: Data loader
        device: Computation device
        
    Returns:
        Dictionary with evaluation metrics (MSE, R2)
    """
    if device is None:
        device = get_device()
    
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model.predict(X_batch)
            all_preds.append(preds)
            all_targets.append(y_batch)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = calculate_metrics(all_targets, all_preds)
    
    return metrics

def predict(
    model: GradientBoostingRegressor,
    data_loader: DataLoader,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate predictions.
    
    Args:
        model: GradientBoostingRegressor model
        data_loader: Data loader
        device: Computation device
        
    Returns:
        Tensor with predictions
    """
    if device is None:
        device = get_device()
    
    model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            preds = model.predict(X_batch)
            all_preds.append(preds)
    
    return torch.cat(all_preds, dim=0)

# ============== Metrics ==============

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    Calculate standard regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with MSE, R2 score
    """
    # Ensure tensors are on CPU for numpy operations
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()
    
    # MSE
    mse = np.mean((y_true_np - y_pred_np) ** 2)
    
    # R2 score
    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # MAE
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'mae': float(mae)
    }

# ============== Artifact Saving ==============

def save_artifacts(
    model: GradientBoostingRegressor,
    history: Dict[str, List[float]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    output_dir: str = "./output",
    task_name: str = "boosting"
) -> None:
    """
    Save model artifacts and report.
    
    Args:
        model: Trained model
        history: Training history
        train_metrics: Training metrics
        val_metrics: Validation metrics
        output_dir: Output directory
        task_name: Task name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters
    model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        'n_estimators': model.n_estimators,
        'learning_rate': model.learning_rate,
        'max_depth': model.max_depth,
        'subsample': model.subsample,
        'trees': model.trees,
        'initial_prediction': model.initial_prediction
    }, model_path)
    
    # Save metrics
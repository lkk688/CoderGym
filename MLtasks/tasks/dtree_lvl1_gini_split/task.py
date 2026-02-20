"""
Decision Tree Classifier (Gini Impurity, From Scratch)
CART-style binary splits with Gini impurity
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "decision_tree_gini",
        "task_type": "classification",
        "description": "CART-style binary splits with Gini impurity",
        "input_type": "tabular",
        "output_type": "discrete"
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Get the computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecisionTreeNode:
    """Node in the decision tree."""
    def __init__(self, feature_idx: int = None, threshold: float = None, 
                 left: 'DecisionTreeNode' = None, right: 'DecisionTreeNode' = None,
                 value: int = None, prob: Dict[int, float] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf class value
        self.prob = prob if prob is not None else {}  # Class probabilities at leaf


class DecisionTree:
    """Decision Tree Classifier using Gini impurity."""
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None
        
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _information_gain(self, y: np.ndarray, left_idx: np.ndarray, 
                         right_idx: np.ndarray) -> float:
        """Calculate information gain using Gini impurity."""
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0.0
        
        parent_gini = self._gini_impurity(y)
        n = len(y)
        n_left = len(left_idx)
        n_right = len(right_idx)
        
        child_gini = (n_left / n) * self._gini_impurity(y[left_idx]) + \
                     (n_right / n) * self._gini_impurity(y[right_idx])
        
        return parent_gini - child_gini
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split for a node."""
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None, -1
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_idx = np.where(feature_values <= threshold)[0]
                right_idx = np.where(feature_values > threshold)[0]
                
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                
                gain = self._information_gain(y, left_idx, right_idx)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_features = X.shape[1]
        
        # Calculate class distribution
        class_counts = Counter(y)
        class_probs = {cls: count / n_samples for cls, count in class_counts.items()}
        majority_class = max(class_counts, key=class_counts.get)
        
        # Stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(class_counts) == 1:
            return DecisionTreeNode(value=majority_class, prob=class_probs)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            return DecisionTreeNode(value=majority_class, prob=class_probs)
        
        # Split the data
        left_idx = np.where(X[:, best_feature] <= best_threshold)[0]
        right_idx = np.where(X[:, best_feature] > best_threshold)[0]
        
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return DecisionTreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Fit the decision tree to the training data."""
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Predict class for a single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for all samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for all samples."""
        probs = []
        for x in X:
            node = self.root
            while node.value is None:
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            # Create probability array
            prob = np.zeros(self.n_classes)
            for cls, p in node.prob.items():
                prob[cls] = p
            probs.append(prob)
        return np.array(probs)


def make_dataloaders(batch_size: int = 32, val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for the Iris dataset."""
    from sklearn.datasets import load_iris
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split into train and validation
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    val_size = int(n_samples * val_ratio)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def build_model() -> DecisionTree:
    """Build the decision tree model."""
    # Hyperparameters
    max_depth = 5
    min_samples_split = 2
    
    model = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
    return model


def train(model: DecisionTree, train_loader: DataLoader, 
          device: torch.device = device) -> float:
    """Train the decision tree model."""
    # Collect all training data
    X_train = []
    y_train = []
    
    for batch_X, batch_y in train_loader:
        X_train.append(batch_X.numpy())
        y_train.append(batch_y.numpy())
    
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return 0.0  # No loss to return for decision tree


def evaluate(model: DecisionTree, data_loader: DataLoader, 
             device: torch.device = device) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Convert to numpy directly (data is already on CPU from dataloader)
            X = batch_X.numpy()
            y = batch_y.numpy()
            
            # Predict
            preds = model.predict(X)
            all_preds.extend(preds)
            all_targets.extend(y)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_targets)
    
    # Calculate per-class metrics
    classes = np.unique(all_targets)
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    
    for cls in classes:
        tp = np.sum((all_preds == cls) & (all_targets == cls))
        fp = np.sum((all_preds == cls) & (all_targets != cls))
        fn = np.sum((all_preds != cls) & (all_targets == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class[cls] = precision
        recall_per_class[cls] = recall
        f1_per_class[cls] = f1
    
    # Macro averages
    precision_macro = np.mean(list(precision_per_class.values()))
    recall_macro = np.mean(list(recall_per_class.values()))
    f1_macro = np.mean(list(f1_per_class.values()))
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }
    
    return metrics


def predict(model: DecisionTree, X: np.ndarray, 
            device: torch.device = device) -> np.ndarray:
    """Make predictions on new data."""
    return model.predict(X)


def save_artifacts(model: DecisionTree, metrics: Dict[str, float], 
                   output_dir: str = "output") -> None:
    """Save model artifacts and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters
    model_path = os.path.join(output_dir, "model_state_dict.json")
    model_data = {
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'n_classes': model.n_classes
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")


def main():
    """Main function to run the decision tree training and evaluation."""
    print("=" * 60)
    print("Decision Tree Classifier (Gini Impurity)")
    print("=" * 60)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(batch_size=32, val_ratio=0.2)
    
    # Count samples
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    print(f"Model: DecisionTree(max_depth={model.max_depth}, min_samples_split={model.min_samples_split})")
    
    # Train model
    print("\nTraining model...")
    train_loss = train(model, train_loader, device)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print("Train Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics, output_dir="output")
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"Train F1 Macro:  {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:    {val_metrics['f1_macro']:.4f}")
    print("=" * 60)
    
    # Quality checks
    print("\nQUALITY CHECKS")
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
    
    # Check 3: Val F1 macro > 0.85
    check3 = val_metrics['f1_macro'] > 0.85
    print(f"{'✓' if check3 else '✗'} Val F1 Macro > 0.85: {val_metrics['f1_macro']:.4f}")
    checks_passed = checks_passed and check3
    
    # Check 4: Accuracy gap < 0.15
    accuracy_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
    check4 = accuracy_gap < 0.15
    print(f"{'✓' if check4 else '✗'} Accuracy gap < 0.15: {accuracy_gap:.4f}")
    checks_passed = checks_passed and check4
    
    print("=" * 60)
    
    if checks_passed:
        print("PASS: All quality checks passed!")
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

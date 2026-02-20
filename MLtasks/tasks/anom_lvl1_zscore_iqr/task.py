"""
Statistical Anomaly Detection using Z-score and IQR methods.
Implements unsupervised anomaly detection with evaluation on synthetic data.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'anomaly_detection_zscore_iqr',
        'task_type': 'unsupervised',
        'description': 'Statistical anomaly detection using Z-score and IQR methods',
        'input_type': 'tabular',
        'output_type': 'binary_classification'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_train=800, n_val=200, anomaly_ratio=0.1, batch_size=32):
    """
    Create synthetic data with anomalies.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        anomaly_ratio: Proportion of anomalies in the data
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Generate normal data from a multivariate distribution
    n_features = 5
    mean = np.zeros(n_features)
    cov = np.eye(n_features) * 0.5
    
    # Add some correlation
    cov[0, 1] = cov[1, 0] = 0.3
    cov[2, 3] = cov[3, 2] = 0.2
    
    # Generate normal samples
    X_normal_train = np.random.multivariate_normal(mean, cov, n_train)
    X_normal_val = np.random.multivariate_normal(mean, cov, n_val)
    
    # Generate anomalies (outliers)
    n_anomalies_train = int(n_train * anomaly_ratio)
    n_anomalies_val = int(n_val * anomaly_ratio)
    
    # Anomalies are far from the mean
    X_anomaly_train = np.random.multivariate_normal(mean + 4, cov * 2, n_anomalies_train)
    X_anomaly_val = np.random.multivariate_normal(mean + 4, cov * 2, n_anomalies_val)
    
    # Combine normal and anomaly data
    X_train = np.vstack([X_normal_train, X_anomaly_train])
    X_val = np.vstack([X_normal_val, X_anomaly_val])
    
    # Create labels (0 = normal, 1 = anomaly)
    y_train = np.array([0] * n_train + [1] * n_anomalies_train)
    y_val = np.array([0] * n_val + [1] * n_anomalies_val)
    
    # Shuffle the data
    train_indices = np.random.permutation(len(X_train))
    val_indices = np.random.permutation(len(X_val))
    
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    X_val = X_val[val_indices]
    y_val = y_val[val_indices]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset, val_dataset


class ZScoreIQRAnomalyDetector:
    """
    Statistical anomaly detector using Z-score and IQR methods.
    """
    
    def __init__(self, z_threshold=3.0, iqr_multiplier=1.5):
        """
        Initialize the anomaly detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
            iqr_multiplier: IQR multiplier for anomaly detection
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.mean = None
        self.std = None
        self.q1 = None
        self.q3 = None
        self.iqr = None
        self.fitted = False
    
    def fit(self, X):
        """
        Fit the anomaly detector to the data.
        
        Args:
            X: Training data (numpy array or torch tensor)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Calculate statistics for Z-score method
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        
        # Calculate statistics for IQR method
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        
        self.fitted = True
    
    def predict_zscore(self, X):
        """
        Predict anomalies using Z-score method.
        
        Args:
            X: Data to predict (numpy array or torch tensor)
        
        Returns:
            Binary predictions (0 = normal, 1 = anomaly)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Calculate Z-scores
        z_scores = np.abs((X - self.mean) / self.std)
        
        # Flag anomalies where any feature exceeds threshold
        predictions = (np.max(z_scores, axis=1) > self.z_threshold).astype(int)
        
        return predictions
    
    def predict_iqr(self, X):
        """
        Predict anomalies using IQR method.
        
        Args:
            X: Data to predict (numpy array or torch tensor)
        
        Returns:
            Binary predictions (0 = normal, 1 = anomaly)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Calculate bounds
        lower_bound = self.q1 - self.iqr_multiplier * self.iqr
        upper_bound = self.q3 + self.iqr_multiplier * self.iqr
        
        # Flag anomalies outside bounds
        predictions = ((X < lower_bound) | (X > upper_bound)).any(axis=1).astype(int)
        
        return predictions
    
    def predict_combined(self, X):
        """
        Predict anomalies using combined Z-score and IQR methods.
        
        Args:
            X: Data to predict (numpy array or torch tensor)
        
        Returns:
            Binary predictions (0 = normal, 1 = anomaly)
        """
        zscore_preds = self.predict_zscore(X)
        iqr_preds = self.predict_iqr(X)
        
        # Combine predictions (anomaly if either method flags it)
        return np.maximum(zscore_preds, iqr_preds)
    
    def score_zscore(self, X):
        """
        Calculate anomaly scores using Z-score method.
        
        Args:
            X: Data to score (numpy array or torch tensor)
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        z_scores = np.abs((X - self.mean) / self.std)
        return np.max(z_scores, axis=1)
    
    def score_iqr(self, X):
        """
        Calculate anomaly scores using IQR method.
        
        Args:
            X: Data to score (numpy array or torch tensor)
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        lower_bound = self.q1 - self.iqr_multiplier * self.iqr
        upper_bound = self.q3 + self.iqr_multiplier * self.iqr
        
        # Calculate distance from bounds
        lower_dist = np.maximum(0, self.q1 - X)
        upper_dist = np.maximum(0, X - self.q3)
        
        return np.max(lower_dist + upper_dist, axis=1)
    
    def score_combined(self, X):
        """
        Calculate combined anomaly scores.
        
        Args:
            X: Data to score (numpy array or torch tensor)
        
        Returns:
            Combined anomaly scores
        """
        zscore_scores = self.score_zscore(X)
        iqr_scores = self.score_iqr(X)
        
        # Normalize scores to [0, 1] range and combine
        if zscore_scores.max() > 0:
            zscore_scores = zscore_scores / zscore_scores.max()
        if iqr_scores.max() > 0:
            iqr_scores = iqr_scores / iqr_scores.max()
        
        return (zscore_scores + iqr_scores) / 2


def build_model():
    """
    Build the anomaly detection model.
    
    Returns:
        ZScoreIQRAnomalyDetector instance
    """
    return ZScoreIQRAnomalyDetector(z_threshold=3.0, iqr_multiplier=1.5)


def train(model, train_loader, device):
    """
    Train the anomaly detection model.
    
    Args:
        model: The anomaly detection model
        train_loader: Training data loader
        device: Device for computation
    
    Returns:
        Trained model
    """
    # Collect all training data
    X_train = []
    for X, _ in train_loader:
        X_train.append(X)
    
    X_train = torch.cat(X_train).to(device)
    
    # Fit the model
    model.fit(X_train)
    
    return model


def evaluate(model, data_loader, device):
    """
    Evaluate the anomaly detection model.
    
    Args:
        model: The anomaly detection model
        data_loader: Data loader for evaluation
        device: Device for computation
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Collect all data
    X_all = []
    y_all = []
    
    for X, y in data_loader:
        X_all.append(X)
        y_all.append(y)
    
    X_all = torch.cat(X_all).to(device)
    y_all = torch.cat(y_all).cpu().numpy()
    
    # Make predictions using combined method
    predictions = model.predict_combined(X_all)
    scores = model.score_combined(X_all)
    
    # Calculate metrics
    metrics = {}
    
    # Binary classification metrics
    metrics['precision'] = precision_score(y_all, predictions, zero_division=0)
    metrics['recall'] = recall_score(y_all, predictions, zero_division=0)
    metrics['f1'] = f1_score(y_all, predictions, zero_division=0)
    
    # For AUC, we need probability-like scores
    # Use the negative of anomaly score as "normality" score
    try:
        metrics['roc_auc'] = roc_auc_score(y_all, -scores)
        metrics['pr_auc'] = average_precision_score(y_all, -scores)
    except ValueError:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    # Accuracy
    metrics['accuracy'] = np.mean(predictions == y_all)
    
    # Count anomalies detected
    metrics['anomalies_detected'] = int(np.sum(predictions))
    metrics['true_anomalies'] = int(np.sum(y_all))
    
    return metrics


def predict(model, X, device):
    """
    Make predictions on new data.
    
    Args:
        model: The anomaly detection model
        X: Input data
        device: Device for computation
    
    Returns:
        Predictions and scores
    """
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    
    X = X.to(device)
    
    predictions = model.predict_combined(X)
    scores = model.score_combined(X)
    
    return predictions, scores


def save_artifacts(model, metrics, output_dir='output'):
    """
    Save model artifacts and metrics.
    
    Args:
        model: The trained model
        metrics: Evaluation metrics dictionary
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters
    model_params = {
        'z_threshold': model.z_threshold,
        'iqr_multiplier': model.iqr_multiplier,
        'mean': model.mean.tolist() if model.mean is not None else None,
        'std': model.std.tolist() if model.std is not None else None,
        'q1': model.q1.tolist() if model.q1 is not None else None,
        'q3': model.q3.tolist() if model.q3 is not None else None,
        'iqr': model.iqr.tolist() if model.iqr is not None else None,
        'fitted': model.fitted
    }
    
    with open(os.path.join(output_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=2)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")


def main():
    """Main function to run the anomaly detection task."""
    print("=" * 60)
    print("Statistical Anomaly Detection (Z-score + IQR)")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, train_dataset, val_dataset = make_dataloaders(
        n_train=800, n_val=200, anomaly_ratio=0.1, batch_size=32
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Count anomalies
    y_train = train_dataset.tensors[1].numpy()
    y_val = val_dataset.tensors[1].numpy()
    print(f"Training anomalies: {int(np.sum(y_train))} ({100*np.sum(y_train)/len(y_train):.1f}%)")
    print(f"Validation anomalies: {int(np.sum(y_val))} ({100*np.sum(y_val)/len(y_val):.1f}%)")
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    
    # Train model
    print("\nTraining model...")
    model = train(model, train_loader, device)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print("Train Metrics:")
    for key, value in train_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics, output_dir='output')
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Precision: {train_metrics['precision']:.4f}")
    print(f"Val Precision:   {val_metrics['precision']:.4f}")
    print(f"Train Recall:    {train_metrics['recall']:.4f}")
    print(f"Val Recall:      {val_metrics['recall']:.4f}")
    print(f"Train F1:        {train_metrics['f1']:.4f}")
    print(f"Val F1:          {val_metrics['f1']:.4f}")
    print(f"Train Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)
    
    quality_passed = True
    
    # Check 1: Validation precision > 0.8
    check1 = val_metrics['precision'] > 0.8
    print(f"{'[PASS]' if check1 else '[FAIL]'} Val Precision > 0.8: {val_metrics['precision']:.4f}")
    quality_passed = quality_passed and check1
    
    # Check 2: Validation recall > 0.7
    check2 = val_metrics['recall'] > 0.7
    print(f"{'[PASS]' if check2 else '[FAIL]'} Val Recall > 0.7: {val_metrics['recall']:.4f}")
    quality_passed = quality_passed and check2
    
    # Check 3: Validation F1 > 0.75
    check3 = val_metrics['f1'] > 0.75
    print(f"{'[PASS]' if check3 else '[FAIL]'} Val F1 > 0.75: {val_metrics['f1']:.4f}")
    quality_passed = quality_passed and check3
    
    # Check 4: Accuracy gap < 0.15 (no severe overfitting)
    accuracy_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
    check4 = accuracy_gap < 0.15
    print(f"{'[PASS]' if check4 else '[FAIL]'} Accuracy gap < 0.15: {accuracy_gap:.4f}")
    quality_passed = quality_passed and check4
    
    # Check 5: Precision and recall are reasonable
    check5 = val_metrics['precision'] > 0.5 and val_metrics['recall'] > 0.5
    print(f"{'[PASS]' if check5 else '[FAIL]'} Precision and Recall > 0.5: P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}")
    quality_passed = quality_passed and check5
    
    print("\n" + "=" * 60)
    if quality_passed:
        print("ALL QUALITY CHECKS PASSED")
    else:
        print("SOME QUALITY CHECKS FAILED")
    print("=" * 60)
    
    return quality_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

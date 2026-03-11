"""
Ridge Regression with K-Fold Cross-Validation for Hyperparameter Tuning

Mathematical Formulation:
- Hypothesis: h_theta(X) = X @ theta
- Ridge Objective: J(theta) = (1/2m) * ||X @ theta - y||^2 + lambda * ||theta||^2
- Closed-form Solution: theta = (X^T X + lambda * I)^{-1} X^T y

This implementation uses PyTorch with manual k-fold cross-validation for hyperparameter selection.
The key innovation is implementing CV from scratch to select the optimal regularization parameter.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Output directory for artifacts
OUTPUT_DIR = './output/tasks/ridge_lvl1_cv_hyperparam'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'ridge_regression_cv_hyperparam',
        'description': 'Ridge Regression with k-fold CV for hyperparameter tuning',
        'input_dim': 8,
        'output_dim': 1,
        'model_type': 'ridge_regression',
        'loss_type': 'mse',
        'optimization': 'closed_form_with_cv',
        'dataset': 'california_housing'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(test_size=0.2, val_size=0.2, batch_size=32):
    """
    Load California Housing dataset and create train/val/test splits.
    
    Args:
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # Load California Housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Split train into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler


class RidgeRegressionModel:
    """
    Ridge Regression model with closed-form solution and k-fold cross-validation.
    
    Closed-form solution:
        theta = (X^T X + lambda * I)^{-1} X^T y
    
    Where lambda is the regularization parameter selected via CV.
    """
    
    def __init__(self, lambda_reg=1.0, device=None):
        """
        Initialize Ridge Regression model.
        
        Args:
            lambda_reg: L2 regularization parameter (lambda)
            device: Device for computation
        """
        self.lambda_reg = lambda_reg
        self.device = device if device is not None else get_device()
        self.theta = None
        self.fitted = False
    
    def fit(self, X, y):
        """
        Fit Ridge Regression using closed-form solution.
        
        theta = (X^T X + lambda * I)^{-1} X^T y
        
        Args:
            X: Input features of shape (N, D)
            y: Target values of shape (N, 1)
        """
        X = X.to(self.device)
        y = y.to(self.device)
        
        N, D = X.shape
        
        # Add bias term (intercept)
        X_bias = torch.cat([torch.ones(N, 1, device=self.device), X], dim=1)
        
        # Closed-form solution: theta = (X^T X + lambda * I)^{-1} X^T y
        XTX = X_bias.T @ X_bias
        reg_matrix = self.lambda_reg * torch.eye(D + 1, device=self.device)
        reg_matrix[0, 0] = 0  # Don't regularize bias term
        
        # Solve using torch.linalg.solve for numerical stability
        self.theta = torch.linalg.solve(XTX + reg_matrix, X_bias.T @ y)
        self.fitted = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input features of shape (N, D)
        
        Returns:
            Predictions of shape (N, 1)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = X.to(self.device)
        N = X.shape[0]
        
        # Add bias term
        X_bias = torch.cat([torch.ones(N, 1, device=self.device), X], dim=1)
        
        return X_bias @ self.theta
    
    def compute_mse(self, X, y):
        """Compute Mean Squared Error."""
        y_pred = self.predict(X)
        return torch.mean((y_pred - y.to(self.device)) ** 2).item()
    
    def compute_r2(self, X, y):
        """Compute R2 score."""
        y = y.to(self.device)
        y_pred = self.predict(X)
        
        ss_res = torch.sum((y - y_pred) ** 2).item()
        ss_tot = torch.sum((y - torch.mean(y)) ** 2).item()
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return r2


def k_fold_cross_validation(X, y, lambda_values, k_folds=5, device=None):
    """
    Perform k-fold cross-validation to select best lambda.
    
    Args:
        X: Input features tensor
        y: Target values tensor
        lambda_values: List of lambda values to try
        k_folds: Number of folds
        device: Computation device
    
    Returns:
        best_lambda, cv_scores_dict
    """
    N = X.shape[0]
    fold_size = N // k_folds
    indices = torch.randperm(N)
    
    cv_scores = {lam: [] for lam in lambda_values}
    
    print(f"\nPerforming {k_folds}-fold cross-validation...")
    
    for fold in range(k_folds):
        # Create fold indices
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k_folds - 1 else N
        
        val_indices = indices[val_start:val_end]
        train_indices = torch.cat([indices[:val_start], indices[val_end:]])
        
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        # Try each lambda
        for lam in lambda_values:
            model = RidgeRegressionModel(lambda_reg=lam, device=device)
            model.fit(X_train_fold, y_train_fold)
            mse = model.compute_mse(X_val_fold, y_val_fold)
            cv_scores[lam].append(mse)
        
        print(f"  Fold {fold + 1}/{k_folds} complete")
    
    # Compute mean CV score for each lambda
    mean_cv_scores = {lam: np.mean(scores) for lam, scores in cv_scores.items()}
    std_cv_scores = {lam: np.std(scores) for lam, scores in cv_scores.items()}
    
    # Select best lambda (lowest mean CV MSE)
    best_lambda = min(mean_cv_scores, key=mean_cv_scores.get)
    
    print(f"\nCross-validation results:")
    for lam in lambda_values:
        print(f"  lambda={lam:8.4f}: MSE={mean_cv_scores[lam]:.6f} ± {std_cv_scores[lam]:.6f}")
    print(f"\nBest lambda: {best_lambda}")
    
    return best_lambda, {
        'mean_scores': mean_cv_scores,
        'std_scores': std_cv_scores,
        'all_scores': cv_scores
    }


def build_model(lambda_reg=1.0, device=None):
    """Build Ridge Regression model."""
    return RidgeRegressionModel(lambda_reg=lambda_reg, device=device)


def train(model, train_loader):
    """
    Train Ridge Regression model.
    
    Args:
        model: RidgeRegressionModel instance
        train_loader: Training data loader
    
    Returns:
        Trained model
    """
    # Collect all training data
    X_list, y_list = [], []
    for X_batch, y_batch in train_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    
    X_train = torch.cat(X_list, dim=0)
    y_train = torch.cat(y_list, dim=0)
    
    # Fit model
    model.fit(X_train, y_train)
    
    return model


def evaluate(model, data_loader, split_name='Validation'):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader
        split_name: Name of the split (for printing)
    
    Returns:
        Dictionary with metrics
    """
    # Collect all data
    X_list, y_list = [], []
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Compute metrics
    mse = model.compute_mse(X, y)
    r2 = model.compute_r2(X, y)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'split': split_name
    }
    
    print(f"\n{split_name} Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R2:   {r2:.6f}")
    
    return metrics


def predict(model, X):
    """Make predictions on new data."""
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    return model.predict(X)


def save_artifacts(model, cv_results, train_metrics, val_metrics, test_metrics):
    """
    Save model artifacts and visualizations.
    
    Args:
        model: Trained model
        cv_results: Cross-validation results
        train_metrics: Training metrics
        val_metrics: Validation metrics
        test_metrics: Test metrics
    """
    # Save model parameters
    torch.save({
        'theta': model.theta,
        'lambda_reg': model.lambda_reg
    }, os.path.join(OUTPUT_DIR, 'ridge_model.pt'))
    
    # Plot CV results
    lambda_values = sorted(cv_results['mean_scores'].keys())
    mean_scores = [cv_results['mean_scores'][lam] for lam in lambda_values]
    std_scores = [cv_results['std_scores'][lam] for lam in lambda_values]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(lambda_values, mean_scores, yerr=std_scores, marker='o', capsize=5)
    plt.xscale('log')
    plt.xlabel('Lambda (Regularization Parameter)', fontsize=12)
    plt.ylabel('Cross-Validation MSE', fontsize=12)
    plt.title('Ridge Regression: Hyperparameter Tuning via Cross-Validation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cv_lambda_selection.png'), dpi=150)
    plt.close()
    
    # Plot train/val/test comparison
    splits = ['Train', 'Validation', 'Test']
    mse_values = [train_metrics['mse'], val_metrics['mse'], test_metrics['mse']]
    r2_values = [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(splits, mse_values, color=['blue', 'orange', 'green'], alpha=0.7)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Mean Squared Error by Split', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(splits, r2_values, color=['blue', 'orange', 'green'], alpha=0.7)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Score by Split', fontsize=13)
    ax2.axhline(y=0.7, color='r', linestyle='--', label='Threshold (0.7)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\nArtifacts saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    print("=" * 70)
    print("Task: Ridge Regression with K-Fold Cross-Validation")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"\nTask Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading California Housing dataset...")
    train_loader, val_loader, test_loader, scaler = make_dataloaders(
        test_size=0.2, val_size=0.2, batch_size=512
    )
    
    # Collect training data for CV
    X_list, y_list = [], []
    for X_batch, y_batch in train_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    X_train = torch.cat(X_list, dim=0)
    y_train = torch.cat(y_list, dim=0)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {sum(len(y) for _, y in val_loader)}")
    print(f"  Test samples: {sum(len(y) for _, y in test_loader)}")
    
    # Perform k-fold cross-validation to select best lambda
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda, cv_results = k_fold_cross_validation(
        X_train, y_train, lambda_values, k_folds=5, device=device
    )
    
    # Build model with best lambda
    print(f"\n{'=' * 70}")
    print(f"Training final model with best lambda={best_lambda}")
    print(f"{'=' * 70}")
    model = build_model(lambda_reg=best_lambda, device=device)
    
    # Train model
    model = train(model, train_loader)
    print("\nModel training complete!")
    
    # Evaluate on all splits
    train_metrics = evaluate(model, train_loader, split_name='Train')
    val_metrics = evaluate(model, val_loader, split_name='Validation')
    test_metrics = evaluate(model, test_loader, split_name='Test')
    
    # Save artifacts
    save_artifacts(model, cv_results, train_metrics, val_metrics, test_metrics)
    
    # Validation checks
    print(f"\n{'=' * 70}")
    print("VALIDATION CHECKS")
    print(f"{'=' * 70}")
    
    # Check 1: Test R2 should be > 0.7
    test_r2_threshold = 0.7
    test_r2_pass = test_metrics['r2'] > test_r2_threshold
    print(f"✓ Test R² > {test_r2_threshold}: {test_metrics['r2']:.6f} - {'PASS' if test_r2_pass else 'FAIL'}")
    
    # Check 2: Test MSE should be reasonable (< 1.0)
    test_mse_threshold = 1.0
    test_mse_pass = test_metrics['mse'] < test_mse_threshold
    print(f"✓ Test MSE < {test_mse_threshold}: {test_metrics['mse']:.6f} - {'PASS' if test_mse_pass else 'FAIL'}")
    
    # Check 3: No overfitting (train R2 - test R2 < 0.15)
    overfit_margin = train_metrics['r2'] - test_metrics['r2']
    no_overfit = overfit_margin < 0.15
    print(f"✓ No severe overfitting (margin < 0.15): {overfit_margin:.6f} - {'PASS' if no_overfit else 'FAIL'}")
    
    # Check 4: CV selected reasonable lambda
    reasonable_lambda = 0.001 <= best_lambda <= 1000.0
    print(f"✓ Reasonable lambda selected: {best_lambda} - {'PASS' if reasonable_lambda else 'FAIL'}")
    
    # Final verdict
    all_checks_pass = test_r2_pass and test_mse_pass and no_overfit and reasonable_lambda
    
    print(f"\n{'=' * 70}")
    if all_checks_pass:
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print(f"{'=' * 70}")
        sys.exit(0)
    else:
        print("✗ SOME VALIDATION CHECKS FAILED!")
        print(f"{'=' * 70}")
        sys.exit(1)

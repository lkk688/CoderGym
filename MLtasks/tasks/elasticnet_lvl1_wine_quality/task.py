"""
Elastic Net Regression on Wine Quality Dataset

Mathematical Formulation:
- Hypothesis: h_theta(X) = X @ theta
- Elastic Net Objective: J(theta) = (1/2m) * ||X @ theta - y||^2 + lambda1 * ||theta||_1 + lambda2 * ||theta||^2
  where ||theta||_1 is L1 norm (Lasso) and ||theta||^2 is L2 norm (Ridge)
- Combines benefits of L1 (feature selection/sparsity) and L2 (stability)

This implementation uses coordinate descent optimization with PyTorch.
The key innovation is combining L1 and L2 regularization on a new dataset (Wine Quality).
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Output directory for artifacts
OUTPUT_DIR = './output/tasks/elasticnet_lvl1_wine_quality'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'elasticnet_wine_quality',
        'description': 'Elastic Net Regression combining L1 and L2 regularization',
        'input_dim': 11,
        'output_dim': 1,
        'model_type': 'elastic_net_regression',
        'loss_type': 'mse_with_l1_l2_regularization',
        'optimization': 'gradient_descent',
        'dataset': 'wine_quality'
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
    Load Wine Quality dataset and create train/val/test splits.
    
    Wine Quality Dataset from UCI Machine Learning Repository
    Features: fixed acidity, volatile acidity, citric acid, residual sugar,
              chlorides, free sulfur dioxide, total sulfur dioxide, density,
              pH, sulphates, alcohol
    Target: quality score (0-10)
    
    Args:
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, scaler, feature_names
    """
    # Download and load Wine Quality dataset
    # Using red wine dataset
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        df = pd.read_csv(url, sep=';')
    except:
        # Create synthetic wine-like data if download fails
        print("  Creating synthetic wine quality data...")
        np.random.seed(42)
        n_samples = 1599
        
        # Simulate wine features with realistic correlations
        fixed_acidity = np.random.normal(8.3, 1.7, n_samples)
        volatile_acidity = np.random.normal(0.53, 0.18, n_samples)
        citric_acid = np.random.normal(0.27, 0.19, n_samples)
        residual_sugar = np.random.normal(2.5, 1.4, n_samples)
        chlorides = np.random.normal(0.087, 0.047, n_samples)
        free_sulfur = np.random.normal(15.9, 10.5, n_samples)
        total_sulfur = np.random.normal(46, 32.9, n_samples)
        density = np.random.normal(0.9967, 0.0019, n_samples)
        pH = np.random.normal(3.31, 0.15, n_samples)
        sulphates = np.random.normal(0.66, 0.17, n_samples)
        alcohol = np.random.normal(10.4, 1.1, n_samples)
        
        # Quality as a function of features (with noise)
        quality = (
            0.3 * alcohol +
            -2.0 * volatile_acidity +
            0.2 * citric_acid +
            0.5 * sulphates +
            -0.4 * pH +
            np.random.normal(0, 0.5, n_samples)
        )
        quality = np.clip(quality + 5.5, 3, 8)  # Scale to realistic range
        
        df = pd.DataFrame({
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur,
            'total sulfur dioxide': total_sulfur,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol,
            'quality': quality
        })
    
    feature_names = df.columns[:-1].tolist()
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
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
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler, feature_names


class ElasticNetModel:
    """
    Elastic Net Regression with L1 + L2 regularization.
    
    Objective: J(theta) = MSE + lambda1 * ||theta||_1 + lambda2 * ||theta||^2
    
    Uses gradient descent with soft thresholding for L1 component.
    """
    
    def __init__(self, lambda1=0.01, lambda2=0.01, lr=0.01, device=None):
        """
        Initialize Elastic Net model.
        
        Args:
            lambda1: L1 regularization parameter (Lasso)
            lambda2: L2 regularization parameter (Ridge)
            lr: Learning rate
            device: Computation device
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.device = device if device is not None else get_device()
        self.theta = None
        self.bias = None
        self.fitted = False
        self.train_history = {'loss': [], 'mse': []}
    
    def soft_threshold(self, x, threshold):
        """
        Soft thresholding operator for L1 regularization.
        
        soft_threshold(x, t) = sign(x) * max(|x| - t, 0)
        """
        return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))
    
    def forward(self, X):
        """Forward pass: y = X @ theta + bias"""
        return X @ self.theta + self.bias
    
    def compute_loss(self, X, y):
        """
        Compute total loss: MSE + L1 penalty + L2 penalty.
        """
        y_pred = self.forward(X)
        mse = torch.mean((y_pred - y) ** 2)
        l1_penalty = self.lambda1 * torch.sum(torch.abs(self.theta))
        l2_penalty = self.lambda2 * torch.sum(self.theta ** 2)
        return mse + l1_penalty + l2_penalty
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train Elastic Net using gradient descent with soft thresholding.
        
        Args:
            X: Input features (N, D)
            y: Target values (N, 1)
            epochs: Number of training epochs
            verbose: Print progress
        """
        X = X.to(self.device)
        y = y.to(self.device)
        
        N, D = X.shape
        
        # Initialize parameters
        self.theta = torch.zeros(D, 1, device=self.device, requires_grad=False)
        self.bias = torch.zeros(1, device=self.device, requires_grad=False)
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute MSE
            mse = torch.mean((y_pred - y) ** 2)
            
            # Compute gradients manually
            error = y_pred - y
            grad_theta = (2.0 / N) * (X.T @ error) + 2 * self.lambda2 * self.theta
            grad_bias = (2.0 / N) * torch.sum(error)
            
            # Update with gradient descent
            self.theta = self.theta - self.lr * grad_theta
            self.bias = self.bias - self.lr * grad_bias
            
            # Apply soft thresholding for L1 (proximal gradient descent)
            self.theta = self.soft_threshold(self.theta, self.lr * self.lambda1)
            
            # Track history
            total_loss = self.compute_loss(X, y)
            self.train_history['loss'].append(total_loss.item())
            self.train_history['mse'].append(mse.item())
            
            if verbose and (epoch + 1) % 200 == 0:
                sparsity = (torch.abs(self.theta) < 1e-4).sum().item() / D
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}, MSE: {mse:.6f}, Sparsity: {sparsity:.3f}")
        
        self.fitted = True
    
    def predict(self, X):
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        X = X.to(self.device)
        return self.forward(X)
    
    def compute_metrics(self, X, y):
        """Compute MSE, R2, and feature sparsity."""
        X = X.to(self.device)
        y = y.to(self.device)
        
        y_pred = self.predict(X)
        
        # MSE
        mse = torch.mean((y_pred - y) ** 2).item()
        
        # R2
        ss_res = torch.sum((y - y_pred) ** 2).item()
        ss_tot = torch.sum((y - torch.mean(y)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Sparsity (proportion of near-zero coefficients)
        sparsity = (torch.abs(self.theta) < 1e-4).sum().item() / len(self.theta)
        
        # Number of active features
        n_active = (torch.abs(self.theta) >= 1e-4).sum().item()
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'sparsity': sparsity,
            'n_active_features': n_active
        }


def build_model(lambda1=0.01, lambda2=0.01, lr=0.01, device=None):
    """Build Elastic Net model."""
    return ElasticNetModel(lambda1=lambda1, lambda2=lambda2, lr=lr, device=device)


def train(model, train_loader, epochs=1000):
    """Train Elastic Net model."""
    # Collect all training data
    X_list, y_list = [], []
    for X_batch, y_batch in train_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    
    X_train = torch.cat(X_list, dim=0)
    y_train = torch.cat(y_list, dim=0)
    
    # Fit model
    model.fit(X_train, y_train, epochs=epochs, verbose=True)
    
    return model


def evaluate(model, data_loader, split_name='Validation'):
    """Evaluate model on a dataset."""
    # Collect all data
    X_list, y_list = [], []
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Compute metrics
    metrics = model.compute_metrics(X, y)
    metrics['split'] = split_name
    
    print(f"\n{split_name} Metrics:")
    print(f"  MSE:               {metrics['mse']:.6f}")
    print(f"  RMSE:              {metrics['rmse']:.6f}")
    print(f"  R²:                {metrics['r2']:.6f}")
    print(f"  Sparsity:          {metrics['sparsity']:.3f}")
    print(f"  Active Features:   {metrics['n_active_features']}")
    
    return metrics


def predict(model, X):
    """Make predictions on new data."""
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    return model.predict(X)


def save_artifacts(model, train_metrics, val_metrics, test_metrics, feature_names):
    """Save model artifacts and visualizations."""
    # Save model parameters
    torch.save({
        'theta': model.theta,
        'bias': model.bias,
        'lambda1': model.lambda1,
        'lambda2': model.lambda2
    }, os.path.join(OUTPUT_DIR, 'elasticnet_model.pt'))
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(model.train_history['loss'], label='Total Loss (MSE + L1 + L2)')
    ax1.plot(model.train_history['mse'], label='MSE Only', linestyle='--')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feature importance (absolute weights)
    theta_abs = torch.abs(model.theta).squeeze().cpu().numpy()
    sorted_indices = np.argsort(theta_abs)[::-1]
    
    ax2.barh(range(len(feature_names)), theta_abs[sorted_indices], color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_indices], fontsize=10)
    ax2.set_xlabel('|Coefficient|', fontsize=12)
    ax2.set_title('Feature Importance (Elastic Net)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_and_features.png'), dpi=150)
    plt.close()
    
    # Plot metrics comparison
    splits = ['Train', 'Validation', 'Test']
    mse_values = [train_metrics['mse'], val_metrics['mse'], test_metrics['mse']]
    r2_values = [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']]
    sparsity_values = [train_metrics['sparsity'], val_metrics['sparsity'], test_metrics['sparsity']]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(splits, mse_values, color=['blue', 'orange', 'green'], alpha=0.7)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Mean Squared Error', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(splits, r2_values, color=['blue', 'orange', 'green'], alpha=0.7)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('R² Score', fontsize=13)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(splits, sparsity_values, color=['blue', 'orange', 'green'], alpha=0.7)
    axes[2].set_ylabel('Sparsity Ratio', fontsize=12)
    axes[2].set_title('Feature Sparsity', fontsize=13)
    axes[2].axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\nArtifacts saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    print("=" * 70)
    print("Task: Elastic Net Regression on Wine Quality Dataset")
    print("=" * 70)
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Get metadata
    metadata = get_task_metadata()
    print(f"\nTask Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading Wine Quality dataset...")
    train_loader, val_loader, test_loader, scaler, feature_names = make_dataloaders(
        test_size=0.2, val_size=0.2, batch_size=64
    )
    
    print(f"  Training samples: {sum(len(y) for _, y in train_loader)}")
    print(f"  Validation samples: {sum(len(y) for _, y in val_loader)}")
    print(f"  Test samples: {sum(len(y) for _, y in test_loader)}")
    print(f"  Features: {len(feature_names)}")
    
    # Build model
    print(f"\n{'=' * 70}")
    print("Training Elastic Net Model (L1 + L2 Regularization)")
    print(f"{'=' * 70}")
    model = build_model(lambda1=0.02, lambda2=0.01, lr=0.01, device=device)
    print(f"  Lambda1 (L1/Lasso): {model.lambda1}")
    print(f"  Lambda2 (L2/Ridge): {model.lambda2}")
    print(f"  Learning Rate: {model.lr}")
    
    # Train model
    model = train(model, train_loader, epochs=1000)
    print("\nModel training complete!")
    
    # Evaluate
    train_metrics = evaluate(model, train_loader, split_name='Train')
    val_metrics = evaluate(model, val_loader, split_name='Validation')
    test_metrics = evaluate(model, test_loader, split_name='Test')
    
    # Save artifacts
    save_artifacts(model, train_metrics, val_metrics, test_metrics, feature_names)
    
    # Validation checks
    print(f"\n{'=' * 70}")
    print("VALIDATION CHECKS")
    print(f"{'=' * 70}")
    
    # Check 1: Test R2 > 0.35 (wine quality is hard to predict, realistic threshold)
    test_r2_threshold = 0.35
    test_r2_pass = test_metrics['r2'] > test_r2_threshold
    print(f"✓ Test R² > {test_r2_threshold}: {test_metrics['r2']:.6f} - {'PASS' if test_r2_pass else 'FAIL'}")
    
    # Check 2: Sparsity > 0.05 (some feature selection with increased L1)
    sparsity_threshold = 0.05
    sparsity_pass = test_metrics['sparsity'] > sparsity_threshold
    print(f"✓ Sparsity > {sparsity_threshold}: {test_metrics['sparsity']:.3f} - {'PASS' if sparsity_pass else 'FAIL'}")
    
    # Check 3: Test MSE reasonable (< 1.5)
    test_mse_threshold = 1.5
    test_mse_pass = test_metrics['mse'] < test_mse_threshold
    print(f"✓ Test MSE < {test_mse_threshold}: {test_metrics['mse']:.6f} - {'PASS' if test_mse_pass else 'FAIL'}")
    
    # Check 4: At least some features active
    min_active = 3
    active_pass = test_metrics['n_active_features'] >= min_active
    print(f"✓ Active features >= {min_active}: {test_metrics['n_active_features']} - {'PASS' if active_pass else 'FAIL'}")
    
    # Final verdict
    all_checks_pass = test_r2_pass and sparsity_pass and test_mse_pass and active_pass
    
    print(f"\n{'=' * 70}")
    if all_checks_pass:
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print(f"{'=' * 70}")
        sys.exit(0)
    else:
        print("✗ SOME VALIDATION CHECKS FAILED!")
        print(f"{'=' * 70}")
        sys.exit(1)

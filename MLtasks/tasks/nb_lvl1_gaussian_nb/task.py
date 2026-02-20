"""
Gaussian Naive Bayes Implementation from First Principles
Compares with sklearn GaussianNB
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
OUTPUT_DIR = '/Developer/AIserver/output/tasks/nb_lvl1_gaussian_nb'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'gaussian_nb',
        'task_type': 'classification',
        'description': 'Gaussian Naive Bayes from first principles',
        'input_type': 'continuous',
        'output_type': 'categorical'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=1000, n_features=10, test_size=0.2, batch_size=32):
    """
    Create classification dataset and dataloaders.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        test_size: Proportion of data for validation
        batch_size: Batch size for training
    
    Returns:
        train_loader, val_loader, X_train, X_val, y_train, y_val
    """
    # Generate classification dataset with Gaussian features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.1),
        n_classes=3,
        class_sep=1.5,
        random_state=42
    )
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
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
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val


class GaussianNBModel(nn.Module):
    """
    Gaussian Naive Bayes implemented as a PyTorch module.
    
    This is a simple implementation that pre-computes the parameters
    and uses them for prediction without gradient computation.
    """
    def __init__(self, n_features, n_classes):
        super(GaussianNBModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_priors_ = None
        self.theta_ = None  # Mean of each feature per class
        self.sigma_ = None  # Variance of each feature per class
        
    def fit(self, X, y):
        """
        Fit the Gaussian NB model using first principles.
        
        Args:
            X: Training features (numpy array or torch tensor)
            y: Training labels (numpy array or torch tensor)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
            
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.theta_ = np.zeros((self.n_classes, n_features))
        self.sigma_ = np.zeros((self.n_classes, n_features))
        self.class_priors_ = np.zeros(self.n_classes)
        
        # Compute parameters for each class
        for c in range(self.n_classes):
            # Get samples belonging to class c
            X_c = X[y == c]
            
            # Compute mean (theta) and variance (sigma) for each feature
            self.theta_[c, :] = X_c.mean(axis=0)
            self.sigma_[c, :] = X_c.var(axis=0) + 1e-9  # Add small epsilon for numerical stability
            
            # Compute class prior
            self.class_priors_[c] = X_c.shape[0] / n_samples
    
    def _compute_log_likelihood(self, X):
        """
        Compute log likelihood using Gaussian PDF.
        
        log P(x|c) = sum_j log(N(x_j | theta[c,j], sigma[c,j]))
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            log_likelihood of shape (n_samples, n_classes)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            
        n_samples = X.shape[0]
        log_likelihood = np.zeros((n_samples, self.n_classes))
        
        for c in range(self.n_classes):
            # Gaussian log PDF: -0.5 * log(2*pi*sigma) - 0.5 * (x - mu)^2 / sigma
            mean = self.theta_[c]
            var = self.sigma_[c]
            
            # Compute log likelihood for each feature
            log_prob = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2) / var
            log_likelihood[:, c] = np.sum(log_prob, axis=1)
            
        return log_likelihood
    
    def _compute_log_posterior(self, X):
        """
        Compute log posterior: log P(y=c|x) ∝ log P(x|c) + log P(y=c)
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            log_posterior of shape (n_samples, n_classes)
        """
        log_likelihood = self._compute_log_likelihood(X)
        log_prior = np.log(self.class_priors_)
        
        # Add log prior to log likelihood (broadcast)
        log_posterior = log_likelihood + log_prior
        
        return log_posterior
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples, n_classes)
        """
        log_posterior = self._compute_log_posterior(X)
        
        # Use log-sum-exp trick for numerical stability
        max_log = np.max(log_posterior, axis=1, keepdims=True)
        exp_posterior = np.exp(log_posterior - max_log)
        probabilities = exp_posterior / np.sum(exp_posterior, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """
        Predict class labels using argmax of log-posterior.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        log_posterior = self._compute_log_posterior(X)
        return np.argmax(log_posterior, axis=1)
    
    def forward(self, X):
        """PyTorch forward method - returns log probabilities."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        log_posterior = self._compute_log_posterior(X)
        return torch.FloatTensor(log_posterior)


def build_model(n_features, n_classes, device):
    """
    Build the Gaussian NB model.
    
    Args:
        n_features: Number of features
        n_classes: Number of classes
        device: Computation device
        
    Returns:
        model: GaussianNBModel instance
    """
    model = GaussianNBModel(n_features, n_classes)
    model.device = device
    return model


def train(model, train_loader, device, epochs=1):
    """
    Train the Gaussian NB model.
    
    Note: Gaussian NB is a closed-form solution, so it doesn't need
    iterative training. We just fit it once on the training data.
    
    Args:
        model: GaussianNBModel instance
        train_loader: Training data loader
        device: Computation device
        epochs: Number of epochs (ignored for GBN)
        
    Returns:
        model: Trained model
    """
    # Get all training data
    X_train = []
    y_train = []
    
    for batch_X, batch_y in train_loader:
        X_train.append(batch_X)
        y_train.append(batch_y)
    
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    
    # Fit the model (closed-form solution)
    model.fit(X_train, y_train)
    
    return model


def evaluate(model, data_loader, device, dataset_name='validation'):
    """
    Evaluate the model and compute metrics.
    
    Args:
        model: Trained GaussianNBModel
        data_loader: Data loader for evaluation
        device: Computation device
        dataset_name: Name of the dataset for logging
        
    Returns:
        metrics: Dictionary of metrics (MSE, R2, accuracy, etc.)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Move to device
            batch_X = batch_X.to(device)
            
            # Get predictions
            preds = model.predict(batch_X)
            
            all_preds.append(preds)
            all_targets.append(batch_y.numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    
    # For regression-style metrics (treating as continuous for comparison)
    mse = mean_squared_error(all_targets, all_preds)
    
    # R2 score (for continuous approximation)
    r2 = r2_score(all_targets, all_preds)
    
    metrics = {
        f'{dataset_name}_accuracy': accuracy,
        f'{dataset_name}_mse': mse,
        f'{dataset_name}_r2': r2
    }
    
    return metrics


def predict(model, X, device):
    """
    Make predictions on new data.
    
    Args:
        model: Trained GaussianNBModel
        X: Input features
        device: Computation device
        
    Returns:
        predictions: Predicted labels
    """
    model.eval()
    
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
        elif isinstance(X, torch.Tensor):
            X = X.to(device)
        
        predictions = model.predict(X)
    
    return predictions


def save_artifacts(model, metrics, X_train, y_train, X_val, y_val):
    """
    Save model artifacts and visualizations.
    
    Args:
        model: Trained model
        metrics: Dictionary of metrics
        X_train, y_train: Training data
        X_val, y_val: Validation data
    """
    # Save model parameters
    np.savez(
        os.path.join(OUTPUT_DIR, 'model_parameters.npz'),
        theta=model.theta_,
        sigma=model.sigma_,
        class_priors=model.class_priors_
    )
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value:.4f}\n')
    
    # Create visualization of feature distributions per class
    if hasattr(model, 'theta_') and model.theta_ is not None:
        n_classes = model.theta_.shape[0]
        n_features = min(model.theta_.shape[1], 4)  # Plot first 4 features
        
        fig, axes = plt.subplots(n_classes, n_features, figsize=(4*n_features, 3*n_classes))
        
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        if n_features == 1:
            axes = axes.reshape(-1, 1)
        
        for c in range(n_classes):
            for f in range(n_features):
                # Plot histogram of training data for this class and feature
                X_c = X_train[y_train == c]
                axes[c, f].hist(X_c[:, f], bins=30, alpha=0.5, label='Data', density=True)
                
                # Plot Gaussian PDF
                x_range = np.linspace(X_c[:, f].min(), X_c[:, f].max(), 100)
                pdf = (1 / np.sqrt(2 * np.pi * model.sigma_[c, f])) * \
                      np.exp(-0.5 * (x_range - model.theta_[c, f])**2 / model.sigma_[c, f])
                axes[c, f].plot(x_range, pdf, 'r-', label='Gaussian')
                
                axes[c, f].set_xlabel(f'Feature {f}')
                axes[c, f].set_ylabel('Density')
                axes[c, f].set_title(f'Class {c}, Feature {f}')
                axes[c, f].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions.png'), dpi=150)
        plt.close()
    
    # Save comparison with sklearn
    sklearn_metrics = compare_with_sklearn(X_train, y_train, X_val, y_val)
    with open(os.path.join(OUTPUT_DIR, 'sklearn_comparison.txt'), 'w') as f:
        for key, value in sklearn_metrics.items():
            f.write(f'{key}: {value:.4f}\n')


def compare_with_sklearn(X_train, y_train, X_val, y_val):
    """
    Compare our implementation with sklearn GaussianNB.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        comparison_metrics: Dictionary of comparison metrics
    """
    # Train sklearn model
    sklearn_model = SklearnGaussianNB()
    sklearn_model.fit(X_train, y_train)
    
    # Make predictions
    sklearn_preds = sklearn_model.predict(X_val)
    
    # Create and fit our model for comparison
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    
    # We'll use a temporary instance
    temp_model = GaussianNBModel(n_features, n_classes)
    temp_model.fit(X_train, y_train)
    our_preds = temp_model.predict(X_val)
    
    # Compute metrics
    sklearn_accuracy = accuracy_score(y_val, sklearn_preds)
    our_accuracy = accuracy_score(y_val, our_preds)
    
    comparison_metrics = {
        'sklearn_accuracy': sklearn_accuracy,
        'our_accuracy': our_accuracy,
        'accuracy_difference': abs(sklearn_accuracy - our_accuracy)
    }
    
    return comparison_metrics


def main():
    """Main function to run the Gaussian NB task."""
    print("=" * 60)
    print("Gaussian Naive Bayes Implementation")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating datasets...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=1000,
        n_features=10,
        test_size=0.2,
        batch_size=32
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Build model
    print("\nBuilding model...")
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = build_model(n_features, n_classes, device)
    
    # Train model
    print("\nTraining model...")
    model = train(model, train_loader, device, epochs=1)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device, dataset_name='train')
    print(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
    print(f"Train MSE: {train_metrics['train_mse']:.4f}")
    print(f"Train R2: {train_metrics['train_r2']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device, dataset_name='validation')
    print(f"Validation Accuracy: {val_metrics['validation_accuracy']:.4f}")
    print(f"Validation MSE: {val_metrics['validation_mse']:.4f}")
    print(f"Validation R2: {val_metrics['validation_r2']:.4f}")
    
    # Compare with sklearn
    print("\nComparing with sklearn GaussianNB...")
    comparison_metrics = compare_with_sklearn(X_train, y_train, X_val, y_val)
    print(f"Sklearn Accuracy: {comparison_metrics['sklearn_accuracy']:.4f}")
    print(f"Our Accuracy: {comparison_metrics['our_accuracy']:.4f}")
    print(f"Accuracy Difference: {comparison_metrics['accuracy_difference']:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    all_metrics = {**train_metrics, **val_metrics}
    save_artifacts(model, all_metrics, X_train, y_train, X_val, y_val)
    
    print("\nDone! Artifacts saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()

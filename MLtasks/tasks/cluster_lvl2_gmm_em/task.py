import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    r2_score,
    mean_squared_error
)
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


OUTPUT_DIR = "output"


def get_task_metadata() -> Dict:
    """Get task metadata."""
    return {
        "task": "cluster_lvl2_gmm_em",
        "description": "Gaussian Mixture Model with EM algorithm"
    }


def get_device() -> torch.device:
    """Get device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GaussianMixtureModel:
    """Gaussian Mixture Model with EM algorithm."""
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'diag',
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_ = None
        
    def _initialize_parameters(self, X: torch.Tensor) -> None:
        """Initialize model parameters using k-means++ style initialization."""
        torch.manual_seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = torch.ones(self.n_components) / self.n_components
        
        # Initialize means using k-means++ style
        self.means_ = torch.zeros(self.n_components, n_features)
        
        # First mean is random
        idx = torch.randint(0, n_samples, (1,))
        self.means_[0] = X[idx].squeeze()
        
        # Subsequent means are chosen based on distance
        for k in range(1, self.n_components):
            distances = torch.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float('inf')
                for j in range(k):
                    dist = torch.sum((X[i] - self.means_[j]) ** 2)
                    min_dist = min(min_dist, dist.item())
                distances[i] = min_dist
            
            probs = distances / distances.sum()
            idx = torch.multinomial(probs, 1)
            self.means_[k] = X[idx].squeeze()
        
        # Initialize covariances
        if self.covariance_type == 'diag':
            self.covariances_ = torch.ones(self.n_components, n_features) * 0.1
        else:
            self.covariances_ = torch.eye(n_features).unsqueeze(0).repeat(self.n_components, 1, 1) * 0.1
    
    def _e_step(self, X: torch.Tensor) -> torch.Tensor:
        """E-step: compute responsibilities."""
        n_samples = X.shape[0]
        
        # Compute log probabilities for each component
        log_probs = torch.zeros(n_samples, self.n_components)
        
        for k in range(self.n_components):
            diff = X - self.means_[k]
            
            if self.covariance_type == 'diag':
                var = self.covariances_[k]
                log_det = torch.sum(torch.log(var + 1e-10))
                inv_var = 1.0 / (var + 1e-10)
                mahalanobis = torch.sum(diff ** 2 * inv_var, dim=1)
            else:
                cov = self.covariances_[k] + 1e-6 * torch.eye(X.shape[1])
                sign, log_det = torch.linalg.slogdet(cov)
                inv_cov = torch.linalg.inv(cov)
                mahalanobis = torch.sum(diff @ inv_cov * diff, dim=1)
            
            log_probs[:, k] = torch.log(self.weights_[k] + 1e-10) - 0.5 * log_det - 0.5 * mahalanobis
        
        # Normalize to get responsibilities
        log_probs_max = log_probs.max(dim=1, keepdim=True).values
        probs = torch.exp(log_probs - log_probs_max)
        responsibilities = probs / probs.sum(dim=1, keepdim=True)
        
        return responsibilities
    
    def _m_step(self, X: torch.Tensor, responsibilities: torch.Tensor) -> None:
        """M-step: update parameters."""
        n_samples, n_features = X.shape
        
        # Update weights
        N_k = responsibilities.sum(dim=0)
        self.weights_ = N_k / n_samples
        
        # Update means
        for k in range(self.n_components):
            self.means_[k] = (responsibilities[:, k:k+1] * X).sum(dim=0) / (N_k[k] + 1e-10)
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            
            if self.covariance_type == 'diag':
                weighted_diff = responsibilities[:, k:k+1] * diff ** 2
                self.covariances_[k] = weighted_diff.sum(dim=0) / (N_k[k] + 1e-10) + 1e-6
            else:
                cov = torch.zeros(n_features, n_features)
                for i in range(n_samples):
                    cov += responsibilities[i, k] * torch.outer(diff[i], diff[i])
                self.covariances_[k] = cov / (N_k[k] + 1e-10) + 1e-6 * torch.eye(n_features)
    
    def _compute_log_likelihood(self, X: torch.Tensor) -> float:
        """Compute log-likelihood of data."""
        n_samples = X.shape[0]
        
        log_probs = torch.zeros(n_samples, self.n_components)
        
        for k in range(self.n_components):
            diff = X - self.means_[k]
            
            if self.covariance_type == 'diag':
                var = self.covariances_[k]
                log_det = torch.sum(torch.log(var + 1e-10))
                inv_var = 1.0 / (var + 1e-10)
                mahalanobis = torch.sum(diff ** 2 * inv_var, dim=1)
            else:
                cov = self.covariances_[k] + 1e-6 * torch.eye(X.shape[1])
                sign, log_det = torch.linalg.slogdet(cov)
                inv_cov = torch.linalg.inv(cov)
                mahalanobis = torch.sum(diff @ inv_cov * diff, dim=1)
            
            log_probs[:, k] = torch.log(self.weights_[k] + 1e-10) - 0.5 * log_det - 0.5 * mahalanobis
        
        # Log-sum-exp trick
        log_probs_max = log_probs.max(dim=1, keepdim=True).values
        log_likelihood = (log_probs_max.squeeze() + torch.log(torch.exp(log_probs - log_probs_max).sum(dim=1))).sum()
        
        return log_likelihood.item()
    
    def fit(self, X: torch.Tensor) -> 'GaussianMixtureModel':
        """Fit the GMM using EM algorithm."""
        X = X.clone().detach().float()
        n_samples = X.shape[0]
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        # EM iterations
        prev_log_likelihood = float('-inf')
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            prev_log_likelihood = log_likelihood
        
        self.log_likelihood_ = self._compute_log_likelihood(X)
        
        return self
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict responsibilities for each sample."""
        X = X.clone().detach().float()
        return self._e_step(X)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster assignments."""
        X = X.clone().detach().float()
        responsibilities = self._e_step(X)
        return responsibilities.argmax(dim=1)
    
    def score(self, X: torch.Tensor) -> float:
        """Compute average log-likelihood."""
        X = X.clone().detach().float()
        return self._compute_log_likelihood(X) / X.shape[0]


def make_dataloaders(
    n_samples: int = 1000,
    n_features: int = 2,
    n_clusters: int = 4,
    batch_size: int = 128,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Create dataloaders for training and validation."""
    # Generate synthetic data
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=n_features,
        random_state=random_state,
        cluster_std=1.0
    )
    
    # Split into train and validation
    split_idx = int(n_samples * (1 - test_size))
    
    X_train = torch.FloatTensor(X[:split_idx])
    X_val = torch.FloatTensor(X[split_idx:])
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, torch.LongTensor(y_train))
    val_dataset = TensorDataset(X_val, torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val


def build_model(
    n_components: int = 4,
    covariance_type: str = 'diag',
    max_iter: int = 100,
    tol: float = 1e-4
) -> GaussianMixtureModel:
    """Build GMM model."""
    return GaussianMixtureModel(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        tol=tol
    )


def train(
    model: GaussianMixtureModel,
    train_loader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Train the GMM model."""
    # Collect all training data
    X_all = []
    for X, _ in train_loader:
        X_all.append(X)
    X_all = torch.cat(X_all, dim=0).to(device)
    
    history = {'log_likelihood': []}
    
    # Fit the model
    model.fit(X_all)
    
    # Record log-likelihood
    history['log_likelihood'].append(model.log_likelihood_)
    
    if verbose:
        print(f"  Final log-likelihood: {model.log_likelihood_:.4f}")
    
    return history


def evaluate(
    model: GaussianMixtureModel,
    data_loader: DataLoader,
    device: torch.device,
    y_true: np.ndarray
) -> Dict[str, float]:
    """Evaluate the model."""
    metrics = {}
    
    # Collect all data
    X_all = []
    for X, _ in data_loader:
        X_all.append(X)
    X_all = torch.cat(X_all, dim=0).to(device)
    
    # Get predictions
    predictions = model.predict(X_all).cpu().numpy()
    
    # Compute log-likelihood
    metrics['log_likelihood'] = model._compute_log_likelihood(X_all)
    
    # R2 score (using means as predictions)
    X_np = X_all.cpu().numpy()
    means_np = model.means_.cpu().numpy()
    predictions_expanded = means_np[predictions]
    metrics['r2'] = r2_score(X_np, predictions_expanded)
    
    # MSE
    metrics['mse'] = mean_squared_error(X_np, predictions_expanded)
    
    # Find best mapping between predicted and true labels
    n_clusters = len(np.unique(y_true))
    n_classes = len(np.unique(y_true))
    
    # Compute confusion matrix
    confusion = np.zeros((n_clusters, n_classes))
    for pred, true in zip(predictions, y_true):
        confusion[pred, true] += 1
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(-confusion)
    accuracy = confusion[row_ind, col_ind].sum() / len(y_true)
    metrics['accuracy'] = accuracy
    
    # Adjusted Rand Index
    metrics['ari'] = adjusted_rand_score(y_true, predictions)
    
    # Silhouette score
    metrics['silhouette'] = silhouette_score(X_all.cpu().numpy(), predictions)
    
    return metrics


def predict(
    model: GaussianMixtureModel,
    X: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Predict cluster assignments.
    
    Args:
        model: Trained GMM model
        X: Input data
        device: Device to use
        
    Returns:
        Cluster assignments
    """
    X = X.to(device)
    return model.predict(X)


def save_artifacts(
    model: GaussianMixtureModel,
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Save model artifacts.
    
    Args:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters
    torch.save({
        'weights': model.weights_.cpu() if hasattr(model.weights_, 'cpu') else model.weights_,
        'means': model.means_.cpu() if hasattr(model.means_, 'cpu') else model.means_,
        'covariances': model.covariances_.cpu() if hasattr(model.covariances_, 'cpu') else model.covariances_,
        'log_likelihood': model.log_likelihood_,
    }, os.path.join(output_dir, 'model.pt'))
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save metadata
    metadata = get_task_metadata()
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot log-likelihood history
    if len(history.get('log_likelihood', [])) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(history['log_likelihood'], marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title('GMM Training - Log-Likelihood Progression')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'log_likelihood.png'))
        plt.close()
    
    # Plot clusters if 2D
    if hasattr(model, 'means_') and model.means_.shape[1] == 2:
        try:
            # Generate sample points
            x_min, x_max = model.means_[:, 0].min() - 1, model.means_[:, 0].max() + 1
            y_min, y_max = model.means_[:, 1].min() - 1, model.means_[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            
            # Compute probabilities
            grid_torch = torch.FloatTensor(grid)
            probs = model.predict_proba(grid_torch).numpy()
            
            # Plot decision boundaries
            plt.figure(figsize=(10, 8))
            for k in range(model.n_components):
                plt.contour(xx, yy, probs[:, k].reshape(xx.shape), levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
                           alpha=0.3, colors=f'C{k}')
            
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('GMM Cluster Decision Boundaries')
            plt.savefig(os.path.join(output_dir, 'clusters.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save cluster plot: {e}")


def main():
    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=1000,
        n_features=2,
        n_clusters=4,
        batch_size=128,
        test_size=0.2,
        random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\n[2] Building model...")
    model = build_model(
        n_components=4,
        covariance_type='diag',
        max_iter=100,
        tol=1e-4
    )
    print(f"Model: GMM with {model.n_components} components, {model.covariance_type} covariance")
    
    # Train model
    print("\n[3] Training model...")
    history = train(model, train_loader, device)
    
    # Evaluate model
    print("\n[4] Evaluating model...")
    metrics = evaluate(model, val_loader, device, y_val)
    
    print("  Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")
    
    # Save artifacts
    print("\n[5] Saving artifacts...")
    save_artifacts(model, history, metrics)
    
    print("\nDone! Artifacts saved to:", OUTPUT_DIR)
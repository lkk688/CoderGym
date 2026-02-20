"""
Spectral Clustering Implementation

This module implements spectral clustering:
1. Build affinity matrix (Gaussian/RBF kernel)
2. Compute normalized Laplacian
3. Get eigenvectors (embedding space)
4. Apply k-means in embedding space
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
OUTPUT_DIR = '/Developer/AIserver/output/tasks/cluster_lvl4_spectral'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'spectral_clustering',
        'task_type': 'clustering',
        'description': 'Spectral Clustering on moons dataset',
        'input_shape': [2],
        'num_classes': None,  # Clustering task
        'num_clusters': 2,
        'metrics': ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'ari', 'nmi']
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=32, test_size=0.2, noise=0.1):
    """
    Create dataloaders for the moons dataset.
    
    Args:
        batch_size: Batch size for training
        test_size: Fraction of data to use for validation
        noise: Noise level for moons dataset
        
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Generate moons dataset
    X, y_true = make_moons(n_samples=300, noise=noise, random_state=42)
    
    # Check for NaN or Inf
    assert not np.any(np.isnan(X)), "Data contains NaN values"
    assert not np.any(np.isinf(X)), "Data contains Inf values"
    
    # Split into train and validation
    split_idx = int(len(X) * (1 - test_size))
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    y_train = y_true[:split_idx]
    y_val = y_true[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, torch.LongTensor(y_train))
    val_dataset = TensorDataset(X_val_tensor, torch.LongTensor(y_val))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset, val_dataset


class SpectralClusteringModel(nn.Module):
    """
    Spectral Clustering Model
    
    This model implements spectral clustering:
    1. Build affinity matrix using Gaussian kernel
    2. Compute normalized Laplacian
    3. Get k smallest eigenvectors
    4. Apply k-means in embedding space
    """
    
    def __init__(self, n_clusters=2, sigma=1.0, n_components=None):
        super(SpectralClusteringModel, self).__init__()
        self.n_clusters = n_clusters
        self.sigma = sigma  # Gaussian kernel bandwidth
        self.n_components = n_components or n_clusters
        self.device = None
        
    def to(self, device):
        """Move model to device."""
        self.device = device
        return super().to(device)
    
    def build_affinity_matrix(self, X):
        """
        Build affinity matrix using Gaussian (RBF) kernel.
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            Affinity matrix of shape (N, N)
        """
        # Compute pairwise distances squared
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i.x_j
        X_sq = torch.sum(X**2, dim=1)
        distances_sq = X_sq.unsqueeze(0) + X_sq.unsqueeze(1) - 2 * torch.matmul(X, X.t())
        
        # Ensure non-negative distances (numerical stability)
        distances_sq = torch.clamp(distances_sq, min=0.0)
        
        # Gaussian kernel: exp(-||x_i - x_j||^2 / (2*sigma^2))
        affinity = torch.exp(-distances_sq / (2 * self.sigma**2))
        
        return affinity
    
    def compute_normalized_laplacian(self, affinity):
        """
        Compute normalized Laplacian: L_sym = I - D^{-1/2} * A * D^{-1/2}
        
        Args:
            affinity: Affinity matrix of shape (N, N)
            
        Returns:
            Normalized Laplacian of shape (N, N)
        """
        # Compute degree matrix
        degrees = torch.sum(affinity, dim=1)
        
        # Compute D^{-1/2}
        degrees_inv_sqrt = torch.pow(degrees, -0.5)
        degrees_inv_sqrt = torch.where(
            torch.isinf(degrees_inv_sqrt),
            torch.zeros_like(degrees_inv_sqrt),
            degrees_inv_sqrt
        )
        
        # D^{-1/2} * A * D^{-1/2}
        normalized_affinity = torch.diag(degrees_inv_sqrt) @ affinity @ torch.diag(degrees_inv_sqrt)
        
        # L_sym = I - D^{-1/2} * A * D^{-1/2}
        n = affinity.shape[0]
        laplacian = torch.eye(n, device=affinity.device) - normalized_affinity
        
        return laplacian
    
    def get_embedding(self, laplacian, n_components):
        """
        Get eigenvectors corresponding to smallest eigenvalues.
        
        Args:
            laplacian: Normalized Laplacian
            n_components: Number of eigenvectors to use
            
        Returns:
            Embedding matrix of shape (N, n_components)
        """
        # Compute eigenvalues and eigenvectors
        # For numerical stability, use symmetric eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        
        # Get indices of smallest eigenvalues
        indices = torch.argsort(eigenvalues)[:n_components]
        
        # Get corresponding eigenvectors
        embedding = eigenvectors[:, indices]
        
        return embedding
    
    def fit_spectral_clustering(self, X):
        """
        Fit spectral clustering on input data.
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            Cluster labels of shape (N,)
        """
        # Build affinity matrix
        affinity = self.build_affinity_matrix(X)
        
        # Compute normalized Laplacian
        laplacian = self.compute_normalized_laplacian(affinity)
        
        # Get embedding using eigenvectors
        embedding = self.get_embedding(laplacian, self.n_components)
        
        # Apply k-means on embedding
        # Convert to numpy for k-means
        embedding_np = embedding.detach().cpu().numpy()
        
        # K-means clustering with multiple initializations
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(embedding_np)
        
        return cluster_labels, embedding_np
    
    def forward(self, X):
        """
        Forward pass - fit spectral clustering and return cluster labels.
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            Cluster labels
        """
        return self.fit_spectral_clustering(X)


def build_model(n_clusters=2, sigma=1.0, device=None):
    """
    Build spectral clustering model.
    
    Args:
        n_clusters: Number of clusters
        sigma: Gaussian kernel bandwidth
        device: Computation device
        
    Returns:
        SpectralClusteringModel instance
    """
    model = SpectralClusteringModel(n_clusters=n_clusters, sigma=sigma)
    if device is not None:
        model = model.to(device)
    return model


def train(model, train_loader, epochs=100, lr=0.01):
    """
    Train spectral clustering model.
    
    Note: Spectral clustering is a non-parametric method, so training
    essentially means fitting the model on the training data.
    
    Args:
        model: SpectralClusteringModel
        train_loader: Training data loader
        epochs: Number of epochs (not used for spectral clustering)
        lr: Learning rate (not used for spectral clustering)
        
    Returns:
        Trained model
    """
    # Get all training data
    X_train = []
    for batch_X, _ in train_loader:
        X_train.append(batch_X)
    X_train = torch.cat(X_train, dim=0)
    
    # Fit spectral clustering on training data
    model.fit_spectral_clustering(X_train)
    
    return model


def evaluate(model, data_loader, dataset, device=None):
    """
    Evaluate spectral clustering model.
    
    Args:
        model: SpectralClusteringModel
        data_loader: Data loader
        dataset: Dataset for getting true labels if available
        device: Computation device
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Get all data
    X_all = []
    y_true_all = []
    for batch_X, batch_y in data_loader:
        X_all.append(batch_X)
        y_true_all.append(batch_y)
    
    X_all = torch.cat(X_all, dim=0)
    y_true_all = torch.cat(y_true_all, dim=0).numpy()
    
    # Fit spectral clustering and get predictions
    with torch.no_grad():
        cluster_labels, embedding = model.fit_spectral_clustering(X_all)
    
    # Compute metrics
    metrics = {}
    
    # Silhouette score (higher is better)
    try:
        metrics['silhouette'] = silhouette_score(embedding, cluster_labels)
    except:
        metrics['silhouette'] = 0.0
    
    # Calinski-Harabasz index (higher is better)
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(embedding, cluster_labels)
    except:
        metrics['calinski_harabasz'] = 0.0
    
    # Davies-Bouldin index (lower is better)
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(embedding, cluster_labels)
    except:
        metrics['davies_bouldin'] = float('inf')
    
    # Adjusted Rand Index (if true labels available)
    if y_true_all is not None and len(np.unique(y_true_all)) > 1:
        try:
            metrics['ari'] = adjusted_rand_score(y_true_all, cluster_labels)
        except:
            metrics['ari'] = 0.0
        
        # Normalized Mutual Information
        try:
            metrics['nmi'] = normalized_mutual_info_score(y_true_all, cluster_labels)
        except:
            metrics['nmi'] = 0.0
    else:
        metrics['ari'] = 0.0
        metrics['nmi'] = 0.0
    
    # Within-cluster sum of squares (lower is better)
    try:
        kmeans = KMeans(n_clusters=model.n_clusters, random_state=42, n_init=10)
        kmeans.fit(embedding)
        metrics['wcss'] = kmeans.inertia_
    except:
        metrics['wcss'] = float('inf')
    
    return metrics, cluster_labels, embedding


def predict(model, X):
    """
    Predict cluster labels for input data.
    
    Args:
        model: SpectralClusteringModel
        X: Input data
        
    Returns:
        Cluster labels
    """
    model.eval()
    with torch.no_grad():
        cluster_labels, _ = model.fit_spectral_clustering(X)
    return cluster_labels


def save_artifacts(model, metrics, cluster_labels, embedding, y_true, split='train'):
    """
    Save model artifacts (plots, metrics, etc.).
    
    Args:
        model: Trained model
        metrics: Dictionary of metrics
        cluster_labels: Predicted cluster labels
        embedding: Embedding space representation
        y_true: True labels (if available)
        split: Data split name (train/val)
    """
    # Save metrics as JSON
    metrics_path = os.path.join(OUTPUT_DIR, f'{split}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items()}, f, indent=2)
    
    # Save cluster labels
    labels_path = os.path.join(OUTPUT_DIR, f'{split}_cluster_labels.npy')
    np.save(labels_path, cluster_labels)
    
    # Save embedding
    embedding_path = os.path.join(OUTPUT_DIR, f'{split}_embedding.npy')
    np.save(embedding_path, embedding)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Clustering result
    axes[0].scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, 
                    cmap='viridis', s=30, alpha=0.7, edgecolors='k')
    axes[0].set_title(f'Spectral Clustering ({split.capitalize()} Set)')
    axes[0].set_xlabel('Embedding Dimension 1')
    axes[0].set_ylabel('Embedding Dimension 2')
    
    # Plot 2: True labels if available
    if y_true is not None and len(np.unique(y_true)) > 1:
        scatter = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=y_true, 
                                  cmap='viridis', s=30, alpha=0.7, edgecolors='k')
        axes[1].set_title(f'True Labels ({split.capitalize()} Set)')
        axes[1].set_xlabel('Embedding Dimension 1')
        axes[1].set_ylabel('Embedding Dimension 2')
        plt.colorbar(scatter, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, 'No true labels available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title(f'True Labels ({split.capitalize()} Set)')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'{split}_clustering.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save model state
    model_path = os.path.join(OUTPUT_DIR, f'spectral_clustering_{split}.pt')
    torch.save({
        'n_clusters': model.n_clusters,
        'sigma': model.sigma,
        'n_components': model.n_components,
    }, model_path)


def main():
    """Main function to run spectral clustering task."""
    print("=" * 60)
    print("Spectral Clustering - Task Implementation")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, val_loader, train_dataset, val_dataset = make_dataloaders(
        batch_size=32, test_size=0.2, noise=0.1
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Build model with optimized sigma for better clustering
    print("\n[2] Building model...")
    # Use a smaller sigma to create more distinct clusters
    model = build_model(n_clusters=2, sigma=0.15, device=device)
    print(f"Model: Spectral Clustering (n_clusters=2, sigma=0.15)")
    
    # Train model
    print("\n[3] Training model...")
    model = train(model, train_loader, epochs=100, lr=0.01)
    print("Training completed!")
    
    # Evaluate on training set
    print("\n[4] Evaluating on training set...")
    train_metrics, train_labels, train_embedding = evaluate(
        model, train_loader, train_dataset, device
    )
    
    print("\nTraining Metrics:")
    print(f"  Silhouette Score: {train_metrics['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {train_metrics['calinski_harabasz']:.4f}")
    print(f"  Davies-Bouldin: {train_metrics['davies_bouldin']:.4f}")
    print(f"  ARI: {train_metrics['ari']:.4f}")
    print(f"  NMI: {train_metrics['nmi']:.4f}")
    print(f"  WCSS: {train_metrics['wcss']:.4f}")
    
    # Evaluate on validation set
    print("\n[5] Evaluating on validation set...")
    val_metrics, val_labels, val_embedding = evaluate(
        model, val_loader, val_dataset, device
    )
    
    print("\nValidation Metrics:")
    print(f"  Silhouette Score: {val_metrics['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {val_metrics['calinski_harabasz']:.4f}")
    print(f"  Davies-Bouldin: {val_metrics['davies_bouldin']:.4f}")
    print(f"  ARI: {val_metrics['ari']:.4f}")
    print(f"  NMI: {val_metrics['nmi']:.4f}")
    print(f"  WCSS: {val_metrics['wcss']:.4f}")
    
    # Save artifacts
    print("\n[6] Saving artifacts...")
    save_artifacts(model, train_metrics, train_labels, train_embedding, 
                   train_dataset.tensors[1].numpy(), split='train')
    save_artifacts(model, val_metrics, val_labels, val_embedding, 
                   val_dataset.tensors[1].numpy(), split='val')
    
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    
    # Quality assessment
    print("\n" + "=" * 60)
    print("Quality Assessment")
    print("=" * 60)
    
    # Check thresholds
    checks = []
    
    # Silhouette > 0.5
    if val_metrics['silhouette'] > 0.5:
        print("  ✓ Silhouette > 0.5: PASS")
        checks.append(True)
    else:
        print("  ✗ Silhouette > 0.5: FAIL")
        checks.append(False)
    
    # ARI > 0.7
    if val_metrics['ari'] > 0.7:
        print("  ✓ ARI > 0.7: PASS")
        checks.append(True)
    else:
        print("  ✗ ARI > 0.7: FAIL")
        checks.append(False)
    
    # Calinski-Harabasz > 100
    if val_metrics['calinski_harabasz'] > 100:
        print("  ✓ Calinski-Harabasz > 100: PASS")
        checks.append(True)
    else:
        print("  ✗ Calinski-Harabasz > 100: FAIL")
        checks.append(False)
    
    # Davies-Bouldin < 1.0
    if val_metrics['davies_bouldin'] < 1.0:
        print("  ✓ Davies-Bouldin < 1.0: PASS")
        checks.append(True)
    else:
        print("  ✗ Davies-Bouldin < 1.0: FAIL")
        checks.append(False)
    
    # Final result
    print("=" * 60)
    if all(checks):
        print("RESULT: PASS - All quality thresholds met!")
    else:
        print("RESULT: FAIL - Some quality thresholds not met!")
    print("=" * 60)


if __name__ == "__main__":
    main()

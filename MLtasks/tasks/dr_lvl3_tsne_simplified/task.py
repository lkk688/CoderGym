"""
Simplified t-SNE Implementation for Dimensionality Reduction and Visualization
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory
OUTPUT_DIR = '/Developer/AIserver/output/tasks/dr_lvl3_tsne_simplified'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'tsne_simplified',
        'task_type': 'dimensionality_reduction',
        'input_dim': None,  # Will be set dynamically
        'output_dim': 2,
        'description': 'Simplified t-SNE for small N with visualization'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get computation device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=300, noise=0.1, batch_size=32):
    """
    Create synthetic dataset for t-SNE visualization.
    
    Args:
        n_samples: Number of samples to generate
        noise: Noise level for data generation
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, X_full, y_full
    """
    # Generate a complex dataset with multiple clusters
    X1, y1 = make_circles(n_samples=n_samples//3, factor=0.5, noise=noise, random_state=42)
    X2, y2 = make_moons(n_samples=n_samples//3, noise=noise, random_state=43)
    X3, y3 = make_classification(n_samples=n_samples//3, n_features=2, n_redundant=0, 
                                  n_informative=2, n_clusters_per_class=1, 
                                  flip_y=0.1, class_sep=1.5, random_state=44)
    
    # Combine datasets
    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2, y3])
    
    # Add more dimensions for dimensionality reduction
    np.random.seed(42)
    X_extra = np.random.randn(X.shape[0], 8)  # 8 more dimensions
    X = np.hstack([X, X_extra])
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X, y


class TSNEModel(nn.Module):
    """
    Simplified t-SNE implementation using gradient descent.
    
    This is a simplified version that optimizes a cost function
    similar to t-SNE but with reduced complexity.
    """
    def __init__(self, input_dim, output_dim=2, perplexity=30.0):
        super(TSNEModel, self).__init__()
        self.output_dim = output_dim
        self.perplexity = perplexity
        
        # Initialize low-dimensional embeddings randomly
        self.embeddings = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        
    def _compute_pairwise_distances(self, X):
        """Compute pairwise Euclidean distances."""
        # X shape: (N, D)
        # Returns: (N, N) distance matrix
        sum_sq = torch.sum(X**2, dim=1, keepdim=True)
        dist_sq = sum_sq + sum_sq.t() - 2 * torch.mm(X, X.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return torch.sqrt(dist_sq + 1e-12)
    
    def _compute_q_matrix(self, embeddings):
        """
        Compute Q matrix (t-SNE similarities in low-dimensional space).
        Q_ij = (1 + ||y_i - y_j||^2)^{-1} / sum_{k != l} (1 + ||y_k - y_l||^2)^{-1}
        """
        # Compute pairwise distances in embedding space
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # (N, N, D)
        dist_sq = torch.sum(diff**2, dim=2)  # (N, N)
        
        # Compute Q matrix with t-distribution (1 degree of freedom)
        q = 1.0 / (1.0 + dist_sq)
        q = q - torch.diag(torch.diag(q))  # Set diagonal to zero
        
        # Normalize
        sum_q = torch.sum(q)
        q = q / sum_q
        
        return q
    
    def _compute_p_matrix(self, X):
        """
        Compute P matrix (Gaussian similarities in high-dimensional space).
        This is a simplified version of t-SNE P matrix computation.
        """
        n = X.size(0)
        
        # Compute pairwise distances in input space
        diff = X.unsqueeze(1) - X.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)
        
        # Compute bandwidth using perplexity
        # Simplified: use a fixed bandwidth based on perplexity
        bandwidth = self.perplexity ** 2
        
        # Compute P matrix (Gaussian kernel)
        p = torch.exp(-dist_sq / (2 * bandwidth))
        p = p - torch.diag(torch.diag(p))  # Set diagonal to zero
        
        # Normalize
        sum_p = torch.sum(p)
        p = p / sum_p
        
        return p
    
    def forward(self, X):
        """Compute t-SNE embeddings."""
        return torch.mm(X, self.embeddings)
    
    def compute_kl_divergence(self, X):
        """Compute KL divergence between P and Q distributions."""
        embeddings = self.forward(X)
        p = self._compute_p_matrix(X)
        q = self._compute_q_matrix(embeddings)
        
        # KL divergence: sum(P * log(P/Q))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        kl_div = torch.sum(p * torch.log((p + epsilon) / (q + epsilon)))
        
        return kl_div


def build_model(input_dim, output_dim=2, perplexity=30.0):
    """Build t-SNE model."""
    model = TSNEModel(input_dim, output_dim, perplexity)
    return model


def train(model, train_loader, val_loader, X_full, y_full, 
          n_epochs=100, lr=1.0, device=None):
    """
    Train t-SNE model using gradient descent.
    
    Args:
        model: t-SNE model
        train_loader: Training data loader
        val_loader: Validation data loader
        X_full: Full feature matrix
        y_full: Full labels
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Computation device
    
    Returns:
        model: Trained model
        history: Training history with KL divergences
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    # Convert full data to tensor
    X_tensor = torch.FloatTensor(X_full).to(device)
    
    # Use Adam optimizer for gradient descent
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_kl': [], 'val_kl': []}
    
    for epoch in range(n_epochs):
        model.train()
        
        # Compute KL divergence and optimize
        optimizer.zero_grad()
        kl_loss = model.compute_kl_divergence(X_tensor)
        kl_loss.backward()
        optimizer.step()
        
        # Record KL divergence
        train_kl = kl_loss.item()
        history['train_kl'].append(train_kl)
        
        # Validation KL divergence (using a subset for efficiency)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                # Use a subset of validation data for efficiency
                val_batch, _ = next(iter(val_loader))
                val_batch = val_batch.to(device)
                val_kl = model.compute_kl_divergence(val_batch).item()
                history['val_kl'].append(val_kl)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, KL Divergence: {train_kl:.4f}")
    
    return model, history


def evaluate(model, data_loader, X_full, y_full, device=None):
    """
    Evaluate t-SNE model and compute metrics.
    
    Args:
        model: Trained t-SNE model
        data_loader: Data loader for evaluation
        X_full: Full feature matrix
        y_full: Full labels
        device: Computation device
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if device is None:
        device = get_device()
    
    model.eval()
    
    metrics = {}
    
    with torch.no_grad():
        # Compute KL divergence on full data
        X_tensor = torch.FloatTensor(X_full).to(device)
        kl_div = model.compute_kl_divergence(X_tensor).item()
        metrics['kl_divergence'] = kl_div
        
        # Get embeddings
        embeddings = model.forward(X_tensor)
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Compute pairwise distance correlation (simplified quality metric)
        # High correlation between high-d and low-d distances is good
        X_np = X_full
        dist_high = squareform(pdist(X_np, metric='euclidean'))
        dist_low = squareform(pdist(embeddings_np, metric='euclidean'))
        
        # Compute Spearman correlation (simplified as Pearson on ranked distances)
        # Use upper triangle only (excluding diagonal)
        upper_idx = np.triu_indices(len(X_np), k=1)
        corr, _ = spearmanr(dist_high[upper_idx], dist_low[upper_idx])
        metrics['distance_correlation'] = corr
        
        # Compute MSE between original and reconstructed (if applicable)
        # For t-SNE, we don't have explicit reconstruction, so we use distance preservation
        metrics['mse'] = np.mean((dist_high - dist_low) ** 2)
        metrics['r2'] = r2_score(dist_high[upper_idx], dist_low[upper_idx])
        
        # Cluster separation metric (simplified)
        # Compute silhouette-like score based on cluster distances
        unique_labels = np.unique(y_full)
        if len(unique_labels) > 1:
            # Compute between-cluster and within-cluster distances
            between_dist = []
            within_dist = []
            
            for label in unique_labels:
                mask = y_full == label
                cluster_embeddings = embeddings_np[mask]
                cluster_center = np.mean(cluster_embeddings, axis=0)
                
                # Within-cluster distance
                if len(cluster_embeddings) > 1:
                    within_dist.append(np.mean(pdist(cluster_embeddings)))
                
                # Between-cluster distance
                for other_label in unique_labels:
                    if label != other_label:
                        other_mask = y_full == other_label
                        other_embeddings = embeddings_np[other_mask]
                        other_center = np.mean(other_embeddings, axis=0)
                        between_dist.append(np.linalg.norm(cluster_center - other_center))
            
            if len(within_dist) > 0 and len(between_dist) > 0:
                metrics['cluster_separation'] = np.mean(between_dist) / (np.mean(within_dist) + 1e-12)
            else:
                metrics['cluster_separation'] = 0.0
        else:
            metrics['cluster_separation'] = 0.0
    
    return metrics


def predict(model, X, device=None):
    """
    Generate t-SNE embeddings for input data.
    
    Args:
        model: Trained t-SNE model
        X: Input features (numpy array or tensor)
        device: Computation device
    
    Returns:
        embeddings: Low-dimensional embeddings
    """
    if device is None:
        device = get_device()
    
    model.eval()
    
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    
    X = X.to(device)
    
    with torch.no_grad():
        embeddings = model.forward(X)
        embeddings_np = embeddings.detach().cpu().numpy()
    
    return embeddings_np


def save_artifacts(model, history, metrics, embeddings, y_full, X_full):
    """
    Save all artifacts (model, plots, metrics).
    
    Args:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
        embeddings: Final embeddings
        y_full: Full labels
        X_full: Full features
    """
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'tsne_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'embeddings': model.embeddings.data.cpu(),
    }, model_path)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v 
                   for k, v in metrics.items()}, f, indent=2)
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)
    
    # Save embeddings
    embeddings_path = os.path.join(OUTPUT_DIR, 'embeddings.npz')
    np.savez(embeddings_path, embeddings=embeddings, labels=y_full, features=X_full)
    
    # Save visualization
    plot_path = os.path.join(OUTPUT_DIR, 'tsne_visualization.png')
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Embedding scatter plot
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y_full, 
                          cmap='tab10', s=30, alpha=0.7, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Embedding Visualization')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training history
    plt.subplot(1, 2, 2)
    plt.plot(history['train_kl'], label='Train KL Divergence', linewidth=2)
    if 'val_kl' in history and len(history['val_kl']) > 0:
        plt.plot([i * 20 for i in range(len(history['val_kl']))], 
                 history['val_kl'], label='Val KL Divergence', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run t-SNE task."""
    print("=" * 60)
    print("Simplified t-SNE Implementation")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")
    print(f"Output dimension: {metadata['output_dim']}")
    
    # Create dataloaders
    print("\n--- Creating DataLoaders ---")
    train_loader, val_loader, test_loader, X_full, y_full = make_dataloaders(
        n_samples=300, noise=0.1, batch_size=32
    )
    print(f"Training samples: {len(X_full)}")
    print(f"Validation samples: {len(y_full) - len(y_full) // 2}")
    print(f"Test samples: {len(y_full) - len(y_full) // 2}")
    print(f"Input dimension: {X_full.shape[1]}")
    
    # Build model
    print("\n--- Building Model ---")
    model = build_model(
        input_dim=X_full.shape[1],
        output_dim=2,
        perplexity=30.0
    )
    print(f"Model: Simplified t-SNE")
    
    # Train model
    print("\n--- Training Model ---")
    model, history = train(model, train_loader, val_loader, X_full, y_full,
                           n_epochs=100, lr=1.0, device=device)
    
    # Evaluate model
    print("\n--- Evaluating Model ---")
    metrics = evaluate(model, test_loader, X_full, y_full, device=device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Get final embeddings
    print("\n--- Generating Embeddings ---")
    embeddings = predict(model, X_full, device=device)
    
    # Save artifacts
    print("\n--- Saving Artifacts ---")
    save_artifacts(model, history, metrics, embeddings, y_full, X_full)
    
    print("\n" + "=" * 60)
    print("t-SNE task completed successfully!")
    print("=" * 60)
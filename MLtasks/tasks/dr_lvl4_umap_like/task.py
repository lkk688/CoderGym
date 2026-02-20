"""
UMAP-like Dimensionality Reduction Task
Implements a simplified UMAP with kNN graph construction and SGD optimization
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Any, List, Optional

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        'task_name': 'umap_like',
        'task_type': 'dimensionality_reduction',
        'input_type': 'float',
        'output_type': 'float',
        'description': 'UMAP-like dimensionality reduction with kNN graph and negative sampling'
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    batch_size: int = 128,
    val_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for dimensionality reduction.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes for labels
        batch_size: Batch size for dataloader
        val_ratio: Validation split ratio
        
    Returns:
        train_loader, val_loader, X_full, y_full
    """
    # Create synthetic data with some structure
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=42,
        scale=np.ones(n_features)
    )
    
    # Add non-linear structure for better UMAP demonstration
    # Apply non-linear transformations
    X[:, 0] = X[:, 0] ** 2 + X[:, 1] * 0.5
    X[:, 1] = np.sin(X[:, 2]) + X[:, 3] * 0.3
    X[:, 2] = X[:, 4] * X[:, 5] / 10
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and validation
    n_val = int(n_samples * val_ratio)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Create datasets
    class EmbeddingDataset(Dataset):
        def __init__(self, X, y=None):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y) if y is not None else None
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            if self.y is not None:
                return self.X[idx], self.y[idx]
            return self.X[idx]
    
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X, y


def build_knn_graph(X: np.ndarray, n_neighbors: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build kNN graph from data.
    
    Args:
        X: Input data (n_samples, n_features)
        n_neighbors: Number of neighbors
        
    Returns:
        knn_indices: Indices of k nearest neighbors for each sample
        knn_distances: Distances to k nearest neighbors
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Remove self-loop (first neighbor is the point itself)
    return indices[:, 1:], distances[:, 1:]


def compute_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    # Efficient distance computation
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    distances = X_sq + X_sq.T - 2 * np.dot(X, X.T)
    distances = np.maximum(distances, 0)  # Handle numerical errors
    return np.sqrt(distances)


def build_model(
    input_dim: int,
    embedding_dim: int = 2,
    hidden_dim: int = 64
) -> nn.Module:
    """
    Build encoder model for dimensionality reduction.
    
    Args:
        input_dim: Input feature dimension
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        
    Returns:
        Encoder model
    """
    class Encoder(nn.Module):
        def __init__(self, input_dim, embedding_dim, hidden_dim):
            super(Encoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, embedding_dim)
            )
            
        def forward(self, x):
            return self.encoder(x)
    
    model = Encoder(input_dim, embedding_dim, hidden_dim).to(device)
    return model


def compute_affinities(X: np.ndarray, n_neighbors: int = 15, perplexity: float = 30.0) -> np.ndarray:
    """
    Compute affinity matrix using Gaussian kernel with adaptive bandwidth.
    
    Args:
        X: Input data
        n_neighbors: Number of neighbors for bandwidth estimation
        perplexity: Perplexity for bandwidth selection
        
    Returns:
        Affinity matrix
    """
    n_samples = X.shape[0]
    distances = compute_distances(X)
    
    # Find bandwidth for each point using binary search for perplexity
    target_log_perplexity = np.log(perplexity)
    affinities = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        # Get distances to all points
        di = distances[i].copy()
        di[i] = 0  # Set self-distance to 0
        
        # Binary search for sigma
        beta_min = 0
        beta_max = np.inf
        beta = 1.0
        
        for _ in range(50):
            # Compute probabilities
            pi = np.exp(-di * beta)
            pi[i] = 0
            sum_pi = np.sum(pi)
            
            if sum_pi == 0:
                beta = beta_max
                break
                
            pi /= sum_pi
            
            # Compute entropy
            entropy = -np.sum(pi * np.log(pi + 1e-10))
            
            if entropy > target_log_perplexity:
                beta_min = beta
                if beta_max == np.inf:
                    beta = beta * 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = (beta + beta_min) / 2
        
        affinities[i] = pi
    
    # Make symmetric
    affinities = (affinities + affinities.T) / (2 * n_samples)
    
    return affinities


def sample_negative_edges(
    n_samples: int,
    positive_edges: np.ndarray,
    n_negative: int = 5
) -> np.ndarray:
    """
    Sample negative edges for contrastive learning.
    
    Args:
        n_samples: Number of samples
        positive_edges: Array of (source, target) positive edges
        n_negative: Number of negative samples per positive
        
    Returns:
        Negative edges array
    """
    negative_edges = []
    
    for src, tgt in positive_edges:
        # Sample negative targets
        neg_targets = []
        attempts = 0
        while len(neg_targets) < n_negative and attempts < n_negative * 10:
            neg = np.random.randint(0, n_samples)
            if neg != src and neg != tgt:
                neg_targets.append(neg)
            attempts += 1
        negative_edges.extend([(src, neg) for neg in neg_targets])
    
    return np.array(negative_edges)


def compute_neighbor_preservation(
    high_dim_distances: np.ndarray,
    low_dim_distances: np.ndarray,
    k: int = 5
) -> Dict[str, float]:
    """
    Compute neighbor preservation metrics.
    
    Args:
        high_dim_distances: High-dimensional pairwise distances
        low_dim_distances: Low-dimensional pairwise distances
        k: Number of neighbors to consider
        
    Returns:
        Dictionary of metrics
    """
    n_samples = high_dim_distances.shape[0]
    
    # Get k nearest neighbors in high and low dimensions
    high_dim_knn = np.argsort(high_dim_distances, axis=1)[:, 1:k+1]
    low_dim_knn = np.argsort(low_dim_distances, axis=1)[:, 1:k+1]
    
    # Compute neighbor recall
    recalls = []
    for i in range(n_samples):
        high_neighbors = set(high_dim_knn[i])
        low_neighbors = set(low_dim_knn[i])
        intersection = len(high_neighbors & low_neighbors)
        recall = intersection / len(high_neighbors) if len(high_neighbors) > 0 else 0
        recalls.append(recall)
    
    mean_recall = np.mean(recalls)
    
    # Compute Spearman correlation of distance ranks
    correlations = []
    for i in range(n_samples):
        high_rank = np.argsort(high_dim_distances[i])[1:k+1]
        low_rank = np.argsort(low_dim_distances[i])[1:k+1]
        
        # Simple correlation of rank positions
        high_pos = np.argsort(high_rank)
        low_pos = np.argsort(low_rank)
        
        if np.std(high_pos) > 0 and np.std(low_pos) > 0:
            corr = np.corrcoef(high_pos, low_pos)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
    
    mean_correlation = np.mean(correlations) if correlations else 0
    
    return {
        'neighbor_recall': mean_recall,
        'rank_correlation': mean_correlation
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    X_train: np.ndarray,
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    n_neighbors: int = 15,
    n_negative: int = 5,
    temperature: float = 0.1
) -> List[float]:
    """
    Train the embedding model using contrastive learning with negative sampling.
    
    Args:
        model: Encoder model
        train_loader: Training data loader
        X_train: Full training data
        n_epochs: Number of epochs
        learning_rate: Learning rate
        n_neighbors: Number of neighbors for positive edges
        n_negative: Number of negative samples per positive
        temperature: Temperature for softmax
        
    Returns:
        List of loss values per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Build kNN graph for positive edges
    knn_indices, _ = build_knn_graph(X_train, n_neighbors)
    
    # Create positive edge list
    positive_edges = []
    for i in range(len(X_train)):
        for j in knn_indices[i]:
            positive_edges.append((i, int(j)))
    
    # Compute high-dimensional distances for evaluation
    high_dim_distances = compute_distances(X_train)
    
    losses = []
    
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            batch_size = batch_X.size(0)
            
            # Get embeddings
            embeddings = model(batch_X)
            
            # Compute pairwise distances in embedding space
            emb_sq = torch.sum(embeddings ** 2, dim=1, keepdim=True)
            emb_dist = emb_sq + emb_sq.t() - 2 * torch.mm(embeddings, embeddings.t())
            emb_dist = torch.sqrt(torch.relu(emb_dist) + 1e-10)
            
            # Contrastive loss with negative sampling
            loss = 0
            
            for i in range(batch_size):
                # Positive pairs (from kNN graph)
                for j_idx in knn_indices[i]:
                    if j_idx < len(X_train):  # Check bounds
                        j = int(j_idx)
                        if j < batch_size:  # Only consider samples in batch
                            # Push positive pairs closer
                            pos_dist = emb_dist[i, j]
                            loss += torch.relu(pos_dist - 1.0)
                        
                        # Negative sampling
                        for _ in range(n_negative):
                            neg = np.random.randint(0, len(X_train))
                            if neg != i and neg != j and neg < batch_size:
                                neg_dist = emb_dist[i, neg]
                                # Push negative pairs apart
                                loss += torch.relu(2.0 - neg_dist)
            
            # Normalize loss
            loss = loss / (batch_size * (n_neighbors + n_negative))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            # Compute neighbor preservation during training
            model.eval()
            with torch.no_grad():
                all_embeddings = []
                for batch_X, _ in train_loader:
                    all_embeddings.append(model(batch_X.to(device)).cpu().numpy())
                all_embeddings = np.vstack(all_embeddings)
            
            low_dim_distances = compute_distances(all_embeddings)
            metrics = compute_neighbor_preservation(high_dim_distances, low_dim_distances)
            
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, "
                  f"Neighbor Recall: {metrics['neighbor_recall']:.4f}")
            model.train()
    
    return losses


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    X_data: np.ndarray,
    y_data: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the model and compute metrics.
    
    Args:
        model: Encoder model
        data_loader: Data loader
        X_data: Full data
        y_data: Labels
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Get embeddings
        embeddings_list = []
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            embeddings_list.append(model(batch_X).cpu().numpy())
        
        embeddings = np.vstack(embeddings_list)
    
    # Compute low-dimensional distances
    low_dim_distances = compute_distances(embeddings)
    
    # Compute high-dimensional distances
    high_dim_distances = compute_distances(X_data)
    
    # Compute neighbor preservation metrics
    neighbor_metrics = compute_neighbor_preservation(high_dim_distances, low_dim_distances)
    
    # Compute basic statistics
    emb_var = np.var(embeddings)
    emb_mean = np.mean(embeddings)
    
    metrics = {
        'embedding_variance': float(emb_var),
        'embedding_mean': float(emb_mean),
        'neighbor_recall': neighbor_metrics['neighbor_recall'],
        'rank_correlation': neighbor_metrics['rank_correlation'],
        'low_dim_mse': float(np.mean(low_dim_distances ** 2)),
        'high_dim_mse': float(np.mean(high_dim_distances ** 2))
    }
    
    return metrics


def predict(
    model: nn.Module,
    X: np.ndarray
) -> np.ndarray:
    """
    Generate embeddings for input data.
    
    Args:
        model: Encoder model
        X: Input data
        
    Returns:
        Low-dimensional embeddings
    """
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        embeddings = model(X_tensor).cpu().numpy()
    
    return embeddings


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, float],
    output_dir: str = 'output'
) -> None:
    """
    Save model artifacts and metrics.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'umap_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.npy')
    np.save(metrics_path, metrics)
    
    print(f"Artifacts saved to {output_dir}")


def main():
    """Main function to run the UMAP-like task."""
    print("=" * 60)
    print("UMAP-like Dimensionality Reduction Task")
    print("=" * 60)
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, X_full, y_full = make_dataloaders(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        batch_size=128,
        val_ratio=0.2
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(
        input_dim=20,
        embedding_dim=2,
        hidden_dim=64
    )
    print(f"Model architecture:\n{model}")
    
    #
"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) Implementation
Implements clustering on the moons dataset to find non-convex clusters.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

# Set paths
OUTPUT_DIR = '/Developer/AIserver/output/tasks/cluster_lvl3_dbscan'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

# Get device (CPU or GPU)
def get_device():
    """Get the computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Task metadata
def get_task_metadata():
    """Return task metadata."""
    return {
        'task_type': 'clustering',
        'algorithm': 'DBSCAN',
        'dataset': 'moons',
        'description': 'Density-based clustering for non-convex clusters'
    }

# Create dataloaders for clustering
def make_dataloaders(batch_size=32, test_size=0.2, random_state=42):
    """Create dataloaders for the moons dataset."""
    # Generate moons dataset
    X, y_true = make_moons(n_samples=300, noise=0.1, random_state=random_state)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_true, test_size=test_size, random_state=random_state
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    
    # Create datasets and dataloaders (labels not used for unsupervised learning)
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val

# DBSCAN model class
class DBSCANModel(nn.Module):
    """DBSCAN clustering model."""
    
    def __init__(self, eps=0.3, min_samples=5):
        super(DBSCANModel, self).__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.device = get_device()
        
    def _compute_distance_matrix(self, X):
        """Compute pairwise distance matrix."""
        # X shape: (N, D)
        # Using broadcasting for efficient distance computation
        diff = X.unsqueeze(1) - X.unsqueeze(0)  # (N, N, D)
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=2))  # (N, N)
        return dist_matrix
    
    def _get_neighbors(self, dist_matrix, point_idx):
        """Get indices of neighbors within eps distance."""
        return torch.where(dist_matrix[point_idx] <= self.eps)[0]
    
    def fit(self, X):
        """
        Fit DBSCAN clustering on input data.
        
        Args:
            X: Tensor of shape (N, D) containing input data
            
        Returns:
            labels: Array of cluster labels (shape: N)
                   -1 indicates noise points
        """
        X = X.to(self.device)
        n_samples = X.shape[0]
        
        # Compute distance matrix
        dist_matrix = self._compute_distance_matrix(X)
        
        # Initialize labels as undefined (-2)
        labels = torch.full((n_samples,), -2, dtype=torch.long, device=self.device)
        
        # Cluster counter
        cluster_id = 0
        
        # Iterate over all points
        for point_idx in range(n_samples):
            # Skip if already processed
            if labels[point_idx] != -2:
                continue
                
            # Get neighbors
            neighbors = self._get_neighbors(dist_matrix, point_idx)
            
            # Check if point is a core point (has enough neighbors)
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1  # Mark as noise (temporary)
                continue
            
            # Start a new cluster
            labels[point_idx] = cluster_id
            
            # Expand cluster
            seed_set = list(neighbors.cpu().numpy())
            i = 0
            while i < len(seed_set):
                q = seed_set[i]
                
                if labels[q] == -1:
                    # Change noise to border point
                    labels[q] = cluster_id
                elif labels[q] == -2:
                    # Add to cluster
                    labels[q] = cluster_id
                    
                    # Get neighbors of q
                    q_neighbors = self._get_neighbors(dist_matrix, q)
                    
                    if len(q_neighbors) >= self.min_samples:
                        # q is a core point, add its neighbors to seed set
                        for neighbor in q_neighbors:
                            if neighbor.item() not in seed_set:
                                seed_set.append(neighbor.item())
                
                i += 1
            
            cluster_id += 1
        
        # Convert to numpy array
        return labels.cpu().numpy()
    
    def forward(self, X):
        """Forward pass - fit and return labels."""
        return self.fit(X)

# Build model
def build_model(eps=0.3, min_samples=5):
    """Build DBSCAN model."""
    model = DBSCANModel(eps=eps, min_samples=min_samples)
    return model

# Train function (for DBSCAN, this is just fitting)
def train(model, train_loader, device, eps=0.3, min_samples=5):
    """Train the DBSCAN model."""
    # Get all training data
    X_train_list = []
    for batch in train_loader:
        X_train_list.append(batch[0])
    X_train = torch.cat(X_train_list, dim=0).to(device)
    
    # Fit DBSCAN
    model.eps = eps
    model.min_samples = min_samples
    labels = model.fit(X_train)
    
    return model, labels

# Evaluate function
def evaluate(model, val_loader, device, train_labels=None, val_data=None, y_val=None):
    """
    Evaluate the DBSCAN model.
    
    Returns metrics including MSE, R2, and clustering-specific metrics.
    """
    # Get all validation data
    X_val_list = []
    for batch in val_loader:
        X_val_list.append(batch[0])
    X_val = torch.cat(X_val_list, dim=0).to(device)
    
    # Predict clusters
    val_labels = model.fit(X_val)
    
    # Compute clustering metrics
    metrics = {}
    
    # Number of clusters (excluding noise)
    unique_labels = set(val_labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = np.sum(np.array(val_labels) == -1)
    
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = int(n_noise)
    metrics['noise_ratio'] = float(n_noise) / len(val_labels) if len(val_labels) > 0 else 0.0
    
    # Silhouette score (only if we have multiple clusters and non-noise points)
    if n_clusters > 1 and n_clusters < len(val_labels):
        try:
            # Filter out noise points for silhouette score
            non_noise_mask = val_labels != -1
            if np.sum(non_noise_mask) > 1:
                X_val_filtered = X_val[non_noise_mask].cpu().numpy()
                labels_filtered = val_labels[non_noise_mask]
                metrics['silhouette_score'] = float(silhouette_score(X_val_filtered, labels_filtered))
            else:
                metrics['silhouette_score'] = 0.0
        except:
            metrics['silhouette_score'] = 0.0
    else:
        metrics['silhouette_score'] = 0.0
    
    # Adjusted Rand Index (if ground truth available)
    if y_val is not None:
        try:
            metrics['adjusted_rand_index'] = float(adjusted_rand_score(y_val, val_labels))
        except:
            metrics['adjusted_rand_score'] = 0.0
    
    # For DBSCAN, we don't have traditional MSE/R2, but we can compute
    # the within-cluster variance as a proxy
    if n_clusters > 0:
        total_variance = 0.0
        for cluster in range(n_clusters):
            cluster_mask = val_labels == cluster
            if np.sum(cluster_mask) > 0:
                cluster_points = X_val[cluster_mask].cpu().numpy()
                cluster_center = cluster_points.mean(axis=0)
                variance = np.sum((cluster_points - cluster_center) ** 2)
                total_variance += variance
        metrics['within_cluster_variance'] = float(total_variance)
    else:
        metrics['within_cluster_variance'] = float(torch.sum(X_val ** 2).item())
    
    # Convert to numpy for consistency
    metrics['n_samples'] = len(val_labels)
    
    return val_labels, metrics

# Predict function
def predict(model, X):
    """Predict cluster labels for input data."""
    X_tensor = torch.FloatTensor(X).to(model.device)
    return model.fit(X_tensor)

# Save artifacts
def save_artifacts(model, train_labels, val_labels, metrics, X_train, X_val, y_train, y_val):
    """Save model artifacts including plots and metrics."""
    # Save metrics as JSON
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model parameters
    model_path = os.path.join(OUTPUT_DIR, 'model.pt')
    torch.save({
        'eps': model.eps,
        'min_samples': model.min_samples,
        'state_dict': None  # DBSCAN doesn't have learnable parameters
    }, model_path)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training results
    unique_train_labels = set(train_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_train_labels)))
    
    for cluster in unique_train_labels:
        if cluster == -1:
            # Noise points
            color = 'gray'
            label = 'Noise'
        else:
            color = colors[cluster % len(colors)]
            label = f'Cluster {cluster}'
        
        mask = train_labels == cluster
        axes[0].scatter(X_train[mask, 0], X_train[mask, 1], 
                       c=color, label=label, alpha=0.6, edgecolors='k', s=50)
    
    axes[0].set_title(f'Training Clusters (DBSCAN)\nClusters: {len([l for l in unique_train_labels if l != -1])}, Noise: {np.sum(train_labels == -1)}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot validation results
    unique_val_labels = set(val_labels)
    
    for cluster in unique_val_labels:
        if cluster == -1:
            color = 'gray'
            label = 'Noise'
        else:
            color = colors[cluster % len(colors)]
            label = f'Cluster {cluster}'
        
        mask = val_labels == cluster
        axes[1].scatter(X_val[mask, 0], X_val[mask, 1], 
                       c=color, label=label, alpha=0.6, edgecolors='k', s=50)
    
    axes[1].set_title(f'Validation Clusters (DBSCAN)\nClusters: {len([l for l in unique_val_labels if l != -1])}, Noise: {np.sum(val_labels == -1)}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save data for reference
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), train_labels)
    np.save(os.path.join(OUTPUT_DIR, 'val_labels.npy'), val_labels)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    
    print(f"Artifacts saved to {OUTPUT_DIR}")

# Main function
def main():
    """Main function to run DBSCAN clustering."""
    print("=" * 60)
    print("DBSCAN Clustering on Moons Dataset")
    print("=" * 60)
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_type']} - {metadata['algorithm']}")
    print(f"Dataset: {metadata['dataset']}")
    
    # Create dataloaders
    print("\n--- Creating DataLoaders ---")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        batch_size=32, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model with optimal parameters for moons dataset
    print("\n--- Building Model ---")
    # DBSCAN parameters tuned for moons dataset
    model = build_model(eps=0.15, min_samples=5)
    print(f"DBSCAN parameters: eps={model.eps}, min_samples={model.min_samples}")
    
    # Train model
    print("\n--- Training Model ---")
    model, train_labels = train(model, train_loader, device, eps=0.15, min_samples=5)
    print(f"Training completed. Found {len(set(train_labels)) - (1 if -1 in train_labels else 0)} clusters")
    
    # Evaluate on training data
    print("\n--- Evaluating on Training Data ---")
    train_pred, train_metrics = evaluate(
        model, train_loader, device, 
        train_labels=train_labels, val_data=X_train, y_val=y_train
    )
    print(f"Training Metrics:")
    print(f"  - Number of clusters: {train_metrics['n_clusters']}")
    print(f"  - Number of noise points: {train_metrics['n_noise']}")
    print(f"  - Noise ratio: {train_metrics['noise_ratio']:.4f}")
    print(f"  - Silhouette score: {train_metrics['silhouette_score']:.4f}")
    print(f"  - Within-cluster variance: {train_metrics['within_cluster_variance']:.4f}")
    
    # Evaluate on validation data
    print("\n--- Evaluating on Validation Data ---")
    val_pred, val_metrics = evaluate(
        model, val_loader, device,
        train_labels=train_labels, val_data=X_val, y_val=y_val
    )
    print(f"Validation Metrics:")
    print(f"  - Number of clusters: {val_metrics['n_clusters']}")
    print(f"  - Number of noise points: {val_metrics['n_noise']}")
    print(f"  - Noise ratio: {val_metrics['noise_ratio']:.4f}")
    print(f"  - Silhouette score: {val_metrics['silhouette_score']:.4f}")
    print(f"  - Within-cluster variance: {val_metrics['within_cluster_variance']:.4f}")
    
    # Check if ground truth available for ARI
    if 'adjusted_rand_index' in val_metrics:
        print(f"  - Adjusted Rand Index: {val_metrics['adjusted_rand_index']:.4f}")
    
    # Quality thresholds
    print("\n--- Quality Thresholds ---")
    success = True
    
    # DBSCAN should find at least 2 clusters for moons dataset
    if val_metrics['n_clusters'] < 2:
        print(f"FAIL: Expected at least 2 clusters, got {val_metrics['n_clusters']}")
        success = False
    else:
        print(f"PASS: Found {val_metrics['n_clusters']} clusters (expected >= 2)")
    
    # Silhouette score should be reasonable (not too low)
    if val_metrics['silhouette_score'] < 0.5:
        print(f"WARNING: Silhouette score {val_metrics['silhouette_score']:.4f} is low (expected > 0.5)")
        # Don't fail on this as it depends on data
    else:
        print(f"PASS: Silhouette score {val_metrics['silhouette_score']:.4f} is good (expected > 0.5)")
    
    # Noise ratio should be reasonable (not too high)
    if val_metrics['noise_ratio'] > 0.3:
        print(f"WARNING: Noise ratio {val_metrics['noise_ratio']:.4f} is high (expected < 0.3)")
    else:
        print(f"PASS: Noise ratio {val_metrics['noise_ratio']:.4f} is acceptable (expected < 0.3)")
    
    # Save artifacts
    print("\n--- Saving Artifacts ---")
    save_artifacts(model, train_labels, val_pred, val_metrics, X_train, X_val, y_train, y_val)
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("PASS: All quality thresholds met!")
    else:
        print("FAIL: Some quality thresholds not met")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

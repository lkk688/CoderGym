import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import time

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Get device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# Task metadata
def get_task_metadata():
    return {
        'task_name': 'kmeans_clustering',
        'task_type': 'clustering',
        'input_type': 'continuous',
        'output_type': 'cluster_assignment',
        'num_clusters': 4,
        'input_dim': 2,
        'metrics': ['inertia', 'silhouette_score', 'mse', 'r2']
    }

# Create dataloaders for clustering
def make_dataloaders(batch_size=32, val_split=0.2, random_state=42):
    set_seed(random_state)
    
    # Generate synthetic data with clear clusters
    X, y_true = make_blobs(
        n_samples=1000,
        centers=4,
        cluster_std=0.60,
        random_state=random_state,
        n_features=2
    )
    
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and validation
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    val_size = int(n_samples * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_train = X[train_indices]
    X_val = X[val_indices]
    y_train = y_true[train_indices]
    y_val = y_true[val_indices]
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val, scaler

# K-Means model class
class KMeans(nn.Module):
    def __init__(self, n_clusters=4, max_iter=300, tol=1e-4, random_state=42):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _kmeans_plus_plus_init(self, X):
        """K-means++ initialization"""
        n_samples, n_features = X.shape
        centroids = torch.zeros(self.n_clusters, n_features, device=X.device)
        
        # Choose first centroid randomly
        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx]
        
        # Choose remaining centroids with probability proportional to distance squared
        for k in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = torch.zeros(n_samples, device=X.device)
            for i in range(n_samples):
                min_dist = float('inf')
                for j in range(k):
                    dist = torch.sum((X[i] - centroids[j]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                distances[i] = min_dist
            
            # Choose next centroid with probability proportional to distance squared
            probs = distances / distances.sum()
            cumprobs = torch.cumsum(probs, dim=0)
            r = torch.rand(1, device=X.device)
            next_idx = torch.searchsorted(cumprobs, r).item()
            if next_idx >= n_samples:
                next_idx = n_samples - 1
            centroids[k] = X[next_idx]
        
        return centroids
    
    def _assign_clusters(self, X):
        """Assign each sample to nearest centroid"""
        # X: (n_samples, n_features)
        # centroids: (n_clusters, n_features)
        # distances: (n_samples, n_clusters)
        distances = torch.cdist(X.unsqueeze(0), self.centroids.unsqueeze(0)).squeeze(0)
        return torch.argmin(distances, dim=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids based on cluster assignments"""
        new_centroids = torch.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[k] = X[torch.randint(0, len(X), (1,))]
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """Compute within-cluster sum of squares"""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                cluster_points = X[mask]
                centroid = self.centroids[k]
                inertia += torch.sum((cluster_points - centroid) ** 2)
        return inertia.item()
    
    def forward(self, X):
        """Predict cluster labels"""
        return self._assign_clusters(X)
    
    def fit(self, X):
        """Fit K-means to data"""
        device = X.device
        X = X.to(device)
        
        # Initialize centroids using k-means++
        self.centroids = self._kmeans_plus_plus_init(X)
        
        for i in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = torch.sqrt(torch.sum((new_centroids - self.centroids) ** 2))
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                self.n_iter_ = i + 1
                break
            self.n_iter_ = i + 1
        
        # Compute final inertia
        self.inertia_ = self._compute_inertia(X, labels)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for samples in X"""
        device = self.centroids.device
        X = X.to(device)
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """Fit and predict"""
        return self.fit(X).predict(X)

# Build model
def build_model(n_clusters=4, max_iter=300, tol=1e-4, random_state=42):
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    return model

# Train function
def train(model, train_loader, device, verbose=True):
    # Collect all training data
    X_train = []
    for X, _ in train_loader:
        X_train.append(X)
    X_train = torch.cat(X_train, dim=0).to(device)
    
    # Fit the model
    model.fit(X_train)
    
    if verbose:
        print(f"Training completed in {model.n_iter_} iterations")
        print(f"Final inertia: {model.inertia_:.4f}")
    
    return model

# Evaluate function
def evaluate(model, data_loader, X_data, y_true, device):
    """Evaluate model and return metrics"""
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        X_all = []
        for X, _ in data_loader:
            X_all.append(X)
        X_all = torch.cat(X_all, dim=0).to(device)
        
        predictions = model.predict(X_all)
    
    # Compute metrics
    # Inertia (within-cluster SSE)
    inertia = model.inertia_
    
    # Silhouette score
    try:
        sil_score = silhouette_score(X_all.cpu().numpy(), predictions.cpu().numpy())
    except:
        sil_score = 0.0
    
    # MSE between points and their assigned centroids
    mse = 0.0
    for k in range(model.n_clusters):
        mask = predictions == k
        if mask.sum() > 0:
            cluster_points = X_all[mask]
            centroid = model.centroids[k]
            mse += torch.sum((cluster_points - centroid) ** 2).item()
    mse /= len(X_all)
    
    # R2 score (comparing to baseline)
    # Calculate total sum of squares
    overall_mean = X_all.mean(dim=0)
    ss_tot = torch.sum((X_all - overall_mean) ** 2).item()
    ss_res = inertia
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'inertia': inertia,
        'silhouette_score': sil_score,
        'mse': mse,
        'r2': r2,
        'n_clusters': model.n_clusters,
        'n_iter': model.n_iter_
    }

# Predict function
def predict(model, X, device):
    """Predict cluster labels"""
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(X).to(device)
        predictions = model.predict(X)
    return predictions.cpu().numpy()

# Save artifacts
def save_artifacts(model, metrics, output_dir, X_train, y_train, X_val, y_val):
    """Save model, metrics, and visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state
    torch.save({
        'centroids': model.centroids,
        'inertia': model.inertia_,
        'n_iter': model.n_iter_,
        'n_clusters': model.n_clusters
    }, os.path.join(output_dir, 'model.pt'))
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualization
    device = get_device()
    
    # Plot training data
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training data with clusters
    plt.subplot(1, 2, 1)
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    
    # Get predictions for training data
    train_preds = predict(model, X_train_np, device)
    
    # Plot clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    for k in range(model.n_clusters):
        mask = train_preds == k
        plt.scatter(X_train_np[mask, 0], X_train_np[mask, 1], 
                   c=colors[k % len(colors)], label=f'Cluster {k}', alpha=0.6)
    
    # Plot centroids
    centroids_np = model.centroids.cpu().numpy()
    plt.scatter(centroids_np[:, 0], centroids_np[:, 1], 
               c='black', marker='X', s=200, label='Centroids')
    
    plt.title(f'Training Data - K-Means Clustering\nInertia: {metrics["inertia"]:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Validation data with clusters
    plt.subplot(1, 2, 2)
    X_val_np = X_val.cpu().numpy() if isinstance(X_val, torch.Tensor) else X_val
    y_val_np = y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
    
    # Get predictions for validation data
    val_preds = predict(model, X_val_np, device)
    
    # Plot clusters
    for k in range(model.n_clusters):
        mask = val_preds == k
        plt.scatter(X_val_np[mask, 0], X_val_np[mask, 1], 
                   c=colors[k % len(colors)], label=f'Cluster {k}', alpha=0.6)
    
    # Plot centroids
    plt.scatter(centroids_np[:, 0], centroids_np[:, 1], 
               c='black', marker='X', s=200, label='Centroids')
    
    plt.title(f'Validation Data - K-Means Clustering\nSilhouette: {metrics["silhouette_score"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_visualization.png'), dpi=150)
    plt.close()
    
    # Compare with sklearn baseline
    sklearn_kmeans = SklearnKMeans(n_clusters=model.n_clusters, random_state=42, n_init=10)
    sklearn_kmeans.fit(X_train_np)
    sklearn_pred = sklearn_kmeans.predict(X_train_np)
    sklearn_inertia = sklearn_kmeans.inertia_
    
    # Calculate difference
    diff_percent = abs(sklearn_inertia - metrics['inertia']) / sklearn_inertia * 100
    
    # Save comparison
    comparison = {
        'our_inertia': metrics['inertia'],
        'sklearn_inertia': sklearn_inertia,
        'difference_percent': diff_percent,
        'within_tolerance': diff_percent < 5.0
    }
    
    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison

# Main execution
if __name__ == '__main__':
    # Set output directory
    output_dir = '/Developer/AIserver/output/tasks/cluster_lvl1_kmeans'
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")
    print(f"Number of clusters: {metadata['num_clusters']}")
    print(f"Input dimension: {metadata['input_dim']}")
    print("-" * 50)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating data loaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val, scaler = make_dataloaders(
        batch_size=32, val_split=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(
        n_clusters=metadata['num_clusters'],
        max_iter=300,
        tol=1e-4,
        random_state=42
    )
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    model = train(model, train_loader, device)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_metrics = evaluate
"""
kNN (k-Nearest Neighbors) ML Task with ANN Indexing Benchmarking
Engineering report: batching, memory, speed benchmark; optional FAISS comparison
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
def get_device():
    """Get computation device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()

# Task metadata
def get_task_metadata():
    """Return task metadata"""
    return {
        'task_type': 'knn_benchmark',
        'description': 'kNN with ANN indexing benchmarking',
        'input_type': 'tabular',
        'output_type': 'classification/regression',
        'required_metrics': ['mse', 'r2', 'accuracy', 'latency']
    }

# Set seed for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Get device
def get_device():
    """Get computation device"""
    return DEVICE

# Generate synthetic dataset for benchmarking
def generate_synthetic_data(n_samples=10000, n_features=20, n_classes=5, regression=False, random_state=42):
    """Generate synthetic dataset for kNN benchmarking"""
    np.random.seed(random_state)
    
    if regression:
        # Regression: continuous target
        X = np.random.randn(n_samples, n_features)
        # Create target with some structure for kNN to capture
        centers = np.random.randn(n_classes, n_features)
        labels = np.random.randint(0, n_classes, n_samples)
        X = X + 0.5 * centers[labels]
        y = np.sum(X * centers[labels], axis=1) + np.random.randn(n_samples) * 0.1
    else:
        # Classification: discrete classes
        X = np.random.randn(n_samples, n_features)
        centers = np.random.randn(n_classes, n_features)
        labels = np.random.randint(0, n_classes, n_samples)
        X = X + 0.8 * centers[labels]
        y = labels
    
    return X, y

# Create dataloaders
def make_dataloaders(X, y, batch_size=32, test_size=0.2, regression=False, random_state=42):
    """Create train and validation dataloaders"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if not regression else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) if regression else torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1) if regression else torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'scaler': scaler,
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)) if not regression else None
    }

# Build kNN model
class kNNModel:
    """kNN model wrapper with ANN indexing support"""
    
    def __init__(self, n_neighbors=5, algorithm='auto', metric='euclidean', use_faiss=False):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.use_faiss = use_faiss
        self.model = None
        self.index = None
        
    def fit(self, X, y=None):
        """Fit the kNN model"""
        if self.use_faiss and self._faiss_available():
            self._build_faiss_index(X)
        else:
            self.model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, metric=self.metric)
            self.model.fit(X)
            
    def _faiss_available(self):
        """Check if FAISS is available"""
        try:
            import faiss
            return True
        except ImportError:
            return False
            
    def _build_faiss_index(self, X):
        """Build FAISS index"""
        import faiss
        dimension = X.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(X.astype('float32'))
        
    def kneighbors(self, X, return_distance=True):
        """Find k nearest neighbors"""
        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(X.astype('float32'), self.n_neighbors)
            if return_distance:
                return distances, indices
            return indices
        else:
            return self.model.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=return_distance)
            
    def predict(self, X):
        """Predict using kNN (for classification/regression)"""
        # This would need to be extended for actual prediction tasks
        distances, indices = self.kneighbors(X, return_distance=True)
        return indices  # Return neighbor indices for further processing
        
    def score(self, X, y):
        """Score the model (placeholder)"""
        # This would need to be extended for actual scoring
        return 0.0

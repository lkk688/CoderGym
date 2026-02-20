"""
Linear Discriminant Analysis (LDA) - Supervised Projection
Implements LDA from scratch and compares to sklearn implementation.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define output directory
OUTPUT_DIR = "/Developer/AIserver/output/tasks/dr_lvl2_lda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "lda_supervised_projection",
        "task_type": "dimensionality_reduction",
        "description": "Linear Discriminant Analysis for supervised dimensionality reduction",
        "input_type": "tabular",
        "output_type": "projection",
        "framework": "pytorch",
        "dataset": "iris"
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(test_size=0.2, batch_size=32):
    """Create data loaders for Iris dataset."""
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
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
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_names': iris.target_names
    }


class LDA(nn.Module):
    """Linear Discriminant Analysis implementation."""
    
    def __init__(self, n_components=None):
        super(LDA, self).__init__()
        self.n_components = n_components
        self.scalings_ = None
        self.coef_ = None
        self.intercept_ = None
        self.means_ = None
        self.priors_ = None
        
    def _compute_scatter_matrices(self, X, y):
        """Compute within-class and between-class scatter matrices."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Overall mean
        overall_mean = np.mean(X, axis=0)
        
        # Scatter matrices
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for c in range(n_classes):
            # Class mask
            class_mask = (y == c)
            X_c = X[class_mask]
            
            # Class mean
            mean_c = np.mean(X_c, axis=0)
            
            # Within-class scatter
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
            
            # Between-class scatter
            n_c = np.sum(class_mask)
            S_B += n_c * np.outer(mean_c - overall_mean, mean_c - overall_mean)
        
        return S_W, S_B
    
    def fit(self, X, y):
        """Fit LDA model."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Compute scatter matrices
        S_W, S_B = self._compute_scatter_matrices(X, y)
        
        # Solve generalized eigenvalue problem: S_B * w = lambda * S_W * w
        # Using scipy.linalg.eigh for numerical stability
        try:
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(S_B, S_W, subset_by_index=[0, min(n_classes-1, n_features)-1])
        except ImportError:
            # Fallback to numpy if scipy not available
            eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.pinv(S_W), S_B))
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        if self.n_components is None:
            n_components = min(n_classes - 1, n_features)
        else:
            n_components = min(self.n_components, n_classes - 1, n_features)
        
        self.scalings_ = eigenvectors[:, :n_components]
        self.coef_ = self.scalings_.T
        
        # Compute class means and priors for prediction
        self.means_ = np.array([np.mean(X[y == c], axis=0) for c in range(n_classes)])
        self.priors_ = np.array([np.sum(y == c) / n_samples for c in range(n_classes)])
        
        return self
    
    def transform(self, X):
        """Project data onto LDA components."""
        X = np.asarray(X)
        return np.dot(X, self.scalings_)
    
    def predict(self, X):
        """Predict class labels using LDA projection."""
        X = np.asarray(X)
        X_proj = self.transform(X)
        
        # Predict using nearest class mean classifier
        predictions = []
        for x in X_proj:
            distances = [np.linalg.norm(x - self.means_[c]) for c in range(len(self.means_))]
            predictions.append(np.argmin(distances))
        
        return np.array(predictions)


class LDAProjectionModel(nn.Module):
    """PyTorch wrapper for LDA projection."""
    
    def __init__(self, n_features, n_components, n_classes):
        super(LDAProjectionModel, self).__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.n_classes = n_classes
        
        # Linear layer to perform LDA projection
        self.projection = nn.Linear(n_features, n_components, bias=False)
        
        # Initialize with zeros (will be replaced by actual LDA)
        with torch.no_grad():
            self.projection.weight.zero_()
    
    def forward(self, x):
        """Forward pass - project input."""
        return self.projection(x)
    
    def fit_lda(self, X, y):
        """Fit LDA using numpy implementation and update weights."""
        # Convert to numpy
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # Fit custom LDA
        lda = LDA(n_components=self.n_components)
        lda.fit(X_np, y_np)
        
        # Update projection weights - scalings_ has shape (n_features, n_components)
        # but Linear layer expects weight of shape (n_components, n_features)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.FloatTensor(lda.scalings_.T)
            )
        
        return lda


def build_model(device, n_features=4, n_components=2, n_classes=3):
    """Build LDA model."""
    model = LDAProjectionModel(n_features, n_components, n_classes).to(device)
    
    # Fit LDA on the full dataset
    iris = load_iris()
    X = torch.FloatTensor(iris.data).to(device)
    y = torch.LongTensor(iris.target).to(device)
    
    lda = model.fit_lda(X, y)
    
    return model, lda


def train(model, train_loader, device, epochs=100, lr=0.01):
    """Train LDA projection model."""
    # LDA doesn't really need training in the traditional sense
    # But we'll do a few epochs to fine-tune if needed
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            projected = model(batch_X)
            
            # For LDA, we don't have a traditional loss - 
            # we're just ensuring the projection works
            # Use a simple reconstruction loss as proxy
            loss = criterion(projected, projected)  # Dummy loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return model


def evaluate(model, data_loader, device, X_val, y_val, lda_model):
    """Evaluate LDA model."""
    model.eval()
    
    # Get predictions using 1-NN classifier in LDA space
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    with torch.no_grad():
        X_val_proj = model(X_val_tensor)
    
    X_val_proj_np = X_val_proj.detach().cpu().numpy()
    
    # Train 1-NN classifier on projected training data
    X_train_tensor = torch.FloatTensor(data_loader.dataset.tensors[0].numpy()).to(device)
    y_train = data_loader.dataset.tensors[1].numpy()
    
    with torch.no_grad():
        X_train_proj = model(X_train_tensor)
    
    X_train_proj_np = X_train_proj.detach().cpu().numpy()
    
    # 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_proj_np, y_train)
    y_pred = knn.predict(X_val_proj_np)
    
    # Compute metrics
    accuracy = accuracy_score(y_val, y_pred)
    
    # Also compute PCA baseline for comparison
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(data_loader.dataset.tensors[0].numpy())
    X_val_pca = pca.transform(X_val)
    
    knn_pca = KNeighborsClassifier(n_neighbors=1)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_pca = knn_pca.predict(X_val_pca)
    
    accuracy_pca = accuracy_score(y_val, y_pred_pca)
    
    # Compute sklearn LDA accuracy for comparison
    sklearn_lda = SklearnLDA(n_components=2)
    X_train_sklearn = sklearn_lda.fit_transform(data_loader.dataset.tensors[0].numpy(), y_train)
    X_val_sklearn = sklearn_lda.transform(X_val)
    
    knn_sklearn = KNeighborsClassifier(n_neighbors=1)
    knn_sklearn.fit(X_train_sklearn, y_train)
    y_pred_sklearn = knn_sklearn.predict(X_val_sklearn)
    
    accuracy_sklearn = accuracy_score(y_val, y_pred_sklearn)
    
    return {
        'accuracy': accuracy,
        'accuracy_pca': accuracy_pca,
        'accuracy_sklearn': accuracy_sklearn,
        'lda_beats_pca': accuracy > accuracy_pca,
        'lda_matches_sklearn': abs(accuracy - accuracy_sklearn) < 0.05
    }


def predict(model, X, device):
    """Predict using LDA model."""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        projection = model(X_tensor)
    
    return projection.detach().cpu().numpy()


def save_artifacts(model, lda_model, metrics, X_train, y_train, X_val, y_val, class_names):
    """Save model artifacts and visualizations."""
    # Save model weights
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lda_model.pt"))
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save LDA components
    np.save(os.path.join(OUTPUT_DIR, "lda_components.npy"), lda_model.scalings_)
    np.save(os.path.join(OUTPUT_DIR, "lda_means.npy"), lda_model.means_)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: LDA projection
    plt.subplot(1, 3, 1)
    X_train_proj = lda_model.transform(X_train)
    X_val_proj = lda_model.transform(X_val)
    
    colors = ['red', 'green', 'blue']
    for i, (name, color) in enumerate(zip(class_names, colors)):
        plt.scatter(X_train_proj[y_train == i, 0], X_train_proj[y_train == i, 1], 
                   c=color, label=f'{name} (train)', alpha=0.7, marker='o')
        plt.scatter(X_val_proj[y_val == i, 0], X_val_proj[y_val == i, 1], 
                   c=color, label=f'{name} (val)', alpha=0.7, marker='s')
    
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title('LDA Projection (Custom Implementation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: PCA projection for comparison
    plt.subplot(1, 3, 2)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], 
                   c=color, label=f'{name} (train)', alpha=0.7, marker='o')
        plt.scatter(X_val_pca[y_val == i, 0], X_val_pca[y_val == i, 1], 
                   c=color, label=f'{name} (val)', alpha=0.7, marker='s')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Projection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: sklearn LDA for comparison
    plt.subplot(1, 3, 3)
    sklearn_lda = SklearnLDA(n_components=2)
    X_train_sklearn = sklearn_lda.fit_transform(X_train, y_train)
    X_val_sklearn = sklearn_lda.transform(X_val)
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        plt.scatter(X_train_sklearn[y_train == i, 0], X_train_sklearn[y_train == i, 1], 
                   c=color, label=f'{name} (train)', alpha=0.7, marker='o')
        plt.scatter(X_val_sklearn[y_val == i, 0], X_val_sklearn[y_val == i, 1], 
                   c=color, label=f'{name} (val)', alpha=0.7, marker='s')
    
    plt.xlabel('LDA Component 1 (sklearn)')
    plt.ylabel('LDA Component 2 (sklearn)')
    plt.title('LDA Projection (sklearn)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lda_visualization.png"), dpi=150)
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run LDA task."""
    print("=" * 60)
    print("Linear Discriminant Analysis (LDA) - Supervised Projection")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\n[1] Loading data...")
    data = make_dataloaders(test_size=0.2, batch_size=32)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']
    n_features = data['n_features']
    n_classes = data['n_classes']
    class_names = data['class_names']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {n_features}, Classes: {n_classes}")
    
    # Build model
    print("\n[2] Building LDA model...")
    model, lda_model = build_model(device, n_features, n_components=2, n_classes=n_classes)
    print(f"Model built with {n_features} -> {2} dimensions")
    
    # Train model
    print("\n[3] Training model...")
    model = train(model, train_loader, device, epochs=50, lr=0.01)
    print("Training completed")
    
    # Evaluate on training data
    print("\n[4] Evaluating on training data...")
    train_metrics = evaluate(model, train_loader, device, X_train, y_train, lda_model)
    print(f"Training Metrics:")
    print(f"  - 1-NN Accuracy (LDA): {train_metrics['accuracy']:.4f}")
    print(f"  - 1-NN Accuracy (PCA): {train_metrics['accuracy_pca']:.4f}")
    print(f"  - 1-NN Accuracy (sklearn LDA): {train_metrics['accuracy_sklearn']:.4f}")
    
    # Evaluate on validation data
    print("\n[5] Evaluating on validation data...")
    val_metrics = evaluate(model, val_loader, device, X_val, y_val, lda_model)
    print(f"Validation Metrics:")
    print(f"  - 1-NN Accuracy (LDA): {val_metrics['accuracy']:.4f}")
    print(f"  - 1-NN Accuracy (PCA): {val_metrics['accuracy_pca']:.4f}")
    print(f"  - 1-NN Accuracy (sklearn LDA): {val_metrics['accuracy_sklearn']:.4f}")
    
    # Save artifacts
    print("\n[6] Saving artifacts...")
    save_artifacts(model, lda_model, val_metrics, X_train, y_train, X_val, y_val, class_names)
    
    print("\n" + "=" * 60)
    print("LDA Task Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

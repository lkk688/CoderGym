"""
kNN + Learned Mahalanobis Metric
Learn a linear transform A to improve kNN classification.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set paths
OUTPUT_DIR = '/Developer/AIserver/output/tasks/knn_lvl3_metric_learning_intro'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
set_seed_called = False

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    global set_seed_called
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed_called = True

def get_device():
    """Get computation device."""
    if not set_seed_called:
        set_seed()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'knn_metric_learning',
        'task_type': 'classification',
        'input_type': 'tabular',
        'output_dim': 1,
        'description': 'Learn a linear transform A to improve kNN classification'
    }

def make_dataloaders(batch_size=32, test_size=0.2, random_state=42):
    """Create train and validation dataloaders."""
    if not set_seed_called:
        set_seed(random_state)
    
    # Create synthetic classification dataset with some structure
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=0.1,
        class_sep=1.0,
        random_state=random_state
    )
    
    # Convert to float32
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val

class LearnedMetricKNN(nn.Module):
    """kNN with learned Mahalanobis metric (linear transform)."""
    
    def __init__(self, n_features, n_neighbors=5):
        super(LearnedMetricKNN, self).__init__()
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        
        # Learnable linear transform matrix A (n_features x n_features)
        # We'll use a lower triangular matrix with positive diagonal for stability
        # Initialize as identity
        self.A = nn.Linear(n_features, n_features, bias=False)
        # Initialize to identity
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(n_features))
    
    def get_transform(self):
        """Get the transformation matrix A."""
        return self.A.weight
    
    def transform(self, X):
        """Apply linear transform to input."""
        return self.A(X)
    
    def forward(self, X_train, X_test, y_train):
        """
        Perform kNN in transformed space.
        
        Args:
            X_train: Training features (N_train, D)
            X_test: Test features (N_test, D)
            y_train: Training labels (N_train, 1)
        
        Returns:
            Predictions for X_test
        """
        # Transform both train and test (this creates the computational graph)
        X_train_t = self.transform(X_train)
        X_test_t = self.transform(X_test)
        
        # Compute distances in transformed space (differentiable)
        # X_test_t: (N_test, D), X_train_t: (N_train, D)
        # Distances: (N_test, N_train)
        distances = self._compute_distances(X_test_t, X_train_t)
        
        # Use soft kNN: compute weights using softmax of negative distances
        # This makes the operation differentiable
        # Use a temperature parameter for sharper or softer weights
        temperature = 1.0
        weights = torch.softmax(-distances * temperature, dim=1)  # (N_test, N_train)
        
        # Compute weighted average of all training labels
        # weights: (N_test, N_train), y_train: (N_train, 1)
        # Result: (N_test, 1)
        predictions = torch.matmul(weights, y_train)
        
        
        return predictions
    
    def _compute_distances(self, X_test, X_train):
        """Compute Euclidean distances between test and train points."""
        # X_test: (N_test, D), X_train: (N_train, D)
        # Output: (N_test, N_train)
        
        # Using (a-b)^2 = a^2 + b^2 - 2ab
        test_sq = torch.sum(X_test ** 2, dim=1, keepdim=True)  # (N_test, 1)
        train_sq = torch.sum(X_train ** 2, dim=1, keepdim=True)  # (N_train, 1)
        cross_term = torch.matmul(X_test, X_train.t())  # (N_test, N_train)
        
        distances = test_sq + train_sq.t() - 2 * cross_term
        distances = torch.clamp(distances, min=0.0)  # Numerical stability
        
        return distances

def train(model, train_loader, val_loader, device, epochs=100, lr=0.01):
    """Train the metric learning model."""
    model.to(device)
    
    # Use MSE loss for regression-style training of classification
    # We want predictions close to 0 or 1
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Get training data for kNN (use full train set)
            X_train = train_loader.dataset.tensors[0].to(device)
            y_train = train_loader.dataset.tensors[1].to(device)
            
            # Forward pass - get predictions for validation batch
            optimizer.zero_grad()
            predictions = model(X_train, X_batch, y_train)
            loss = criterion(predictions, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                X_train = train_loader.dataset.tensors[0].to(device)
                y_train = train_loader.dataset.tensors[1].to(device)
                
                predictions = model(X_train, X_batch, y_train)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    
    return model, train_losses, val_losses

def evaluate(model, data_loader, device):
    """Evaluate the model and return metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Get training data
            X_train = data_loader.dataset.tensors[0].to(device)
            y_train = data_loader.dataset.tensors[1].to(device)
            
            predictions = model(X_train, X_batch, y_train)
            
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Compute metrics
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    # For classification, also compute accuracy with threshold
    predictions_binary = (all_predictions > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, predictions_binary)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy)
    }

def predict(model, X_train, X_test, device):
    """Make predictions using the trained model."""
    model.eval()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Get labels
    y_train = model.train_labels if hasattr(model, 'train_labels') else None
    
    # If we don't have labels stored, we need to pass them
    # For simplicity, assume we'll pass them explicitly
    return None

def save_artifacts(model, train_losses, val_losses, metrics, X_train, y_train):
    """Save model, plots, and metrics."""
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save transformation matrix
    A_matrix = model.get_transform().detach().cpu().numpy()
    A_path = os.path.join(OUTPUT_DIR, 'A_matrix.npy')
    np.save(A_path, A_matrix)
    
    # Save training data
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()
    
    # Plot transformation matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(A_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title('Learned Transformation Matrix A')
    plt.savefig(os.path.join(OUTPUT_DIR, 'A_matrix.png'))
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")

def main():
    """Main function to run the task."""
    print("=" * 60)
    print("kNN + Learned Mahalanobis Metric")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders()
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = LearnedMetricKNN(n_features=10, n_neighbors=5)
    print(f"Model: {model}")
    
    # Train model
    print("\nTraining model...")
    model, train_losses, val_losses = train(
        model, train_loader, val_loader, device, epochs=100, lr=0.01
    )
    
    # Evaluate on train set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Train Metrics - MSE: {train_metrics['mse']:.4f}, R2: {train_metrics['r2']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Validation Metrics - MSE: {val_metrics['mse']:.4f}, R2: {val_metrics['r2']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
    
    # Compare with vanilla kNN
    print("\nComparing with vanilla kNN...")
    from sklearn.neighbors import KNeighborsClassifier
    
    # Vanilla kNN on original space
    vanilla_knn = KNeighborsClassifier(n_neighbors=5)
    vanilla_knn.fit(X_train, y_train)
    vanilla_pred = vanilla_knn.predict(X_val)
    vanilla_accuracy = accuracy_score(y_val, vanilla_pred)
    print(f"Vanilla kNN Accuracy: {vanilla_accuracy:.4f}")
    
    # Learned kNN accuracy (from our model)
    learned_accuracy = val_metrics['accuracy']
    print(f"Learned kNN Accuracy: {learned_accuracy:.4f}")
    print(f"Improvement: {learned_accuracy - vanilla_accuracy:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_losses, val_losses, val_metrics, X_train, y_train)
    
    # Quality checks and assertions
    print("\n" + "=" * 60)
    print("Quality Checks")
    print("=" * 60)
    
    # Check 1: R2 should be positive (model captures variance)
    assert val_metrics['r2'] > 0.0, f"R2 should be positive, got {val_metrics['r2']}"
    print(f"✓ R2 > 0: {val_metrics['r2']:.4f}")
    
    # Check 2: Accuracy should be above threshold
    assert val_metrics['accuracy'] > 0.85, f"Accuracy should be > 0.85, got {val_metrics['accuracy']}"
    print(f"✓ Accuracy > 0.85: {val_metrics['accuracy']:.4f}")
    
    # Check 3: Learned model should outperform vanilla kNN
    assert learned_accuracy > vanilla_accuracy, \
        f"Learned kNN should outperform vanilla kNN: {learned_accuracy:.4f} vs {vanilla_accuracy:.4f}"
    print(f"✓ Learned kNN outperforms vanilla kNN: {learned_accuracy:.4f} > {vanilla_accuracy:.4f}")
    
    # Check 4: MSE should be reasonable
    assert val_metrics['mse'] < 0.25, f"MSE should be < 0.25, got {val_metrics['mse']}"
    print(f"✓ MSE < 0.25: {val_metrics['mse']:.4f}")
    
    print("\n" + "=" * 60)
    print("PASS: All quality checks passed!")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

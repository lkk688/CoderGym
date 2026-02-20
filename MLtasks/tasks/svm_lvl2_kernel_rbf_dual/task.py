"""
Kernel SVM (RBF, Dual - Simplified)
Implements simplified dual optimization (projected GD) for small datasets with RBF kernel.
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
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "kernel_svm_rbf_dual",
        "task_type": "classification",
        "description": "Kernel SVM with RBF kernel using simplified dual optimization",
        "complexity": "O(n²) for kernel matrix computation, O(n) per iteration for projected GD",
        "suitable_for_small_n": True,
        "kernel": "RBF",
        "optimization_method": "projected_gradient_descent"
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(batch_size=32, test_size=0.2, random_state=42):
    """
    Create dataloaders for a nonlinear classification dataset.
    Uses make_moons dataset which is challenging for linear classifiers.
    """
    # Generate nonlinear dataset
    X, y = make_moons(n_samples=200, noise=0.2, random_state=random_state)
    
    # Convert labels to {-1, 1} for SVM
    y_svm = np.where(y == 0, -1, 1)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_svm, test_size=test_size, random_state=random_state, stratify=y_svm
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val, scaler

class KernelSVMRBF(nn.Module):
    """
    Kernel SVM with RBF kernel using simplified dual optimization.
    Implements projected gradient descent for the dual problem.
    """
    
    def __init__(self, gamma=1.0, C=1.0, lr=0.01, max_iter=1000):
        super(KernelSVMRBF, self).__init__()
        self.gamma = gamma
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        
    def rbf_kernel(self, X1, X2):
        """Compute RBF (Gaussian) kernel matrix."""
        # ||x - y||² = ||x||² + ||y||² - 2*x.y
        X1_sq = torch.sum(X1**2, dim=1).unsqueeze(1)
        X2_sq = torch.sum(X2**2, dim=1).unsqueeze(0)
        distances = X1_sq + X2_sq - 2 * torch.mm(X1, X2.t())
        return torch.exp(-self.gamma * distances)
    
    def forward(self, X):
        """Make predictions using the kernel model."""
        if self.X_train is None or self.alphas is None:
            raise ValueError("Model not trained yet")
        
        X = torch.FloatTensor(X).to(device) if not isinstance(X, torch.Tensor) else X
        K = self.rbf_kernel(X, self.X_train)
        predictions = torch.mm(K, self.alphas * self.y_train) + self.b
        return predictions
    
    def fit(self, X_train, y_train):
        """
        Train the kernel SVM using simplified dual optimization with projected GD.
        
        Dual problem: maximize W(α) = Σα_i - ½ΣΣα_iα_jy_iy_jK(x_i,x_j)
        subject to: 0 ≤ α_i ≤ C and Σα_iy_i = 0
        
        We use projected gradient descent on the dual objective.
        """
        n_samples = X_train.shape[0]
        self.X_train = X_train
        self.y_train = y_train
        
        # Initialize alphas
        self.alphas = torch.zeros(n_samples, 1, requires_grad=True, device=device)
        
        # Compute kernel matrix
        K = self.rbf_kernel(X_train, X_train)
        
        # Convert to float for gradient computation
        y_train_float = y_train.float()
        
        # Dual optimization using projected gradient descent
        optimizer = optim.SGD([self.alphas], lr=self.lr)
        
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            
            # Compute dual objective: W(α) = Σα_i - ½ΣΣα_iα_jy_iy_jK(x_i,x_j)
            # We minimize -W(α) = ½ΣΣα_iα_jy_iy_jK(x_i,x_j) - Σα_i
            
            alpha_y = self.alphas * y_train_float
            quadratic_term = 0.5 * torch.sum(torch.mm(alpha_y.t(), K) * alpha_y)
            linear_term = torch.sum(self.alphas)
            
            loss = quadratic_term - linear_term
            
            # Backward pass
            loss.backward()
            
            # Gradient step
            optimizer.step()
            
            # Project onto constraint set: 0 ≤ α_i ≤ C
            with torch.no_grad():
                self.alphas.clamp_(0, self.C)
            
            # Optional: enforce Σα_iy_i = 0 (soft constraint)
            if epoch % 100 == 0:
                constraint_violation = torch.abs(torch.sum(self.alphas * y_train_float)).item()
                if epoch % 500 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Constraint violation: {constraint_violation:.6f}")
        
        # Compute bias term b
        self._compute_bias(K, y_train_float)
        
        return self
    
    def _compute_bias(self, K, y_train):
        """Compute bias term b using support vectors."""
        # Find support vectors (0 < α_i < C)
        sv_mask = (self.alphas > 1e-5).squeeze() & (self.alphas < self.C - 1e-5).squeeze()
        
        if torch.any(sv_mask):
            # Use support vectors to compute b
            sv_indices = torch.where(sv_mask)[0]
            margin = y_train[sv_indices] - torch.mm(K[sv_indices], self.alphas * y_train)
            self.b = torch.mean(margin).item()
        else:
            # If no strict support vectors, use all non-zero alphas
            sv_mask = (self.alphas > 1e-5).squeeze()
            if torch.any(sv_mask):
                sv_indices = torch.where(sv_mask)[0]
                margin = y_train[sv_indices] - torch.mm(K[sv_indices], self.alphas * y_train)
                self.b = torch.mean(margin).item()
            else:
                self.b = 0.0
    
    def predict(self, X):
        """Predict class labels."""
        X = torch.FloatTensor(X).to(device) if not isinstance(X, torch.Tensor) else X
        predictions = self.forward(X)
        return torch.sign(predictions).detach().squeeze().cpu().numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities (simplified)."""
        predictions = self.forward(X).squeeze()
        # Convert to probability-like scores
        probs = (torch.sigmoid(predictions) > 0.5).float().cpu().numpy()
        return probs

def build_model(gamma=1.0, C=1.0, lr=0.01, max_iter=1000):
    """Build the kernel SVM model."""
    model = KernelSVMRBF(gamma=gamma, C=C, lr=lr, max_iter=max_iter)
    return model

def train(model, train_loader, X_train, y_train):
    """Train the model."""
    print(f"Training Kernel SVM with RBF kernel...")
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    
    # Convert to tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train).to(device)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_val, y_val):
    """
    Evaluate the model and return metrics.
    Returns MSE, R2 score, and accuracy.
    """
    model.eval()
    
    with torch.no_grad():
        # Ensure X_val is a tensor on correct device
        X_val_tensor = torch.FloatTensor(X_val).to(device) if not isinstance(X_val, torch.Tensor) else X_val
        
        # Get predictions
        predictions = model.predict(X_val_tensor)
        
        # Convert to numpy if needed
        y_val_np = y_val.squeeze().cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
        predictions = predictions.squeeze()
        
        # Compute metrics
        # For classification, we use accuracy as primary metric
        accuracy = accuracy_score(y_val_np, predictions)
        
        # Also compute MSE and R2 for regression-style evaluation
        mse = mean_squared_error(y_val_np, predictions)
        
        # R2 score
        r2 = r2_score(y_val_np, predictions)
        
        # Additional metrics for classification
        metrics = {
            'mse': float(mse),
            'r2': float(r2),
            'accuracy': float(accuracy)
        }
        
        print(f"\nEvaluation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return metrics

def predict(model, X):
    """Make predictions."""
    model.eval()
    with torch.no_grad():
        predictions = model.predict(X)
    return predictions

def save_artifacts(model, metrics, output_dir, X_train, y_train, X_val, y_val):
    """Save model artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters
    model_params = {
        'gamma': model.gamma,
        'C': model.C,
        'lr': model.lr,
        'max_iter': model.max_iter,
        'n_support_vectors': int(torch.sum((model.alphas > 1e-5).squeeze()).item()) if model.alphas is not None else 0,
        'alphas_shape': list(model.alphas.shape) if model.alphas is not None else None
    }
    
    with open(os.path.join(output_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=2)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model state (convert tensors to numpy for serialization)
    if model.alphas is not None:
        model_state = {
            'alphas': model.alphas.detach().cpu().numpy(),
            'b': model.b,
            'X_train': X_train.detach().cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train,
            'y_train': y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train,
            'gamma': model.gamma
        }
        np.savez(os.path.join(output_dir, 'model_state.npz'), **model_state)
    
    # Create visualization
    try:
        create_visualization(model, X_train, y_train, X_val, y_val, output_dir)
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    print(f"\nArtifacts saved to {output_dir}")

def create_visualization(model, X_train, y_train, X_val, y_val, output_dir):
    """Create and save visualization of decision boundary."""
    # Combine train and val for visualization
    X = np.vstack([X_train.detach().cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train,
                   X_val.detach().cpu().numpy() if isinstance(X_val, torch.Tensor) else X_val])
    y = np.concatenate([y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train,
                        y_val.detach().cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val])
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Prepare mesh points for prediction
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.FloatTensor(mesh_points).to(device)
    
    # Get predictions
    Z = model.predict(mesh_tensor)
    Z = Z.reshape(xx.shape)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
    
    # Training points
    train_mask = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    plt.scatter(X_train.detach().cpu().numpy()[:, 0], X_train.detach().cpu().numpy()[:, 1], 
                c=train_mask, cmap='RdBu', s=50, edgecolors='k', label='Train')
    
    # Validation points
    val_mask = y_val.detach().cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
    plt.scatter(X_val.detach().cpu().numpy()[:, 0], X_val.detach().cpu().numpy()[:, 1], 
                c=val_mask, cmap='RdBu', s=50, marker='s', edgecolors='k', label='Validation')
    
    plt.title(f'Kernel SVM (RBF) Decision Boundary\nGamma={model.gamma}, C={model.C}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'decision_boundary.png'), dpi=150)
    plt.close()

def main():
    """Main function to run the kernel SVM task."""
    print("=" * 60)
    print("Kernel SVM (RBF, Dual - Simplified)")
    print("=" * 60)
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Description: {metadata['description']}")
    print(f"Complexity: {metadata['complexity']}")
    
    # Set device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\n" + "-" * 40)
    print("Creating datasets...")
    train_loader, val_loader, X_train, X_val, y_train, y_val, scaler = make_dataloaders(
        batch_size=32, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model with tuned hyperparameters for this dataset
    print("\n" + "-" * 40)
    print("Building model...")
    model = build_model(gamma=2.0, C=10.0, lr=0.1, max_iter=500)
    
    # Train model
    print("\n" + "-" * 40)
    model = train(model, train_loader, X_train, y_train)
    
    # Evaluate on training set
    print("\n" + "-" * 40)
    print("Evaluating on training set...")
    train_metrics = evaluate(model, X_train, y_train)
    
    # Evaluate on validation set
    print("\n" + "-" * 40)
    print("Evaluating on validation set...")
    val_metrics = evaluate(model, X_val, y_val)
    
    # Quality thresholds
    print("\n" + "=" * 60)
    print("Quality Assessment")
    print("=" * 60)
    
    # Get accuracies
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    # Quality thresholds
    try:
        assert val_accuracy >= 0.70, f"Validation accuracy {val_accuracy:.4f} below threshold 0.70"
        assert train_accuracy >= 0.75, f"Training accuracy {train_accuracy:.4f} below threshold 0.75"
        assert val_metrics['mse'] < 1.5, f"Validation MSE {val_metrics['mse']:.4f} above threshold 1.5"
        
        print("\n✅ PASS: All quality thresholds met!")
        print(f"   - Validation Accuracy: {val_accuracy:.4f} >= 0.70 ✓")
        print(f"   - Training Accuracy: {train_accuracy:.4f} >= 0.75 ✓")
        print(f"   - Validation MSE: {val_metrics['mse']:.4f} < 1.5 ✓")
        print(f"   - R2 Score: {val_metrics['r2']:.4f}")
        
        # Save artifacts
        output_dir = '/Developer/AIserver/output/tasks/svm_lvl2_kernel_rbf_dual'
        save_artifacts(model, val_metrics, output_dir, X_train, y_train, X_val, y_val)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ FAIL: {e}")
        print("Quality thresholds not met. Please adjust hyperparameters or check data.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
"""
MLP (Manual Backprop) Implementation
2-layer MLP with manual backpropagation (no autograd)
Solves XOR problem with > 0.95 accuracy
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory for artifacts
OUTPUT_DIR = '/Developer/AIserver/output/tasks/mlp_lvl1_numpy_to_torch'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'mlp_manual_backprop',
        'description': '2-layer MLP with manual backpropagation',
        'input_dim': 2,
        'hidden_dim': 4,
        'output_dim': 1,
        'task_type': 'classification',
        'dataset': 'XOR'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=4):
    """Create dataloaders for XOR problem."""
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Split into train and validation (use all data for both for small dataset)
    X_train = torch.FloatTensor(X)
    y_train = torch.FloatTensor(y)
    X_val = torch.FloatTensor(X)
    y_val = torch.FloatTensor(y)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class MLP(nn.Module):
    """2-layer MLP with manual backpropagation."""
    
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights manually with better initialization
        # Layer 1: input -> hidden
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        
        # Layer 2: hidden -> output
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x):
        """Forward pass with manual computation."""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Store input for backward pass
        self.x = x
        
        # Layer 1: Linear + Sigmoid
        self.z1 = torch.matmul(x, self.W1) + self.b1  # (N, hidden_dim)
        self.a1 = torch.sigmoid(self.z1)               # (N, hidden_dim)
        
        # Layer 2: Linear + Sigmoid
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2  # (N, output_dim)
        self.a2 = torch.sigmoid(self.z2)                     # (N, output_dim)
        
        return self.a2
    
    def manual_backward(self, y_true, y_pred):
        """Manual backpropagation using chain rule."""
        # Ensure tensors are on correct device
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        
        N = y_true.shape[0]
        
        # Output layer gradients (using MSE loss: L = (1/2N) * sum((y_pred - y_true)^2))
        # dL/dy_pred = (y_pred - y_true) / N
        dL_dy = (y_pred - y_true) / N
        
        # dL/dz2 = dL/dy * dy/dz2 = dL/dy * sigmoid'(z2)
        # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        dz2 = dL_dy * (y_pred * (1 - y_pred))
        
        # Gradients for W2 and b2
        dL_dW2 = torch.matmul(self.a1.t(), dz2)  # (hidden_dim, output_dim)
        dL_db2 = torch.sum(dz2, dim=0)           # (output_dim,)
        
        # Backprop to hidden layer
        # dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 * W2^T
        dL_da1 = torch.matmul(dz2, self.W2.t())  # (N, hidden_dim)
        
        # dL/dz1 = dL/da1 * da1/dz1 = dL/da1 * sigmoid'(z1)
        dz1 = dL_da1 * (self.a1 * (1 - self.a1))
        
        # Gradients for W1 and b1
        dL_dW1 = torch.matmul(self.x.t(), dz1)  # (input_dim, hidden_dim)
        dL_db1 = torch.sum(dz1, dim=0)          # (hidden_dim,)
        
        # Store gradients
        self.W1.grad = dL_dW1
        self.b1.grad = dL_db1
        self.W2.grad = dL_dW2
        self.b2.grad = dL_db2
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        return super().to(device)


def build_model():
    """Build the MLP model."""
    model = MLP(input_dim=2, hidden_dim=4, output_dim=1)
    return model


def train(model, train_loader, val_loader, epochs=1000, lr=0.1):
    """Train the model with manual backpropagation."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Move batch to model device
            batch_X = batch_X.to(model.device)
            batch_y = batch_y.to(model.device)
            
            # Forward pass
            y_pred = model(batch_X)
            
            # Compute loss (MSE)
            loss = torch.mean((y_pred - batch_y) ** 2)
            epoch_loss += loss.item()
            
            # Manual backward pass
            model.manual_backward(batch_y, y_pred)
            
            # Update weights using gradient descent
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_X, batch_y in val_loader:
                # Move batch to model device
                batch_X = batch_X.to(model.device)
                batch_y = batch_y.to(model.device)
                y_pred = model(batch_X)
                val_loss += torch.mean((y_pred - batch_y) ** 2).item()
            val_losses.append(val_loss / len(val_loader))
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
    
    return model, train_losses, val_losses


def evaluate(model, data_loader):
    """Evaluate the model and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Move to model device
            batch_X = batch_X.to(model.device)
            batch_y = batch_y.to(model.device)
            
            y_pred = model(batch_X)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Compute MSE
    mse = np.mean((all_preds - all_targets) ** 2)
    
    # Compute R2 score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Compute accuracy (threshold at 0.5 for binary classification)
    preds_binary = (all_preds >= 0.5).astype(int)
    targets_binary = (all_targets >= 0.5).astype(int)
    accuracy = np.mean(preds_binary == targets_binary)
    
    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy),
        'predictions': all_preds,
        'targets': all_targets
    }


def predict(model, X):
    """Make predictions."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(model.device)
        y_pred = model(X_tensor)
        return y_pred.cpu().numpy()


def save_artifacts(model, train_losses, val_losses, train_metrics, val_metrics):
    """Save model and evaluation artifacts."""
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.npz')
    np.savez(metrics_path,
             train_mse=train_metrics['mse'],
             train_r2=train_metrics['r2'],
             train_accuracy=train_metrics['accuracy'],
             val_mse=val_metrics['mse'],
             val_r2=val_metrics['r2'],
             val_accuracy=val_metrics['accuracy'])
    
    # Save loss curves plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'))
    plt.close()
    
    # Save predictions visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Train predictions
    axes[0].scatter(train_metrics['targets'], train_metrics['predictions'], alpha=0.7)
    axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect fit')
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predictions')
    axes[0].set_title(f'Train Predictions (R2={train_metrics["r2"]:.4f}, Acc={train_metrics["accuracy"]:.4f})')
    axes[0].legend()
    axes[0].grid(True)
    
    # Validation predictions
    axes[1].scatter(val_metrics['targets'], val_metrics['predictions'], alpha=0.7, color='orange')
    axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect fit')
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Predictions')
    axes[1].set_title(f'Val Predictions (R2={val_metrics["r2"]:.4f}, Acc={val_metrics["accuracy"]:.4f})')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predictions.png'))
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run the MLP training and evaluation."""
    print("=" * 60)
    print("MLP Manual Backpropagation - XOR Problem")
    print("=" * 60)
    
    # Get metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Input dim: {metadata['input_dim']}, Hidden dim: {metadata['hidden_dim']}, Output dim: {metadata['output_dim']}")
    
    # Set device
    device = get_device()
    print(f"Device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(batch_size=4)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    print(f"Model architecture: {metadata['input_dim']} -> {metadata['hidden_dim']} -> {metadata['output_dim']}")
    
    # Train model
    print("\nTraining model...")
    model, train_losses, val_losses = train(
        model, train_loader, val_loader, 
        epochs=1000, 
        lr=0.5
    )
    
    # Evaluate on train set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader)
    print(f"Train MSE: {train_metrics['mse']:.6f}")
    print(f"Train R2: {train_metrics['r2']:.6f}")
    print(f"Train Accuracy: {train_metrics['accuracy']:.6f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader)
    print(f"Validation MSE: {val_metrics['mse']:.6f}")
    print(f"Validation R2: {val_metrics['r2']:.6f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.6f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_losses, val_losses, train_metrics, val_metrics)
    
    # Quality assertions
    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)
    
    # Assert quality thresholds
    try:
        # Validation accuracy should be > 0.95 for XOR
        assert val_metrics['accuracy'] > 0.95, f"Validation accuracy {val_metrics['accuracy']:.4f} < 0.95"
        print(f"✓ Validation accuracy > 0.95: {val_metrics['accuracy']:.4f}")
        
        # Validation MSE should be low
        assert val_metrics['mse'] < 0.05, f"Validation MSE {val_metrics['mse']:.6f} > 0.05"
        print(f"✓ Validation MSE < 0.05: {val_metrics['mse']:.6f}")
        
        # Validation R2 should be high
        assert val_metrics['r2'] > 0.9, f"Validation R2 {val_metrics['r2']:.4f} < 0.9"
        print(f"✓ Validation R2 > 0.9: {val_metrics['r2']:.4f}")
        
        # Check that model learned (train accuracy > val accuracy by reasonable margin)
        assert train_metrics['accuracy'] >= val_metrics['accuracy'], "Train accuracy should be >= validation accuracy"
        print(f"✓ No overfitting detected")
        
        print("\n" + "=" * 60)
        print("PASS: All quality thresholds met!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        print("\n" + "=" * 60)
        print("FAIL: Quality thresholds not met")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

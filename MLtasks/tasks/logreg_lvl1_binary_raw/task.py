"""
Binary Logistic Regression with Manual Gradients (Raw Tensors)
No autograd, manual sigmoid, manual gradient descent.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Task metadata
def get_task_metadata():
    return {
        'task_type': 'binary_classification',
        'input_dim': 2,
        'output_dim': 1,
        'description': 'Binary logistic regression with manual gradients'
    }

# Build synthetic dataset: Two Gaussian clusters
def make_dataloaders(batch_size=32, val_ratio=0.2):
    """
    Create two Gaussian clusters for binary classification.
    Class 0: centered at (-1, -1)
    Class 1: centered at (1, 1)
    """
    n_samples = 400
    n_class = n_samples // 2
    
    # Generate two Gaussian clusters with reduced variance for better separability
    cluster_0 = np.random.randn(n_class, 2) * 0.8 + np.array([-1, -1])
    cluster_1 = np.random.randn(n_class, 2) * 0.8 + np.array([1, 1])
    
    X = np.vstack([cluster_0, cluster_1])
    y = np.array([0] * n_class + [1] * n_class)
    
    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train and validation
    val_size = int(len(X) * val_ratio)
    X_val, X_train = X[:val_size], X[val_size:]
    y_val, y_train = y[:val_size], y[val_size:]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val

# Logistic regression model with manual parameters
class LogisticRegressionManual(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionManual, self).__init__()
        self.input_dim = input_dim
        # Initialize weights and bias
        self.weights = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def sigmoid(self, z):
        """Manual sigmoid function: σ(z) = 1 / (1 + e^(-z))"""
        return 1.0 / (1.0 + torch.exp(-z))
    
    def forward(self, x):
        """Forward pass with manual sigmoid"""
        z = torch.matmul(x, self.weights) + self.bias
        return self.sigmoid(z)
    
    def get_params(self):
        return self.weights.data.cpu().numpy(), self.bias.data.cpu().numpy()

# Manual gradient computation
def compute_gradients(model, X, y_true):
    """
    Compute gradients manually.
    For log loss: L = -[y*log(σ(z)) + (1-y)*log(1-σ(z))]
    Gradient w.r.t. weights: X^T * (σ(z) - y)
    Gradient w.r.t. bias: sum(σ(z) - y)
    """
    batch_size = X.size(0)
    
    # Forward pass
    z = torch.matmul(X, model.weights) + model.bias
    y_pred = model.sigmoid(z)
    
    # Compute error
    error = y_pred - y_true
    
    # Gradients
    grad_weights = torch.matmul(X.t(), error) / batch_size
    grad_bias = torch.sum(error) / batch_size
    
    return grad_weights, grad_bias

# Train function
def train(model, train_loader, val_loader, device, learning_rate=0.1, epochs=100):
    """Train the model with manual gradients"""
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # Move to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Compute gradients manually
            grad_weights, grad_bias = compute_gradients(model, X_batch, y_batch)
            
            # Manual gradient descent update
            with torch.no_grad():
                model.weights -= learning_rate * grad_weights
                model.bias -= learning_rate * grad_bias
        
        # Compute training loss
        train_loss = evaluate_loss(model, train_loader, device)
        if (epoch + 1) % 100 == 0:
            val_loss = evaluate_loss(model, val_loader, device)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model

def evaluate_loss(model, data_loader, device):
    """Compute average log loss"""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            # Log loss: -[y*log(σ(z)) + (1-y)*log(1-σ(z))]
            eps = 1e-7
            loss = -torch.mean(y_batch * torch.log(y_pred + eps) + 
                              (1 - y_batch) * torch.log(1 - y_pred + eps))
            total_loss += loss.item()
            count += 1
    
    return total_loss / count

# Evaluate function
def evaluate(model, data_loader, device):
    """
    Evaluate model and return metrics: MSE, R2, accuracy
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
    
    # Concatenate
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    
    # Convert to binary predictions (threshold at 0.5)
    y_pred_binary = (y_pred >= 0.5).float()
    
    # Compute MSE
    mse = torch.mean((y_pred - y_true) ** 2).item()
    
    # Compute R2 score
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-7)).item()
    
    # Compute accuracy
    accuracy = torch.mean((y_pred_binary == y_true).float()).item()
    
    # Compute precision, recall, F1
    tp = torch.sum((y_pred_binary == 1) & (y_true == 1)).float()
    fp = torch.sum((y_pred_binary == 1) & (y_true == 0)).float()
    fn = torch.sum((y_pred_binary == 0) & (y_true == 1)).float()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'accuracy': accuracy,
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }
    
    return metrics

# Predict function
def predict(model, X, device):
    """Make predictions for input X"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        return model(X_tensor).cpu().numpy()

# Save artifacts
def save_artifacts(model, metrics, output_dir):
    """Save model and metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state
    torch.save({
        'weights': model.weights.data.cpu(),
        'bias': model.bias.data.cpu(),
        'input_dim': model.input_dim
    }, os.path.join(output_dir, 'model.pt'))
    
    # Save metrics
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)
    
    # Save model parameters
    weights, bias = model.get_params()
    np.save(os.path.join(output_dir, 'weights.npy'), weights)
    np.save(os.path.join(output_dir, 'bias.npy'), bias)
    
    print(f"Artifacts saved to {output_dir}")

# Main execution
if __name__ == '__main__':
    # Configuration
    OUTPUT_DIR = '/Developer/AIserver/output/tasks/logreg_lvl1_binary_raw'
    LEARNING_RATE = 0.5
    EPOCHS = 1000
    BATCH_SIZE = 32
    
    print("=" * 60)
    print("Binary Logistic Regression with Manual Gradients")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_type']}")
    print(f"Input dimension: {metadata['input_dim']}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        batch_size=BATCH_SIZE, val_ratio=0.2
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding model...")
    model = LogisticRegressionManual(input_dim=metadata['input_dim'])
    print(f"Initial weights: {model.weights.data.cpu().numpy().flatten()}")
    print(f"Initial bias: {model.bias.data.cpu().numpy()}")
    
    # Train model
    print(f"\nTraining for {EPOCHS} epochs...")
    model = train(model, train_loader, val_loader, device, 
                  learning_rate=LEARNING_RATE, epochs=EPOCHS)
    
    # Evaluate on training set
    print("\n" + "-" * 60)
    print("Evaluating on TRAINING set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Training Metrics:")
    print(f"  MSE:  {train_metrics['mse']:.4f}")
    print(f"  R2:   {train_metrics['r2']:.4f}")
    print(f"  Acc:  {train_metrics['accuracy']:.4f}")
    print(f"  Prec: {train_metrics['precision']:.4f}")
    print(f"  Rec:  {train_metrics['recall']:.4f}")
    print(f"  F1:   {train_metrics['f1']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on VALIDATION set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Validation Metrics:")
    print(f"  MSE:  {val_metrics['mse']:.4f}")
    print(f"  R2:   {val_metrics['r2']:.4f}")
    print(f"  Acc:  {val_metrics['accuracy']:.4f}")
    print(f"  Prec: {val_metrics['precision']:.4f}")
    print(f"  Rec:  {val_metrics['recall']:.4f}")
    print(f"  F1:   {val_metrics['f1']:.4f}")
    
    # Final model parameters
    weights, bias = model.get_params()
    print(f"\nFinal weights: {weights.flatten()}")
    print(f"Final bias: {bias}")
    
    # Assert quality thresholds
    print("\n" + "=" * 60)
    print("Quality Assertions...")
    print("=" * 60)
    
    # Check validation accuracy > 0.90
    assert val_metrics['accuracy'] > 0.90, \
        f"Validation accuracy {val_metrics['accuracy']:.4f} < 0.90"
    print(f"✓ Validation accuracy > 0.90: {val_metrics['accuracy']:.4f}")
    
    # Check R2 > 0.80
    assert val_metrics['r2'] > 0.80, \
        f"Validation R2 {val_metrics['r2']:.4f} < 0.80"
    print(f"✓ Validation R2 > 0.80: {val_metrics['r2']:.4f}")
    
    # Check MSE < 0.1
    assert val_metrics['mse'] < 0.1, \
        f"Validation MSE {val_metrics['mse']:.4f} >= 0.1"
    print(f"✓ Validation MSE < 0.1: {val_metrics['mse']:.4f}")
    
    # Check accuracy on both splits (no overfitting)
    assert abs(train_metrics['accuracy'] - val_metrics['accuracy']) < 0.1, \
        f"Overfitting detected: train_acc={train_metrics['accuracy']:.4f}, val_acc={val_metrics['accuracy']:.4f}"
    print(f"✓ No significant overfitting: train_acc={train_metrics['accuracy']:.4f}, val_acc={val_metrics['accuracy']:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("PASS: All quality thresholds met!")
    print("=" * 60)
    print(f"Final validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Final validation R2: {val_metrics['r2']:.4f}")
    print(f"Final validation MSE: {val_metrics['mse']:.4f}")
    
    exit(0)

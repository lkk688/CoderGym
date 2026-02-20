#!/usr/bin/env python3
"""
Multivariate Linear Regression using torch.autograd with visualization.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata():
    """Return task metadata."""
    return {
        'name': 'linear_regression_autograd',
        'description': 'Multivariate Linear Regression using torch.autograd',
        'input_shape': [5],
        'output_shape': [1],
        'task_type': 'regression'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get computation device (CUDA if available, else CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


class LinearRegressionModel(nn.Module):
    """Simple linear regression model."""
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def make_dataloaders(n_samples=1000, n_features=5, train_ratio=0.8, batch_size=32, device=None):
    """
    Create synthetic multivariate linear regression data and dataloaders.
    
    Y = X @ w + b + noise
    """
    if device is None:
        device = torch.device('cpu')
    
    # True parameters
    true_w = torch.tensor([2.5, -1.3, 0.8, -0.5, 1.1], dtype=torch.float32)
    true_b = torch.tensor([3.0], dtype=torch.float32)
    
    # Generate synthetic data
    X = torch.randn(n_samples, n_features)
    noise = 0.5 * torch.randn(n_samples, 1)
    y = X @ true_w.unsqueeze(1) + true_b.unsqueeze(0) + noise
    
    # Split into train and validation
    n_train = int(n_samples * train_ratio)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    print(f"Training samples: {n_train}, Validation samples: {n_samples - n_train}")
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    val_dataset = TensorDataset(X_val.to(device), y_val.to(device))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train.cpu(), y_train.cpu(), X_val.cpu(), y_val.cpu()


def build_model(input_dim, output_dim, device):
    """Build and return the model."""
    model = LinearRegressionModel(input_dim, output_dim)
    model = model.to(device)
    print(f"Model architecture: {model}")
    return model


def train(model, train_loader, val_loader, device, epochs=100, lr=0.01, verbose=True):
    """
    Train the linear regression model using autograd.
    
    Loss function: J(θ) = (1/2m) * Σ(hθ(x^(i)) - y^(i))^2
    Gradient: ∇J(θ) = (1/m) * Σ(hθ(x^(i)) - y^(i)) * x^(i)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return history


def evaluate(model, data_loader, device, return_loss_only=False):
    """
    Evaluate the model on the given data loader.
    
    Returns metrics: loss, MSE, MAE, R2
    """
    model.eval()
    criterion = nn.MSELoss()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    avg_loss = total_loss / len(data_loader)
    
    metrics = {
        'loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
    
    if return_loss_only:
        return avg_loss
    
    return metrics


def predict(model, X, device):
    """Generate predictions for input X."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        return outputs.cpu().numpy()


def save_artifacts(model, history, save_dir='output'):
    """Save model artifacts and visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'linreg_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save history
    history_path = os.path.join(save_dir, 'linreg_history.npz')
    np.savez(history_path, 
             train_loss=history['train_loss'],
             val_loss=history['val_loss'])
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_dir, 'linreg_lvl2_loss.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Artifacts saved to {save_dir}")
    
    return save_dir


def main():
    """Main function to run the linear regression task."""
    print("=" * 60)
    print("Starting Multivariate Linear Regression Task...")
    print("=" * 60)
    
    # Set device
    device = get_device()
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, y_train, X_val, y_val = make_dataloaders(
        n_samples=1000, 
        n_features=5, 
        train_ratio=0.8, 
        batch_size=32,
        device=device
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(input_dim=5, output_dim=1, device=device)
    
    # Train model
    print("\nTraining model...")
    history = train(model, train_loader, val_loader, device, epochs=100, lr=0.01, verbose=True)
    
    # Evaluate on train set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print("Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print("Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, history, save_dir='output')
    
    # Print final results
    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    print(f"MSE (Train): {train_metrics['mse']:.4f}")
    print(f"MSE (Val):   {val_metrics['mse']:.4f}")
    print(f"MAE (Train): {train_metrics['mae']:.4f}")
    print(f"MAE (Val):   {val_metrics['mae']:.4f}")
    print(f"R² (Train):  {train_metrics['r2']:.4f}")
    print(f"R² (Val):    {val_metrics['r2']:.4f}")
    print("=" * 60)
    
    # Quality checks
    print("\nQuality Checks:")
    all_passed = True
    
    # Check R² > 0.9 on train
    if train_metrics['r2'] > 0.9:
        print(f"  ✓ Train R² > 0.9: {train_metrics['r2']:.4f}")
    else:
        print(f"  ✗ Train R² > 0.9: {train_metrics['r2']:.4f} (FAILED)")
        all_passed = False
    
    # Check R² > 0.8 on validation
    if val_metrics['r2'] > 0.8:
        print(f"  ✓ Val R² > 0.8: {val_metrics['r2']:.4f}")
    else:
        print(f"  ✗ Val R² > 0.8: {val_metrics['r2']:.4f} (FAILED)")
        all_passed = False
    
    # Check MSE < 1.0 on validation
    if val_metrics['mse'] < 1.0:
        print(f"  ✓ Val MSE < 1.0: {val_metrics['mse']:.4f}")
    else:
        print(f"  ✗ Val MSE < 1.0: {val_metrics['mse']:.4f} (FAILED)")
        all_passed = False
    
    # Check loss decreased
    if history['train_loss'][-1] < history['train_loss'][0]:
        print(f"  ✓ Loss decreased: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    else:
        print(f"  ✗ Loss did not decrease: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f} (FAILED)")
        all_passed = False
    
    # Check R² difference < 0.15 (no severe overfitting)
    r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
    if r2_diff < 0.15:
        print(f"  ✓ R² difference < 0.15: {r2_diff:.4f}")
    else:
        print(f"  ✗ R² difference < 0.15: {r2_diff:.4f} (FAILED)")
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("PASS: All quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())

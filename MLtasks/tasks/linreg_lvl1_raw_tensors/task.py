"""
Linear Regression using Raw PyTorch Tensors

Mathematical Formulation:
- Hypothesis: h_theta(x) = theta_0 + theta_1 * x
- Cost Function (MSE): J(theta) = (1/2m) * sum((h_theta(x_i) - y_i)^2)
- Gradient Descent: theta = theta - lr * grad(theta)

This implementation uses ONLY PyTorch tensors without torch.nn, torch.optim, or autograd.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure output directory exists
OUTPUT_DIR = '/Developer/AIserver/output/tasks/linreg_lvl1_raw_tensors'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'linear_regression_raw_tensors',
        'description': 'Univariate Linear Regression using raw PyTorch tensors',
        'input_dim': 1,
        'output_dim': 1,
        'model_type': 'linear_regression',
        'loss_type': 'mse',
        'optimization': 'gradient_descent'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=200, train_ratio=0.8, noise_std=0.5, batch_size=32):
    """
    Create synthetic dataset: y = 2x + 3 + noise
    
    Args:
        n_samples: Number of samples to generate
        train_ratio: Ratio of training data
        noise_std: Standard deviation of noise
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, X_train, X_val, y_train, y_val
    """
    # Generate synthetic data: y = 2x + 3 + noise
    X = np.random.uniform(-5, 5, n_samples)
    y = 2 * X + 3 + np.random.normal(0, noise_std, n_samples)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(1)  # Shape: (n_samples, 1)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)  # Shape: (n_samples, 1)
    
    # Split into train and validation
    n_train = int(n_samples * train_ratio)
    X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val


class LinearRegressionModel:
    """
    Linear Regression model implemented from scratch using PyTorch tensors.
    
    Hypothesis: h_theta(x) = theta_0 + theta_1 * x
    Parameters:
        - theta_0: bias term (intercept)
        - theta_1: weight (slope)
    """
    
    def __init__(self, device=None):
        self.device = device if device is not None else get_device()
        # Initialize parameters (theta_0 = bias, theta_1 = weight)
        self.theta_0 = torch.zeros(1, requires_grad=False).to(self.device)
        self.theta_1 = torch.zeros(1, requires_grad=False).to(self.device)
    
    def forward(self, X):
        """
        Forward pass: h_theta(x) = theta_0 + theta_1 * x
        
        Args:
            X: Input tensor of shape (N, 1)
        Returns:
            Predictions of shape (N, 1)
        """
        return self.theta_0 + self.theta_1 * X
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error (MSE) loss.
        
        MSE = (1/2m) * sum((y_pred - y_true)^2)
        
        Args:
            y_pred: Predictions of shape (N, 1)
            y_true: Ground truth of shape (N, 1)
        Returns:
            MSE loss (scalar)
        """
        m = y_true.shape[0]
        mse = torch.mean((y_pred - y_true) ** 2) / 2
        return mse
    
    def compute_gradients(self, y_pred, y_true, X):
        """
        Compute gradients manually.
        
        dJ/dtheta_0 = (1/m) * sum(y_pred - y_true)
        dJ/dtheta_1 = (1/m) * sum((y_pred - y_true) * x)
        
        Args:
            y_pred: Predictions of shape (N, 1)
            y_true: Ground truth of shape (N, 1)
            X: Input of shape (N, 1)
        Returns:
            grad_theta_0, grad_theta_1
        """
        m = y_true.shape[0]
        errors = y_pred - y_true
        
        grad_theta_0 = torch.mean(errors)
        grad_theta_1 = torch.mean(errors * X)
        
        return grad_theta_0, grad_theta_1
    
    def update_parameters(self, grad_theta_0, grad_theta_1, lr):
        """
        Update parameters using gradient descent.
        
        theta = theta - lr * grad(theta)
        
        Args:
            grad_theta_0: Gradient for bias
            grad_theta_1: Gradient for weight
            lr: Learning rate
        """
        with torch.no_grad():
            self.theta_0 -= lr * grad_theta_0
            self.theta_1 -= lr * grad_theta_1
    
    def fit(self, train_loader, val_loader=None, epochs=1000, lr=0.01, verbose=True):
        """
        Train the model using gradient descent.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Whether to print progress
        
        Returns:
            dict with loss_history and val_loss_history
        """
        loss_history = []
        val_loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                # Move to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss.item()
                n_batches += 1
                
                # Compute gradients
                grad_theta_0, grad_theta_1 = self.compute_gradients(y_pred, y_batch, X_batch)
                
                # Update parameters
                self.update_parameters(grad_theta_0, grad_theta_1, lr)
            
            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            
            # Compute validation loss
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, return_dict=False)
                val_loss_history.append(val_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        return {
            'loss_history': loss_history,
            'val_loss_history': val_loss_history
        }
    
    def evaluate(self, data_loader, return_dict=True):
        """
        Evaluate the model on a data loader.
        
        Computes MSE, R2 score, and parameter accuracy.
        
        Args:
            data_loader: Data loader to evaluate on
            return_dict: Whether to return as dict
        
        Returns:
            dict with metrics or float (MSE)
        """
        self.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.forward(X_batch)
                all_preds.append(y_pred)
                all_targets.append(y_batch)
        
        # Concatenate
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)
        
        # Compute MSE
        mse = torch.mean((y_pred - y_true) ** 2).item()
        
        # Compute R2 score
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Parameter accuracy (how close to true values: theta_0=3.0, theta_1=2.0)
        theta_0_error = abs(self.theta_0.item() - 3.0)
        theta_1_error = abs(self.theta_1.item() - 2.0)
        
        metrics = {
            'mse': mse,
            'r2': r2,
            'theta_0': self.theta_0.item(),
            'theta_1': self.theta_1.item(),
            'theta_0_error': theta_0_error,
            'theta_1_error': theta_1_error,
            'theta_0_true': 3.0,
            'theta_1_true': 2.0
        }
        
        if return_dict:
            return metrics
        return mse
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input tensor of shape (N, 1)
        
        Returns:
            Predictions tensor of shape (N, 1)
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            return self.forward(X)
    
    def eval(self):
        """Set model to evaluation mode."""
        pass  # No-op for this simple model
    
    def state_dict(self):
        """Return model state for saving."""
        return {
            'theta_0': self.theta_0,
            'theta_1': self.theta_1
        }
    
    def load_state_dict(self, state_dict):
        """Load model state."""
        self.theta_0 = state_dict['theta_0']
        self.theta_1 = state_dict['theta_1']


def build_model(device=None):
    return LinearRegressionModel(device=device)


def train(model, train_loader, val_loader, epochs=10):
    """Train the model."""
    return model.fit(train_loader, val_loader, epochs=epochs)
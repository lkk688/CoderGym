"""
Linear Regression Task: PyTorch vs Sklearn Comparison
Industrial Comparison with EDA, Preprocessing, and Standardized Output
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory
OUTPUT_DIR = '/Developer/AIserver/output/tasks/linreg_lvl4_sklearn_production'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_task_metadata():
    """Return task metadata as dictionary."""
    return {
        "task_name": "linear_regression",
        "task_type": "regression",
        "dataset": "california_housing",
        "frameworks": ["pytorch", "sklearn"],
        "metrics": ["mse", "rmse", "r2", "mae"],
        "output_dir": OUTPUT_DIR
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(test_size=0.2, batch_size=64):
    """
    Load and preprocess California housing data.
    Returns train/val dataloaders and scalers.
    """
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Split data
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Manual StandardScaler for PyTorch
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_full = X_scaler.fit_transform(X_train_full)
    X_val = X_scaler.transform(X_val)
    
    # Reshape y for scaling
    y_train_full = y_train_full.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    y_train_full = y_scaler.fit_transform(y_train_full).flatten()
    y_val = y_scaler.transform(y_val).flatten()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_full)
    y_train_tensor = torch.FloatTensor(y_train_full).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Store original data for sklearn comparison
    sklearn_data = {
        'X_train': X_train_full,
        'X_val': X_val,
        'y_train': y_train_full,
        'y_val': y_val,
        'feature_names': data.feature_names,
        'target_name': data.target_names if hasattr(data, 'target_names') else ['MedHouseVal']
    }
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'X_train': X_train_full,
        'X_val': X_val,
        'y_train': y_train_full,
        'y_val': y_val,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'sklearn_data': sklearn_data,
        'n_features': X_train_full.shape[1]
    }


class PyTorchLinearRegression(nn.Module):
    """PyTorch Linear Regression model with sklearn-style API."""
    
    def __init__(self, n_features):
        super(PyTorchLinearRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, train_loader, val_loader=None, epochs=100, lr=0.01, verbose=False):
        """Fit the model using training data."""
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = self(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                    val_losses.append(val_loss / len(val_loader))
            
            if verbose and (epoch + 1) % 10 == 0:
                val_loss_str = f"{val_losses[-1]:.4f}" if val_loader else "N/A"
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, "
                      f"Val Loss: {val_loss_str}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses if val_loader else None
        }
    
    def predict(self, X):
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(device)
            else:
                X = X.to(device)
            outputs = self(X)
            return outputs.cpu().numpy().flatten()
    
    def save(self, filepath):
        """Save model state dict."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model state dict."""
        self.load_state_dict(torch.load(filepath))


def build_model(n_features):
    """Build and return PyTorch linear regression model."""
    model = PyTorchLinearRegression(n_features)
    return model


def train(model, dataloaders, epochs=100, lr=0.01, verbose=True):
    """Train the model and return training history."""
    history = model.fit(
        dataloaders['train_loader'],
        val_loader=dataloaders['val_loader'],
        epochs=epochs,
        lr=lr,
        verbose=verbose
    )
    return history


def evaluate(model, dataloaders, y_scaler):
    """
    Evaluate model on validation set and return metrics.
    Returns dict with MSE, RMSE, R2, MAE.
    """
    model.eval()
    with torch.no_grad():
        # Get predictions for validation set
        X_val = dataloaders['X_val']
        y_val = dataloaders['y_val']
        
        if isinstance(X_val, torch.Tensor):
            X_val = X_val.cpu().numpy()
        if isinstance(y_val, torch.Tensor):
            y_val = y_val.cpu().numpy()
        
        y_pred = model.predict(X_val)
        
        # Inverse transform to get actual values
        y_val_actual = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_actual = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_val_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_actual, y_pred_actual)
        mae = mean_absolute_error(y_val_actual, y_pred_actual)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mae': float(mae)
        }


def predict(model, X):
    """Make predictions on new data."""
    return model.predict(X)


def save_artifacts(model, dataloaders, metrics, sklearn_metrics, history, metadata):
    """Save all artifacts to output directory."""
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'pytorch_model.pth')
    model.save(model_path)
    
    # Save sklearn model
    sklearn_model_path = os.path.join(OUTPUT_DIR, 'sklearn_model.npz')
    np.savez(sklearn_model_path, 
             coef=dataloaders.get('sklearn_model_coef'),
             intercept=dataloaders.get('sklearn_model_intercept'))
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    all_metrics = {
        'metadata': metadata,
        'pytorch_metrics': metrics,
        'sklearn_metrics': sklearn_metrics,
        'comparison': {
            'r2_diff': abs(metrics['r2'] - sklearn_metrics['r2']),
            'rmse_diff': abs(metrics['rmse'] - sklearn_metrics['rmse'])
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save EDA plots
    # Correlation matrix
    X_train = dataloaders['X_train']
    X_train_inv = dataloaders['X_scaler'].inverse_transform(X_train)
    y_train = dataloaders['y_train']
    y_train_inv = dataloaders['y_scaler'].inverse_transform(y_train.reshape(-1, 1)).flatten()
    
    # Create correlation matrix plot
    feature_names = dataloaders['sklearn_data']['feature_names']
    all_data = np.column_stack([X_train_inv, y_train_inv])
    all_features = feature_names + ['MedHouseVal']
    
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(all_data.T)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                xticklabels=all_features, yticklabels=all_features,
                cmap='coolwarm', center=0)
    plt.title('Correlation Matrix - California Housing')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=150)
    plt.close()
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_train_inv, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('MedHouseVal')
    plt.ylabel('Frequency')
    plt.title('Target Distribution - California Housing')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'), dpi=150)
    plt.close()
    
    # Training loss curve
    if history and 'train_losses' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_losses'], label='Train Loss')
        if history.get('val_losses'):
            plt.plot(history['val_losses'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_plot.png'), dpi=150)
        plt.close()


def main():
    """Main function to run the complete ML pipeline."""
    print("=" * 60)
    print("Linear Regression: PyTorch vs Sklearn Comparison")
    print("=" * 60)
    
    # Get metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Device: {get_device()}")
    
    # Create dataloaders
    print("\n[1/5] Loading and preprocessing data...")
    dataloaders = make_dataloaders(test_size=0.2, batch_size=64)
    print(f"Training samples: {len(dataloaders['X_train'])}")
    print(f"Validation samples: {len(dataloaders['X_val'])}")
    print(f"Features: {dataloaders['n_features']}")
    
    # Build and train PyTorch model
    print("\n[2/5] Training PyTorch model...")
    pytorch_model = build_model(dataloaders['n_features'])
    print(f"Model architecture: {pytorch_model}")
    
    history = train(pytorch_model, dataloaders, epochs=100, lr=0.01, verbose=True)
    
    # Train sklearn model for comparison
    print("\n[3/5] Training Sklearn model...")
    sklearn_model = LinearRegression()
    sklearn_model.fit(dataloaders['X_train'], dataloaders['y_train'])
    
    # Evaluate on validation set
    print("\n[4/5] Evaluating models...")
    pytorch_metrics = evaluate(pytorch_model, dataloaders, dataloaders['y_scaler'])
    
    # Sklearn evaluation
    y_val_pred_sklearn = sklearn_model.predict(dataloaders['X_val'])
    y_val_actual = dataloaders['y_scaler'].inverse_transform(
        dataloaders['y_val'].reshape(-1, 1)
    ).flatten()
    y_val_pred_sklearn_inv = dataloaders['y_scaler'].inverse_transform(
        y_val_pred_sklearn.reshape(-1, 1)
    ).flatten()
    
    sklearn_metrics = {
        'mse': float(mean_squared_error(y_val_actual, y_val_pred_sklearn_inv)),
        'rmse': float(np.sqrt(mean_squared_error(y_val_actual, y_val_pred_sklearn_inv))),
        'r2': float(r2_score(y_val_actual, y_val_pred_sklearn_inv)),
        'mae': float(mean_absolute_error(y_val_actual, y_val_pred_sklearn_inv))
    }
    
    # Evaluate on train set for both models
    print("\nEvaluating on TRAIN set...")
    y_train_pred_pytorch = pytorch_model.predict(dataloaders['X_train'])
    y_train_actual = dataloaders['y_scaler'].inverse_transform(
        dataloaders['y_train'].reshape(-1, 1)
    ).flatten()
    y_train_pred_pytorch_inv = dataloaders['y_scaler'].inverse_transform(
        y_train_pred_pytorch.reshape(-1, 1)
    ).flatten()
    
    pytorch_train_metrics = {
        'mse': float(mean_squared_error(y_train_actual, y_train_pred_pytorch_inv)),
        'rmse': float(np.sqrt(mean_squared_error(y_train_actual, y_train_pred_pytorch_inv))),
        'r2': float(r2_score(y_train_actual, y_train_pred_pytorch_inv)),
        'mae': float(mean_absolute_error(y_train_actual, y_train_pred_pytorch_inv))
    }
    
    y_train_pred_sklearn = sklearn_model.predict(dataloaders['X_train'])
    y_train_pred_sklearn_inv = dataloaders['y_scaler'].inverse_transform(
        y_train_pred_sklearn.reshape(-1, 1)
    ).flatten()
    
    sklearn_train_metrics = {
        'mse': float(mean_squared_error(y_train_actual, y_train_pred_sklearn_inv)),
        'rmse': float(np.sqrt(mean_squared_error(y_train_actual, y_train_pred_sklearn_inv))),
        'r2': float(r2_score(y_train_actual, y_train_pred_sklearn_inv)),
        'mae': float(mean_absolute_error(y_train_actual, y_train_pred_sklearn_inv))
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS - VALIDATION SET")
    print("=" * 60)
    print(f"\nPyTorch Model:")
    print(f"  MSE:  {pytorch_metrics['mse']:.6f}")
    print(f"  RMSE: {pytorch_metrics['rmse']:.6f}")
    print(f"  R2:   {pytorch_metrics['r2']:.6f}")
    print(f"  MAE:  {pytorch_metrics['mae']:.6f}")
    
    print(f"\nSklearn Model:")
    print(f"  MSE:  {sklearn_metrics['mse']:.6f}")
    print(f"  RMSE: {sklearn_metrics['rmse']:.6f}")
    print(f"  R2:   {sklearn_metrics['r2']:.6f}")
    print(f"  MAE:  {sklearn_metrics['mae']:.6f}")
    
    print("\n" + "=" * 60)
    print("RESULTS - TRAIN SET")
    print("=" * 60)
    print(f"\nPyTorch Model (Train):")
    print(f"  MSE:  {pytorch_train_metrics['mse']:.6f}")
    print(f"  RMSE: {pytorch_train_metrics['rmse']:.6f}")
    print(f"  R2:   {pytorch_train_metrics['r2']:.6f}")
    print(f"  MAE:  {pytorch_train_metrics['mae']:.6f}")
    
    print(f"\nSklearn Model (Train):")
    print(f"  MSE:  {sklearn_train_metrics['mse']:.6f}")
    print(f"  RMSE: {sklearn_train_metrics['rmse']:.6f}")
    print(f"  R2:   {sklearn_train_metrics['r2']:.6f}")
    print(f"  MAE:  {sklearn_train_metrics['mae']:.6f}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)
    r2_diff = abs(pytorch_metrics['r2'] - sklearn_metrics['r2'])
    rmse_diff = abs(pytorch_metrics['rmse'] - sklearn_metrics['rmse'])
    
    print(f"\nR2 Difference: {r2_diff:.6f}")
    print(f"RMSE Difference: {rmse_diff:.6f}")
    
    if r2_diff < 0.01:
        print("\n✓ Both models perform similarly (R2 < 0.01 difference)")
    else:
        print("\n✗ Significant performance difference between models")
    
    # Save artifacts
    print("\n[5/5] Saving artifacts...")
    save_artifacts(pytorch_model, dataloaders, pytorch_metrics, sklearn_metrics, history, metadata)
    
    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("Task completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

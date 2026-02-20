"""
MLP Hyperparameter Search Task
Grid/random search over depth/width/lr/weight_decay; select best by val metric.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
OUTPUT_DIR = '/Developer/AIserver/output/tasks/mlp_lvl4_hparam_sweep'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the ML task."""
    return {
        'task_type': 'regression',
        'input_dim': 1,
        'output_dim': 1,
        'description': 'MLP hyperparameter search for regression task',
        'metrics': ['mse', 'r2', 'mae'],
        'hyperparameters': ['depth', 'width', 'learning_rate', 'weight_decay']
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the appropriate device (GPU or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    n_samples: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    noise_std: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for the regression task.
    Uses a sine wave with noise as the target function.
    """
    # Generate synthetic data
    X = np.linspace(-3, 3, n_samples)
    y = np.sin(X) + np.random.randn(n_samples) * noise_std
    
    # Split data
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


class MLP(nn.Module):
    """Multi-layer Perceptron for regression."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super(MLP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers).to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def to(self, device):
        self.device = device
        self.network = self.network.to(device)
        return self


def build_model(
    input_dim: int = 1,
    output_dim: int = 1,
    depth: int = 3,
    width: int = 64,
    dropout: float = 0.1
) -> MLP:
    """Build an MLP with specified architecture."""
    hidden_dims = [width] * depth
    model = MLP(input_dim, output_dim, hidden_dims, dropout)
    return model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    epochs: int = 100,
    patience: int = 10,
    device: torch.device = None
) -> Dict[str, List[float]]:
    """
    Train the model with early stopping.
    Returns training history.
    """
    if device is None:
        device = get_device()
    
    model.to(device)
    model.train()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_r2 = 0.0
        
        with torch.no_grad():
            all_preds = []
            all_targets = []
            
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Calculate R2 score
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        val_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R2: {val_r2:.4f}")
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    return history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Evaluate the model on the given data loader.
    Returns metrics: MSE, R2, MAE.
    """
    if device is None:
        device = get_device()
    
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_mae = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            # MSE
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # MAE
            mae = torch.mean(torch.abs(outputs - y_batch)).item()
            total_mae += mae
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    n_samples = len(data_loader)
    
    # Calculate R2 score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mse': total_loss / n_samples,
        'r2': r2,
        'mae': total_mae / n_samples
    }


def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device = None
) -> np.ndarray:
    """Make predictions on input data."""
    if device is None:
        device = get_device()
    
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
    
    return outputs.cpu().numpy()


def save_artifacts(
    model: nn.Module,
    history: Dict[str, List[float]],
    best_config: Dict[str, Any],
    metrics: Dict[str, float],
    output_dir: str = OUTPUT_DIR
) -> None:
    """Save model, history, and metrics to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save history
    history_path = os.path.join(output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save best config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Save training plot
    plot_path = os.path.join(output_dir, 'training_plot.png')
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Val R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.title('Validation R2 Score')
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Artifacts saved to {output_dir}")


def hyperparameter_search(
    search_type: str = 'grid',
    n_iterations: int = 10,
    device: torch.device = None
) -> Tuple[Dict[str, Any], Dict[str, Any], nn.Module]:
    """
    Perform hyperparameter search over depth, width, learning_rate, weight_decay.
    Returns best config, metrics, and model.
    """
    if device is None:
        device = get_device()
    
    # Define search space
    depth_options = [2, 3, 4]
    width_options = [32, 64, 128]
    lr_options = [0.001, 0.01]
    weight_decay_options = [0.0, 0.0001, 0.001]
    
    # Create dataloaders
    train_loader, val_loader, _ = make_dataloaders(n_samples=800, batch_size=32)
    
    best_config = None
    best_val_r2 = float('-inf')
    best_model = None
    best_history = None
    sweep_results = []
    
    if search_type == 'grid':
        # Grid search over all combinations
        configs = []
        for depth in depth_options:
            for width in width_options:
                for lr in lr_options:
                    for wd in weight_decay_options:
                        configs.append({
                            'depth': depth,
                            'width': width,
                            'learning_rate': lr,
                            'weight_decay': wd
                        })
    else:
        # Random search
        np.random.seed(42)
        configs = []
        for _ in range(n_iterations):
            configs.append({
                'depth': np.random.choice(depth_options),
                'width': np.random.choice(width_options),
                'learning_rate': np.random.choice(lr_options),
                'weight_decay': np.random.choice(weight_decay_options)
            })
    
    print(f"Searching over {len(configs)} configurations...")
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Config: {config}")
        
        # Build model
        model = build_model(
            input_dim=1,
            output_dim=1,
            depth=config['depth'],
            width=config['width']
        )
        
        # Train
        history = train(
            model,
            train_loader,
            val_loader,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            epochs=50,
            device=device
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        
        # Record results
        result = {
            'config': config,
            'val_metrics': val_metrics
        }
        sweep_results.append(result)
        
        print(f"  Val MSE: {val_metrics['mse']:.6f}, Val R2: {val_metrics['r2']:.4f}")
        
        # Update best
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_config = config
            best_model = model
            best_history = history
    
    # Prepare leaderboard
    leaderboard = []
    for result in sweep_results:
        config = result['config']
        metrics = result['val_metrics']
        leaderboard.append({
            'depth': config['depth'],
            'width': config['width'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'val_mse': metrics['mse'],
            'val_r2': metrics['r2'],
            'val_mae': metrics['mae']
        })
    
    # Sort by R2 descending
    leaderboard.sort(key=lambda x: x['val_r2'], reverse=True)
    
    metrics = {
        'sweep': leaderboard,
        'best_config': best_config,
        'best_val_r2': best_val_r2
    }
    
    return best_config, metrics, best_model


def main():
    """Main function to run the MLP hyperparameter search task."""
    print("=" * 60)
    print("MLP Hyperparameter Search Task")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Perform hyperparameter search
    print("\n" + "-" * 60)
    print("Starting Hyperparameter Search...")
    print("-" * 60)
    
    best_config, metrics, best_model = hyperparameter_search(
        search_type='grid',
        device=device
    )
    
    print("\n" + "-" * 60)
    print("Hyperparameter Search Complete!")
    print("-" * 60)
    print(f"\nBest Configuration: {best_config}")
    print(f"Best Validation R2: {metrics['best_val_r2']:.4f}")
    
    # Create dataloaders for final evaluation
    train_loader, val_loader, test_loader = make_dataloaders(n_samples=1000, batch_size=32)
    
    # Evaluate on train set
    print("\n" + "-" * 60)
    print("Evaluating on Training Set...")
    print("-" * 60)
    train_metrics = evaluate(best_model, train_loader, device)
    print(f"Train MSE: {train_metrics['mse']:.6f}")
    print(f"Train R2: {train_metrics['r2']:.4f}")
    print(f"Train MAE: {train_metrics['mae']:.6f}")
    
    # Evaluate on validation set
    print("\n" + "-" * 60)
    print("Evaluating on Validation Set...")
    print("-" * 60)
    val_metrics = evaluate(best_model, val_loader, device)
    print(f"Val MSE: {val_metrics['mse']:.6f}")
   ```python
print(f"Val R2: {val_metrics['r2']:.4f}")
    print(f"Val MAE: {val_metrics['mae']:.6f}")
    
    # Evaluate on test set
    print("\n" + "-" * 60)
    print("Evaluating on Test Set...")
    print("-" * 60)
    test_metrics = evaluate(best_model, test_loader, device)
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    
    # Save artifacts
    print("\n" + "-" * 60)
    print("Saving Artifacts...")
    print("-" * 60)
    save_artifacts(best_model, best_history, best_config, metrics)
    
    print("\n" + "=" * 60)
    print("Task Complete!")
    print("=" * 60)
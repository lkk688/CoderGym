import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple, Optional

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the task."""
    return {
        "task_name": "mlp_regression",
        "description": "MLP with schedulers, AMP, gradient clipping, and checkpointing",
        "input_type": "float",
        "output_type": "float",
        "metrics": ["mse", "r2"],
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Get the appropriate device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(
    n_samples: int = 1000,
    n_features: int = 10,
    train_ratio: float = 0.8,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, int]:
    """Create synthetic regression dataset and dataloaders."""
    # Generate synthetic data with non-linear relationships
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create non-linear target with some noise
    y = (
        2.0 * X[:, 0] + 
        1.5 * X[:, 1] ** 2 + 
        0.8 * np.sin(2 * X[:, 2]) +
        0.5 * X[:, 3] * X[:, 4] +
        np.random.randn(n_samples) * 0.1
    ).astype(np.float32)
    
    # Split data
    n_train = int(n_samples * train_ratio)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, n_features

class MLP(nn.Module):
    """Multi-Layer Perceptron for regression."""
    
    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2):
        super(MLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def build_model(input_dim: int, device: torch.device) -> MLP:
    """Build and initialize the model."""
    model = MLP(input_dim=input_dim, hidden_dims=[64, 32, 16], dropout=0.2)
    model = model.to(device)
    return model

def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    save_dir: str = "output"
) -> Dict[str, Any]:
    """Train the model with schedulers, AMP, and gradient clipping."""
    
    # Initialize scaler for AMP (using new API)
    scaler = GradScaler('cuda') if use_amp else None
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate schedulers
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_state_dict = None
    checkpoint_path = os.path.join(save_dir, "best_checkpoint.pth")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            if use_amp and scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in train_loader:  # Use train loader for quick validation
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(train_loader)
        
        # Update schedulers
        scheduler.step()
        scheduler_plateau.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, checkpoint_path)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    # Save training curves
    curves_path = os.path.join(save_dir, "training_curves.json")
    with open(curves_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'checkpoint_path': checkpoint_path
    }

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    
    model.eval()
    all_preds = []
    all_targets = []
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y.to(device))
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    return {
        'loss': total_loss / len(data_loader),
        'mse': mse,
        'r2': r2
    }

def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """Make predictions on new data."""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
    
    return outputs.cpu().numpy()

def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    save_dir: str = "output"
) -> None:
    """Save model artifacts and metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Artifacts saved to {save_dir}")

def main():
    """Main function to run the complete training pipeline."""
    print("=" * 60)
    print("MLP Training with Schedulers, AMP, Gradient Clipping")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, n_features = make_dataloaders(
        n_samples=1000,
        n_features=10,
        train_ratio=0.8,
        batch_size=32
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(input_dim=n_features, device=device)
    print(f"Model architecture:\n{model}")
    
    # Train model
    print("\nTraining model...")
    save_dir = "output"
    train_result = train(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=100,
        lr=0.001,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=torch.cuda.is_available(),
        save_dir=save_dir
    )
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'best_val_loss': train_result['best_val_loss']
    }
    save_artifacts(model, all_metrics, save_dir)
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nTrain MSE: {train_metrics['mse']:.4f}")
    print(f"Val MSE:   {val_metrics['mse']:.4f}")
    print(f"Train R²:  {train_metrics['r2']:.4f}")
    print(f"Val R²:    {val_metrics['r2']:.4f}")
    
    # Quality checks
    print("\nQuality Checks:")
    checks_passed = True
    
    # Check 1: Train R² > 0.8
    check1 = train_metrics['r2'] > 0.8
    print(f"  {'✓' if check1 else '✗'} Train R² > 0.8: {train_metrics['r2']:.4f}")
    checks_passed = checks_passed and check1
    
    # Check 2: Val R² > 0.7
    check2 = val_metrics['r2'] > 0.7
    print(f"  {'✓' if check2 else '✗'} Val R² > 0.7: {val_metrics['r2']:.4f}")
    checks_passed = checks_passed and check2
    
    # Check 3: Val MSE < 1.0
    check3 = val_metrics['mse'] < 1.0
    print(f"  {'✓' if check3 else '✗'} Val MSE < 1.0: {val_metrics['mse']:.4f}")
    checks_passed = checks_passed and check3
    
    # Check 4: R² difference < 0.15 (no severe overfitting)
    r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
    check4 = r2_diff < 0.15
    print(f"  {'✓' if check4 else '✗'} R² difference < 0.15: {r2_diff:.4f}")
    checks_passed = checks_passed and check4
    
    # Check 5: Final loss decreased
    history = train_result['history']
    loss_decreased = history['train_loss'][-1] < history['train_loss'][0]
    check5 = loss_decreased
    print(f"  {'✓' if check5 else '✗'} Loss decreased: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    checks_passed = checks_passed and check5
    
    # Final summary
    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    sys.exit(main())

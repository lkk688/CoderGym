import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Set seeds for reproducibility
def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "polynomial_regression_ridge_sgd",
        "task_type": "regression",
        "input_type": "continuous",
        "output_type": "continuous",
        "description": "Polynomial regression with L2 regularization using torch.nn + torch.optim"
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    """Get device (cuda/cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(n_train=800, n_val=200, batch_size=32, noise_std=0.5, degree=2, device=None):
    """Create training and validation dataloaders with polynomial data."""
    if device is None:
        device = get_device()
    
    # Generate x values
    x_train = np.linspace(-3, 3, n_train)
    x_val = np.linspace(-3, 3, n_val)
    
    # Generate polynomial features manually: y = x^2 + noise
    # For degree 2: [1, x, x^2]
    X_train = np.column_stack([x_train**i for i in range(degree + 1)])
    X_val = np.column_stack([x_val**i for i in range(degree + 1)])
    
    # True coefficients for y = 0.5 + 0.3*x + 0.7*x^2 + noise
    true_coeffs = np.array([0.5, 0.3, 0.7])
    y_train = X_train @ true_coeffs + np.random.randn(n_train) * noise_std
    y_val = X_val @ true_coeffs + np.random.randn(n_val) * noise_std
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val

class PolynomialRegressionModel(nn.Module):
    """Polynomial regression model with linear layer."""
    def __init__(self, input_dim):
        super(PolynomialRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    @property
    def device(self):
        return next(self.parameters()).device

def build_model(input_dim, device=None):
    """Build the polynomial regression model."""
    if device is None:
        device = get_device()
    
    model = PolynomialRegressionModel(input_dim).to(device)
    return model

def train(model, train_loader, val_loader, device=None, epochs=100, lr=0.01, momentum=0.9, weight_decay=0.01):
    """Train the polynomial regression model with SGD and momentum."""
    if device is None:
        device = get_device()
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Move to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate(model, data_loader, device=None):
    """Evaluate the model and return metrics."""
    if device is None:
        device = get_device()
    
    model.eval()
    criterion = nn.MSELoss()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'mse': mse,
        'r2': r2
    }
    
    return metrics

def predict(model, X, device=None):
    """Make predictions."""
    if device is None:
        device = get_device()
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy()

def save_artifacts(model, train_losses, val_losses, X_train, y_train, X_val, y_val, 
                   output_dir="output", filename_prefix="linreg_lvl3"):
    """Save model artifacts and visualization."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, f"{filename_prefix}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f"{filename_prefix}_history.json")
    history = {
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses]
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"{filename_prefix}_fit.png")
    plot_fit(model, X_train, y_train, X_val, y_val, train_losses, val_losses, viz_path)
    
    print(f"Artifacts saved to {output_dir}")

def plot_fit(model, X_train, y_train, X_val, y_val, train_losses, val_losses, save_path):
    """Create and save visualization of the fit."""
    # Sort by x for smooth curves
    train_sorted = np.argsort(X_train[:, 1])  # Sort by x (second column)
    val_sorted = np.argsort(X_val[:, 1])
    
    x_train_sorted = X_train[train_sorted, 1]
    y_train_sorted = y_train[train_sorted]
    x_val_sorted = X_val[val_sorted, 1]
    y_val_sorted = y_val[val_sorted]
    
    # Get predictions
    y_train_pred = predict(model, X_train)
    y_val_pred = predict(model, X_val)
    
    y_train_pred_sorted = y_train_pred[train_sorted]
    y_val_pred_sorted = y_val_pred[val_sorted]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and fit
    ax1.scatter(x_train_sorted, y_train_sorted, alpha=0.5, label='Train data', s=15)
    ax1.scatter(x_val_sorted, y_val_sorted, alpha=0.5, label='Val data', s=15)
    ax1.plot(x_train_sorted, y_train_pred_sorted, 'r-', linewidth=2, label='Train fit')
    ax1.plot(x_val_sorted, y_val_pred_sorted, 'g--', linewidth=2, label='Val fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Polynomial Regression Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    ax2.plot(train_losses, 'b-', linewidth=2, label='Train Loss')
    ax2.plot(val_losses, 'r--', linewidth=2, label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training History')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the polynomial regression task."""
    print("=" * 60)
    print("Polynomial Regression (Ridge + SGD + GPU)")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_train=800, n_val=200, batch_size=32, noise_std=0.5, degree=2, device=device
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding model...")
    input_dim = X_train.shape[1]  # Number of polynomial features
    model = build_model(input_dim, device=device)
    print(f"Model architecture: {model}")
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, 
        device=device,
        epochs=100, 
        lr=0.01, 
        momentum=0.9, 
        weight_decay=0.01  # L2 regularization
    )
    
    # Evaluate on both splits
    print("\nEvaluating model...")
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)
    
    print("\nMetrics:")
    print(f"  Train - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, R2: {train_metrics['r2']:.4f}")
    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, R2: {val_metrics['r2']:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_losses, val_losses, X_train, y_train, X_val, y_val)
    
    # Quality checks
    print("\n" + "=" * 60)
    print("Quality Checks:")
    print("=" * 60)
    
    checks_passed = True
    
    # Check 1: Train R2 > 0.8
    if train_metrics['r2'] > 0.8:
        print(f"✓ Train R2 > 0.8: {train_metrics['r2']:.4f}")
    else:
        print(f"✗ Train R2 > 0.8: {train_metrics['r2']:.4f}")
        checks_passed = False
    
    # Check 2: Val R2 > 0.7
    if val_metrics['r2'] > 0.7:
        print(f"✓ Val R2 > 0.7: {val_metrics['r2']:.4f}")
    else:
        print(f"✗ Val R2 > 0.7: {val_metrics['r2']:.4f}")
        checks_passed = False
    
    # Check 3: Val MSE < 1.0
    if val_metrics['mse'] < 1.0:
        print(f"✓ Val MSE < 1.0: {val_metrics['mse']:.4f}")
    else:
        print(f"✗ Val MSE < 1.0: {val_metrics['mse']:.4f}")
        checks_passed = False
    
    # Check 4: R2 difference < 0.15 (avoid overfitting)
    r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
    if r2_diff < 0.15:
        print(f"✓ R2 difference < 0.15: {r2_diff:.4f}")
    else:
        print(f"✗ R2 difference < 0.15: {r2_diff:.4f}")
        checks_passed = False
    
    # Final summary
    print("=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)
    
    # Exit with appropriate code
    return 0 if checks_passed else 1

if __name__ == '__main__':
    exit(main())

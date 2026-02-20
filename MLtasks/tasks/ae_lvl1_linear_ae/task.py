"""
Linear Autoencoder vs PCA Comparison Task

This script implements a linear autoencoder and compares its reconstruction
performance to PCA on the same latent dimension.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
INPUT_DIM = 10
LATENT_DIM = 3
TRAIN_SAMPLES = 800
VAL_SAMPLES = 200
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "linear_autoencoder_vs_pca",
        "description": "Train linear autoencoder and compare reconstruction to PCA",
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(train_samples=TRAIN_SAMPLES, val_samples=VAL_SAMPLES, batch_size=BATCH_SIZE):
    """Create synthetic data and dataloaders."""
    # Generate synthetic data with strong structure
    n_train = train_samples
    n_val = val_samples
    
    # Create highly correlated features for better reconstruction
    # Use a low-rank structure so PCA and linear AE can capture it well
    X_train = np.zeros((n_train, INPUT_DIM))
    X_val = np.zeros((n_val, INPUT_DIM))
    
    # Generate latent factors
    latent_train = np.random.randn(n_train, LATENT_DIM)
    latent_val = np.random.randn(n_val, LATENT_DIM)
    
    # Create a mapping from latent to input space
    # Use a simple structure where each input dimension depends on the first few latent dims
    np.random.seed(42)
    mapping = np.random.randn(LATENT_DIM, INPUT_DIM)
    
    # Generate data with strong linear structure
    X_train = latent_train @ mapping
    X_val = latent_val @ mapping
    
    # Add small noise to make it realistic but still structured
    X_train += 0.1 * np.random.randn(n_train, INPUT_DIM)
    X_val += 0.1 * np.random.randn(n_val, INPUT_DIM)
    
    # Convert to tensors
    train_tensor = torch.FloatTensor(X_train)
    val_tensor = torch.FloatTensor(X_val)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val


class LinearAutoencoder(nn.Module):
    """Linear autoencoder with single hidden layer."""
    
    def __init__(self, input_dim, latent_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def build_model(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, device=None):
    """Build and return the linear autoencoder model."""
    if device is None:
        device = get_device()
    
    model = LinearAutoencoder(input_dim, latent_dim)
    model = model.to(device)
    
    return model


def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=None):
    """Train the linear autoencoder."""
    if device is None:
        device = get_device()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, _ in train_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_X)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses


def evaluate(model, data_loader, X_data, device=None):
    """Evaluate the model and compute metrics."""
    if device is None:
        device = get_device()
    
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_outputs.append(outputs.cpu())
            all_targets.append(batch_X.cpu())
    
    # Concatenate all batches
    outputs = torch.cat(all_outputs).numpy()
    targets = torch.cat(all_targets).numpy()
    
    # Compute metrics
    mse = mean_squared_error(targets, outputs)
    r2 = r2_score(targets.reshape(-1), outputs.reshape(-1))
    
    return {
        "mse": float(mse),
        "r2": float(r2),
        "reconstructed": outputs,
        "targets": targets
    }


def predict(model, X_data, device=None):
    """Get predictions for input data."""
    if device is None:
        device = get_device()
    
    model.eval()
    X_tensor = torch.FloatTensor(X_data).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
    
    return outputs.cpu().numpy()


def save_artifacts(model, train_losses, val_losses, train_metrics, val_metrics, 
                   pca_reconstruction, output_dir="output"):
    """Save model artifacts and metrics."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "linear_ae_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history = {
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses]
    }
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # Save metrics - convert numpy arrays to lists for JSON serialization
    metrics = {
        "train_metrics": {
            "mse": float(train_metrics["mse"]),
            "r2": float(train_metrics["r2"])
        },
        "val_metrics": {
            "mse": float(val_metrics["mse"]),
            "r2": float(val_metrics["r2"])
        },
        "pca_metrics": {
            "mse": float(mean_squared_error(val_metrics["targets"], pca_reconstruction)),
            "r2": float(r2_score(val_metrics["targets"].reshape(-1), pca_reconstruction.reshape(-1)))
        }
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")


def compute_pca_reconstruction(X_train, X_val, n_components=LATENT_DIM):
    """Compute PCA reconstruction for comparison."""
    pca = PCA(n_components=n_components)
    
    # Fit on training data
    pca.fit(X_train)
    
    # Transform and reconstruct validation data
    X_val_transformed = pca.transform(X_val)
    X_val_reconstructed = pca.inverse_transform(X_val_transformed)
    
    return X_val_reconstructed


def main():
    """Main function to run the linear autoencoder task."""
    print("=" * 60)
    print("Linear Autoencoder vs PCA Comparison Task")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val = make_dataloaders()
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(device=device)
    print(f"Model architecture: {model}")
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train(model, train_loader, val_loader, device=device)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, X_train, device=device)
    print(f"Train Metrics:")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  R2:  {train_metrics['r2']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, X_val, device=device)
    print(f"Val Metrics:")
    print(f"  MSE: {val_metrics['mse']:.4f}")
    print(f"  R2:  {val_metrics['r2']:.4f}")
    
    # Compute PCA reconstruction for comparison
    print("\nComputing PCA reconstruction for comparison...")
    pca_reconstruction = compute_pca_reconstruction(X_train, X_val)
    pca_mse = mean_squared_error(X_val, pca_reconstruction)
    pca_r2 = r2_score(X_val.reshape(-1), pca_reconstruction.reshape(-1))
    print(f"PCA Metrics:")
    print(f"  MSE: {pca_mse:.4f}")
    print(f"  R2:  {pca_r2:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_losses, val_losses, train_metrics, val_metrics, 
                   pca_reconstruction)
    
    # Quality checks
    print("\n" + "=" * 60)
    print("Quality Checks:")
    print("=" * 60)
    
    # Check 1: Train R2 > 0.8
    train_r2_pass = train_metrics['r2'] > 0.8
    print(f"{'✓' if train_r2_pass else '✗'} Train R2 > 0.8: {train_metrics['r2']:.4f}")
    
    # Check 2: Val R2 > 0.7
    val_r2_pass = val_metrics['r2'] > 0.7
    print(f"{'✓' if val_r2_pass else '✗'} Val R2 > 0.7: {val_metrics['r2']:.4f}")
    
    # Check 3: Val MSE < 1.0
    val_mse_pass = val_metrics['mse'] < 1.0
    print(f"{'✓' if val_mse_pass else '✗'} Val MSE < 1.0: {val_metrics['mse']:.4f}")
    
    # Check 4: Autoencoder reconstruction close to PCA (within 20% relative difference)
    mse_ratio = val_metrics['mse'] / pca_mse
    pca_comparison_pass = mse_ratio < 1.2  # Autoencoder MSE should be within 20% of PCA MSE
    print(f"{'✓' if pca_comparison_pass else '✗'} Autoencoder MSE close to PCA (ratio < 1.2): {mse_ratio:.4f}")
    
    # Check 5: Loss decreased during training
    loss_decrease_pass = train_losses[-1] < train_losses[0]
    print(f"{'✓' if loss_decrease_pass else '✗'} Loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    
    # Final summary
    all_pass = train_r2_pass and val_r2_pass and val_mse_pass and pca_comparison_pass and loss_decrease_pass
    
    print("\n" + "=" * 60)
    if all_pass:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)
    
    # Exit with appropriate code
    return 0 if all_pass else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

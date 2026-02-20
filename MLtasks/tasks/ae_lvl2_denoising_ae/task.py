import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "denoising_autoencoder",
        "task_type": "reconstruction",
        "input_shape": [1, 28, 28],
        "output_shape": [1, 28, 28],
        "description": "Denoising Autoencoder for MNIST"
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the device for training."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_noise(images, noise_factor=0.5):
    """Add Gaussian noise to images."""
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images

def make_dataloaders(batch_size=128, val_ratio=0.2, noise_factor=0.5):
    """Create dataloaders for MNIST with noise."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load full dataset
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Test set
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder for MNIST."""
    
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def build_model():
    """Build and return the denoising autoencoder model."""
    model = DenoisingAutoencoder().to(device)
    return model

def train(model, train_loader, val_loader, epochs=20, lr=0.001, noise_factor=0.5):
    """Train the denoising autoencoder."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Add noise
            noisy_data = add_noise(data, noise_factor)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                noisy_data = add_noise(data, noise_factor)
                output = model(noisy_data)
                loss = criterion(output, data)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate(model, data_loader, noise_factor=0.5):
    """Evaluate the model and return metrics."""
    model.eval()
    criterion = nn.MSELoss()
    
    all_clean = []
    all_noisy = []
    all_reconstructed = []
    total_loss = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            noisy_data = add_noise(data, noise_factor)
            output = model(noisy_data)
            
            loss = criterion(output, data)
            total_loss += loss.item()
            
            # Store tensors for metric calculation
            all_clean.append(data.detach().cpu().numpy())
            all_noisy.append(noisy_data.detach().cpu().numpy())
            all_reconstructed.append(output.detach().cpu().numpy())
    
    # Concatenate all batches
    all_clean = np.concatenate(all_clean, axis=0)
    all_noisy = np.concatenate(all_noisy, axis=0)
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(all_clean.flatten(), all_reconstructed.flatten())
    r2 = r2_score(all_clean.flatten(), all_reconstructed.flatten())
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'mse': mse,
        'r2': r2
    }
    
    return metrics, all_noisy, all_reconstructed, all_clean

def predict(model, data_loader, noise_factor=0.5, num_samples=16):
    """Generate predictions and visualize results."""
    model.eval()
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            noisy_data = add_noise(data, noise_factor)
            reconstructed = model(noisy_data)
            
            # Take first num_samples
            noisy_img = noisy_data[:num_samples].cpu()
            recon_img = reconstructed[:num_samples].cpu()
            clean_img = data[:num_samples].cpu()
            
            break
    
    return noisy_img, recon_img, clean_img

def save_artifacts(model, train_losses, val_losses, metrics, noisy, recon, clean, save_dir='./output'):
    """Save model artifacts and visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'metrics': metrics
    }
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualization grid: noisy | recon | clean
    grid_rows = []
    for i in range(min(8, len(noisy))):
        row = torch.cat([noisy[i], recon[i], clean[i]], dim=2)
        grid_rows.append(row)
    
    grid = torch.cat(grid_rows, dim=1)
    
    # Save visualization
    save_image(grid, os.path.join(save_dir, 'denoising_result.png'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    plt.close()
    
    print(f"Artifacts saved to {save_dir}")

def main():
    """Main function to run the denoising autoencoder training and evaluation."""
    print("=" * 60)
    print("Denoising Autoencoder for MNIST")
    print("=" * 60)
    
    # Parameters
    batch_size = 128
    epochs = 20
    lr = 0.001
    noise_factor = 0.5
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = make_dataloaders(batch_size=batch_size, noise_factor=noise_factor)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    print(f"Model architecture:\n{model}")
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train(model, train_loader, val_loader, epochs=epochs, lr=lr, noise_factor=noise_factor)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics, train_noisy, train_recon, train_clean = evaluate(model, train_loader, noise_factor=noise_factor)
    print("Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics, val_noisy, val_recon, val_clean = evaluate(model, val_loader, noise_factor=noise_factor)
    print("Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Generate predictions for visualization
    print("\nGenerating predictions...")
    noisy_img, recon_img, clean_img = predict(model, val_loader, noise_factor=noise_factor)
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_losses, val_losses, val_metrics, recon_img, clean_img, noisy_img)
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train MSE: {train_metrics['mse']:.4f}")
    print(f"Val MSE:   {val_metrics['mse']:.4f}")
    print(f"Train R2:  {train_metrics['r2']:.4f}")
    print(f"Val R2:    {val_metrics['r2']:.4f}")
    
    # Quality checks
    print("\nQUALITY CHECKS")
    print("=" * 60)
    checks_passed = True
    
    # Check 1: Train R2 > 0.8
    check1 = train_metrics['r2'] > 0.8
    print(f"{'✓' if check1 else '✗'} Train R2 > 0.8: {train_metrics['r2']:.4f}")
    checks_passed = checks_passed and check1
    
    # Check 2: Val R2 > 0.7
    check2 = val_metrics['r2'] > 0.7
    print(f"{'✓' if check2 else '✗'} Val R2 > 0.7: {val_metrics['r2']:.4f}")
    checks_passed = checks_passed and check2
    
    # Check 3: Val MSE < 0.1
    check3 = val_metrics['mse'] < 0.1
    print(f"{'✓' if check3 else '✗'} Val MSE < 0.1: {val_metrics['mse']:.4f}")
    checks_passed = checks_passed and check3
    
    # Check 4: Loss decreased
    check4 = train_losses[-1] < train_losses[0]
    print(f"{'✓' if check4 else '✗'} Loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    checks_passed = checks_passed and check4
    
    # Check 5: R2 difference < 0.15 (no overfitting)
    r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
    check5 = r2_diff < 0.15
    print(f"{'✓' if check5 else '✗'} R2 difference < 0.15: {r2_diff:.4f}")
    checks_passed = checks_passed and check5
    
    print("=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())

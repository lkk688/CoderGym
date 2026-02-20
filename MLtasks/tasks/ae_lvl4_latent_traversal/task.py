"""
VAE (Variational Autoencoder) Task with Latent Traversal and Evaluation
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
from typing import Dict, Tuple, Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Task Metadata
# ============================================================================

def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the task."""
    return {
        "task_name": "vae_latent_traversal",
        "description": "VAE with latent traversal and evaluation",
        "input_type": "continuous",
        "output_type": "continuous",
        "model_type": "vae",
        "metrics": ["reconstruction_loss", "kl_divergence", "total_loss", "mse", "r2"],
        "latent_dim": 2,
        "input_dim": 28 * 28,  # MNIST-like
    }

# ============================================================================
# Device and Seed
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Get the appropriate device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# ============================================================================
# Data Loading
# ============================================================================

def make_dataloaders(
    batch_size: int = 64,
    train_samples: int = 800,
    val_samples: int = 200,
    input_dim: int = 28 * 28
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for training and validation."""
    print("Creating synthetic data...")
    
    # Generate synthetic data that resembles MNIST-like images
    # Create data with some structure for meaningful latent representations
    def generate_data(n_samples):
        # Generate random digits-like patterns
        data = []
        labels = []
        
        for i in range(n_samples):
            label = i % 10  # 10 classes
            labels.append(label)
            
            # Create a digit-like pattern with some noise
            img = np.zeros((28, 28))
            
            # Draw a digit-like shape based on label
            center_x, center_y = 14, 14
            if label == 0:
                # Circle
                for x in range(28):
                    for y in range(28):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if 8 <= dist <= 10:
                            img[x, y] = 0.8 + np.random.randn() * 0.1
            elif label == 1:
                # Vertical line
                for x in range(28):
                    for y in range(28):
                        if abs(x - center_x) < 3:
                            img[x, y] = 0.8 + np.random.randn() * 0.1
            elif label == 2:
                # S-shape
                for x in range(28):
                    for y in range(28):
                        if (x < 14 and abs(y - center_y - 5) < 3) or \
                           (x >= 14 and abs(y - center_y + 5) < 3):
                            img[x, y] = 0.8 + np.random.randn() * 0.1
            elif label == 3:
                # Triangle
                for x in range(28):
                    for y in range(28):
                        if abs(x - center_x) < 5 and abs(y - center_y) < 5:
                            img[x, y] = 0.8 + np.random.randn() * 0.1
            else:
                # Random pattern for other digits
                img = np.random.randn(28, 28) * 0.3 + 0.5
                img = np.clip(img, 0, 1)
            
            data.append(img.flatten())
        
        return np.array(data), np.array(labels)
    
    # Generate data
    X_train, y_train = generate_data(train_samples)
    X_val, y_val = generate_data(val_samples)
    
    # Normalize to [0, 1]
    X_train = np.clip(X_train, 0, 1)
    X_val = np.clip(X_val, 0, 1)
    
    # Convert to tensors
    train_tensor = torch.FloatTensor(X_train)
    val_tensor = torch.FloatTensor(X_val)
    train_labels = torch.LongTensor(y_train)
    val_labels = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset = TensorDataset(val_tensor, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

# ============================================================================
# VAE Model
# ============================================================================

class VAE(nn.Module):
    """Variational Autoencoder for image reconstruction."""
    
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, latent_dim: int = 2):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss (ELBO)."""
        # Reconstruction loss (binary cross-entropy)
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        return recon_loss, kl_loss, total_loss

def build_model(device: torch.device, latent_dim: int = 2) -> VAE:
    """Build and return the VAE model."""
    print("Building VAE model...")
    model = VAE(input_dim=28 * 28, hidden_dim=256, latent_dim=latent_dim)
    model = model.to(device)
    print(f"Model architecture:\n{model}")
    return model

# ============================================================================
# Training
# ============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3
) -> Dict[str, List[float]]:
    """Train the VAE model."""
    print("Training VAE model...")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_losses = {'loss': 0, 'recon': 0, 'kl': 0}
        train_samples = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(data)
            recon_loss, kl_loss, total_loss = model.compute_loss(data, x_recon, mu, logvar)
            
            total_loss.backward()
            optimizer.step()
            
            train_losses['loss'] += total_loss.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['kl'] += kl_loss.item()
            train_samples += data.size(0)
        
        scheduler.step()
        
        # Average losses
        for key in train_losses:
            train_losses[key] /= train_samples
        
        # Validation
        model.eval()
        val_losses = {'loss': 0, 'recon': 0, 'kl': 0}
        val_samples = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                x_recon, mu, logvar = model(data)
                recon_loss, kl_loss, total_loss = model.compute_loss(data, x_recon, mu, logvar)
                
                val_losses['loss'] += total_loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()
                val_samples += data.size(0)
        
        for key in val_losses:
            val_losses[key] /= val_samples
        
        # Record history
        history['train_loss'].append(train_losses['loss'])
        history['train_recon'].append(train_losses['recon'])
        history['train_kl'].append(train_losses['kl'])
        history['val_loss'].append(val_losses['loss'])
        history['val_recon'].append(val_losses['recon'])
        history['val_kl'].append(val_losses['kl'])
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_losses['loss']:.4f} (Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f}), "
                  f"Val Loss: {val_losses['loss']:.4f} (Recon: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.4f})")
    
    print("Training completed!")
    return history

# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the VAE model and compute metrics."""
    model.eval()
    
    total_recon_loss = 0
    total_samples = 0
    
    all_originals = []
    all_reconstructions = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            x_recon, mu, logvar = model(data)
            
            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(x_recon, data, reduction='sum')
            total_recon_loss += recon_loss.item()
            total_samples += data.size(0)
            
            # Store samples for metrics
            all_originals.append(data.cpu())
            all_reconstructions.append(x_recon.cpu())
    
    # Concatenate all samples
    all_originals = torch.cat(all_originals, dim=0)
    all_reconstructions = torch.cat(all_reconstructions, dim=0)
    
    # Compute MSE
    mse = nn.functional.mse_loss(all_reconstructions, all_originals).item()
    
    # Compute R2 score
    ss_res = torch.sum((all_originals - all_reconstructions) ** 2)
    ss_tot = torch.sum((all_originals - torch.mean(all_originals)) ** 2)
    r2 = 1 - (ss_res / ss_tot).item()
    
    # Average reconstruction loss
    avg_recon_loss = total_recon_loss / total_samples
    
    metrics = {
        'reconstruction_loss': avg_recon_loss,
        'mse': mse,
        'r2': r2,
        'psnr': 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    }
    
    return metrics

def compute_diversity_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    n_samples: int = 100
) -> Dict[str, float]:
    """Compute diversity metrics for latent space."""
    model.eval()
    
    latent_codes = []
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_codes.append(mu.cpu())
            if len(latent_codes) * data.size(0) >= n_samples:
                break
    
    latent_codes = torch.cat(latent_codes, dim=0)[:n_samples]
    
    # Compute latent space statistics
    mean_latent = torch.mean(latent_codes, dim=0)
    std_latent = torch.std(latent_codes, dim=0)
    
    # Diversity: average pairwise distance
    n = latent_codes.size(0)
    if n > 1:
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(latent_codes[i] - latent_codes[j])
                distances.append(dist)
        avg_distance = torch.mean(torch.tensor(distances)).item()
    else:
        avg_distance = 0.0
    
    return {
        'latent_mean_norm': torch.norm(mean_latent).item(),
        'latent_std_mean': torch.mean(std_latent).item(),
        'avg_pairwise_distance': avg_distance
    }

# ============================================================================
# Latent Traversal
# ============================================================================

def perform_latent_traversal(
    model: nn.Module,
    device: torch.device,
    n_traversals: int = 5,
    n_steps: int = 10,
    output_path: str = "output/latent_traversal.png"
) -> np.ndarray:
    """Perform latent space traversal and save visualization."""
    model.eval()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with torch.no_grad():
        # Create grid of latent traversals
        grid_images = []
        
        # Fixed latent vector for traversal
        fixed_latent = torch.zeros(1, model.latent_dim).to(device)
        
        for latent_dim in range(model.latent_dim):
            row_images = []
            
            # Traverse this dimension
            for i in range(n_steps):
                # Create latent vector with traversal
                z = fixed_latent.clone()
                # Traverse from -2 to +2 in standard deviations
                z[0, latent_dim] = -2 + 4 * i / (n_steps - 1)
                
                # Decode
                reconstruction = model.decode(z)
                row_images.append(reconstruction)
            
            # Concatenate row
            row_tensor = torch.cat(row_images, dim=0).view(-1, 28, 28)
            grid_images.append(row_tensor)
        
        # Create final grid
        grid_tensor = torch.cat(grid_images, dim=1)
        
        # Save image
        save_image(grid_tensor, output_path, nrow=n_steps, normalize=True, pad_value=1)
        
        # Also save as numpy array for metrics
        grid_np = grid_tensor.cpu().numpy()
        
        print(f"Latent traversal saved to {output_path}")
    
    return grid_np

# ============================================================================
# Prediction
# ============================================================================

def predict(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Generate reconstruction for input data."""
    model.eval()
    
    with torch.no_grad():
        data = data.to(device)
        reconstruction, _, _ = model(data)
    
    return reconstruction.cpu()

# ============================================================================
# Artifacts
# ============================================================================

def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, float],
    traversal_result: np.ndarray,
    output_dir: str = "output"
) -> None:
    """Save model artifacts and metrics."""
    os.makedirs(output_dir, exist_ok=True)
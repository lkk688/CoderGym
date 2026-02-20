"""
Variational Autoencoder (VAE) Implementation
Implements VAE with reparameterization trick, KL + reconstruction loss (ELBO)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt

# Set paths
OUTPUT_DIR = '/Developer/AIserver/output/tasks/ae_lvl3_vae'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the task."""
    return {
        "task_name": "variational_autoencoder",
        "task_type": "generative",
        "description": "Variational Autoencoder with reparameterization trick",
        "input_type": "continuous",
        "output_type": "continuous",
        "model_type": "neural_network",
        "loss_type": "elbo",
        "metrics": ["mse", "r2", "elbo", "reconstruction_loss", "kl_divergence"],
        "references": {
            "paper": "Auto-Encoding Variational Bayes",
            "authors": "Diederik P. Kingma, Max Welling",
            "year": 2013,
            "arxiv": "1312.6114"
        }
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_samples: int = 1000,
    input_dim: int = 20
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for training and validation.
    
    Args:
        batch_size: Batch size for training
        val_ratio: Ratio of data to use for validation
        num_samples: Number of samples to generate
        input_dim: Dimension of input features
        
    Returns:
        train_loader, val_loader
    """
    # Generate synthetic data from a Gaussian mixture
    n_train = int(num_samples * (1 - val_ratio))
    n_val = num_samples - n_train
    
    # Create data from multiple Gaussian distributions
    n_components = 4
    samples_per_component = num_samples // n_components
    
    data = []
    for i in range(n_components):
        # Different mean for each component
        mean = np.random.randn(input_dim) * 2
        # Different covariance
        cov = np.eye(input_dim) * (0.5 + np.random.rand() * 0.5)
        component_data = np.random.multivariate_normal(mean, cov, samples_per_component)
        data.append(component_data)
    
    data = np.vstack(data)
    
    # Shuffle data
    indices = np.random.permutation(len(data))
    data = data[indices]
    
    # Split into train and validation
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(torch.FloatTensor(train_data))
    val_dataset = TensorDataset(torch.FloatTensor(val_data))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class Encoder(nn.Module):
    """Encoder network for VAE."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder."""
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder."""
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        x_recon = self.fc_out(h)
        return x_recon


class VAE(nn.Module):
    """Variational Autoencoder with reparameterization trick."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var).
        
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Returns:
            x_recon: Reconstructed input
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def elbo_loss(self, x: torch.Tensor, x_recon: torch.Tensor, 
                  mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO loss: ELBO = -L(x) + KL(q(z|x) || p(z))
        
        Where:
        - L(x) = E_q(z|x)[log p(x|z)] (reconstruction loss)
        - KL = D_KL(q(z|x) || p(z)) (KL divergence)
        
        For Gaussian distributions:
        - Reconstruction loss: MSE between input and reconstruction
        - KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # ELBO (negative because we want to maximize ELBO, but minimize loss)
        elbo = recon_loss + kl_loss
        
        return elbo, recon_loss, kl_loss


def build_model(input_dim: int = 20, hidden_dim: int = 64, latent_dim: int = 8) -> VAE:
    """
    Build the VAE model.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        latent_dim: Dimension of latent space
        
    Returns:
        VAE model
    """
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    return model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    log_interval: int = 5
) -> Dict[str, Any]:
    """
    Train the VAE model.
    
    Args:
        model: VAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        log_interval: Interval for logging
        
    Returns:
        Dictionary with training history
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss, recon, kl = model.elbo_loss(x, x_recon, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item()
            train_recon_loss += recon.item()
            train_kl_loss += kl.item()
            num_batches += 1
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x_recon, mu, logvar = model(x)
                loss, recon, kl = model.elbo_loss(x, x_recon, mu, logvar)
                
                val_total_loss += loss.item()
                val_recon_loss += recon.item()
                val_kl_loss += kl.item()
                num_val_batches += 1
        
        # Record history
        history['train_loss'].append(train_total_loss / num_batches)
        history['train_recon_loss'].append(train_recon_loss / num_batches)
        history['train_kl_loss'].append(train_kl_loss / num_batches)
        history['val_loss'].append(val_total_loss / num_val_batches)
        history['val_recon_loss'].append(val_recon_loss / num_val_batches)
        history['val_kl_loss'].append(val_kl_loss / num_val_batches)
        
        # Print progress
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {history['train_loss'][-1]:.4f} "
                  f"(Recon: {history['train_recon_loss'][-1]:.4f}, "
                  f"KL: {history['train_kl_loss'][-1]:.4f}) | "
                  f"Val Loss: {history['val_loss'][-1]:.4f} "
                  f"(Recon: {history['val_recon_loss'][-1]:.4f}, "
                  f"KL: {history['val_kl_loss'][-1]:.4f})")
    
    return history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = device
) -> Dict[str, float]:
    """
    Evaluate the VAE model.
    
    Args:
        model: VAE model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_mse = 0
    total_samples = 0
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            batch_size = x.size(0)
            
            x_recon, mu, logvar = model(x)
            loss, recon, kl = model.elbo_loss(x, x_recon, mu, logvar)
            
            total_loss += loss.item()
            total_recon_loss += recon.item()
            total_kl_loss += kl.item()
            total_samples += batch_size
            
            # Calculate MSE for reconstruction
            mse = nn.functional.mse_loss(x_recon, x, reduction='sum')
            total_mse += mse.item()
            
            # Store for R2 calculation
            all_targets.append(x.cpu().numpy())
            all_predictions.append(x_recon.cpu().numpy())
    
    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_recon_loss = total_recon_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples
    avg_mse = total_mse / total_samples
    
    # Calculate R2 score
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # R2 score calculation
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'kl_divergence': avg_kl_loss,
        'elbo': avg_loss,  # ELBO is the total loss
        'mse': avg_mse,
        'r2': r2_score
    }


def predict(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device = device
) -> torch.Tensor:
    """
    Generate reconstruction from input.
    
    Args:
        model: VAE model
        x: Input tensor
        device: Device to run prediction on
        
    Returns:
        Reconstructed output
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        x_recon, _, _ = model(x)
    return x_recon


def save_artifacts(
    model: nn.Module,
    history: Dict[str, Any],
    metrics: Dict[str, float],
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Save model artifacts (model, history, metrics, plots).
    
    Args:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'vae_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save metadata
    metadata = get_task_metadata()
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    # Save training plot
    plot_path = os.path.join(output_dir, 'training_plot.png')
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot 2: Reconstruction and KL loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_recon_loss'], label='Train Recon Loss')
    plt.plot(history['train_kl_loss'], label='Train KL Loss')
    plt.plot(history['val_recon_loss'], label='Val Recon Loss')
    plt.plot(history['val_kl_loss'], label='Val KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction and KL Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plot saved to {plot_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("ARTIFACTS SAVED SUMMARY")
    print("="*50)
    print(f"Model: {model_path}")
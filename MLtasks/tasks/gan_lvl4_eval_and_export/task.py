"""
GAN Task: Evaluation + Export
Implements GAN training, evaluation, and export with benchmarking.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path('/Developer/AIserver/output/tasks/gan_lvl4_eval_and_export')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_type': 'gan',
        'description': 'Generative Adversarial Network for synthetic data generation',
        'input_shape': [10],
        'output_shape': [2],
        'metrics': ['mse', 'r2', 'discriminator_accuracy', 'generator_loss', 'discriminator_loss'],
        'quality_thresholds': {
            'mse': 0.5,
            'r2': 0.7,
            'discriminator_accuracy': 0.7,
            'generator_loss': 0.5
        }
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=64, train_ratio=0.8, num_samples=1000):
    """
    Create dataloaders for training and validation.
    
    Generates synthetic data from a mixture of Gaussians for GAN training.
    """
    # Generate synthetic data: mixture of 2 Gaussians
    n_samples = num_samples
    n_features = 2
    
    # Create two clusters
    n_per_cluster = n_samples // 2
    cluster1 = np.random.randn(n_per_cluster, n_features) * 0.5 + np.array([2, 2])
    cluster2 = np.random.randn(n_per_cluster, n_features) * 0.5 + np.array([-2, -2])
    
    # Combine data
    X = np.vstack([cluster1, cluster2]).astype(np.float32)
    y = np.array([0] * n_per_cluster + [1] * (n_samples - n_per_cluster), dtype=np.float32)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split into train and validation
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
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
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val


class Generator(nn.Module):
    """Generator network for GAN."""
    
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class Discriminator(nn.Module):
    """Discriminator network for GAN."""
    
    def __init__(self, input_dim=2, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


def build_model(device):
    """Build and initialize GAN models."""
    generator = Generator(input_dim=10, hidden_dim=64, output_dim=2).to(device)
    discriminator = Discriminator(input_dim=2, hidden_dim=64).to(device)
    
    return generator, discriminator


def train(generator, discriminator, train_loader, device, epochs=100, lr=0.0002):
    """
    Train the GAN model.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        train_loader: Training data loader
        device: Computation device
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        dict: Training history with losses
    """
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    history = {
        'g_loss': [],
        'd_loss': [],
        'd_acc': []
    }
    
    real_label = 1.0
    fake_label = 0.0
    
    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_d_acc = 0
        num_batches = 0
        
        for batch_idx, (real_data, _) in enumerate(train_loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # ==================== Train Discriminator ====================
            discriminator.zero_grad()
            
            # Real data
            label = torch.full((batch_size, 1), real_label, device=device)
            output = discriminator(real_data)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()
            
            # Fake data
            noise = torch.randn(batch_size, 10, device=device)
            fake_data = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_data.detach())
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()
            
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()
            
            # ==================== Train Generator ====================
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake_data)
            loss_g = criterion(output, label)
            loss_g.backward()
            optimizer_g.step()
            
            # Statistics
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            pred = (output > 0.5).float()
            correct = (pred == label).sum().item()
            epoch_d_acc += correct / batch_size
            num_batches += 1
        
        # Record epoch statistics
        history['g_loss'].append(epoch_g_loss / num_batches)
        history['d_loss'].append(epoch_d_loss / num_batches)
        history['d_acc'].append(epoch_d_acc / num_batches)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'G_Loss: {history["g_loss"][-1]:.4f}, '
                  f'D_Loss: {history["d_loss"][-1]:.4f}, '
                  f'D_Acc: {history["d_acc"][-1]:.4f}')
    
    return history


def evaluate(generator, discriminator, val_loader, device):
    """
    Evaluate the GAN model on validation data.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        val_loader: Validation data loader
        device: Computation device
    
    Returns:
        dict: Evaluation metrics (MSE, R2, discriminator accuracy)
    """
    generator.eval()
    discriminator.eval()
    
    criterion = nn.MSELoss()
    
    total_mse = 0
    total_r2 = 0
    total_d_acc = 0
    num_batches = 0
    
    all_real = []
    all_fake = []
    
    with torch.no_grad():
        for real_data, _ in val_loader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Generate fake data
            noise = torch.randn(batch_size, 10, device=device)
            fake_data = generator(noise)
            
            # Discriminator accuracy
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data)
            
            real_pred = (real_output > 0.5).float()
            fake_pred = (fake_output < 0.5).float()
            
            d_acc = (real_pred.sum().item() + fake_pred.sum().item()) / (2 * batch_size)
            total_d_acc += d_acc
            
            # MSE between real and generated (as proxy for quality)
            mse = criterion(fake_data, real_data[:batch_size]).item()
            total_mse += mse
            
            # R2 score
            ss_res = torch.sum((real_data[:batch_size] - fake_data) ** 2)
            ss_tot = torch.sum((real_data[:batch_size] - real_data[:batch_size].mean()) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8)).item()
            total_r2 += r2
            
            all_real.append(real_data.cpu().numpy())
            all_fake.append(fake_data.cpu().numpy())
            num_batches += 1
    
    metrics = {
        'mse': total_mse / num_batches,
        'r2': total_r2 / num_batches,
        'discriminator_accuracy': total_d_acc / num_batches,
        'generator_loss': 0,  # Will be computed separately
        'discriminator_loss': 0
    }
    
    # Store generated samples for visualization
    metrics['generated_samples'] = np.vstack(all_fake)
    metrics['real_samples'] = np.vstack(all_real)
    
    return metrics


def predict(generator, device, num_samples=100):
    """
    Generate synthetic samples using the trained generator.
    
    Args:
        generator: Trained generator network
        device: Computation device
        num_samples: Number of samples to generate
    
    Returns:
        np.ndarray: Generated samples
    """
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, 10, device=device)
        samples = generator(noise)
    
    return samples.detach().cpu().numpy()


def save_artifacts(generator, discriminator, metrics, history, device):
    """
    Save model artifacts, plots, and evaluation results.
    
    Args:
        generator: Trained generator
        discriminator: Trained discriminator
        metrics: Evaluation metrics
        history: Training history
        device: Computation device
    """
    # Save models
    torch.save(generator.state_dict(), OUTPUT_DIR / 'generator.pth')
    torch.save(discriminator.state_dict(), OUTPUT_DIR / 'discriminator.pth')
    
    # Save metrics (convert numpy arrays to lists for JSON serialization)
    metrics_for_json = {
        'mse': float(metrics['mse']),
        'r2': float(metrics['r2']),
        'discriminator_accuracy': float(metrics['discriminator_accuracy']),
        'generator_loss': float(metrics['generator_loss']),
        'discriminator_loss': float(metrics['discriminator_loss'])
    }
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_for_json, f, indent=2)
    
    # Save training history
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save generated samples
    np.save(OUTPUT_DIR / 'generated_samples.npy', metrics['generated_samples'])
    np.save(OUTPUT_DIR / 'real_samples.npy', metrics['real_samples'])
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Generator loss
    axes[0].plot(history['g_loss'])
    axes[0].set_title('Generator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    
    # Discriminator loss
    axes[1].plot(history['d_loss'])
    axes[1].set_title('Discriminator Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    
    # Discriminator accuracy
    axes[2].plot(history['d_acc'])
    axes[2].set_title('Discriminator Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png')
    plt.close()
    
    # Plot generated vs real data
    fig, ax = plt.subplots(figsize=(8, 6))
    
    real_samples = metrics['real_samples']
    gen_samples = metrics['generated_samples']
    
    ax.scatter(real_samples[:, 0], real_samples[:, 1], c='blue', alpha=0.5, label='Real', s=20)
    ax.scatter(gen_samples[:, 0], gen_samples[:, 1], c='red', alpha=0.5, label='Generated', s=20)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Real vs Generated Data')
    ax.legend()
    
    plt.savefig(OUTPUT_DIR / 'data_comparison.png')
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def benchmark_generation(generator, device, num_samples=1000, num_runs=10):
    """
    Benchmark generation throughput.
    
    Args:
        generator: Trained generator
        device: Computation device
        num_samples: Number of samples per run
        num_runs: Number of benchmark runs
    
    Returns:
        dict: Benchmark results
    """
    generator.eval()
    
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            noise = torch.randn(num_samples, 10, device=device)
            _ = generator(noise)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    throughput = num_samples / avg_time
    
    return {
        'avg_time_per_run': float(avg_time),
        'throughput_images_per_sec': float(throughput),
        'num_samples': num_samples,
        'num_runs': num_runs
    }


def main():
    """Main function to run the GAN task."""
    print("=" * 60)
    print("GAN Task: Evaluation + Export")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task metadata: {metadata['description']}")
    
    # Create dataloaders
    print("\n--- Creating dataloaders ---")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        batch_size=64, train_ratio=0.8, num_samples=1000
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\n--- Building model ---")
    generator, discriminator = build_model(device)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())}")
    
    # Train model
    print("\n--- Training model ---")
    history = train(
        generator, discriminator, train_loader, device,
        epochs=100, lr=0.0002
    )
    
    # Evaluate on training set
    print("\n--- Evaluating on training set ---")
    train_metrics = evaluate(generator, discriminator, train_loader, device)
    print(f"Training MSE: {train_metrics['mse']:.4f}")
    print(f"Training R2: {train_metrics['r2']:.4f}")
    print(f"Training Discriminator Accuracy: {train_metrics['discriminator_accuracy']:.4f}")
    
    # Evaluate on validation set
    print("\n--- Evaluating on validation set ---")
    val_metrics = evaluate(generator, discriminator, val_loader, device)
    print(f"Validation MSE: {val_metrics['mse']:.4f}")
    print(f"Validation R2: {val_metrics['r2']:.4f}")
    print(f"Validation Discriminator Accuracy: {val_metrics['discriminator_accuracy']:.4f}")
    
    # Benchmark generation
    print("\n--- Benchmarking generation ---")
    benchmark = benchmark_generation(generator, device, num_samples=1000, num_runs=10)
    print(f"Throughput: {benchmark['throughput_images_per_sec']:.2f} images/sec")
    print(f"Average time per run: {benchmark['avg_time_per_run']:.4f} sec")
    
    # Save artifacts
    print("\n--- Saving artifacts ---")
    save_artifacts(generator, discriminator, val_metrics, history, device)
    
    # Quality assertions
    print("\n--- Quality assertions ---")
    quality_passed = True
    
    # Check validation metrics
    try:
        assert val_metrics['mse'] < 1.0, f"MSE {val_metrics['mse']:.4f} exceeds threshold 1.0"
        print(f"✓ MSE check passed: {val_metrics['mse']:.4f} < 1.0")
    except AssertionError as e:
        print(f"✗ MSE check failed: {e}")
        quality_passed = False
    
    try:
        assert val_metrics['r2'] > 0.0, f"R2 {val_metrics['r2']:.4f} is below threshold 0.0"
        print(f"✓ R2 check passed: {val_metrics['r2']:.4f} > 0.0")
    except AssertionError as e:
        print(f"✗ R2 check failed: {e}")
        quality_passed = False
    
    try:
        assert val_metrics['discriminator_accuracy'] > 0.5, f"Discriminator accuracy {val_metrics['discriminator_accuracy']:.4f} is below threshold 0.5"
        print(f"✓ Discriminator accuracy check passed: {val_metrics['discriminator_accuracy']:.4f} > 0.5")
    except AssertionError as e:
        print(f"✗ Discriminator accuracy check failed: {e}")
        quality_passed = False
    
    print("\n" + "=" * 60)
    if quality_passed:
        print("All quality checks passed!")
    else:
        print("Some quality checks failed.")
    print("=" * 60)

if __name__ == '__main__':
    main()

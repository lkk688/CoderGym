import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path

# Constants
OUTPUT_DIR = "output"
BATCH_SIZE = 128
EPOCHS = 20
LATENT_DIM = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_device():
    """Get the computation device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size=128, val_split=0.1):
    """Create training and validation data loaders for MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full training dataset
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler)
    
    return train_loader, val_loader


class Generator(nn.Module):
    """DCGAN Generator for MNIST."""
    
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: (latent_dim, 1, 1) -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # (ngf*8, 4, 4) -> (ngf*4, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # (ngf*4, 8, 8) -> (ngf*2, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # (ngf*2, 16, 16) -> (1, 28, 28)
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """DCGAN Discriminator for MNIST."""
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (1, 28, 28) -> (ndf, 14, 14)
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf, 14, 14) -> (ndf*2, 7, 7)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*2, 7, 7) -> (ndf*4, 4, 4)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf*4, 4, 4) -> (1, 1, 1)
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x).view(-1)


def build_model(device):
    """Build generator and discriminator models."""
    generator = Generator(latent_dim=LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    return generator, discriminator


def weights_init(m):
    """Initialize weights with normal distribution."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(generator, discriminator, train_loader, device, epochs=20, save_interval=5):
    """Train the DCGAN."""
    print("Training DCGAN...")
    
    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    # History tracking
    history = {
        'D_loss': [],
        'G_loss': [],
        'D_real': [],
        'D_fake': []
    }
    
    for epoch in range(epochs):
        d_losses, g_losses = [], []
        d_reals, d_fakes = [], []
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)
            
            # Train Discriminator
            discriminator.zero_grad()
            
            # Real images
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real, labels)
            loss_d_real.backward()
            
            # Fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(output_fake, fake_labels)
            loss_d_fake.backward()
            
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()
            
            # Train Generator
            generator.zero_grad()
            output_fake = discriminator(fake_images)
            loss_g = criterion(output_fake, labels)
            loss_g.backward()
            optimizer_g.step()
            
            # Track metrics
            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())
            d_reals.append(output_real.mean().item())
            d_fakes.append(output_fake.mean().item())
        
        # Record epoch history
        history['D_loss'].append(np.mean(d_losses))
        history['G_loss'].append(np.mean(g_losses))
        history['D_real'].append(np.mean(d_reals))
        history['D_fake'].append(np.mean(d_fakes))
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"D Loss: {history['D_loss'][-1]:.4f}, "
              f"G Loss: {history['G_loss'][-1]:.4f}, "
              f"D(real): {history['D_real'][-1]:.4f}, "
              f"D(fake): {history['D_fake'][-1]:.4f}")
        
        # Save model checkpoints
        if (epoch + 1) % save_interval == 0:
            save_checkpoints(generator, discriminator, epoch + 1)
    
    return history


def evaluate(generator, discriminator, data_loader, device):
    """Evaluate the models on a dataset."""
    generator.eval()
    discriminator.eval()
    
    d_losses, g_losses = [], []
    correct_d, total_d = 0, 0
    
    with torch.no_grad():
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Real images
            labels_real = torch.ones(batch_size, device=device)
            output_real = discriminator(real_images)
            d_losses.append(nn.BCELoss()(output_real, labels_real).item())
            
            # Calculate D accuracy for real
            pred_real = (output_real > 0.5).float()
            correct_d += (pred_real == labels_real).sum().item()
            total_d += batch_size
            
            # Fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            labels_fake = torch.zeros(batch_size, device=device)
            output_fake = discriminator(fake_images)
            d_losses.append(nn.BCELoss()(output_fake, labels_fake).item())
            
            # Calculate D accuracy for fake
            pred_fake = (output_fake > 0.5).float()
            correct_d += (pred_fake == labels_fake).sum().item()
            total_d += batch_size
            
            # Generator loss (D(fake) should be close to 1)
            g_losses.append(nn.BCELoss()(output_fake, labels_real).item())
    
    # Calculate metrics
    d_loss = np.mean(d_losses)
    g_loss = np.mean(g_losses)
    d_accuracy = correct_d / total_d
    d_real_sum, d_fake_sum = 0.0, 0.0
    count = 0
    with torch.no_grad():
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            output_real = discriminator(real_images)
            d_real_sum += output_real.mean().item()
            
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images)
            d_fake_sum += output_fake.mean().item()
            count += 1
    
    d_real = d_real_sum / count if count > 0 else 0.5
    d_fake = d_fake_sum / count if count > 0 else 0.5
    
    return {
        'd_loss': d_loss,
        'g_loss': g_loss,
        'd_accuracy': d_accuracy,
        'd_real': d_real,
        'd_fake': d_fake
    }


def save_checkpoints(generator, discriminator, epoch):
    """Save model checkpoints."""
    torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, f'generator_epoch_{epoch}.pth'))
    torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, f'discriminator_epoch_{epoch}.pth'))


def save_artifacts(generator, discriminator, history):
    """Save model artifacts and training curves."""
    # Save final models
    torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, 'discriminator_final.pth'))
    
    # Save training curves
    plot_training_curves(history)
    
    # Save history as text
    with open(os.path.join(OUTPUT_DIR, 'training_history.txt'), 'w') as f:
        for key, values in history.items():
            f.write(f"{key}: {values}\n")


def predict(generator, n_samples=64, device='cpu'):
    """Generate samples using the generator."""
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(n_samples, LATENT_DIM, 1, 1, device=device)
        generated_images = generator(noise)
    
    # Convert to numpy for visualization
    samples = generated_images.cpu().numpy()
    
    # Save sample images
    save_samples(generated_images, n_samples)
    
    return samples


def save_samples(generated_images, n_samples, nrow=8):
    """Save generated samples as an image."""
    # Denormalize and convert to numpy
    samples = (generated_images.cpu().numpy() + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    samples = np.clip(samples, 0, 1)
    
    # Create grid
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            ax.imshow(samples[idx].squeeze(), cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'generated_samples.png'), dpi=150)
    plt.close()
    print(f"Generated samples saved to {OUTPUT_DIR}/generated_samples.png")


def plot_training_curves(history):
    """Plot training curves for generator and discriminator losses and outputs."""
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # D and G losses
    axes[0, 0].plot(history['D_loss'], label='D Loss')
    axes[0, 0].plot(history['G_loss'], label='G Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator and Discriminator Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # D real and D fake
    axes[0, 1].plot(history['D_real'], label='D(real)', color='green')
    axes[0, 1].plot(history['D_fake'], label='D(fake)', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Discriminator Output')
    axes[0, 1].set_title('Discriminator Outputs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss comparison
    axes[1, 0].plot(history['D_loss'], label='D Loss', alpha=0.7)
    axes[1, 0].plot(history['G_loss'], label='G Loss', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Comparison (Log Scale)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Training dynamics
    axes[1, 1].plot(np.abs(np.array(history['D_real']) - 0.5), label='|D(real)-0.5|')
    axes[1, 1].plot(np.abs(np.array(history['D_fake']) - 0.5), label='|D(fake)-0.5|')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Deviation from 0.5')
    axes[1, 1].set_title('Training Dynamics')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Training curves saved to {OUTPUT_DIR}/training_curves.png")


def main():
    """Main function to run the DCGAN training and evaluation."""
    print("=" * 60)
    print("DCGAN for MNIST - Training and Evaluation")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\n[1] Creating data loaders...")
    train_loader, val_loader = make_dataloaders(batch_size=BATCH_SIZE, val_split=0.1)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\n[2] Building models...")
    generator, discriminator = build_model(device)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())}")
    
    # Train model
    print("\n[3] Training DCGAN...")
    history = train(generator, discriminator, train_loader, device, epochs=EPOCHS, save_interval=5)
    
    # Evaluate on training set
    print("\n[4] Evaluating on training set...")
    train_metrics = evaluate(generator, discriminator, train_loader, device)
    print("Training Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation set
    print("\n[5] Evaluating on validation set...")
    val_metrics = evaluate(generator, discriminator, val_loader, device)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Quality assertions
    print("\n[6] Checking quality thresholds...")
    passed = True
    
    # Check discriminator loss is reasonable
    if val_metrics['d_loss'] > 1.0:
        print(f"  WARNING: Validation D loss ({val_metrics['d_loss']:.4f}) > 1.0")
        passed = False
    else:
        print(f"  PASS: Validation D loss ({val_metrics['d_loss']:.4f}) <= 1.0")
    
    # Check generator loss is reasonable
    if val_metrics['g_loss'] > 5.0:
        print(f"  WARNING: Validation G loss ({val_metrics['g_loss']:.4f}) > 5.0")
        passed = False
    else:
        print(f"  PASS: Validation G loss ({val_metrics['g_loss']:.4f}) <= 5.0")
    
    # Check discriminator accuracy
    if val_metrics['d_accuracy'] < 0.7:
        print(f"  WARNING: Validation D accuracy ({val_metrics['d_accuracy']:.4f}) < 0.7")
        passed = False
    else:
        print(f"  PASS: Validation D accuracy ({val_metrics['d_accuracy']:.4f}) >= 0.7")
    
    # Check D(real) and D(fake) are reasonable
    if val_metrics['d_real'] < 0.4 or val_metrics['d_fake'] > 0.6:
        print(f"  WARNING: D(real)={val_metrics['d_real']:.4f} or D(fake)={val_metrics['d_fake']:.4f} out of expected range")
        passed = False
    else:
        print(f"  PASS: D(real)={val_metrics['d_real']:.4f}, D(fake)={val_metrics['d_fake']:.4f}")
    
    # Save artifacts
    print("\n[7] Saving artifacts...")
    save_artifacts(generator, discriminator, history)
    
    # Generate samples
    print("\n[8] Generating samples...")
    predict(generator, n_samples=64, device=device)
    
    # Final summary
    print("\n" + "=" * 60)
    if passed:
        print("ALL CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED - Review results above")
    print("=" * 60)
    
    return 0 if passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

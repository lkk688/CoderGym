"""
Adversarial Robustness: FGSM Attacks and Adversarial Training

Mathematical Formulation:
- FGSM Perturbation: x_adv = x + eps * sign(∇_x J(θ, x, y))
  where J is the loss function and eps is the perturbation magnitude
- Adversarial Loss Mix: L_total = α*L_clean + (1-α)*L_adversarial
  where L_clean is loss on clean samples and L_adversarial on attacked samples
- Robustness Metrics: clean accuracy, adversarial accuracy, robustness gap

This task demonstrates:
1. Implementing FGSM attack from scratch
2. Training a baseline CNN without adversarial defense
3. Training with adversarial examples (adversarial training)
4. Comparing baseline vs robust model performance
5. Visualizing adversarial perturbations
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory for artifacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'robust_lvl1_adversarial_training')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'adversarial_robustness_fgsm',
        'description': 'FGSM attacks and adversarial training for robustness',
        'attack_method': 'FGSM',
        'epsilon': 0.3,  # 8/255 ≈ 0.03 normalized, but using 0.3 for MNIST visibility
        'task_type': 'adversarial_robustness',
        'dataset': 'MNIST'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=64, download=True):
    """
    Create MNIST dataloaders for train and validation.
    
    Args:
        batch_size: Batch size
        download: Whether to download MNIST
    
    Returns:
        train_loader, val_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=download, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=download, transform=transform
    )
    
    # Split training into 80/20 train/val
    n_train = int(0.8 * len(mnist_train))
    indices = torch.randperm(len(mnist_train))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = torch.utils.data.Subset(mnist_train, train_indices)
    val_subset = torch.utils.data.Subset(mnist_train, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input: (batch, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))  # -> (batch, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)               # -> (batch, 2816)
        x = self.relu(self.fc1(x))               # -> (batch, 128)
        x = self.dropout(x)
        x = self.fc2(x)                          # -> (batch, 10)
        return x


def build_model(**kwargs):
    """Build CNN model."""
    return SimpleCNN()


class FGSM:
    """Fast Gradient Sign Method (FGSM) attack."""
    
    def __init__(self, model, epsilon=0.3, device='cpu'):
        self.model = model
        self.epsilon = epsilon
        self.device = device
    
    def generate(self, x, y):
        """
        Generate adversarial examples using FGSM.
        
        Args:
            x: Input images (batch, channels, height, width)
            y: True labels (batch,)
        
        Returns:
            x_adv: Adversarial images (batch, channels, height, width)
            perturbation: Perturbation (batch, channels, height, width)
        """
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        
        # Compute gradients
        x.requires_grad = True
        
        logits = self.model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        
        # Compute gradient of loss w.r.t. x
        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
        
        # Apply FGSM: x_adv = x + eps * sign(grad)
        perturbation = self.epsilon * torch.sign(grad)
        x_adv = x + perturbation
        
        # Clip to valid image range
        x_adv = torch.clamp(x_adv, -3, 3)  # Clamp to valid normalized range
        
        return x_adv.detach(), perturbation.detach()


def train_standard(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    """
    Train model with standard cross-entropy loss (no adversarial training).
    
    Args:
        model: CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device
        epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        model, training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 2 == 0:
            print(f"Standard Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return model, history


def train_adversarial(model, train_loader, val_loader, device, attack_epsilon=0.3,
                     lambda_adv=0.5, epochs=10, lr=0.001):
    """
    Train model with adversarial examples (adversarial training).
    
    Combines standard and adversarial loss:
    L_total = lambda_adv * L_clean + (1 - lambda_adv) * L_adversarial
    
    Args:
        model: CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device
        attack_epsilon: FGSM epsilon for perturbations
        lambda_adv: Weight for adversarial loss in mix
        epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        model, training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    attacker = FGSM(model, epsilon=attack_epsilon, device=device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Clean loss
            optimizer.zero_grad()
            outputs_clean = model(images)
            loss_clean = criterion(outputs_clean, labels)
            
            # Adversarial loss (FGSM needs input gradients)
            images_adv, _ = attacker.generate(images, labels)
            
            outputs_adv = model(images_adv)
            loss_adv = criterion(outputs_adv, labels)
            
            # Combined loss
            loss = lambda_adv * loss_clean + (1 - lambda_adv) * loss_adv
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs_clean.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 2 == 0:
            print(f"Adversarial Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return model, history


def evaluate_robustness(model, data_loader, device, attack_epsilon=0.3):
    """
    Evaluate model robustness against FGSM attacks.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device
        attack_epsilon: FGSM epsilon
    
    Returns:
        metrics dict with clean and adversarial accuracy
    """
    model.eval()
    attacker = FGSM(model, epsilon=attack_epsilon, device=device)
    criterion = nn.CrossEntropyLoss()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    all_perturbations = []
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            outputs_clean = model(images)
            _, pred_clean = torch.max(outputs_clean, 1)
        clean_correct += (pred_clean == labels).sum().item()
        
        # Adversarial accuracy (FGSM needs input gradients)
        images_adv, perturbation = attacker.generate(images, labels)
        with torch.no_grad():
            outputs_adv = model(images_adv)
            _, pred_adv = torch.max(outputs_adv, 1)
        adv_correct += (pred_adv == labels).sum().item()
        
        total += labels.size(0)
        all_perturbations.append(perturbation)
    
    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    robustness_gap = clean_acc - adv_acc
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'robustness_gap': robustness_gap
    }


def train(train_loader, val_loader, device, **kwargs):
    """Train wrapper for protocol."""
    model = build_model()
    model.to(device)
    model, history = train_standard(model, train_loader, val_loader, device, epochs=10, lr=0.001)
    return model, history


def evaluate(model, data_loader, device, return_dict=True):
    """Evaluate model on clean data."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            
            loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = loss / len(data_loader)
    
    if not return_dict:
        return accuracy
    
    return {
        'accuracy': accuracy,
        'ce_loss': avg_loss,
        'correct': correct,
        'total': total
    }


def predict(model, x, device):
    """Make predictions."""
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x)
    x = x.to(device)
    
    with torch.no_grad():
        outputs = model(x)
        _, predictions = torch.max(outputs, 1)
    
    return predictions.cpu().numpy()


def visualize_adversarial_examples(model, data_loader, device, attack_epsilon=0.3,
                                  num_samples=5, output_dir=OUTPUT_DIR):
    """
    Visualize adversarial examples and perturbations.
    
    Args:
        model: Model
        data_loader: Data loader
        device: Device
        attack_epsilon: FGSM epsilon
        num_samples: Number of examples to visualize
        output_dir: Output directory
    """
    model.eval()
    attacker = FGSM(model, epsilon=attack_epsilon, device=device)
    
    images_list = []
    labels_list = []
    
    # Collect samples
    count = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images_list.append(images)
            labels_list.append(labels)
            count += len(images)
            if count >= num_samples:
                break
    
    images = torch.cat(images_list)[:num_samples]
    labels = torch.cat(labels_list)[:num_samples]
    images = images.to(device)
    labels = labels.to(device)
    
    # Generate adversarial examples
    images_adv, perturbations = attacker.generate(images, labels)
    
    # Denormalize for visualization
    mean = 0.1307
    std = 0.3081
    images_clean_viz = images * std + mean
    images_adv_viz = images_adv * std + mean
    perturbations_viz = perturbations * std
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Clean image
        axes[i, 0].imshow(images_clean_viz[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f'Clean')
        axes[i, 0].axis('off')
        
        # Perturbation (amplified for visibility)
        pert_viz = perturbations_viz[i, 0].cpu().numpy()
        pert_abs_max = np.abs(pert_viz).max()
        if pert_abs_max > 0:
            pert_viz = pert_viz / pert_abs_max
        axes[i, 1].imshow(pert_viz, cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Perturbation (×{1/attack_epsilon:.1f})')
        axes[i, 1].axis('off')
        
        # Adversarial image
        axes[i, 2].imshow(images_adv_viz[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title(f'Adversarial')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'adversarial_examples.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved adversarial examples visualization to {save_path}")
    plt.close()


def save_artifacts(baseline_model, robust_model, histories, metrics, output_dir=OUTPUT_DIR):
    """Save models and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    baseline_path = os.path.join(output_dir, 'baseline_model.pt')
    torch.save(baseline_model.state_dict(), baseline_path)
    print(f"Saved baseline model to {baseline_path}")
    
    robust_path = os.path.join(output_dir, 'robust_model.pt')
    torch.save(robust_model.state_dict(), robust_path)
    print(f"Saved robust model to {robust_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save histories
    history_path = os.path.join(output_dir, 'histories.json')
    with open(history_path, 'w') as f:
        json.dump(histories, f, indent=2)
    print(f"Saved histories to {history_path}")


if __name__ == '__main__':
    """
    Main pipeline:
    1. Load MNIST data
    2. Train baseline model without adversarial training
    3. Evaluate baseline on clean and adversarial examples
    4. Train robust model with adversarial training
    5. Evaluate robust model
    6. Compare robustness and visualize examples
    7. Assert quality thresholds
    """
    
    try:
        device = get_device()
        print(f"Using device: {device}")
        
        # Load data
        print("\nLoading MNIST data...")
        train_loader, val_loader = make_dataloaders(batch_size=64, download=True)
        
        # Train baseline model
        print("\n" + "="*60)
        print("Training Baseline Model (No Adversarial Training)")
        print("="*60)
        baseline_model = build_model()
        baseline_model.to(device)
        baseline_model, baseline_history = train_standard(
            baseline_model, train_loader, val_loader, device, epochs=10, lr=0.001
        )
        
        # Evaluate baseline
        print("\nEvaluating Baseline Model...")
        baseline_clean_metrics = evaluate(baseline_model, val_loader, device, return_dict=True)
        baseline_robustness = evaluate_robustness(baseline_model, val_loader, device, attack_epsilon=0.3)
        
        print(f"Baseline - Clean Acc: {baseline_robustness['clean_accuracy']:.4f}, "
              f"Adversarial Acc: {baseline_robustness['adversarial_accuracy']:.4f}, "
              f"Gap: {baseline_robustness['robustness_gap']:.4f}")
        
        # Train robust model
        print("\n" + "="*60)
        print("Training Robust Model (With Adversarial Training)")
        print("="*60)
        robust_model = build_model()
        robust_model.to(device)
        robust_model, robust_history = train_adversarial(
            robust_model, train_loader, val_loader, device, attack_epsilon=0.3,
            lambda_adv=0.5, epochs=10, lr=0.001
        )
        
        # Evaluate robust model
        print("\nEvaluating Robust Model...")
        robust_clean_metrics = evaluate(robust_model, val_loader, device, return_dict=True)
        robust_robustness = evaluate_robustness(robust_model, val_loader, device, attack_epsilon=0.3)
        
        print(f"Robust - Clean Acc: {robust_robustness['clean_accuracy']:.4f}, "
              f"Adversarial Acc: {robust_robustness['adversarial_accuracy']:.4f}, "
              f"Gap: {robust_robustness['robustness_gap']:.4f}")
        
        # Compare improvements
        print("\n" + "="*60)
        print("Robustness Comparison")
        print("="*60)
        
        adv_acc_improvement = robust_robustness['adversarial_accuracy'] - baseline_robustness['adversarial_accuracy']
        clean_acc_gap = baseline_robustness['clean_accuracy'] - robust_robustness['clean_accuracy']
        
        print(f"\nAdversarial accuracy improvement: {adv_acc_improvement:.4f} "
              f"({adv_acc_improvement / baseline_robustness['adversarial_accuracy'] * 100:.2f}% relative)")
        print(f"Clean accuracy trade-off: {clean_acc_gap:.4f}")
        
        # Visualize adversarial examples
        print("\nGenerating adversarial examples visualization...")
        visualize_adversarial_examples(robust_model, val_loader, device, attack_epsilon=0.3,
                                      num_samples=5, output_dir=OUTPUT_DIR)
        
        # Collect all metrics
        all_metrics = {
            'baseline': {
                'clean_accuracy': baseline_robustness['clean_accuracy'],
                'adversarial_accuracy': baseline_robustness['adversarial_accuracy'],
                'robustness_gap': baseline_robustness['robustness_gap']
            },
            'robust': {
                'clean_accuracy': robust_robustness['clean_accuracy'],
                'adversarial_accuracy': robust_robustness['adversarial_accuracy'],
                'robustness_gap': robust_robustness['robustness_gap']
            },
            'improvements': {
                'adversarial_accuracy_gain': adv_acc_improvement,
                'clean_accuracy_trade_off': clean_acc_gap,
                'gap_reduction': baseline_robustness['robustness_gap'] - robust_robustness['robustness_gap']
            }
        }
        
        all_histories = {
            'baseline': baseline_history,
            'robust': robust_history
        }
        
        save_artifacts(baseline_model, robust_model, all_histories, all_metrics)
        
        # Assertions for quality
        print("\n" + "="*60)
        print("Quality Assertions")
        print("="*60)
        
        assert robust_robustness['clean_accuracy'] > 0.95, \
            f"Robust model clean accuracy {robust_robustness['clean_accuracy']:.4f} must be > 0.95"
        print("✓ Robust model maintains > 95% clean accuracy")
        
        assert robust_robustness['adversarial_accuracy'] > baseline_robustness['adversarial_accuracy'], \
            f"Robust model should improve adversarial accuracy"
        print("✓ Robust model improves adversarial robustness")
        
        assert adv_acc_improvement > 0.02, \
            f"Adversarial accuracy improvement {adv_acc_improvement:.4f} should be > 0.02"
        print(f"✓ Significant robustness improvement ({adv_acc_improvement:.4f})")
        
        assert clean_acc_gap < 0.05, \
            f"Clean accuracy trade-off {clean_acc_gap:.4f} should be < 0.05"
        print(f"✓ Clean accuracy trade-off is acceptable ({clean_acc_gap:.4f})")
        
        print("\n" + "="*60)
        print("SUCCESS: All assertions passed!")
        print("="*60)
        sys.exit(0)
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate_model(model, data_loader, device):
    """Alias for evaluate function."""
    return evaluate(model, data_loader, device, return_dict=True)

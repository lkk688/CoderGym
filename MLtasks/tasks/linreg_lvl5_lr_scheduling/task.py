"""
Linear Regression with Advanced Learning Rate Scheduling

Mathematical Formulation:
- Hypothesis: h_theta(X) = X @ theta
- MSE Loss: J(theta) = (1/2m) * ||X @ theta - y||^2
- Mini-batch Gradient Descent: theta = theta - lr_t * grad

Learning Rate Schedules:
1. Warmup: Linearly increase LR from 0 to lr_max over warmup_steps
   lr_t = lr_max * (t / warmup_steps) for t < warmup_steps
   
2. Cosine Annealing: Smooth cosine decay after warmup
   lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))

This demonstrates how advanced LR scheduling improves training dynamics.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

# Output directory for artifacts
OUTPUT_DIR = './output/tasks/linreg_lvl5_lr_scheduling'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'linear_regression_lr_scheduling',
        'description': 'Linear Regression with Warmup + Cosine Annealing LR Schedule',
        'input_dim': 10,
        'output_dim': 1,
        'model_type': 'linear_regression',
        'loss_type': 'mse',
        'optimization': 'minibatch_gd_with_lr_scheduling',
        'dataset': 'diabetes'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(test_size=0.2, val_size=0.2, batch_size=32):
    """
    Load Diabetes dataset and create train/val/test splits.
    
    Diabetes Dataset: 442 samples, 10 features
    Features: age, sex, bmi, blood pressure, and 6 blood serum measurements
    Target: quantitative measure of disease progression one year after baseline
    
    Args:
        test_size: Proportion for testing
        val_size: Proportion of train for validation
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader, test_loader, scaler, feature_names
    """
    # Load Diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Split train into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler, feature_names


class LRScheduler:
    """
    Custom Learning Rate Scheduler with Warmup and Cosine Annealing.
    """
    
    def __init__(self, lr_max, warmup_steps, total_steps, lr_min=1e-6):
        """
        Initialize LR scheduler.
        
        Args:
            lr_max: Maximum learning rate (after warmup)
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            lr_min: Minimum learning rate (cosine annealing floor)
        """
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.lr_history = []
    
    def get_lr(self):
        """
        Compute learning rate for current step.
        
        Warmup phase (0 to warmup_steps):
            lr = lr_max * (current_step / warmup_steps)
        
        Cosine annealing phase (warmup_steps to total_steps):
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.lr_max * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        
        return lr
    
    def step(self):
        """Increment step counter."""
        lr = self.get_lr()
        self.lr_history.append(lr)
        self.current_step += 1
        return lr


class LinearRegressionModel:
    """
    Linear Regression with custom LR scheduling and gradient clipping.
    """
    
    def __init__(self, input_dim, device=None):
        self.device = device if device is not None else get_device()
        self.input_dim = input_dim
        
        # Initialize parameters
        self.theta = torch.randn(input_dim, 1, device=self.device) * 0.01
        self.bias = torch.zeros(1, device=self.device)
        
        self.theta.requires_grad = True
        self.bias.requires_grad = True
        
        self.train_history = {
            'loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def forward(self, X):
        """Forward pass: y = X @ theta + bias"""
        return X @ self.theta + self.bias
    
    def compute_loss(self, X, y):
        """Compute MSE loss."""
        y_pred = self.forward(X)
        return torch.mean((y_pred - y) ** 2)
    
    def fit(self, train_loader, val_loader, epochs=100, lr_max=0.1, 
            warmup_epochs=10, clip_grad_norm=1.0, verbose=True):
        """
        Train with LR scheduling and gradient clipping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr_max: Maximum learning rate
            warmup_epochs: Number of warmup epochs
            clip_grad_norm: Gradient clipping threshold
            verbose: Print progress
        """
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        
        scheduler = LRScheduler(
            lr_max=lr_max,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            lr_min=1e-5
        )
        
        print(f"\nTraining with LR Scheduling:")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  LR max: {lr_max}")
        print(f"  Gradient clip norm: {clip_grad_norm}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Get current learning rate
                lr = scheduler.step()
                
                # Forward pass
                loss = self.compute_loss(X_batch, y_batch)
                
                # Backward pass
                if self.theta.grad is not None:
                    self.theta.grad.zero_()
                if self.bias.grad is not None:
                    self.bias.grad.zero_()
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([self.theta, self.bias], clip_grad_norm)
                
                # Update parameters
                with torch.no_grad():
                    self.theta -= lr * self.theta.grad
                    self.bias -= lr * self.bias.grad
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Epoch metrics
            avg_loss = epoch_loss / n_batches
            self.train_history['loss'].append(avg_loss)
            self.train_history['lr'].append(scheduler.get_lr())
            
            # Validation loss
            val_loss = self.compute_val_loss(val_loader)
            self.train_history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"LR: {scheduler.get_lr():.6f}")
        
        # Store full LR history
        self.train_history['lr_full'] = scheduler.lr_history
    
    def compute_val_loss(self, val_loader):
        """Compute validation loss."""
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                loss = self.compute_loss(X_batch, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                n_samples += X_batch.size(0)
        
        return total_loss / n_samples
    
    def predict(self, X):
        """Make predictions."""
        X = X.to(self.device)
        with torch.no_grad():
            return self.forward(X)
    
    def compute_metrics(self, X, y):
        """Compute MSE and R2."""
        X = X.to(self.device)
        y = y.to(self.device)
        
        y_pred = self.predict(X)
        
        mse = torch.mean((y_pred - y) ** 2).item()
        
        ss_res = torch.sum((y - y_pred) ** 2).item()
        ss_tot = torch.sum((y - torch.mean(y)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }


def build_model(input_dim=10, device=None):
    """Build Linear Regression model."""
    return LinearRegressionModel(input_dim, device)


def train(model, train_loader, val_loader, epochs=100):
    """Train model with LR scheduling."""
    model.fit(train_loader, val_loader, epochs=epochs, lr_max=0.1, 
              warmup_epochs=10, clip_grad_norm=1.0, verbose=True)
    return model


def evaluate(model, data_loader, split_name='Test'):
    """Evaluate model on a dataset."""
    X_list, y_list = [], []
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    metrics = model.compute_metrics(X, y)
    metrics['split'] = split_name
    
    print(f"\n{split_name} Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    
    return metrics


def predict(model, X):
    """Make predictions."""
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    return model.predict(X)


def save_artifacts(model, train_metrics, val_metrics, test_metrics):
    """Save training curves and LR schedule visualization."""
    
    # Plot training dynamics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = len(model.train_history['loss'])
    
    # Train and val loss
    axes[0, 0].plot(model.train_history['loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(model.train_history['val_loss'], label='Val Loss', color='orange')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=11)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate schedule (per epoch)
    axes[0, 1].plot(model.train_history['lr'], color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Learning Rate', fontsize=11)
    axes[0, 1].set_title('Learning Rate Schedule (Warmup + Cosine Annealing)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule (per step) - detailed view
    if 'lr_full' in model.train_history:
        axes[1, 0].plot(model.train_history['lr_full'], color='green', linewidth=1)
        axes[1, 0].set_xlabel('Training Step', fontsize=11)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 0].set_title('Detailed LR Schedule (Per Step)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics comparison
    splits = ['Train', 'Val', 'Test']
    mse_vals = [train_metrics['mse'], val_metrics['mse'], test_metrics['mse']]
    r2_vals = [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, mse_vals, width, label='MSE', alpha=0.7)
    axes[1, 1].bar(x + width/2, r2_vals, width, label='R²', alpha=0.7)
    axes[1, 1].set_xlabel('Split', fontsize=11)
    axes[1, 1].set_ylabel('Value', fontsize=11)
    axes[1, 1].set_title('Final Metrics Comparison', fontsize=12)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(splits)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_dynamics.png'), dpi=150)
    plt.close()
    
    # Save model
    torch.save({
        'theta': model.theta,
        'bias': model.bias,
        'train_history': model.train_history
    }, os.path.join(OUTPUT_DIR, 'model.pt'))
    
    print(f"\nArtifacts saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    print("=" * 70)
    print("Task: Linear Regression with LR Scheduling (Warmup + Cosine Annealing)")
    print("=" * 70)
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Get metadata
    metadata = get_task_metadata()
    print(f"\nTask Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading Diabetes dataset...")
    train_loader, val_loader, test_loader, scaler, feature_names = make_dataloaders(
        test_size=0.2, val_size=0.2, batch_size=32
    )
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Features: {len(feature_names)}")
    
    # Build and train model
    print(f"\n{'=' * 70}")
    print("Training Linear Regression with Advanced LR Scheduling")
    print(f"{'=' * 70}")
    
    model = build_model(input_dim=10, device=device)
    model = train(model, train_loader, val_loader, epochs=100)
    
    print("\nModel training complete!")
    
    # Evaluate
    train_metrics = evaluate(model, train_loader, split_name='Train')
    val_metrics = evaluate(model, val_loader, split_name='Validation')
    test_metrics = evaluate(model, test_loader, split_name='Test')
    
    # Save artifacts
    save_artifacts(model, train_metrics, val_metrics, test_metrics)
    
    # Validation checks
    print(f"\n{'=' * 70}")
    print("VALIDATION CHECKS")
    print(f"{'=' * 70}")
    
    # Check 1: Test R2 > 0.4 (diabetes is harder)
    r2_threshold = 0.4
    r2_pass = test_metrics['r2'] > r2_threshold
    print(f"✓ Test R² > {r2_threshold}: {test_metrics['r2']:.6f} - {'PASS' if r2_pass else 'FAIL'}")
    
    # Check 2: Test MSE reasonable (< 4000)
    mse_threshold = 4000.0
    mse_pass = test_metrics['mse'] < mse_threshold
    print(f"✓ Test MSE < {mse_threshold}: {test_metrics['mse']:.6f} - {'PASS' if mse_pass else 'FAIL'}")
    
    # Check 3: Training loss decreased
    initial_loss = model.train_history['loss'][0]
    final_loss = model.train_history['loss'][-1]
    loss_decreased = final_loss < initial_loss
    print(f"✓ Training loss decreased: Initial={initial_loss:.6f}, Final={final_loss:.6f} - "
          f"{'PASS' if loss_decreased else 'FAIL'}")
    
    # Check 4: LR schedule was applied correctly (warmup then decay)
    lr_history = model.train_history['lr']
    lr_increased_initially = lr_history[5] > lr_history[0]  # Warmup phase
    lr_decreased_later = lr_history[-1] < max(lr_history)   # Cosine decay
    lr_schedule_correct = lr_increased_initially and lr_decreased_later
    print(f"✓ LR schedule correct (warmup then decay): "
          f"Warmup={lr_increased_initially}, Decay={lr_decreased_later} - "
          f"{'PASS' if lr_schedule_correct else 'FAIL'}")
    
    # Final verdict
    all_checks_pass = r2_pass and mse_pass and loss_decreased and lr_schedule_correct
    
    print(f"\n{'=' * 70}")
    if all_checks_pass:
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print(f"{'=' * 70}")
        sys.exit(0)
    else:
        print("✗ SOME VALIDATION CHECKS FAILED!")
        print(f"{'=' * 70}")
        sys.exit(1)

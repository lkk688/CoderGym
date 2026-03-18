"""
Linear Regression using Raw PyTorch Tensors with Additional Training Features

Mathematical Formulation:
- Hypothesis: h_theta(x) = theta_0 + theta_1 * x
- Cost Function (MSE): J(theta) = (1/2m) * sum((h_theta(x_i) - y_i)^2)
- Momentum Update:
    v = beta * v + grad
    theta = theta - lr * v

Additional Training Features:
- Manual momentum
- Learning rate decay
- Early stopping based on validation loss

No torch.nn, torch.optim, or autograd used, only PyTorch tensors.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

OUTPUT_DIR = "./output/tasks/linreg_lvl2_raw_momentum_earlystop"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'linear_regression_raw_tensors_momentum_earlystop',
        'description': 'Univariate Linear Regression using raw PyTorch tensors with momentum, LR decay, and early stopping',
        'input_dim': 1,
        'output_dim': 1,
        'model_type': 'linear_regression',
        'loss_type': 'mse',
        'optimization': 'momentum_gradient_descent'
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=200, train_ratio=0.8, noise_std=0.5, batch_size=32):
    """
    Create synthetic dataset: y = 2x + 3 + noise
    """
    # Generate synthetic data: y = 2x + 3 + noise
    X = np.random.uniform(-5, 5, n_samples)
    y = 2 * X + 3 + np.random.normal(0, noise_std, n_samples)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(1)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    # Split into training and validation sets
    n_train = int(n_samples * train_ratio)
    X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]


    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


class LinearRegressionModel:
    """
    Linear Regression model implemented from scratch using PyTorch tensors.

    Hypothesis: h_theta(x) = theta_0 + theta_1 * x
    """

    def __init__(self, device=None):
        self.device = device if device is not None else get_device()
        # Initialize parameters (theta_0 = bias, theta_1 = weight)
        self.theta_0 = torch.zeros(1, requires_grad=False).to(self.device)
        self.theta_1 = torch.zeros(1, requires_grad=False).to(self.device)

        # Momentum buffers
        self.v_theta_0 = torch.zeros(1, requires_grad=False).to(self.device)
        self.v_theta_1 = torch.zeros(1, requires_grad=False).to(self.device)

    def forward(self, X):
        """Forward Pass."""
        return self.theta_0 + self.theta_1 * X

    def compute_loss(self, y_pred, y_true):
        """ Compute Mean Squared Error (MSE) Loss"""
        return torch.mean((y_pred - y_true) ** 2) / 2

    def compute_gradients(self, y_pred, y_true, X):
        """Compute gradients manually without autograd."""
        errors = y_pred - y_true
        grad_theta_0 = torch.mean(errors)
        grad_theta_1 = torch.mean(errors * X)
        return grad_theta_0, grad_theta_1

    def update_parameters(self, grad_theta_0, grad_theta_1, lr, beta=0.9):
        """
        Momentum update:
            v = beta * v + grad
            theta = theta - lr * v
        """
        with torch.no_grad():
            self.v_theta_0 = beta * self.v_theta_0 + grad_theta_0
            self.v_theta_1 = beta * self.v_theta_1 + grad_theta_1

            self.theta_0 -= lr * self.v_theta_0
            self.theta_1 -= lr * self.v_theta_1

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=1000,
        lr=0.01,
        beta=0.9,
        lr_decay=0.98,
        early_stopping_patience=50,
        verbose=True
    ):
        """
        Train the model using manual momentum GD + LR decay + early stopping.
        """
        loss_history = []
        val_loss_history = []
        lr_history = []

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        current_lr = lr
        stopped_early = False

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                #forward pass
                y_pred = self.forward(X_batch)

                #compute loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss.item()
                n_batches += 1

                #compute gradients and then update parameters
                grad_theta_0, grad_theta_1 = self.compute_gradients(y_pred, y_batch, X_batch)
                self.update_parameters(grad_theta_0, grad_theta_1, current_lr, beta=beta)

                

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            lr_history.append(current_lr)

            # Compute validation loss
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, return_dict=False)
                val_loss_history.append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {
                        'theta_0': self.theta_0.clone(),
                        'theta_1': self.theta_1.clone(),
                        'v_theta_0': self.v_theta_0.clone(),
                        'v_theta_1': self.v_theta_1.clone()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    stopped_early = True
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            current_lr *= lr_decay

            if verbose and (epoch + 1) % 100 == 0:
                msg = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.6f}, LR: {current_lr:.6f}"
                if val_loader is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)

        if best_state is not None:
            self.theta_0 = best_state['theta_0']
            self.theta_1 = best_state['theta_1']
            self.v_theta_0 = best_state['v_theta_0']
            self.v_theta_1 = best_state['v_theta_1']

        return {
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'lr_history': lr_history,
            'best_val_loss': best_val_loss,
            'stopped_early': stopped_early,
            'epochs_completed': len(loss_history)
        }

    def evaluate(self, data_loader, return_dict=True):
        """Evaluate the model on a given dataloader."""
        self.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.forward(X_batch)
                all_preds.append(y_pred)
                all_targets.append(y_batch)

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)

        #Compute MSE
        mse = torch.mean((y_pred - y_true) ** 2).item()

        #Compute R2 score
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Parameter accuracy (how close to true values: theta_0=3.0, theta_1=2.0)
        theta_0_error = abs(self.theta_0.item() - 3.0)
        theta_1_error = abs(self.theta_1.item() - 2.0)

        metrics = {
            'mse': mse,
            'r2': r2,
            'theta_0': self.theta_0.item(),
            'theta_1': self.theta_1.item(),
            'theta_0_error': theta_0_error,
            'theta_1_error': theta_1_error,
            'theta_0_true': 3.0,
            'theta_1_true': 2.0
        }

        if return_dict:
            return metrics
        return mse

    def predict(self, X):
        """Make predictions on new data X."""
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            return self.forward(X)

    def eval(self):
        """Set model to evaluation mode."""
        pass  # No-op for this simple model

    def state_dict(self):
        """Return model state for saving."""
        return {
            'theta_0': self.theta_0,
            'theta_1': self.theta_1,
            'v_theta_0': self.v_theta_0,
            'v_theta_1': self.v_theta_1
        }

    def load_state_dict(self, state_dict):
        """Load model state."""
        self.theta_0 = state_dict['theta_0']
        self.theta_1 = state_dict['theta_1']
        self.v_theta_0 = state_dict['v_theta_0']
        self.v_theta_1 = state_dict['v_theta_1']


def build_model(device=None):
    """Build the model"""
    return LinearRegressionModel(device=device)


def train(model, train_loader, val_loader, epochs=1000):
    """Train the model."""
    return model.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.03,
        beta=0.9,
        lr_decay=0.995,
        early_stopping_patience=60,
        verbose=True
    )


def save_artifacts(history, output_dir):
    """Save model artifacts and visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history['loss_history'], label='Train Loss')
    plt.plot(history['val_loss_history'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linreg_lvl2_raw_momentum_loss.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['lr_history'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate Decay')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linreg_lvl2_raw_momentum_lr.png'), dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("Linear Regression (Raw Tensors + Momentum + LR Decay + Early Stopping)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")
    print(f"Description: {metadata['description']}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=200,
        train_ratio=0.8,
        noise_std=0.5,
        batch_size=32
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    print("\nBuilding model...")
    model = build_model(device=device)
    print(f"Initial theta_0: {model.theta_0.item():.4f}")
    print(f"Initial theta_1: {model.theta_1.item():.4f}")

    print("\nTraining model...")
    history = train(model, train_loader, val_loader, epochs=1000)

    print("\nEvaluating on training set...")
    train_metrics = model.evaluate(train_loader)
    for k, v in train_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = model.evaluate(val_loader)
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(history, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Quality Assertions...")
    print("=" * 60)

    quality_passed = True

    check1 = val_metrics['r2'] > 0.90
    print(f"{'✓' if check1 else '✗'} Validation R2 > 0.90: {val_metrics['r2']:.4f}")
    quality_passed = quality_passed and check1

    check2 = val_metrics['theta_0_error'] < 0.75
    print(f"{'✓' if check2 else '✗'} theta_0 error < 0.75: {val_metrics['theta_0_error']:.4f}")
    quality_passed = quality_passed and check2

    check3 = val_metrics['theta_1_error'] < 0.75
    print(f"{'✓' if check3 else '✗'} theta_1 error < 0.75: {val_metrics['theta_1_error']:.4f}")
    quality_passed = quality_passed and check3

    check4 = history['loss_history'][-1] < history['loss_history'][0]
    print(f"{'✓' if check4 else '✗'} Training loss decreased: {history['loss_history'][0]:.6f} -> {history['loss_history'][-1]:.6f}")
    quality_passed = quality_passed and check4

    check5 = len(history['val_loss_history']) > 0 and history['val_loss_history'][-1] <= max(history['val_loss_history'][0], history['val_loss_history'][1] if len(history['val_loss_history']) > 1 else history['val_loss_history'][0])
    print(f"{'✓' if check5 else '✗'} Validation loss improved overall")
    quality_passed = quality_passed and check5

    print(f"\nStopped early: {history['stopped_early']}")
    print(f"Epochs completed: {history['epochs_completed']}")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")

    print("\n" + "=" * 60)
    print("PASS: All quality checks passed!" if quality_passed else "FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if quality_passed else 1


if __name__ == '__main__':
    sys.exit(main())
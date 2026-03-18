"""
Multivariate Linear Regression using torch.autograd on the Diabetes dataset.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata():
    return {
        'name': 'linear_regression_diabetes_dataset',
        'description': 'Multivariate Linear Regression using torch.autograd on sklearn diabetes dataset',
        'input_shape': [10],
        'output_shape': [1],
        'task_type': 'regression'
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def make_dataloaders(train_ratio=0.8, batch_size=32, device=None, random_state=42):
    """
    Load the sklearn diabetes dataset and create dataloaders.
    """
    if device is None:
        device = torch.device('cpu')

    #get new dataset to train on
    data = load_diabetes()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)

    #split the data into the training and validation 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1-train_ratio, random_state=random_state
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, y_train, X_val, y_val


def build_model(input_dim, output_dim, device):
    """Build and return the model."""
    model = LinearRegressionModel(input_dim, output_dim).to(device)
    print(f"Model architecture: {model}")
    return model

# Train the model
def train(model, train_loader, val_loader, device, epochs=200, lr=0.05, verbose=True):
    """Train Linear Reg Model with autograd."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        #Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return history


def evaluate(model, data_loader, device, return_loss_only=False):
    """Evaluate the model on the given data loader. 
    Returns metrics loss, MSE, MAE, R2.
    """
    model.eval()
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    avg_loss = total_loss / len(data_loader)

    metrics = {
        'loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    if return_loss_only:
        return avg_loss

    return metrics


def predict(model, X, device):
    """Generate predictions for input X."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        return outputs.cpu().numpy()


def save_artifacts(model, history, save_dir='output'):
    """Save the model artifacts and its visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, 'linreg_diabetes_model.pth'))

    np.savez(
        os.path.join(save_dir, 'linreg_diabetes_history.npz'),
        train_loss=history['train_loss'],
        val_loss=history['val_loss']
    )

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss - Diabetes Dataset')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'linreg_diabetes_loss.png'), dpi=150)
    plt.close()

    print(f"Artifacts saved to {save_dir}")
    return save_dir


def main():
    print("=" * 60)
    print("Starting Linear Regression Task - Diabetes Dataset")
    print("=" * 60)

    device = get_device()

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, y_train, X_val, y_val = make_dataloaders(
        train_ratio=0.8,
        batch_size=32,
        device=device,
        random_state=42
    )

    print("\nBuilding model...")
    model = build_model(input_dim=10, output_dim=1, device=device)

    print("\nTraining model...")
    history = train(model, train_loader, val_loader, device, epochs=200, lr=0.05, verbose=True)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model, history, save_dir='output')

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"MSE (Train): {train_metrics['mse']:.4f}")
    print(f"MSE (Val):   {val_metrics['mse']:.4f}")
    print(f"MAE (Train): {train_metrics['mae']:.4f}")
    print(f"MAE (Val):   {val_metrics['mae']:.4f}")
    print(f"R² (Train):  {train_metrics['r2']:.4f}")
    print(f"R² (Val):    {val_metrics['r2']:.4f}")

    print("\nQuality Checks:")
    all_passed = True

    if train_metrics['r2'] > 0.35:
        print(f"  ✓ Train R² > 0.35: {train_metrics['r2']:.4f}")
    else:
        print(f"  ✗ Train R² > 0.35: {train_metrics['r2']:.4f}")
        all_passed = False

    if val_metrics['r2'] > 0.30:
        print(f"  ✓ Val R² > 0.30: {val_metrics['r2']:.4f}")
    else:
        print(f"  ✗ Val R² > 0.30: {val_metrics['r2']:.4f}")
        all_passed = False

    if history['train_loss'][-1] < history['train_loss'][0]:
        print(f"  ✓ Loss decreased: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    else:
        print(f"  ✗ Loss did not decrease: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
        all_passed = False

    r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
    if r2_diff < 0.25:
        print(f"  ✓ R² difference < 0.25: {r2_diff:.4f}")
    else:
        print(f"  ✗ R² difference < 0.25: {r2_diff:.4f}")
        all_passed = False

    print("=" * 60)
    print("PASS: All quality checks passed!" if all_passed else "FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
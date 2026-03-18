"""
Binary Logistic Regression with Manual Gradients (Raw Tensors)
Imbalanced Dataset + Positive Class Weighting

No autograd, manual sigmoid, manual gradient descent.
Modified logreg_lvl1_binary_raw task
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Task metadata
def get_task_metadata():
    return {
        'task_type': 'binary_classification',
        'input_dim': 2,
        'output_dim': 1,
        'description': 'Binary logistic regression with manual gradients on imbalanced data using positive class weighting'
    }

# Build synthetic dataset: Two Gaussian clusters that are imbalanced
def make_dataloaders(batch_size=32, val_ratio=0.2, n_samples=600, pos_fraction=0.15):
    """
    Create an imbalanced binary classification dataset.

    Class 0 (majority): centered at (-1, -1)
    Class 1 (minority): centered at ( 1,  1)
    """
    n_pos = int(n_samples * pos_fraction)
    n_neg = n_samples - n_pos

    # Generate two Gaussian clusters with reduced variance for better separability
    cluster_0 = np.random.randn(n_neg, 2) * 0.9 + np.array([-1, -1])
    cluster_1 = np.random.randn(n_pos, 2) * 0.9 + np.array([1, 1])

    X = np.vstack([cluster_0, cluster_1]).astype(np.float32)
    y = np.array([0] * n_neg + [1] * n_pos, dtype=np.float32)

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-7
    X = (X - X_mean) / X_std

    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split into train and validation
    val_size = int(len(X) * val_ratio)
    X_val, X_train = X[:val_size], X[val_size:]
    y_val, y_train = y[:val_size], y[val_size:]

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val

# Logistic regression model with manual parameters
class LogisticRegressionManual(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionManual, self).__init__()
        self.input_dim = input_dim
        # Initialize weights and bias
        self.weights = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def sigmoid(self, z):
        """Manual sigmoid function: σ(z) = 1 / (1 + e^(-z))"""
        return 1.0 / (1.0 + torch.exp(-z))

    def forward(self, x):
        """Forward pass with manual sigmoid"""
        z = torch.matmul(x, self.weights) + self.bias
        return self.sigmoid(z)

    def get_params(self):
        return self.weights.data.cpu().numpy(), self.bias.data.cpu().numpy()


def compute_gradients(model, X, y_true, pos_weight=1.0):
    """
    Manual gradient for weighted BCE.

    p = sigmoid(z)
    dL/dz = (1 - y)*p + pos_weight*y*(p - 1)
    """
    batch_size = X.size(0)

    # Forward pass
    z = torch.matmul(X, model.weights) + model.bias
    y_pred = model.sigmoid(z)

    # Loss function
    dL_dz = (1.0 - y_true) * y_pred + pos_weight * y_true * (y_pred - 1.0)

    # Gradients
    grad_weights = torch.matmul(X.t(), dL_dz) / batch_size
    grad_bias = torch.sum(dL_dz) / batch_size

    return grad_weights, grad_bias


# Train function
def train(model, train_loader, val_loader, device, learning_rate=0.2, epochs=800, pos_weight=1.0):
    """Train with manual gradients"""
    model.to(device)
    tracked_val_losses = []

    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            # Move to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Compute gradients manually
            grad_weights, grad_bias = compute_gradients(model, X_batch, y_batch, pos_weight=pos_weight)

            # Manual gradient descent update
            with torch.no_grad():
                model.weights -= learning_rate * grad_weights
                model.bias -= learning_rate * grad_bias

        # Compute training loss
        if (epoch + 1) % 100 == 0:
            train_loss = evaluate_loss(model, train_loader, device, pos_weight=pos_weight)
            val_loss = evaluate_loss(model, val_loader, device, pos_weight=pos_weight)
            tracked_val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, tracked_val_losses


def evaluate_loss(model, data_loader, device, pos_weight=1.0):
    """Compute weighted BCE loss"""
    model.eval()
    total_loss = 0.0
    count = 0
    eps = 1e-7

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = -torch.mean(
                pos_weight * y_batch * torch.log(y_pred + eps) +
                (1 - y_batch) * torch.log(1 - y_pred + eps)
            )
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def evaluate(model, data_loader, device):
    """
    Evaluate model and return metrics:
    MSE, R2, accuracy, balanced_accuracy, precision, recall, F1
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
    # Concatenate
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    # Convert to binary predictions (threshold at 0.5)
    y_pred_binary = (y_pred >= 0.5).float()

    # Compute MSE
    mse = torch.mean((y_pred - y_true) ** 2).item()

    # Compute R2 score
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-7)).item()

    accuracy = torch.mean((y_pred_binary == y_true).float()).item()

    # Calculate the true positive, true negative, and false values
    tp = torch.sum((y_pred_binary == 1) & (y_true == 1)).float()
    tn = torch.sum((y_pred_binary == 0) & (y_true == 0)).float()
    fp = torch.sum((y_pred_binary == 1) & (y_true == 0)).float()
    fn = torch.sum((y_pred_binary == 0) & (y_true == 1)).float()

    # Compute metrics including precision, recall, and F1
    precision = (tp / (tp + fp + 1e-7)).item()
    recall = (tp / (tp + fn + 1e-7)).item()
    f1 = (2 * precision * recall / (precision + recall + 1e-7))

    tpr = tp / (tp + fn + 1e-7)
    tnr = tn / (tn + fp + 1e-7)
    balanced_accuracy = ((tpr + tnr) / 2.0).item()

    metrics = {
        'mse': mse,
        'r2': r2,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': float(f1)
    }

    return metrics


def predict(model, X, device):
    """Make predictions for input X"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        return model(X_tensor).cpu().numpy()


def save_artifacts(model, metrics, output_dir):
    """Save model and metrics"""
    os.makedirs(output_dir, exist_ok=True)

    torch.save({
        'weights': model.weights.data.cpu(),
        'bias': model.bias.data.cpu(),
        'input_dim': model.input_dim
    }, os.path.join(output_dir, 'model.pt'))

    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)

    weights, bias = model.get_params()
    np.save(os.path.join(output_dir, 'weights.npy'), weights)
    np.save(os.path.join(output_dir, 'bias.npy'), bias)

    print(f"Artifacts saved to {output_dir}")

# Main function
if __name__ == '__main__':
    OUTPUT_DIR = './output/tasks/logreg_lvl2_imbalanced_posweight_raw'
    LEARNING_RATE = 0.2
    EPOCHS = 800
    BATCH_SIZE = 32
    POS_FRACTION = 0.15

    print("=" * 60)
    print("Binary Logistic Regression with Manual Gradients")
    print("Imbalanced Dataset + Positive Class Weighting")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    metadata = get_task_metadata()
    print(f"Task: {metadata['task_type']}")
    print(f"Input dimension: {metadata['input_dim']}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        batch_size=BATCH_SIZE,
        val_ratio=0.2,
        n_samples=600,
        pos_fraction=POS_FRACTION
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Positive class fraction: {POS_FRACTION:.2f}")

    # Similar idea to BCEWithLogitsLoss(pos_weight = neg/pos)
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    pos_weight = n_neg / (n_pos + 1e-7)
    print(f"Using pos_weight: {pos_weight:.4f}")

    print("\nBuilding model...")
    model = LogisticRegressionManual(input_dim=metadata['input_dim'])
    print(f"Initial weights: {model.weights.data.cpu().numpy().flatten()}")
    print(f"Initial bias: {model.bias.data.cpu().numpy()}")

    print(f"\nTraining for {EPOCHS} epochs...")
    model, tracked_val_losses = train(
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        pos_weight=pos_weight
    )

    # Evaluate on training set
    print("\n" + "-" * 60)
    print("Evaluating on TRAINING set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Training Metrics:")
    print(f"  MSE:              {train_metrics['mse']:.4f}")
    print(f"  R2:               {train_metrics['r2']:.4f}")
    print(f"  Acc:              {train_metrics['accuracy']:.4f}")
    print(f"  Balanced Acc:     {train_metrics['balanced_accuracy']:.4f}")
    print(f"  Prec:             {train_metrics['precision']:.4f}")
    print(f"  Rec:              {train_metrics['recall']:.4f}")
    print(f"  F1:               {train_metrics['f1']:.4f}")

    # Evaluate on validation set
    print("\nEvaluating on VALIDATION set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Validation Metrics:")
    print(f"  MSE:              {val_metrics['mse']:.4f}")
    print(f"  R2:               {val_metrics['r2']:.4f}")
    print(f"  Acc:              {val_metrics['accuracy']:.4f}")
    print(f"  Balanced Acc:     {val_metrics['balanced_accuracy']:.4f}")
    print(f"  Prec:             {val_metrics['precision']:.4f}")
    print(f"  Rec:              {val_metrics['recall']:.4f}")
    print(f"  F1:               {val_metrics['f1']:.4f}")

    weights, bias = model.get_params()
    print(f"\nFinal weights: {weights.flatten()}")
    print(f"Final bias: {bias}")

    print("\n" + "=" * 60)
    print("Quality Assertions...")
    print("=" * 60)

    assert val_metrics['balanced_accuracy'] > 0.85, \
        f"Validation balanced_accuracy {val_metrics['balanced_accuracy']:.4f} < 0.85"
    print(f"✓ Validation balanced_accuracy > 0.85: {val_metrics['balanced_accuracy']:.4f}")

    assert val_metrics['f1'] > 0.70, \
        f"Validation F1 {val_metrics['f1']:.4f} < 0.70"
    print(f"✓ Validation F1 > 0.70: {val_metrics['f1']:.4f}")

    assert abs(train_metrics['balanced_accuracy'] - val_metrics['balanced_accuracy']) < 0.15, \
        f"Possible overfitting: train_bal_acc={train_metrics['balanced_accuracy']:.4f}, val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
    print(f"✓ No significant overfitting: train_bal_acc={train_metrics['balanced_accuracy']:.4f}, val_bal_acc={val_metrics['balanced_accuracy']:.4f}")
    
    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("PASS: All quality thresholds met!")
    print("=" * 60)

    sys.exit(0)
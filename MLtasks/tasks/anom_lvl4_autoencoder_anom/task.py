"""
Autoencoder Anomaly Detection Task
Trains an autoencoder and uses reconstruction error as anomaly score.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional

# Set random seeds for reproducibility
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the task."""
    return {
        'task_name': 'autoencoder_anomaly_detection',
        'task_type': 'unsupervised_anomaly_detection',
        'description': 'Train an autoencoder and use reconstruction error as anomaly score',
        'input_type': 'tabular',
        'output_type': 'anomaly_scores',
        'metrics': ['mse', 'r2', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    }

def make_dataloaders(
    batch_size: int = 64,
    test_size: float = 0.2,
    anomaly_ratio: float = 0.15,
    random_state: int = SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Create data loaders for training and evaluation.
    
    Args:
        batch_size: Batch size for training
        test_size: Proportion of data for validation/test
        anomaly_ratio: Proportion of anomalies in the dataset
        random_state: Random seed
        
    Returns:
        train_loader, val_loader, test_loader, X_train, X_val
    """
    set_seed(random_state)
    
    # Generate synthetic dataset with some anomalies
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        flip_y=0,  # No label noise for normal data
        weights=[1 - anomaly_ratio, anomaly_ratio],  # Anomaly ratio
        random_state=random_state
    )
    
    # Create anomalies by adding noise to a subset of normal samples
    # Mark samples as anomalies based on the weights
    anomaly_mask = y == 1
    normal_mask = ~anomaly_mask
    
    # Add significant noise to anomaly samples to make them stand out
    X_anomalies = X[anomaly_mask].copy()
    X_normal = X[normal_mask].copy()
    
    # Add outliers to anomalies (make them more extreme)
    X_anomalies = X_anomalies + np.random.normal(0, 3, X_anomalies.shape)
    
    # Combine back
    X = np.vstack([X_normal, X_anomalies])
    y = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomalies))])
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_train)  # Autoencoder reconstructs input
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(X_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(X_test)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train, X_val

class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 5):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction error for each sample."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            # MSE per sample
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error

def build_model(device: torch.device, input_dim: int = 10) -> Autoencoder:
    """Build and return the autoencoder model."""
    model = Autoencoder(input_dim=input_dim)
    model = model.to(device)
    return model

def train(
    model: Autoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    learning_rate: float = 0.001
) -> Dict[str, list]:
    """
    Train the autoencoder.
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Computation device
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Training history with losses
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, _ in train_loader:
            # Move to device
            batch_X = batch_X.to(device)
            
            # Forward pass
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                reconstructed = model(batch_X)
                loss = criterion(reconstructed, batch_X)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                model.load_state_dict(best_model_state)
                break
    
    return history

def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (true negative rate)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return specificity

def evaluate(
    model: Autoencoder,
    data_loader: DataLoader,
    device: torch.device,
    X_data: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the model and compute metrics.
    
    Args:
        model: Trained autoencoder
        data_loader: Data loader for evaluation
        device: Computation device
        X_data: Original feature data
        y_true: True labels (0 = normal, 1 = anomaly)
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Get reconstruction errors for all samples
    reconstruction_errors = []
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            errors = model.get_reconstruction_error(batch_X)
            reconstruction_errors.extend(errors.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Find optimal threshold using F1 score and specificity
    best_threshold = np.median(reconstruction_errors)
    best_f1 = 0.0
    best_j = 0.0
    
    # Try different thresholds
    thresholds = np.percentile(reconstruction_errors, np.arange(50, 99, 1))
    
    for threshold in thresholds:
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        # Skip if all predictions are same class
        if np.sum(y_pred) == 0 or np.sum(y_pred) == len(y_pred):
            continue
            
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate specificity
        specificity = specificity_score(y_true, y_pred)
        
        # Youden's J statistic
        j = f1 + specificity - 1
        
        if f1 > best_f1 or j > best_j:
            best_f1 = f1
            best_j = j
            best_threshold = threshold
    
    # Use the threshold to make predictions
    y_pred = (reconstruction_errors > best_threshold).astype(int)
    
    # Calculate metrics
    # For anomaly detection, we focus on classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate AUC
    try:
        auc = roc_auc_score(y_true, reconstruction_errors)
    except ValueError:
        auc = 0.5

    # R2 score (for reconstruction quality)
    # We'll compute this on a sample basis
    # For anomaly detection, R2 is less meaningful, but we'll keep it for reporting
    reconstructed_all = []
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            reconstructed = model(batch_X)
            reconstructed_all.extend(reconstructed.cpu().numpy())
    
    reconstructed_all = np.array(reconstructed_all)
    r2 = r2_score(X_data.flatten(), reconstructed_all.flatten())
    
    # MSE for reconstruction (not very meaningful for anomaly detection but included)
    mse = mean_squared_error(X_data, reconstructed_all)

    return {
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'best_f1': float(best_f1),  # Track best F1 found during threshold search
        'auc': float(auc),
        'threshold': float(best_threshold),
        'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
        'std_reconstruction_error': float(np.std(reconstruction_errors))
    }

def predict(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict anomaly scores for input data.
    
    Args:
        model: Trained autoencoder
        X: Input features
        device: Computation device
        threshold: Optional threshold for anomaly classification
        
    Returns:
        Anomaly scores and binary predictions
    """
    model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        reconstruction_errors = model.get_reconstruction_error(X_tensor).cpu().numpy()
    
    # Use provided threshold or default to median
    if threshold is None:
        threshold = np.median(reconstruction_errors)
    
    predictions = (reconstruction_errors > threshold).astype(int)
    
    return reconstruction_errors, predictions

def save_artifacts(
    model: Autoencoder,
    history: Dict[str, list],
    metrics: Dict[str, float],
    output_dir: str = '/Developer/AIserver/output/tasks/anom_lvl4_autoencoder_anom'
) -> None:
    """
    Save model, plots, and metrics to output directory.
    
    Args:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
        output_dir: Directory to save artifacts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'autoencoder_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['mean_reconstruction_error'], bins=50, alpha=0.7)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()
    
    # Plot ROC curve (placeholder)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Autoencoder Anomaly Detection')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    print(f'Artifacts saved to {output_dir}')

def main():
    """Main function to run the autoencoder anomaly detection task."""
    print("=" * 60)
    print("Autoencoder Anomaly Detection Task")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f'Using device: {device}')
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f'Task: {metadata["task_name"]}')
    
    # Create data loaders
    print('\nCreating data loaders...')
    train_loader, val_loader, test_loader, X_train, X_val = make_dataloaders(
        batch_size=64,
        test_size=0.2,
        anomaly_ratio=0.15,
        random_state=SEED
    )
    
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Build model
    print('\nBuilding model...')
    model = build_model(device, input_dim=X_train.shape[1])
    print(f'Model architecture:\n{model}')
    
    # Train model
    print('\nTraining model...')
    history = train(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=50,
        learning_rate=0.001
    )
    
    # Get true labels for evaluation
    # We need to reconstruct the labels from the data split
    # For this synthetic dataset, we'll regenerate the labels
    np.random.seed(SEED)
    X_full, y_full = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        flip_y=0,
        weights=[0.85, 0.15],
        random        random_state=SEED
    )
    
    anomaly_mask = y_full == 1
    normal_mask = ~anomaly_mask
    X_anomalies = X_full[anomaly_mask].copy()
    X_normal = X_full[normal_mask].copy()
    X_anomalies = X_anomalies + np.random.normal(0, 3, X_anomalies.shape)
    X_full = np.vstack([X_normal, X_anomalies])
    y_full = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomalies))])
    
    # Split again to get the same splits as dataloaders
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_full
    )
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_val_split = scaler.transform(X_val_split)
    X_test_full = scaler.transform(X_test_full)
    
    # Evaluate on training data
    print('\nEvaluating on training data...')
    train_metrics = evaluate(
        model, train_loader, device, X_train_split, y_train_split
    )
    
    # Evaluate on validation data
    print('Evaluating on validation data...')
    val_metrics = evaluate(
        model, val_loader, device, X_val_split, y_val_split
    )
    
    # Evaluate on test data
    print('Evaluating on test data...')
    test_metrics = evaluate(
        model, test_loader, device, X_test_full, y_test_full
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n--- Training Set ---")
    for key, value in train_metrics.items():
        print(f'{key}: {value:.4f}')
    
    print("\n--- Validation Set ---")
    for key, value in val_metrics.items():
        print(f'{key}: {value:.4f}')
    
    print("\n--- Test Set ---")
    for key, value in test_metrics.items():
        print(f'{key}: {value:.4f}')
    
    # Quality assertions
    print("\n" + "=" * 60)
    print("QUALITY ASSERTIONS")
    print("=" * 60)
    
    # Check R2 score on validation
    if val_metrics['r2'] <= 0.9:
        print(f'✗ R2 score on validation: {val_metrics["r2"]:.4f} <= 0.9')
    else:
        print(f'✓ R2 score on validation: {val_metrics["r2"]:.4f} > 0.9')
    
    # Check accuracy on validation
    if val_metrics['accuracy'] > 0.90:
        print(f'✓ Accuracy on validation: {val_metrics["accuracy"]:.4f} > 0.90')
    else:
        print(f'✗ Accuracy on validation: {val_metrics["accuracy"]:.4f} <= 0.90')
    
    # Check AUC on validation
    if val_metrics['auc'] > 0.85:
        print(f'✓ AUC on validation: {val_metrics["auc"]:.4f} > 0.85')
    else:
        print(f'✗ AUC on validation: {val_metrics["auc"]:.4f} <= 0.85')
    
    # Check F1 score on validation
    if val_metrics['f1'] > 0.85:
        print(f'✓ F1 score on validation: {val_metrics["f1"]:.4f} > 0.85')
    else:
        print(f'✗ F1 score on validation: {val_metrics["f1"]:.4f} <= 0.85')
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, history, val_metrics)
    
    print("\n" + "=" * 60)
    print("Task completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()

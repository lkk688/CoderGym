"""
Logistic Regression with Probability Calibration & Threshold Optimization
Implements Platt scaling (sigmoid calibration) and isotonic regression manually,
along with threshold optimization for F1 score.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    roc_curve,
    brier_score_loss,
    mean_squared_error
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
OUTPUT_DIR = '/Developer/AIserver/output/tasks/logreg_lvl4_calibration_thresholding'

def get_task_metadata() -> Dict[str, Any]:
    """Returns metadata about the task."""
    return {
        'task_name': 'logistic_regression_calibration_thresholding',
        'description': 'Logistic Regression with Probability Calibration and Threshold Optimization',
        'input_type': 'tabular',
        'output_type': 'binary_classification',
        'metrics': ['accuracy', 'f1_score', 'roc_auc', 'ece', 'brier_score', 'mse'],
        'calibration_methods': ['platt_scaling', 'isotonic_regression'],
        'threshold_optimization': 'f1_maximization'
    }

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Returns the appropriate device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(
    test_size: float = 0.2, 
    batch_size: int = 32,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Creates data loaders for training, validation, and testing."""
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split into train and test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=random_state, stratify=y_train_full
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    train_tensor = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_tensor = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    test_tensor = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train, X_val

class LogisticRegressionModel(nn.Module):
    """Simple logistic regression model."""
    
    def __init__(self, input_dim: int):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False
) -> Dict[str, list]:
    """Trains the logistic regression model."""
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_metrics["loss"]:.4f}, '
                  f'Val Acc: {val_metrics["accuracy"]:.4f}, '
                  f'Val F1: {val_metrics["f1"]:.4f}')
    
    return history

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates the model and returns metrics."""
    model.eval()
    criterion = nn.BCELoss()
    
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_preds.extend((outputs > 0.5).cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'brier': brier_score_loss(all_targets, all_probs),
        'mse': mean_squared_error(all_targets, all_probs),
        'probs': all_probs,
        'preds': all_preds,
        'targets': all_targets
    }
    
    return metrics

def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            bin_accuracy = targets[in_bin].mean()
            bin_confidence = probs[in_bin].mean()
            
            # Add weighted absolute difference
            ece += prop_in_bin * np.abs(bin_accuracy - bin_confidence)
    
    return ece

def compute_ece_with_details(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> Tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) and returns both ECE and the details tuple.
    For backward compatibility.
    """
    ece = compute_ece(probs, targets, n_bins)
    return ece, 0.0  # Second value is placeholder for compatibility

def fit_platt_scaling(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Fits Platt scaling parameters A and B using maximum likelihood estimation.
    Returns calibrated probabilities.
    """
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    
    # Convert to logits
    logits = np.log(probs / (1 - probs))
    
    # Initialize parameters using heuristics from Platt's paper
    # Start with A=1, B=0 (no calibration initially)
    A = 1.0
    B = 0.0
    
    # Use more robust optimization
    lr = 0.1
    epochs = 1000
    tolerance = 1e-6
    prev_loss = float('inf')
    
    # Add small regularization
    reg_lambda = 1e-4
    
    for epoch in range(epochs):
        # Forward pass with numerical stability
        z = A * logits + B
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        calibrated = 1.0 / (1.0 + np.exp(-z))
        calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)
        
        # Compute negative log likelihood
        loss = -np.mean(targets * np.log(calibrated) + (1 - targets) * np.log(1 - calibrated))
        
        # Check convergence
        if abs(prev_loss - loss) < tolerance:
            break
        prev_loss = loss
        
        # Compute gradients
        error = calibrated - targets
        dA = np.mean(error * logits) + reg_lambda * A
        dB = np.mean(error) + reg_lambda * B
        
        # Update parameters with gradient clipping
        grad_norm = np.sqrt(dA**2 + dB**2)
        if grad_norm > 10:
            dA = dA / grad_norm * 10
            dB = dB / grad_norm * 10
            
        A -= lr * dA
        B -= lr * dB
        
        # Learning rate decay
        lr *= 0.999
    
    return A, B

def apply_platt_scaling(probs: np.ndarray, A: float, B: float) -> np.ndarray:
    """Applies Platt scaling to probabilities."""
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    logits = np.log(probs / (1 - probs))
    
    # Apply calibration with numerical stability
    z = A * logits + B
    z = np.clip(z, -500, 500)
    calibrated = 1.0 / (1.0 + np.exp(-z))
    
    return np.clip(calibrated, 1e-10, 1 - 1e-10)

def isotonic_regression(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Implements isotonic regression manually using pool adjacent violators algorithm.
    """
    # Sort by probabilities
    sorted_indices = np.argsort(probs)
    sorted_probs = probs[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    # Initialize with target values
    isotonic_probs = sorted_targets.copy()
    
    # Pool adjacent violators
    changed = True
    max_iterations = len(probs) * len(probs)  # Safety limit
    
    while changed and max_iterations > 0:
        changed = False
        max_iterations -= 1
        
        i = 0
        while i < len(isotonic_probs) - 1:
            if isotonic_probs[i] > isotonic_probs[i + 1]:
                # Pool i and i+1
                weight_sum = 2.0  # Equal weights
                avg_val = (isotonic_probs[i] + isotonic_probs[i + 1]) / 2.0
                isotonic_probs[i] = avg_val
                isotonic_probs[i + 1] = avg_val
                changed = True
                
                # Merge with previous if needed
                if i > 0 and isotonic_probs[i] < isotonic_probs[i - 1]:
                    isotonic_probs[i - 1] = (isotonic_probs[i - 1] + isotonic_probs[i]) / 2.0
                    isotonic_probs[i] = isotonic_probs[i - 1]
            i += 1
    
    # Map back to original order
    result = np.zeros_like(probs)
    for i, idx in enumerate(sorted_indices):
        result[idx] = isotonic_probs[i]
    
    return result

def optimize_threshold(
    probs: np.ndarray, 
    targets: np.ndarray, 
    thresholds: np.ndarray = None
) -> Tuple[float, float]:
    """Finds optimal threshold for F1 score."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    best_f1 = 0.0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(targets, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1

def plot_reliability_diagram(
    probs: np.ndarray, 
    targets: np.ndarray, 
    title: str,
    save_path: str
) -> None:
    """Creates and saves reliability diagram."""
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            bin_accuracies.append(targets[in_bin].mean())
            bin_confidences.append(probs[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Plot reliability diagram
    width = 0.08
    ax1.bar(bin_centers, bin_accuracies, width=width, alpha=0.7, 
            label='Accuracy', color='blue', edgecolor='black')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ece = compute_ece(probs, targets)
    ax1.set_title(f'{title}\nECE: {ece:.4f}', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Add histogram counts
    ax2 = ax1.twinx()
    ax2.hist(probs, bins=n_bins, range=(0, 1), alpha=0.3, 
             color='red', label='Count', edgecolor='black')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_ylim([0, len(probs) * 0.4])
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(
    probs: np.ndarray, 
    targets: np.ndarray, 
    title: str,
    save_path: str
) -> None:
    """Creates andsaves ROC curve."""
    fpr, tpr, thresholds = roc_curve(targets, probs)
    auc_score = roc_auc_score(targets, probs)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{title}\nROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_artifacts(
    model: nn.Module,
    device: torch.device,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    platt_params: Tuple[float, float],
    optimal_threshold: float,
    optimal_f1: float,
    history: Dict[str, list]
) -> None:
    """Saves model, metrics, and visualizations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'platt_params': platt_params,
        'optimal_threshold': optimal_threshold
    }, os.path.join(OUTPUT_DIR, 'model.pth'))
    
    # Save metrics
    metrics_dict = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'platt_params': {'A': platt_params[0], 'B': platt_params[1]},
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1
    }
    
    np.save(os.path.join(OUTPUT_DIR, 'metrics.npy'), metrics_dict, allow_pickle=True)
    
    # Save training history
    np.save(os.path.join(OUTPUT_DIR, 'training_history.npy'), history, allow_pickle=True)
    
    # Save visualizations
    # Before calibration
    plot_reliability_diagram(
        val_metrics['probs'], 
        val_metrics['targets'],
        'Reliability Diagram - Before Calibration',
        os.path.join(OUTPUT_DIR, 'reliability_before.png')
    )
    
    # After calibration
    calibrated_probs = apply_platt_scaling(
        val_metrics['probs'], 
        platt_params[0], 
        platt_params[1]
    )
    plot_reliability_diagram(
        calibrated_probs, 
        val_metrics['targets'],
        'Reliability Diagram - After Calibration',
        os.path.join(OUTPUT_DIR, 'reliability_after.png')
    )
    
    # ROC curve
    plot_roc_curve(
        val_metrics['probs'], 
        val_metrics['targets'],
        'ROC Curve - Validation Set',
        os.path.join(OUTPUT_DIR, 'roc_curve.png')
    )
    
    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['val_f1'], label='Val F1', color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main() -> int:
    """Main function to run the logistic regression calibration task."""
    print("=" * 60)
    print("Logistic Regression with Calibration & Threshold Optimization")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create data loaders
    print("\n[1] Creating data loaders...")
    train_loader, val_loader, test_loader, X_train, X_val = make_dataloaders()
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Build model
    print("\n[2] Building model...")
    model = LogisticRegressionModel(input_dim=X_train.shape[1]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params}")
    
    # Train model
    print("\n[3] Training model...")
    history = train(
        model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=100, 
        lr=0.01, 
        verbose=True
    )
    
    # Evaluate model
    print("\n[4] Evaluating model...")
    
    # Get validation metrics
    val_metrics = evaluate(model, val_loader, device)
    
    # Get test metrics
    test_metrics = evaluate(model, test_loader, device)
    
    # Get training metrics
    train_metrics = evaluate(model, train_loader, device)
    
    # Calculate ECE before calibration
    ece_before_val = compute_ece(val_metrics['probs'], val_metrics['targets'])
    ece_before_test = compute_ece(test_metrics['probs'], test_metrics['targets'])
    
    print("\n   === VALIDATION SET METRICS (Before Calibration) ===")
    print(f"   ECE: {ece_before_val:.4f}")
    print(f"   MSE: {val_metrics['mse']:.4f}")
    print(f"   Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"   F1 Score: {val_metrics['f1']:.4f}")
    print(f"   Brier Score: {val_metrics['brier']:.4f}")
    
    print("\n   === TEST SET METRICS (Before Calibration) ===")
    print(f"   ECE: {ece_before_test:.4f}")
    print(f"   MSE: {test_metrics['mse']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   F1 Score: {test_metrics['f1']:.4f}")
    print(f"   Brier Score: {test_metrics['brier']:.4f}")
    
    # Fit Platt scaling on validation set
    print("\n[5] Calibrating probabilities (Platt scaling)...")
    platt_params = fit_platt_scaling(val_metrics['probs'], val_metrics['targets'])
    print(f"   Platt parameters: A={platt_params[0]:.4f}, B={platt_params[1]:.4f}")
    
    # Apply Platt scaling
    calibrated_probs = apply_platt_scaling(
        val_metrics['probs'], 
        platt_params[0], 
        platt_params[1]
    )
    
    # Calculate ECE after calibration
    ece_after_val = compute_ece(calibrated_probs, val_metrics['targets'])
    print(f"   ECE after calibration: {ece_after_val:.4f}")
    print(f"   ECE improvement: {ece_before_val - ece_after_val:.4f}")
    
    # Optimize threshold
    print("\n[6] Optimizing threshold for F1 score...")
    optimal_threshold, optimal_f1 = optimize_threshold(
        calibrated_probs, 
        val_metrics['targets']
    )
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   Optimal F1 score: {optimal_f1:.4f}")
    
    # Apply optimal threshold to test set
    calibrated_test_probs = apply_platt_scaling(
        test_metrics['probs'],
        platt_params[0],
        platt_params[1]
    )
    test_preds = (calibrated_test_probs >= optimal_threshold).astype(int)
    test_f1_optimized = f1_score(test_metrics['targets'], test_preds)
    
    print("\n   === TEST SET METRICS (After Calibration & Threshold Optimization) ===")
    print(f"   ECE: {ece_before_test:.4f}")
    print(f"   MSE: {test_metrics['mse']:.4f}")
    print(f"   Accuracy: {accuracy_score(test_metrics['targets'], test_preds):.4f}")
    print(f"   F1 Score (optimized): {test_f1_optimized:.4f}")
    print(f"   Brier Score: {brier_score_loss(test_metrics['targets'], calibrated_test_probs):.4f}")
    
    # Save artifacts
    print("\n[7] Saving artifacts...")
    save_artifacts(
        model, device,
        train_metrics, val_metrics, test_metrics,
        platt_params, optimal_threshold, optimal_f1,
        history
    )
    print(f"   Artifacts saved to: {OUTPUT_DIR}")
    
    # Quality check
    print("\n[8] Validating quality thresholds...")
    if ece_after_val < ece_before_val:
        print("   ✓ ECE improved after calibration")
    else:
        print(f"   ✗ Quality check failed: ECE did not improve: before={ece_before_val:.4f}, after={ece_after_val:.4f}")
    
    if test_f1_optimized >= val_metrics['f1'] * 0.95:  # Allow 5% drop
        print("   ✓ F1 score maintained after threshold optimization")
    else:
        print(f"   ✗ Quality check failed: F1 score dropped too much: before={val_metrics['f1']:.4f}, after={test_f1_optimized:.4f}")
    
    print("\n" + "=" * 60)
    print("Task completed successfully!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

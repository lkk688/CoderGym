"""
Naive Bayes Production Inference Pipeline
Implements ML Task: NB (Production Inference Pipeline)
- Serialize model stats + fast batched inference
- Include latency benchmark
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "nb_production_inference",
        "task_type": "classification",
        "description": "Naive Bayes Production Inference Pipeline with batched inference and latency benchmark",
        "input_type": "float32",
        "output_type": "int64",
        "model_type": "GaussianNB"
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    """Get computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(n_samples=1000, n_features=10, n_classes=3, batch_size=32, val_ratio=0.2):
    """
    Create synthetic classification dataset and dataloaders.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_classes: Number of classes
        batch_size: Batch size for dataloaders
        val_ratio: Validation split ratio
    
    Returns:
        train_loader, val_loader, test_loader, input_dim, n_classes
    """
    # Generate synthetic data with distinct clusters for NB
    X = np.random.randn(n_samples, n_features)
    
    # Create class-dependent means to make classification easier
    class_means = np.random.randn(n_classes, n_features) * 3
    y = np.random.randint(0, n_classes, n_samples)
    
    # Add class-specific bias to features
    for i in range(n_samples):
        X[i] += class_means[y[i]]
    
    # Add some noise
    X += np.random.randn(n_samples, n_features) * 0.5
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=val_ratio, random_state=42, stratify=y
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, n_features, n_classes

class NaiveBayesModel(nn.Module):
    """
    PyTorch wrapper for GaussianNB that supports batched inference.
    Uses sklearn's GaussianNB internally but provides PyTorch interface.
    """
    def __init__(self, n_features, n_classes):
        super(NaiveBayesModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.sklearn_nb = GaussianNB()
        self.is_trained = False
        
    def forward(self, x):
        """Forward pass for inference."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference")
        
        # Ensure input is on CPU for sklearn compatibility
        if x.is_cuda:
            x = x.cpu()
        
        # Convert to numpy for sklearn
        x_np = x.detach().numpy()
        
        # Get predictions
        predictions = self.sklearn_nb.predict(x_np)
        
        # Get probabilities
        probabilities = self.sklearn_nb.predict_proba(x_np)
        
        return torch.FloatTensor(predictions), torch.FloatTensor(probabilities)
    
    def fit(self, X, y):
        """Train the Naive Bayes model."""
        if X.is_cuda:
            X = X.cpu()
        X_np = X.numpy()
        y_np = y.numpy()
        
        self.sklearn_nb.fit(X_np, y_np)
        self.is_trained = True
        
        # Store model stats for serialization
        self.model_stats = {
            'theta': self.sklearn_nb.theta_.tolist() if hasattr(self.sklearn_nb, 'theta_') else None,
            'var': self.sklearn_nb.var_.tolist() if hasattr(self.sklearn_nb, 'var_') else None,
            'class_prior': self.sklearn_nb.class_prior_.tolist() if hasattr(self.sklearn_nb, 'class_prior_') else None,
            'classes': self.sklearn_nb.classes_.tolist() if hasattr(self.sklearn_nb, 'classes_') else None,
            'n_features': self.n_features,
            'n_classes': self.n_classes
        }
        
        return self
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if X.is_cuda:
            X = X.cpu()
        X_np = X.numpy()
        return self.sklearn_nb.predict_proba(X_np)

def build_model(input_dim, n_classes, device):
    """
    Build the Naive Bayes model.
    
    Args:
        input_dim: Number of input features
        n_classes: Number of classes
        device: Computation device
    
    Returns:
        model: NaiveBayesModel instance
    """
    model = NaiveBayesModel(input_dim, n_classes)
    model.device = device
    return model

def train(model, train_loader, device, epochs=50, lr=0.001):
    """
    Train the Naive Bayes model.
    
    Args:
        model: NaiveBayesModel instance
        train_loader: Training dataloader
        device: Computation device
        epochs: Number of epochs (for compatibility, NB doesn't use epochs)
        lr: Learning rate (not used for NB)
    
    Returns:
        model: Trained model
    """
    print("Training Naive Bayes model...")
    
    # Collect all training data
    X_train = []
    y_train = []
    
    for batch_X, batch_y in train_loader:
        X_train.append(batch_X)
        y_train.append(batch_y)
    
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    
    # Train the model
    model.fit(X_train, y_train)
    
    print(f"Training completed. Classes: {model.model_stats['classes']}")
    
    return model

def evaluate(model, data_loader, device):
    """
    Evaluate the model and compute metrics.
    
    Args:
        model: NaiveBayesModel instance
        data_loader: DataLoader for evaluation
        device: Computation device
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Get predictions
            predictions, probabilities = model(batch_X)
            
            all_predictions.append(predictions)
            all_targets.append(batch_y)
            all_probabilities.append(probabilities)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    all_probabilities = torch.cat(all_probabilities, dim=0).cpu().numpy()
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_targets, all_predictions),
        'precision_macro': precision_score(all_targets, all_predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(all_targets, all_predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(all_targets, all_predictions, average='macro', zero_division=0),
        'mse': mean_squared_error(all_targets, all_predictions),
        'r2': r2_score(all_targets, all_predictions)
    }
    
    return metrics

def predict(model, X, device):
    """
    Perform inference on input data.
    
    Args:
        model: Trained NaiveBayesModel instance
        X: Input tensor
        device: Computation device
    
    Returns:
        predictions: Model predictions
        probabilities: Prediction probabilities
    """
    model.eval()
    
    with torch.no_grad():
        # Ensure input is on device
        X = X.to(device)
        predictions, probabilities = model(X)
    
    return predictions, probabilities

def save_artifacts(model, metrics, output_dir="output"):
    """
    Save model artifacts and metrics.
    
    Args:
        model: Trained model instance
        metrics: Dictionary of evaluation metrics
        output_dir: Output directory path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters
    model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        'model_state_dict': {
            'n_features': model.n_features,
            'n_classes': model.n_classes,
            'is_trained': model.is_trained,
            'model_stats': model.model_stats
        },
        'sklearn_nb': model.sklearn_nb
    }, model_path)
    
    # Save model stats as JSON
    stats_path = os.path.join(output_dir, "model_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(model.model_stats, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")

def benchmark_latency(model, device, n_samples=1000, batch_sizes=[1, 8, 32, 128]):
    """
    Benchmark inference latency and throughput.
    
    Args:
        model: Trained model instance
        device: Computation device
        n_samples: Number of samples to benchmark
        batch_sizes: List of batch sizes to test
    
    Returns:
        benchmarks: Dictionary of benchmark results
    """
    print("\n" + "="*60)
    print("LATENCY BENCHMARK")
    print("="*60)
    
    benchmarks = {}
    
    # Generate random test data
    X_bench = torch.randn(n_samples, model.n_features).to(device)
    
    for batch_size in batch_sizes:
        if batch_size > n_samples:
            continue
            
        # Warm-up
        for _ in range(10):
            _ = model(X_bench[:batch_size])
        
        # Benchmark
        start_time = time.time()
        n_batches = n_samples // batch_size
        
        for i in range(n_batches):
            _ = model(X_bench[i*batch_size:(i+1)*batch_size])
        
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = n_samples / total_time
        
        benchmarks[batch_size] = {
            'batch_size': batch_size,
            'total_time_ms': total_time * 1000,
            'samples_per_second': throughput,
            'latency_ms': total_time * 1000 / n_batches
        }
        
        print(f"Batch Size: {batch_size:4d} | "
              f"Latency: {total_time*1000/n_batches:.2f} ms/batch | "
              f"Throughput: {throughput:.1f} samples/sec")
    
    return benchmarks

def load_model(model_path):
    """
    Load a saved model.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        model: Loaded NaiveBayesModel instance
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = NaiveBayesModel(
        checkpoint['model_state_dict']['n_features'],
        checkpoint['model_state_dict']['n_classes']
    )
    model.is_trained = checkpoint['model_state_dict']['is_trained']
    model.model_stats = checkpoint['model_state_dict']['model_stats']
    model.sklearn_nb = checkpoint['sklearn_nb']
    
    return model

def main():
    """Main function to run the NB production inference pipeline."""
    print("="*60)
    print("NAIVE BAYES PRODUCTION INFERENCE PIPELINE")
    print("="*60)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Set seed
    set_seed(42)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, input_dim, n_classes = make_dataloaders(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        batch_size=32,
        val_ratio=0.2
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Input dimensions: {input_dim}")
    print(f"Number of classes: {n_classes}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(input_dim, n_classes, device)
    print(f"Model: NaiveBayesModel(input_dim={input_dim}, n_classes={n_classes})")
    
    # Train model
    print("\nTraining model...")
    model = train(model, train_loader, device, epochs=50)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print("Train Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print("Test Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Run latency benchmark
    benchmarks = benchmark_latency(model, device)
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'benchmarks': benchmarks
    })
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nTrain Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"Test Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"\nTrain F1 Macro:  {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:    {val_metrics['f1_macro']:.4f}")
    print(f"Test F1 Macro:   {test_metrics['f1_macro']:.4f}")
    
    # Quality checks
    print("\n" + "="*60)
    print("QUALITY CHECKS")
    print("="*60)
    
    quality_passed = True
    
    # Check 1: Train accuracy > 0.85
    check1 = train_metrics['accuracy'] > 0.85
    status1 = "✓" if check1 else "✗"
    print(f"{status1} Train Accuracy > 0.85: {train_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check1
    
    # Check 2: Val accuracy > 0.80
    check2 = val_metrics['accuracy'] > 0.80
    status2 = "✓" if check2 else "✗"
    print(f"{status2} Val Accuracy > 0.80: {val_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check2
    
    # Check 3: Val F1 macro > 0.85
    check3 = val_metrics['f1_macro'] > 0.85
    status3 = "✓" if check3 else "✗"
    print(f"{status3} Val F1 Macro > 0.85: {val_metrics['f1_macro']:.4f}")
    quality_passed = quality_passed and check3
    
    # Check 4: MSE < 1.0
    check4 = val_metrics['mse'] < 1.0
    status4 = "✓" if check4 else "✗"
    print(f"{status4} Val MSE < 1.0: {val_metrics['mse']:.4f}")
    quality_passed = quality_passed and check4
    
    # Check 5: R2 > 0.5
    check5 = val_metrics['r2'] > 0.5
    status5 = "✓" if check5 else "✗"
    print(f"{status5} Val R2 > 0.5: {val_metrics['r2']:.4f}")
    quality_passed = quality_passed and check5
    
    print("\n" + "="*60)
    if quality_passed:
        print("ALL QUALITY CHECKS PASSED ✓")
    else:
        print("SOME QUALITY CHECKS FAILED ✗")
    print("="*60)
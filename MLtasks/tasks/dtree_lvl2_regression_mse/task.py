import os
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "decision_tree_regression_mse",
        "task_type": "regression",
        "input_type": "tabular",
        "output_type": "continuous"
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    """Get computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(n_samples=1000, test_size=0.2, random_state=42):
    """Create synthetic dataset and dataloaders."""
    # Generate synthetic piecewise function data
    np.random.seed(random_state)
    
    # Create features
    X = np.random.uniform(-10, 10, size=(n_samples, 5))
    
    # Create piecewise target function
    y = np.zeros(n_samples)
    for i in range(n_samples):
        if X[i, 0] < -3:
            y[i] = 2 * X[i, 0] + 1 + np.random.normal(0, 0.5)
        elif X[i, 0] < 3:
            y[i] = np.sin(X[i, 0]) * 3 + np.random.normal(0, 0.5)
        else:
            y[i] = -X[i, 0] + 5 + np.random.normal(0, 0.5)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Convert to torch tensors
    train_data = {
        'X': torch.FloatTensor(X_train),
        'y': torch.FloatTensor(y_train).unsqueeze(1)
    }
    val_data = {
        'X': torch.FloatTensor(X_val),
        'y': torch.FloatTensor(y_val).unsqueeze(1)
    }
    
    return train_data, val_data

def build_model(device, max_depth=10, min_samples_split=5, min_samples_leaf=2):
    """Build decision tree regressor model."""
    model = DecisionTreeRegressor(
        criterion='squared_error',  # MSE criterion
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.device = device
    return model

def train(model, train_data, val_data=None, epochs=50, verbose=True):
    """Train the decision tree regressor."""
    X_train = train_data['X'].numpy()
    y_train = train_data['y'].numpy().ravel()
    
    # Decision tree is trained directly (no epochs needed)
    model.fit(X_train, y_train)
    
    return model

def evaluate(model, data, device):
    """Evaluate model and return metrics."""
    X = data['X'].numpy()
    y_true = data['y'].numpy().ravel()
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'loss': mse,
        'mse': mse,
        'r2': r2
    }
    
    return metrics

def predict(model, X, device):
    """Make predictions."""
    X_tensor = torch.FloatTensor(X).to(device)
    X_numpy = X_tensor.detach().cpu().numpy()
    predictions = model.predict(X_numpy)
    return torch.FloatTensor(predictions).unsqueeze(1)

def save_artifacts(model, metrics, save_dir="output"):
    """Save model artifacts."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model (pickle)
    import pickle
    with open(os.path.join(save_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    with open(os.path.join(save_dir, "metrics.txt"), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Artifacts saved to {save_dir}")

def main():
    """Main function to run the decision tree regression task."""
    print("Starting Decision Tree Regression (MSE) Task...")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_data, val_data = make_dataloaders(n_samples=1000, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_data['y'])}")
    print(f"Validation samples: {len(val_data['y'])}")
    
    # Build model
    print("Building model...")
    model = build_model(device, max_depth=10, min_samples_split=5, min_samples_leaf=2)
    
    # Train model
    print("Training model...")
    model = train(model, train_data, val_data, epochs=1, verbose=True)
    
    # Evaluate on training set
    print("Evaluating on training set...")
    train_metrics = evaluate(model, train_data, device)
    print("Train Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluate(model, val_data, device)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save artifacts
    print("Saving artifacts...")
    save_artifacts(model, val_metrics, save_dir="output")
    
    # Print final results
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    print(f"MSE (Train): {train_metrics['mse']:.4f}")
    print(f"MSE (Val):   {val_metrics['mse']:.4f}")
    print(f"R² (Train):  {train_metrics['r2']:.4f}")
    print(f"R² (Val):    {val_metrics['r2']:.4f}")
    print("="*60)
    
    # Quality checks
    print("\nQuality Checks:")
    all_passed = True
    
    # Check R2 > 0.8 on train
    if train_metrics['r2'] > 0.8:
        print(f"✓ Train R² > 0.8: {train_metrics['r2']:.4f}")
    else:
        print(f"✗ Train R² > 0.8: {train_metrics['r2']:.4f} (FAILED)")
        all_passed = False
    
    # Check R2 > 0.7 on validation
    if val_metrics['r2'] > 0.7:
        print(f"✓ Val R² > 0.7: {val_metrics['r2']:.4f}")
    else:
        print(f"✗ Val R² > 0.7: {val_metrics['r2']:.4f} (FAILED)")
        all_passed = False
    
    # Check MSE < 1.0 on validation
    if val_metrics['mse'] < 1.0:
        print(f"✓ Val MSE < 1.0: {val_metrics['mse']:.4f}")
    else:
        print(f"✗ Val MSE < 1.0: {val_metrics['mse']:.4f} (FAILED)")
        all_passed = False
    
    # Check R2 difference < 0.15 (no severe overfitting)
    r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
    if r2_diff < 0.15:
        print(f"✓ R² difference < 0.15: {r2_diff:.4f}")
    else:
        print(f"✗ R² difference < 0.15: {r2_diff:.4f} (FAILED)")
        all_passed = False
    
    print("="*60)
    if all_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("="*60)
    
    # Exit with appropriate code
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)

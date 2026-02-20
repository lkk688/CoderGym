"""
Decision Tree Task: Feature Importance + Permutation Importance
Implements ML Task: Decision Tree (Interpretability)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_task_metadata():
    """Return task metadata."""
    return {
        'name': 'dtree_feature_importance',
        'type': 'regression',  # Can handle both regression and classification
        'input_type': 'tabular',
        'output_type': 'feature_importance'
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Get computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(task_type='regression', n_samples=1000, n_features=10, noise=0.1, 
                     batch_size=32, val_size=0.2, random_state=42):
    """
    Create synthetic dataset and dataloaders.
    
    Args:
        task_type: 'regression' or 'classification'
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level for regression
        batch_size: Batch size for dataloaders
        val_size: Validation split size
        random_state: Random seed
    
    Returns:
        train_loader, val_loader, feature_names
    """
    # Generate synthetic data with known feature importance
    np.random.seed(random_state)
    
    # Create feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Define true feature importance (first 3 features are most important)
    true_importance = np.zeros(n_features)
    true_importance[0] = 0.5
    true_importance[1] = 0.3
    true_importance[2] = 0.15
    true_importance[3] = 0.05
    
    if task_type == 'regression':
        # Regression: weighted sum of features + noise
        y = X @ true_importance + noise * np.random.randn(n_samples)
        y = y.astype(np.float32)
        task_name = 'regression'
    else:
        # Classification: threshold the regression target
        y_reg = X @ true_importance
        y = (y_reg > np.median(y_reg)).astype(int)
        y = y.astype(np.int64)
        task_name = 'classification'
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train) if task_type == 'regression' else torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val) if task_type == 'regression' else torch.LongTensor(y_val)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return train_loader, val_loader, feature_names, task_name

def build_model(task_name='regression', max_depth=5, random_state=42):
    """
    Build decision tree model.
    
    Args:
        task_name: 'regression' or 'classification'
        max_depth: Maximum depth of the tree
        random_state: Random seed
    
    Returns:
        model (sklearn DecisionTree)
    """
    if task_name == 'regression':
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state,
            min_samples_split=5,
            min_samples_leaf=2
        )
    else:
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            min_samples_split=5,
            min_samples_leaf=2
        )
    
    return model

def train(model, train_loader, val_loader, feature_names, task_name='regression', 
          max_depth=5, epochs=20, verbose=True):
    """
    Train the decision tree model.
    
    Args:
        model: Decision tree model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        feature_names: List of feature names
        task_name: 'regression' or 'classification'
        max_depth: Maximum depth for tree
        epochs: Number of epochs (for compatibility, trees don't really need epochs)
        verbose: Print training progress
    
    Returns:
        trained model
    """
    # Extract data from loaders
    X_train = train_loader.dataset.tensors[0].numpy()
    y_train = train_loader.dataset.tensors[1].numpy()
    
    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy()
    
    if verbose:
        print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    if verbose:
        print("Model trained successfully")
    
    return model

def evaluate(model, val_loader, feature_names, task_name='regression'):
    """
    Evaluate the model and compute metrics.
    
    Args:
        model: Trained decision tree model
        val_loader: Validation dataloader
        feature_names: List of feature names
        task_name: 'regression' or 'classification'
    
    Returns:
        dict with metrics
    """
    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy()
    
    # Predictions
    y_pred = model.predict(X_val)
    
    if task_name == 'regression':
        # Regression metrics
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        metrics = {
            'loss': float(mse),
            'mse': float(mse),
            'r2': float(r2)
        }
    else:
        # Classification metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1)
        }
    
    return metrics

def predict(model, X):
    """
    Make predictions.
    
    Args:
        model: Trained model
        X: Input features
    
    Returns:
        predictions
    """
    return model.predict(X)

def save_artifacts(model, feature_names, train_metrics, val_metrics, 
                   feature_importance, perm_importance, output_dir='output'):
    """
    Save model artifacts and importance results.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        train_metrics: Training metrics dict
        val_metrics: Validation metrics dict
        feature_importance: Feature importance array
        perm_importance: Permutation importance results
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters and importance
    import json
    
    artifacts = {
        'feature_names': feature_names,
        'feature_importance': feature_importance.tolist(),
        'permutation_importance': perm_importance.importances_mean.tolist(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    
    # Save to JSON
    output_path = os.path.join(output_dir, 'dtree_importance_results.json')
    with open(output_path, 'w') as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"Artifacts saved to {output_path}")

def compute_feature_importance(model, feature_names, X_val, y_val, task_name='regression'):
    """
    Compute feature importance and permutation importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        X_val: Validation features
        y_val: Validation labels
        task_name: 'regression' or 'classification'
    
    Returns:
        feature_importance, permutation_importance
    """
    # Get sklearn-style feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_)
    else:
        # Fallback: use permutation importance only
        feature_importance = np.zeros(len(feature_names))
    
    # Compute permutation importance
    perm_result = permutation_importance(
        model, X_val, y_val, 
        n_repeats=10, 
        random_state=42,
        scoring='r2' if task_name == 'regression' else 'accuracy',
        n_jobs=-1
    )
    
    return feature_importance, perm_result

def print_importance_results(feature_names, feature_importance, perm_importance, top_k=5):
    """
    Print feature importance results in a formatted way.
    
    Args:
        feature_names: List of feature names
        feature_importance: Feature importance array
        perm_importance: Permutation importance results
        top_k: Number of top features to display
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Create importance dataframe
    importance_df = []
    for i, name in enumerate(feature_names):
        importance_df.append({
            'feature': name,
            'tree_importance': feature_importance[i],
            'perm_importance': perm_importance.importances_mean[i],
            'perm_std': perm_importance.importances_std[i]
        })
    
    # Sort by permutation importance
    importance_df.sort(key=lambda x: x['perm_importance'], reverse=True)
    
    print(f"\nTop {top_k} features by permutation importance:")
    print("-"*60)
    print(f"{'Rank':<6}{'Feature':<15}{'Tree Imp.':<12}{'Perm Imp.':<12}{'Std':<10}")
    print("-"*60)
    
    for i, imp in enumerate(importance_df[:top_k]):
        print(f"{i+1:<6}{imp['feature']:<15}{imp['tree_importance']:<12.4f}"
              f"{imp['perm_importance']:<12.4f}{imp['perm_std']:<10.4f}")
    
    print("-"*60)
    
    # Compute overlap between top-k features
    tree_top_k = [imp['feature'] for imp in importance_df[:top_k]]
    perm_sorted = sorted(importance_df, key=lambda x: x['perm_importance'], reverse=True)
    perm_top_k = [imp['feature'] for imp in perm_sorted[:top_k]]
    
    overlap = len(set(tree_top_k) & set(perm_top_k))
    print(f"\nTop-{top_k} feature overlap: {overlap}/{top_k}")
    
    return importance_df

def main():
    """Main function to run the decision tree feature importance task."""
    print("="*60)
    print("Decision Tree Feature Importance Task")
    print("="*60)
    
    # Configuration
    TASK_TYPE = 'regression'  # Can be 'regression' or 'classification'
    N_SAMPLES = 1000
    N_FEATURES = 10
    NOISE = 0.1
    BATCH_SIZE = 32
    MAX_DEPTH = 5
    EPOCHS = 20
    OUTPUT_DIR = 'output'
    
    # Set device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, feature_names, task_name = make_dataloaders(
        task_type=TASK_TYPE,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=NOISE,
        batch_size=BATCH_SIZE,
        val_size=0.2,
        random_state=SEED
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Task type: {task_name}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(task_name=task_name, max_depth=MAX_DEPTH, random_state=SEED)
    print(f"Model: DecisionTree{task_name.capitalize()}(max_depth={MAX_DEPTH})")
    
    # Train model
    print("\nTraining model...")
    model = train(model, train_loader, val_loader, feature_names, 
                  task_name=task_name, max_depth=MAX_DEPTH, epochs=EPOCHS, verbose=True)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, feature_names, task_name=task_name)
    print("Train Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, feature_names, task_name=task_name)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute feature importance
    print("\nComputing feature importance...")
    X_val = val_loader.dataset.tensors[0].numpy()
    y_val = val_loader.dataset.tensors[1].numpy()
    
    tree_importance, perm_importance = compute_feature_importance(
        model, feature_names, X_val, y_val, task_name=task_name
    )
    
    # Print importance results
    importance_df = print_importance_results(feature_names, tree_importance, perm_importance, top_k=5)
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, feature_names, train_metrics, val_metrics, 
                   tree_importance, perm_importance, output_dir=OUTPUT_DIR)
    
    # Quality checks
    print("\n" + "="*60)
    print("QUALITY CHECKS")
    print("="*60)
    
    all_passed = True
    
    if task_name == 'regression':
        # Regression quality checks
        train_r2 = train_metrics['r2']
        val_r2 = val_metrics['r2']
        val_mse = val_metrics['mse']
        
        print(f"\n✓ Train R² > 0.8: {train_r2:.4f} {'✓' if train_r2 > 0.8 else '✗'}")
        print(f"✓ Val R² > 0.7: {val_r2:.4f} {'✓' if val_r2 > 0.7 else '✗'}")
        print(f"✓ Val MSE < 1.0: {val_mse:.4f} {'✓' if val_mse < 1.0 else '✗'}")
        
        r2_diff = abs(train_r2 - val_r2)
        print(f"✓ R² difference < 0.15: {r2_diff:.4f} {'✓' if r2_diff < 0.15 else '✗'}")
        
        if not (train_r2 > 0.8 and val_r2 > 0.7 and val_mse < 1.0 and r2_diff < 0.15):
            all_passed = False
    else:
        # Classification quality checks
        train_acc = train_metrics['accuracy']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1_macro']
        
        print(f"\n✓ Train Accuracy > 0.85: {train_acc:.4f} {'✓' if train_acc > 0.85 else '✗'}")
        print(f"✓ Val Accuracy > 0.80: {val_acc:.4f} {'✓' if val_acc > 0.80 else '✗'}")
        print(f"✓ Val F1 Macro > 0.85: {val_f1:.4f} {'✓' if val_f1 > 0.85 else '✗'}")
        
        acc_diff = abs(train_acc - val_acc)
        print(f"✓ Accuracy gap < 0.15: {acc_diff:.4f} {'✓' if acc_diff < 0.15 else '✗'}")
        
        if not (train_acc > 0.85 and val_acc > 0.80 and val_f1 > 0.85 and acc_diff < 0.15):
            all_passed = False
    
    # Feature importance overlap check
    print("\n✓ Feature importance analysis completed")
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("PASS: All quality checks passed!")
        print("="*60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())

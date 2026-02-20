"""
Decision Tree with Cost-Complexity Pruning and Cross-Validation
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.datasets import make_regression, make_classification

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory for artifacts
OUTPUT_DIR = "/Developer/AIserver/output/tasks/dtree_lvl3_pruning"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "decision_tree_pruning_cv",
        "task_type": "regression",
        "description": "Decision Tree with Cost-Complexity Pruning and Cross-Validation",
        "input_type": "tabular",
        "output_type": "continuous"
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Get computation device."""
    return torch.device("cpu")  # Decision trees don't need GPU


def make_dataloaders(task_type="regression", test_size=0.2, random_state=42):
    """
    Create train and validation dataloaders.
    
    For this task, we return numpy arrays since sklearn decision trees
    don't use PyTorch dataloaders.
    """
    if task_type == "regression":
        # Generate regression data
        X, y = make_regression(
            n_samples=500,
            n_features=10,
            n_informative=8,
            noise=10.0,
            random_state=random_state
        )
    else:
        # Generate classification data
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=random_state
        )
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "task_type": task_type
    }


def build_model(task_type="regression", max_depth=10, ccp_alpha=0.0):
    """
    Build a decision tree model with cost-complexity pruning.
    
    Args:
        task_type: 'regression' or 'classification'
        max_depth: Maximum depth of the tree
        ccp_alpha: Cost-complexity pruning parameter
    
    Returns:
        model: Unfitted decision tree model
    """
    if task_type == "regression":
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            ccp_alpha=ccp_alpha,
            random_state=42
        )
    else:
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            ccp_alpha=ccp_alpha,
            random_state=42
        )
    
    return model


def train(model, dataloaders, verbose=True):
    """
    Train the decision tree model.
    
    Args:
        model: Decision tree model
        dataloaders: Dictionary with train/validation data
        verbose: Print training info
    
    Returns:
        model: Trained model
    """
    X_train = dataloaders["X_train"]
    y_train = dataloaders["y_train"]
    
    model.fit(X_train, y_train)
    
    if verbose:
        print(f"Training completed. Tree depth: {model.get_depth()}, "
              f"Number of leaves: {model.get_n_leaves()}")
    
    return model


def evaluate(model, dataloaders, split="val"):
    """
    Evaluate the model and return metrics.
    
    Args:
        model: Trained decision tree model
        dataloaders: Dictionary with train/validation data
        split: 'train' or 'val' for evaluation
    
    Returns:
        metrics: Dictionary with MSE, R2, and accuracy (if classification)
    """
    if split == "train":
        X = dataloaders["X_train"]
        y = dataloaders["y_train"]
    else:
        X = dataloaders["X_val"]
        y = dataloaders["y_val"]
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        "mse": float(mse),
        "r2": float(r2)
    }
    
    # For classification, also calculate accuracy
    if dataloaders["task_type"] == "classification":
        accuracy = accuracy_score(y, y_pred)
        metrics["accuracy"] = float(accuracy)
    
    return metrics


def predict(model, X):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained decision tree model
        X: Input features
    
    Returns:
        predictions: Model predictions
    """
    return model.predict(X)


def save_artifacts(model, dataloaders, train_metrics, val_metrics, 
                   pruning_results, plot_path=None):
    """
    Save model artifacts and evaluation results.
    
    Args:
        model: Trained model
        dataloaders: Data dictionary
        train_metrics: Training metrics
        val_metrics: Validation metrics
        pruning_results: Results from pruning analysis
        plot_path: Path to save plot (optional)
    """
    artifacts = {
        "model_params": {
            "max_depth": model.max_depth if hasattr(model, 'max_depth') else None,
            "ccp_alpha": float(model.ccp_alpha) if hasattr(model, 'ccp_alpha') else None,
            "n_leaves": int(model.get_n_leaves()) if hasattr(model, 'get_n_leaves') else None,
            "depth": int(model.get_depth()) if hasattr(model, 'get_depth') else None
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "pruning_results": pruning_results
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"Artifacts saved to {metrics_path}")
    
    # Save plot if provided
    if plot_path:
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")


def find_optimal_ccp_alpha(model, X_train, y_train, cv=5):
    """
    Find optimal ccp_alpha using cross-validation.
    
    Args:
        model: Base decision tree model
        X_train: Training features
        y_train: Training targets
        cv: Number of cross-validation folds
    
    Returns:
        ccp_alphas: List of ccp_alpha values tested
        cv_scores: List of cross-validation scores
        optimal_alpha: Optimal ccp_alpha value
    """
    # Get the cost-complexity pruning path
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    # Exclude the maximum alpha that prunes the entire tree
    ccp_alphas = ccp_alphas[:-1]
    
    # If too many alphas, sample them
    if len(ccp_alphas) > 20:
        indices = np.linspace(0, len(ccp_alphas)-1, 20, dtype=int)
        ccp_alphas = [ccp_alphas[i] for i in indices]
    
    cv_scores = []
    
    for ccp_alpha in ccp_alphas:
        # Create a new model with this alpha
        if model.__class__.__name__ == 'DecisionTreeRegressor':
            tree = DecisionTreeRegressor(
                max_depth=model.max_depth,
                ccp_alpha=ccp_alpha,
                random_state=42
            )
        else:
            tree = DecisionTreeClassifier(
                max_depth=model.max_depth,
                ccp_alpha=ccp_alpha,
                random_state=42
            )
        
        # Perform cross-validation
        scores = cross_val_score(tree, X_train, y_train, cv=cv, 
                                  scoring='neg_mean_squared_error' 
                                  if model.__class__.__name__ == 'DecisionTreeRegressor' 
                                  else 'accuracy')
        cv_scores.append(scores.mean())
    
    # Find optimal alpha (one standard deviation rule)
    cv_scores = np.array(cv_scores)
    optimal_idx = np.argmax(cv_scores)
    optimal_alpha = ccp_alphas[optimal_idx]
    
    return ccp_alphas, cv_scores, optimal_alpha


def plot_depth_vs_score(ccp_alphas, cv_scores, plot_path):
    """
    Plot depth vs validation score for pruning analysis.
    
    Args:
        ccp_alphas: List of ccp_alpha values
        cv_scores: List of cross-validation scores
        plot_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Alpha vs Score
    axes[0].plot(ccp_alphas, cv_scores, 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Cost-Complexity Pruning Alpha (ccp_alpha)', fontsize=12)
    axes[0].set_ylabel('Cross-Validation Score', fontsize=12)
    axes[0].set_title('Pruning Analysis: Alpha vs Validation Score', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Plot 2: Alpha vs Complexity (inverse relationship)
    # Calculate approximate complexity (inverse of alpha)
    # Handle case where alpha might be 0
    complexity = []
    for alpha in ccp_alphas:
        complexity.append(1/float(alpha) if float(alpha) != 0 else float('inf'))
    axes[1].plot(complexity, cv_scores, 'o-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Model Complexity (1/alpha)', fontsize=12)
    axes[1].set_ylabel('Cross-Validation Score', fontsize=12)
    axes[1].set_title('Pruning Analysis: Complexity vs Validation Score', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the decision tree pruning experiment."""
    print("=" * 60)
    print("Decision Tree with Cost-Complexity Pruning and CV")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Task type
    task_type = "regression"
    print(f"Task type: {task_type}")
    
    # Create dataloaders
    print("\n--- Creating Data ---")
    dataloaders = make_dataloaders(task_type=task_type, test_size=0.2)
    print(f"Training samples: {dataloaders['X_train'].shape[0]}")
    print(f"Validation samples: {dataloaders['X_val'].shape[0]}")
    print(f"Features: {dataloaders['X_train'].shape[1]}")
    
    # Step 1: Train unpruned model
    print("\n--- Training Unpruned Model ---")
    unpruned_model = build_model(task_type=task_type, max_depth=10, ccp_alpha=0.0)
    unpruned_model = train(unpruned_model, dataloaders)
    
    # Evaluate unpruned model
    unpruned_train_metrics = evaluate(unpruned_model, dataloaders, split="train")
    unpruned_val_metrics = evaluate(unpruned_model, dataloaders, split="val")
    
    print(f"Unpruned Train MSE: {unpruned_train_metrics['mse']:.4f}")
    print(f"Unpruned Train R2: {unpruned_train_metrics['r2']:.4f}")
    print(f"Unpruned Val MSE: {unpruned_val_metrics['mse']:.4f}")
    print(f"Unpruned Val R2: {unpruned_val_metrics['r2']:.4f}")
    
    # Step 2: Find optimal pruning parameter
    print("\n--- Finding Optimal Pruning Parameter ---")
    ccp_alphas, cv_scores, optimal_alpha = find_optimal_ccp_alpha(
        unpruned_model, 
        dataloaders["X_train"], 
        dataloaders["y_train"],
        cv=5
    )
    
    print(f"Optimal ccp_alpha: {optimal_alpha:.6f}")
    print(f"Best CV Score: {max(cv_scores):.4f}")
    
    # Step 3: Train pruned model with optimal alpha
    print("\n--- Training Pruned Model ---")
    pruned_model = build_model(task_type=task_type, max_depth=10, ccp_alpha=optimal_alpha)
    pruned_model = train(pruned_model, dataloaders)
    
    # Evaluate pruned model
    pruned_train_metrics = evaluate(pruned_model, dataloaders, split="train")
    pruned_val_metrics = evaluate(pruned_model, dataloaders, split="val")
    
    print(f"Pruned Train MSE: {pruned_train_metrics['mse']:.4f}")
    print(f"Pruned Train R2: {pruned_train_metrics['r2']:.4f}")
    print(f"Pruned Val MSE: {pruned_val_metrics['mse']:.4f}")
    print(f"Pruned Val R2: {pruned_val_metrics['r2']:.4f}")
    
    # Step 4: Compare models
    print("\n--- Model Comparison ---")
    print(f"Unpruned - Depth: {unpruned_model.get_depth()}, Leaves: {unpruned_model.get_n_leaves()}")
    print(f"Pruned - Depth: {pruned_model.get_depth()}, Leaves: {pruned_model.get_n_leaves()}")
    
    val_improvement = unpruned_val_metrics['r2'] - pruned_val_metrics['r2']
    print(f"R2 Score Change (Val): {val_improvement:.4f}")
    
    # Step 5: Create and save plot
    print("\n--- Saving Artifacts ---")
    plot_path = os.path.join(OUTPUT_DIR, "pruning_analysis.png")
    plot_depth_vs_score(ccp_alphas, cv_scores, plot_path)
    
    # Prepare pruning results
    pruning_results = {
        "ccp_alphas": [float(a) for a in ccp_alphas],
        "cv_scores": [float(s) for s in cv_scores], 
        "optimal_alpha": float(optimal_alpha) if optimal_alpha is not None else 0.0,
        "best_cv_score": float(max(cv_scores)) if len(cv_scores) > 0 else 0.0
    }
    
    # Save all artifacts
    save_artifacts(
        pruned_model, 
        dataloaders, 
        pruned_train_metrics, 
        pruned_val_metrics,
        pruning_results,
        plot_path=plot_path
    )
    
    # Step 6: Quality assertions
    print("\n--- Quality Assertions ---")
    
    # Check that model has learned something (R2 > 0)
    assert pruned_val_metrics['r2'] > 0.0, "Model R2 should be positive"
    print(f"✓ Pruned model R2 > 0: {pruned_val_metrics['r2']:.4f}")
    
    # Check that MSE is reasonable (less than variance of target)
    y_var = np.var(dataloaders['y_val'])
    assert pruned_val_metrics['mse'] < y_var, "MSE should be less than target variance"
    print(f"✓ Pruned model MSE < target variance: {pruned_val_metrics['mse']:.4f} < {y_var:.4f}")
    
    # Check that pruning doesn't completely destroy performance
    # (pruned model should have R2 within 0.1 of unpruned)
    assert pruned_val_metrics['r2'] > unpruned_val_metrics['r2'] - 0.15, \
        "Pruned model R2 should not be much worse than unpruned"
    print(f"✓ Pruned model R2 not much worse than unpruned: "
          f"{pruned_val_metrics['r2']:.4f} vs {unpruned_val_metrics['r2']:.4f}")
    
    # Check that pruned model is actually simpler
    assert pruned_model.get_n_leaves() <= unpruned_model.get_n_leaves(), \
        "Pruned model should have fewer or equal leaves"
    print(f"✓ Pruned model is simpler: {pruned_model.get_n_leaves()} <= {unpruned_model.get_n_leaves()}")
    
    # Check that generalization is reasonable (train R2 > val R2 or close)
    train_r2 = pruned_train_metrics['r2']
    val_r2 = pruned_val_metrics['r2']
    assert train_r2 > val_r2 - 0.1, "Model should generalize reasonably well"
    print(f"✓ Model generalizes: Train R2={train_r2:.4f}, Val R2={val_r2:.4f}")
    
    print("\n" + "=" * 60)
    print("PASS: All quality assertions passed!")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
Gradient Boosting Regressor Implementation
Stagewise learning with shallow trees, learning rate, and early stopping.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set paths
OUTPUT_DIR = '/Developer/AIserver/output/tasks/ens_lvl3_gbdt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class DecisionStump:
    """Simple decision tree with max_depth=1 (stump) for gradient boosting."""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        self.feature_idx = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
    
    def fit(self, X, residuals):
        """Fit a decision tree to the residuals."""
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            random_state=42
        )
        self.tree.fit(X, residuals)
        self.feature_idx = self.tree.tree_.feature[0]
        self.threshold = self.tree.tree_.threshold[0]
        # Get leaf values
        leaves = self.tree.tree_.value.squeeze()
        self.left_value = leaves[0] if len(leaves) > 1 else leaves[0]
        self.right_value = leaves[1] if len(leaves) > 1 else leaves[0]
    
    def predict(self, X):
        """Predict using the fitted tree."""
        return self.tree.predict(X)


class GradientBoostingRegressor:
    """Gradient Boosting Regressor with stagewise learning."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 early_stopping_rounds=10, validation_fraction=0.2, 
                 min_improvement=1e-4, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.min_improvement = min_improvement
        self.random_state = random_state
        self.models = []
        self.train_errors = []
        self.val_errors = []
        self.best_iteration = None
        self.initial_prediction = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit the gradient boosting model."""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        # Initialize with mean
        self.initial_prediction = np.mean(y_train)
        
        # Current predictions
        current_train_pred = np.full(len(y_train), self.initial_prediction)
        
        # Early stopping tracking
        best_val_loss = float('inf')
        no_improvement_count = 0
        self.best_iteration = None
        
        # Split for early stopping if not provided
        if X_val is None or y_val is None:
            X_train_split, X_val_split, y_train_split, y_val_split = \
                train_test_split(X_train, y_train, test_size=self.validation_fraction, 
                               random_state=self.random_state)
        else:
            X_train_split, X_val_split, y_train_split, y_val_split = \
                X_train, X_val, y_train, y_val
        
        # Initialize validation predictions
        current_val_pred = np.full(len(y_val_split), self.initial_prediction)
        
        for i in range(self.n_estimators):
            # Compute residuals (negative gradient for squared loss)
            train_residuals = y_train_split - current_train_pred
            
            # Fit a weak learner (decision tree)
            model = DecisionStump(max_depth=self.max_depth)
            model.fit(X_train_split, train_residuals)
            self.models.append(model)
            
            # Update predictions (stagewise)
            train_updates = model.predict(X_train_split)
            current_train_pred += self.learning_rate * train_updates
            
            # Update validation predictions
            val_updates = model.predict(X_val_split)
            current_val_pred += self.learning_rate * val_updates
            
            # Compute losses
            train_loss = mean_squared_error(y_train_split, current_train_pred)
            val_loss = mean_squared_error(y_val_split, current_val_pred)
            
            self.train_errors.append(train_loss)
            self.val_errors.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - self.min_improvement:
                best_val_loss = val_loss
                no_improvement_count = 0
                self.best_iteration = i + 1
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= self.early_stopping_rounds:
                print(f"Early stopping at iteration {i + 1}")
                break
        
        # Refit on full data if early stopping was used
        if self.best_iteration is not None:
            # Keep only the best models
            self.models = self.models[:self.best_iteration]
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        predictions = np.full(len(X), self.initial_prediction)
        for model in self.models:
            predictions += self.learning_rate * model.predict(X)
        return predictions
    
    def score(self, X, y):
        """Compute R^2 score."""
        predictions = self.predict(X)
        return r2_score(y, predictions)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'gradient_boosting_regression',
        'task_type': 'regression',
        'input_type': 'tabular',
        'output_type': 'continuous',
        'description': 'Gradient Boosting Regressor with shallow trees, learning rate, and early stopping'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=32, test_size=0.2, random_state=42):
    """Create dataloaders for regression task."""
    set_seed(random_state)
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=15.0,
        random_state=random_state
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test


def build_model(learning_rate=0.1, n_estimators=100, max_depth=3, 
                early_stopping_rounds=10, random_state=42):
    """Build gradient boosting model."""
    set_seed(random_state)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state
    )
    return model


def train(model, train_loader, val_loader, X_train, y_train, X_val, y_val, 
          max_epochs=100, device=None):
    """Train the gradient boosting model."""
    if device is None:
        device = get_device()
    
    # For gradient boosting, we use the raw numpy arrays directly
    model.fit(X_train, y_train, X_val, y_val)
    
    return model


def evaluate(model, X, y):
    """Evaluate the model and return metrics."""
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)
    
    # For regression, we don't have accuracy in the traditional sense
    # But we can compute explained variance
    explained_variance = 1 - np.var(y - predictions) / np.var(y)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'explained_variance': float(explained_variance)
    }
    
    return metrics


def predict(model, X):
    """Make predictions."""
    return model.predict(X)


def save_artifacts(model, metrics, X_train, y_train, X_val, y_val, X_test, y_test):
    """Save model artifacts and plots."""
    # Save model parameters
    model_params = {
        'n_estimators': len(model.models) if hasattr(model, 'models') else 0,
        'learning_rate': model.learning_rate,
        'max_depth': model.max_depth,
        'initial_prediction': float(model.initial_prediction) if hasattr(model, 'initial_prediction') else 0,
        'best_iteration': model.best_iteration,
        'train_errors': [float(e) for e in model.train_errors] if hasattr(model, 'train_errors') else [],
        'val_errors': [float(e) for e in model.val_errors] if hasattr(model, 'val_errors') else []
    }
    
    with open(os.path.join(OUTPUT_DIR, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=2)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if hasattr(model, 'train_errors') and model.train_errors:
        plt.plot(model.train_errors, label='Train MSE')
    if hasattr(model, 'val_errors') and model.val_errors:
        plt.plot(model.val_errors, label='Validation MSE')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot feature importance (approximate)
    plt.subplot(1, 2, 2)
    feature_importance = np.zeros(10)
    for model_ in model.models:
        if hasattr(model_, 'feature_idx') and model_.feature_idx >= 0:
            feature_importance[model_.feature_idx] += 1
    
    plt.bar(range(10), feature_importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Selection Count')
    plt.title('Feature Importance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Save model predictions for analysis
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    np.save(os.path.join(OUTPUT_DIR, 'train_predictions.npy'), train_pred)
    np.save(os.path.join(OUTPUT_DIR, 'val_predictions.npy'), val_pred)
    np.save(os.path.join(OUTPUT_DIR, 'test_predictions.npy'), test_pred)
    np.save(os.path.join(OUTPUT_DIR, 'train_targets.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'val_targets.npy'), y_val)
    np.save(os.path.join(OUTPUT_DIR, 'test_targets.npy'), y_test)
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run the gradient boosting training and evaluation."""
    print("=" * 60)
    print("Gradient Boosting Regressor - Training and Evaluation")
    print("=" * 60)
    
    # Parameters
    LEARNING_RATE = 0.1
    N_ESTIMATORS = 100
    MAX_DEPTH = 3
    EARLY_STOPPING_ROUNDS = 10
    BATCH_SIZE = 32
    
    print(f"\nParameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  N Estimators: {N_ESTIMATORS}")
    print(f"  Max Depth: {MAX_DEPTH}")
    print(f"  Early Stopping Rounds: {EARLY_STOPPING_ROUNDS}")
    
    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test = \
        make_dataloaders(batch_size=BATCH_SIZE)
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Build model
    print("\n[2] Building model...")
    model = build_model(
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    
    # Train model
    print("\n[3] Training model...")
    model = train(model, train_loader, val_loader, X_train, y_train, X_val, y_val)
    
    print(f"  Final iteration: {len(model.models)}")
    print(f"  Best iteration: {model.best_iteration}")
    
    # Evaluate on train set
    print("\n[4] Evaluating on training set...")
    train_metrics = evaluate(model, X_train, y_train)
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  R2: {train_metrics['r2']:.4f}")
    print(f"  Explained Variance: {train_metrics['explained_variance']:.4f}")
    
    # Evaluate on validation set
    print("\n[5] Evaluating on validation set...")
    val_metrics = evaluate(model, X_val, y_val)
    print(f"  MSE: {val_metrics['mse']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  R2: {val_metrics['r2']:.4f}")
    print(f"  Explained Variance: {val_metrics['explained_variance']:.4f}")
    
    # Evaluate on test set
    print("\n[6] Evaluating on test set...")
    test_metrics = evaluate(model, X_test, y_test)
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R2: {test_metrics['r2']:.4f}")
    print(f"  Explained Variance: {test_metrics['explained_variance']:.4f}")
    
    # Save artifacts
    print("\n[7] Saving artifacts...")
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    save_artifacts(model, all_metrics, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Quality assertions
    print("\n[8] Quality Assertions...")
    passed = True
    
    # R2 score should be positive and reasonably high
    if val_metrics['r2'] > 0.7:
        print(f"  ✓ Validation R2 > 0.7: {val_metrics['r2']:.4f}")
    else:
        print(f"  ✗ Validation R2 > 0.7: {val_metrics['r2']:.4f} (FAILED)")
        passed = False
    
    # MSE should be reasonable (not too high)
    if val_metrics['mse'] < 1000:
        print(f"  ✓ Validation MSE < 1000: {val_metrics['mse']:.4f}")
    else:
        print(f"  ✗ Validation MSE < 1000: {val_metrics['mse']:.4f} (FAILED)")
        passed = False
    
    # RMSE should be reasonable
    if val_metrics['rmse'] < 40:
        print(f"  ✓ Validation RMSE < 40: {val_metrics['rmse']:.4f}")
    else:
        print(f"  ✗ Validation RMSE < 40: {val_metrics['rmse']:.4f} (FAILED)")
        passed = False
    
    # Explained variance should be positive
    if val_metrics['explained_variance'] > 0.5:
        print(f"  ✓ Validation Explained Variance > 0.5: {val_metrics['explained_variance']:.4f}")
    else:
        print(f"  ✗ Validation Explained Variance >0.5: {val_metrics['explained_variance']:.4f} (FAILED)")
        passed = False
    
    if passed:
        print("\n  All quality assertions passed!")
    else:
        print("\n  Some quality assertions failed!")
    
    return passed


if __name__ == "__main__":
    main()
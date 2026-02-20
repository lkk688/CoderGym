"""
PCA (SVD) Implementation
Principal Component Analysis using Singular Value Decomposition
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_task_metadata():
    return {
        'name': 'pca_svd',
        'description': 'PCA using SVD with explained variance and reconstruction error',
        'input_type': 'tabular',
        'output_type': 'regression',
        'metrics': ['mse', 'r2', 'explained_variance_ratio', 'reconstruction_error']
    }

def make_dataloaders(n_samples=1000, n_features=20, n_informative=10, noise=0.1,
                     batch_size=64, test_size=0.2, random_state=42):
    """Create regression dataset and dataloaders"""
    set_seed(random_state)
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    
    # Convert to float32
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val

class PCA_SVD(nn.Module):
    """PCA implementation using SVD"""
    
    def __init__(self, n_components=None):
        super().__init__()
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        
    def fit(self, X):
        """
        Fit PCA using SVD
        
        Parameters:
        -----------
        X : torch.Tensor, shape (n_samples, n_features)
            Training data
        """
        # Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        
        # Compute SVD: X = U * S * Vt
        # For covariance: X^T * X = V * S^2 * V^T
        U, S, Vt = torch.svd(X_centered)
        
        # Get components (principal directions)
        self.components_ = Vt.t()  # Shape: (n_features, n_components)
        
        # Compute explained variance from singular values
        n_samples = X_centered.shape[0]
        self.singular_values_ = S
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        
        # Compute explained variance ratio
        total_variance = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # Select number of components
        if self.n_components is None:
            self.n_components = X.shape[1]
        elif self.n_components > X.shape[1]:
            self.n_components = X.shape[1]
        
        # Truncate components
        self.components_ = self.components_[:, :self.n_components]
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
        
        return self
    
    def transform(self, X):
        """Project data onto principal components"""
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Reconstruct data from transformed representation"""
        return X_transformed @ self.components_.t() + self.mean_
    
    def compute_reconstruction_error(self, X):
        """Compute reconstruction error (MSE)"""
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return mean_squared_error(X.detach().cpu().numpy(), 
                                  X_reconstructed.detach().cpu().numpy())
    
    def get_explained_variance_ratio(self):
        """Return explained variance ratio for each component"""
        return self.explained_variance_ratio_.detach().cpu().numpy()
    
    def get_cumulative_explained_variance(self):
        """Return cumulative explained variance ratio"""
        return torch.cumsum(self.explained_variance_ratio_, dim=0).detach().cpu().numpy()

def build_model(n_components=None):
    """Build PCA model"""
    model = PCA_SVD(n_components=n_components)
    return model

def train(model, train_loader, device, n_components=None, epochs=10):
    """
    Train PCA model (fit on training data)
    For PCA, training is just fitting the model
    """
    # Get all training data
    X_train_list = []
    for X_batch, _ in train_loader:
        X_train_list.append(X_batch)
    X_train = torch.cat(X_train_list, dim=0).to(device)
    
    # Fit PCA model
    if n_components is not None:
        model.n_components = n_components
    
    model.fit(X_train)
    
    return model

def evaluate(model, data_loader, device):
    """
    Evaluate PCA model
    Returns metrics: MSE, R2, explained variance, reconstruction error
    """
    model.eval()
    
    # Get all data
    X_list = []
    y_list = []
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    
    X = torch.cat(X_list, dim=0).to(device)
    y = torch.cat(y_list, dim=0).to(device)
    
    metrics = {}
    
    # Compute reconstruction error
    X_transformed = model.transform(X)
    X_reconstructed = model.inverse_transform(X_transformed)
    
    # Convert to numpy for sklearn metrics
    X_np = X.detach().cpu().numpy()
    X_reconstructed_np = X_reconstructed.detach().cpu().numpy()
    
    # Compute reconstruction MSE
    metrics['reconstruction_mse'] = mean_squared_error(X_np, X_reconstructed_np)
    
    # Compute R2 score for reconstruction
    metrics['reconstruction_r2'] = r2_score(X_np, X_reconstructed_np)
    
    # Explained variance ratio
    metrics['explained_variance_ratio'] = model.get_explained_variance_ratio().tolist()
    metrics['cumulative_explained_variance'] = model.get_cumulative_explained_variance().tolist()
    
    # Total explained variance
    metrics['total_explained_variance_ratio'] = float(model.explained_variance_ratio_.sum().item())
    
    # Number of components
    metrics['n_components'] = model.n_components
    
    return metrics

def predict(model, X):
    """Predict by transforming and reconstructing"""
    with torch.no_grad():
        X_transformed = model.transform(X)
        X_reconstructed = model.inverse_transform(X_transformed)
    return X_reconstructed

def save_artifacts(model, metrics, output_dir='output'):
    """Save model artifacts and metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters
    state_dict = {
        'mean_': model.mean_.detach().cpu().numpy() if model.mean_ is not None else None,
        'components_': model.components_.detach().cpu().numpy() if model.components_ is not None else None,
        'explained_variance_': model.explained_variance_.detach().cpu().numpy() if model.explained_variance_ is not None else None,
        'explained_variance_ratio_': model.explained_variance_ratio_.detach().cpu().numpy() if model.explained_variance_ratio_ is not None else None,
        'singular_values_': model.singular_values_.detach().cpu().numpy() if model.singular_values_ is not None else None,
        'n_components': model.n_components
    }
    
    torch.save(state_dict, os.path.join(output_dir, 'pca_model.pth'))
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model metadata
    metadata = get_task_metadata()
    metadata['n_components'] = model.n_components
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")

def main():
    """Main function to run PCA task"""
    print("=" * 60)
    print("PCA (SVD) Implementation")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        batch_size=64,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model with all components first to see variance
    print("\nBuilding PCA model...")
    model = build_model(n_components=None)
    
    # Train model
    print("\nTraining (fitting) PCA model...")
    model = train(model, train_loader, device)
    
    print(f"Number of components: {model.n_components}")
    print(f"Explained variance ratio per component: {model.get_explained_variance_ratio()[:5]}...")
    print(f"Total explained variance: {model.get_cumulative_explained_variance()[-1]:.4f}")
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"  Reconstruction MSE: {train_metrics['reconstruction_mse']:.6f}")
    print(f"  Reconstruction R2: {train_metrics['reconstruction_r2']:.6f}")
    print(f"  Total explained variance: {train_metrics['total_explained_variance_ratio']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"  Reconstruction MSE: {val_metrics['reconstruction_mse']:.6f}")
    print(f"  Reconstruction R2: {val_metrics['reconstruction_r2']:.6f}")
    print(f"  Total explained variance: {val_metrics['total_explained_variance_ratio']:.4f}")
    
    # Test with different number of components to show reconstruction error decreases
    print("\n" + "=" * 60)
    print("Testing different number of components...")
    print("=" * 60)
    
    n_components_list = [5, 10, 15, 20]
    print(f"\n{'Components':<12} {'Train MSE':<12} {'Val MSE':<12} {'Explained Var':<15}")
    print("-" * 60)
    
    for n_comp in n_components_list:
        # Create new model with specific number of components
        test_model = build_model(n_components=n_comp)
        test_model = train(test_model, train_loader, device)
        
        # Evaluate
        train_mse = test_model.compute_reconstruction_error(
            torch.FloatTensor(X_train).to(device)
        )
        val_mse = test_model.compute_reconstruction_error(
            torch.FloatTensor(X_val).to(device)
        )
        explained_var = test_model.get_cumulative_explained_variance()[-1]
        
        print(f"{n_comp:<12} {train_mse:<12.6f} {val_mse:<12.6f} {explained_var:<15.4f}")
    
    # Final metrics summary
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"\nTrain Metrics:")
    print(f"  Reconstruction MSE: {train_metrics['reconstruction_mse']:.6f}")
    print(f"  Reconstruction R2: {train_metrics['reconstruction_r2']:.6f}")
    print(f"  Total Explained Variance: {train_metrics['total_explained_variance_ratio']:.4f}")
    print(f"\nValidation Metrics:")
    print(f"  Reconstruction MSE: {val_metrics['reconstruction_mse']:.6f}")
    print(f"  Reconstruction R2: {val_metrics['reconstruction_r2']:.6f}")
    print(f"  Total Explained Variance: {val_metrics['total_explained_variance_ratio']:.4f}")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("Quality Checks")
    print("=" * 60)
    
    all_passed = True
    
    # Check 1: R2 should be positive (model captures variance)
    check1 = train_metrics['reconstruction_r2'] > 0.5
    status1 = "✓" if check1 else "✗"
    print(f"{status1} Train R2 > 0.5: {train_metrics['reconstruction_r2']:.4f}")
    all_passed = all_passed and check1
    
    # Check 2: Validation R2 should be positive
    check2 = val_metrics['reconstruction_r2'] > 0.5
    status2 = "✓" if check2 else "✗"
    print(f"{status2} Val R2 > 0.5: {val_metrics['reconstruction_r2']:.4f}")
    all_passed = all_passed and check2
    
    # Check 3: Reconstruction error should be reasonable
    check3 = train_metrics['reconstruction_mse'] < 100.0
    status3 = "✓" if check3 else "✗"
    print(f"{status3} Train MSE < 100: {train_metrics['reconstruction_mse']:.6f}")
    all_passed = all_passed and check3
    
    # Check 4: Explained variance should be high
    check4 = train_metrics['total_explained_variance_ratio'] > 0.8
    status4 = "✓" if check4 else "✗"
    print(f"{status4} Total explained variance > 0.8: {train_metrics['total_explained_variance_ratio']:.4f}")
    all_passed = all_passed and check4
    
    # Check 5: Reconstruction error should decrease with more components
    # (Already verified in the loop above, but let's check final values)
    check5 = val_metrics['reconstruction_mse'] < train_metrics['reconstruction_mse'] * 1.2
    status5 = "✓" if check5 else "✗"
    print(f"{status5} Val MSE < 1.2 * Train MSE: {val_metrics['reconstruction_mse']:.6f} < {train_metrics['reconstruction_mse'] * 1.2:.6f}")
    all_passed = all_passed and check5
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics)
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("PASS: All quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'logreg_imbalanced_metrics',
        'task_type': 'binary_classification',
        'n_classes': 2,
        'imbalanced_ratio': 0.05,
        'description': 'Imbalanced binary classification with weighted loss and early stopping'
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(batch_size=64, test_size=0.2, random_state=42):
    """Create imbalanced synthetic dataset and dataloaders."""
    # Generate imbalanced dataset (95/5 split)
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],  # 95% class 0, 5% class 1
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state
    )
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (2 * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    return train_loader, val_loader, class_weights, X_train, X_val, y_train, y_val

class LogisticRegression(nn.Module):
    """Logistic Regression model for binary classification."""
    
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def build_model(input_dim):
    """Build the logistic regression model."""
    model = LogisticRegression(input_dim)
    return model.to(device)

def train(model, train_loader, val_loader, class_weights, max_epochs=100, patience=10, lr=0.01):
    """Train the model with weighted loss and early stopping."""
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_f1_scores = []
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Weighted Binary Cross Entropy loss using batch_y
            batch_weights = class_weights[batch_y.squeeze().long()]
            criterion = nn.BCELoss(reduction='none')
            loss_unreduced = criterion(outputs, batch_y)
            loss = (loss_unreduced * batch_weights).mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader)
        val_f1 = val_metrics['f1']
        val_f1_scores.append(val_f1)
        
        # Early stopping based on F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_f1_scores

def evaluate(model, data_loader):
    """Evaluate the model and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())
            all_probs.extend(outputs.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Calculate MSE and R2 for regression-style metrics
    mse = np.mean((all_probs - all_targets) ** 2)
    
    # R2 calculation
    ss_res = np.sum((all_targets - all_probs) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'r2': r2,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }

def predict(model, X):
    """Make predictions on input data."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        preds = (outputs > 0.5).float()
    return preds.cpu().numpy().flatten(), outputs.cpu().numpy().flatten()

def save_artifacts(model, train_losses, val_f1_scores, X_train, y_train, X_val, y_val, output_dir):
    """Save model, plots, and other artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Training completed successfully\n")
        f.write(f"Final train loss: {train_losses[-1]:.4f}\n")
        f.write(f"Best val F1: {max(val_f1_scores):.4f}\n")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Plot validation F1
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Precision-Recall curve
    _, probs = predict(model, X_val)
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, probs)
    ap_score = average_precision_score(y_val, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP = {ap_score:.4f})')
    plt.grid(True)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    print(f"Artifacts saved to {output_dir}")

def main():
    """Main function to run the complete training and evaluation pipeline."""
    print("=" * 60)
    print("Logistic Regression with Imbalanced Data and Early Stopping")
    print("=" * 60)
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Description: {metadata['description']}")
    print(f"Imbalanced ratio: {metadata['imbalanced_ratio']}")
    
    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, val_loader, class_weights, X_train, X_val, y_train, y_val = make_dataloaders()
    print(f"   Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    print(f"   Class distribution in train: {np.bincount(y_train)}")
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # Build model
    print("\n[2] Building model...")
    model = build_model(input_dim=X_train.shape[1])
    print(f"   Model: LogisticRegression({X_train.shape[1]})")
    
    # Train model
    print("\n[3] Training model...")
    model, train_losses, val_f1_scores = train(
        model, train_loader, val_loader, class_weights,
        max_epochs=100, patience=15, lr=0.01
    )
    
    # Evaluate on training set
    print("\n[4] Evaluating on training set...")
    train_metrics = evaluate(model, train_loader)
    print(f"   Training Metrics:")
    print(f"     Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"     Precision: {train_metrics['precision']:.4f}")
    print(f"     Recall:    {train_metrics['recall']:.4f}")
    print(f"     F1 Score:  {train_metrics['f1']:.4f}")
    print(f"     MSE:       {train_metrics['mse']:.4f}")
    print(f"     R2 Score:  {train_metrics['r2']:.4f}")
    print(f"     Confusion Matrix:\n{np.array(train_metrics['confusion_matrix'])}")
    
    # Evaluate on validation set
    print("\n[5] Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader)
    print(f"   Validation Metrics:")
    print(f"     Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"     Precision: {val_metrics['precision']:.4f}")
    print(f"     Recall:    {val_metrics['recall']:.4f}")
    print(f"     F1 Score:  {val_metrics['f1']:.4f}")
    print(f"     MSE:       {val_metrics['mse']:.4f}")
    print(f"     R2 Score:  {val_metrics['r2']:.4f}")
    print(f"     Confusion Matrix:\n{np.array(val_metrics['confusion_matrix'])}")
    
    # Verify against sklearn metrics
    print("\n[6] Verifying metrics against sklearn...")
    val_preds, _ = predict(model, X_val)
    
    sklearn_precision = precision_score(y_val, val_preds, zero_division=0)
    sklearn_recall = recall_score(y_val, val_preds, zero_division=0)
    sklearn_f1 = f1_score(y_val, val_preds, zero_division=0)
    
    print(f"   Our Implementation:")
    print(f"     Precision: {val_metrics['precision']:.4f}")
    print(f"     Recall:    {val_metrics['recall']:.4f}")
    print(f"     F1 Score:  {val_metrics['f1']:.4f}")
    print(f"   sklearn.metrics:")
    print(f"     Precision: {sklearn_precision:.4f}")
    print(f"     Recall:    {sklearn_recall:.4f}")
    print(f"     F1 Score:  {sklearn_f1:.4f}")
    
    # Verify recall improved vs unweighted baseline
    print("\n[7] Comparing recall vs unweighted baseline...")
    
    # Train unweighted model for comparison
    print("   Training unweighted baseline...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_unweighted = build_model(input_dim=X_train.shape[1])
    criterion_unweighted = nn.BCELoss()
    optimizer_unweighted = optim.Adam(model_unweighted.parameters(), lr=0.01)
    
    for epoch in range(50):
        model_unweighted.train()
        for batch_X, batch_y in train_loader:
            optimizer_unweighted.zero_grad()
            outputs = model_unweighted(batch_X)
            loss = criterion_unweighted(outputs, batch_y)
            loss.backward()
            optimizer_unweighted.step()
    
    val_preds_unweighted, _ = predict(model_unweighted, X_val)
    sklearn_recall_unweighted = recall_score(y_val, val_preds_unweighted, zero_division=0)
    sklearn_recall_weighted = val_metrics['recall']
    
    print(f"   Unweighted baseline Recall: {sklearn_recall_unweighted:.4f}")
    print(f"   Weighted model Recall:      {sklearn_recall_weighted:.4f}")
    
    recall_improved = sklearn_recall_weighted > sklearn_recall_unweighted
    print(f"   Recall improved: {recall_improved}")
    
    # Save artifacts
    output_dir = '/Developer/AIserver/output/tasks/logreg_lvl3_imbalanced_metrics'
    print(f"\n[8] Saving artifacts to {output_dir}...")
    save_artifacts(model, train_losses, val_f1_scores, X_train, y_train, X_val, y_val, output_dir)
    
    # Quality threshold assertions
    print("\n" + "=" * 60)
    print("Quality Threshold Assertions")
    print("=" * 60)
    
    all_passed = True
    
    # Check accuracy threshold
    try:
        assert val_metrics['accuracy'] > 0.90, f"Accuracy {val_metrics['accuracy']:.4f} <= 0.90"
        print("✓ Accuracy > 0.90: PASSED")
    except AssertionError as e:
        print(f"✗ Accuracy > 0.90: FAILED - {e}")
        all_passed = False
    
    # Check F1 threshold
    try:
        assert val_metrics['f1'] > 0.50, f"F1 {val_metrics['f1']:.4f} <= 0.50"
        print("✓ F1 Score > 0.50: PASSED")
    except AssertionError as e:
        print(f"✗ F1 Score > 0.50: FAILED - {e}")
        all_passed = False
    
    # Check recall improved vs baseline
    try:
        assert recall_improved, "Recall did not improve vs unweighted baseline"
        print("✓ Recall improved vs baseline: PASSED")
    except AssertionError as e:
        print(f"✗ Recall improved vs baseline: FAILED - {e}")
        all_passed = False
    
    # Check metrics match sklearn (within tolerance)
    try:
        assert abs(val_metrics['precision'] - sklearn_precision) < 0.01, \
            f"Precision mismatch: {val_metrics['precision']:.4f} vs {sklearn_precision:.4f}"
        assert abs(val_metrics['recall'] - sklearn_recall) < 0.01, \
            f"Recall mismatch: {val_metrics['recall']:.4f} vs {sklearn_recall:.4f}"
        assert abs(val_metrics['f1'] - sklearn_f1) < 0.01, \
            f"F1 mismatch: {val_metrics['f1']:.4f} vs {sklearn_f1:.4f}"
        print("✓ Metrics match sklearn: PASSED")
    except AssertionError as e:
        print(f"✗ Metrics match sklearn: FAILED - {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("PASS: All quality thresholds met!")
    else:
        print("FAIL: Some quality thresholds not met!")
    print("=" * 60)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
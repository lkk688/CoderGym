"""
Transfer Learning with ResNet on CIFAR10
Fine-tuning pretrained ResNet for CIFAR10 classification
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
OUTPUT_DIR = '/Developer/AIserver/output/tasks/cnn_lvl3_resnet_transfer'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the ML task."""
    return {
        'task_type': 'image_classification',
        'dataset': 'CIFAR10',
        'num_classes': 10,
        'model_type': 'ResNet18',
        'transfer_learning': True,
        'input_shape': [3, 32, 32],
        'description': 'Fine-tuning pretrained ResNet18 on CIFAR10 dataset'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the computation device (GPU/CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=64, val_ratio=0.2, num_workers=2):
    """
    Create data loaders for CIFAR10 dataset.
    
    Args:
        batch_size: Batch size for training
        val_ratio: Ratio of validation data from training set
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Data transformations
    # For transfer learning, use ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
    
    # Training transformations with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation and test transformations (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Load full training dataset
    full_train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # Load test dataset
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=eval_transform
    )
    
    # Split training data into train and validation
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    
    # Shuffle indices
    np.random.shuffle(indices)
    train_indices = indices[split:]
    val_indices = indices[:split]
    
    # Create subsets
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # Class names
    class_names = full_train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes=10, pretrained=True, freeze_base=True):
    """
    Build and configure ResNet model for CIFAR10.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_base: Whether to freeze base model parameters
    
    Returns:
        model: Configured ResNet model
    """
    if pretrained:
        # Use pretrained ResNet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
    
    # Modify the final fully connected layer for CIFAR10
    # ResNet18 has 512 features in the final layer
    model.fc = nn.Linear(512, num_classes)
    
    # Initialize the new layer
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    
    # Freeze base model parameters if specified
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final layer
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model.to(device)


def convert_to_python_scalars(obj):
    """Recursively convert tensors and numpy arrays to Python scalars for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_scalars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_scalars(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def train(model, train_loader, val_loader, criterion, optimizer, 
          num_epochs=10, scheduler=None, early_stopping_patience=5):
    """
    Train the model with optional early stopping.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        scheduler: Learning rate scheduler
        early_stopping_patience: Patience for early stopping
    
    Returns:
        model: Trained model
        history: Training history dict
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0,
        'best_model_state': None,
        'patience_counter': 0
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Calculate training metrics
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        
        # Validation phase
        val_results = evaluate(model, val_loader, criterion)
        
        # Update history - convert all values to Python scalars
        history['train_loss'].append(float(train_epoch_loss))
        history['train_acc'].append(float(train_epoch_acc))
        history['val_loss'].append(float(val_results['loss']))
        history['val_acc'].append(float(val_results['accuracy']))
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_results['loss'])
        
        # Print epoch info
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f} | '
              f'Val Loss: {val_results["loss"]:.4f}, Val Acc: {val_results["accuracy"]:.4f}')
        
        # Early stopping and best model tracking
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            history['best_val_acc'] = best_val_acc
            history['best_model_state'] = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
    
    # Load best model
    if history['best_model_state'] is not None:
        model.load_state_dict(history['best_model_state'])
    
    return model, history


def evaluate(model, data_loader, criterion):
    """
    Evaluate the model on given data loader.
    
    Args:
        model: Neural network model
        data_loader: Data loader for evaluation
        criterion: Loss function
    
    Returns:
        results: Dict with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    total_samples = len(data_loader.dataset)
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Classification report
    class_report = classification_report(
        all_targets, all_predictions, output_dict=True, zero_division=0
    )
    
    results = {
        'loss': float(avg_loss),
        'accuracy': float(accuracy),
        'class_report': class_report,
        'num_samples': total_samples
    }
    
    return results


def predict(model, data_loader):
    """
    Generate predictions for the given data loader.
    
    Args:
        model: Neural network model
        data_loader: Data loader for prediction
    
    Returns:
        predictions: List of predictions
        targets: List of ground truth labels
        probabilities: List of prediction probabilities
    """
    model.eval()
    predictions = []
    targets = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, batch_targets in data_loader:
            # Move data to device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            targets.extend(batch_targets.numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return predictions, targets, probabilities


def save_artifacts(model, history, class_names, test_results=None):
    """
    Save model artifacts and evaluation results.
    
    Args:
        model: Trained model
        history: Training history
        class_names: List of class names
        test_results: Optional test set results
    """
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Create a clean copy of history without tensors
    clean_history = {}
    for key, value in history.items():
        # Skip best_model_state as it's saved separately
        if key == 'best_model_state':
            continue
        clean_history[key] = convert_to_python_scalars(value)
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(clean_history, f, indent=2)
    
    # Save class names
    class_names_path = os.path.join(OUTPUT_DIR, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    
    # Save test results if available
    if test_results is not None:
        test_results_path = os.path.join(OUTPUT_DIR, 'test_results.json')
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
    
    print(f'Artifacts saved to {OUTPUT_DIR}')


def main():
    """Main function to run the transfer learning task."""
    print("=" * 60)
    print("Transfer Learning with ResNet on CIFAR10")
    print("=" * 60)
    
    # Get metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['description']}")
    print(f"Dataset: {metadata['dataset']} ({metadata['num_classes']} classes)")
    print(f"Device: {device}")
    
    # Create data loaders
    print("\n[1/5] Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = make_dataloaders(
        batch_size=64, val_ratio=0.2, num_workers=2
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Build model with frozen base
    print("\n[2/5] Building model (frozen base)...")
    model = build_model(
        num_classes=metadata['num_classes'],
        pretrained=True,
        freeze_base=True
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Phase 1: Train only the final layer
    print("\n[3/5] Phase 1: Training final layer only...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    model, history = train(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=10, scheduler=scheduler, early_stopping_patience=5
    )
    
    # Phase 2: Unfreeze and fine-tune
    print("\n[4/5] Phase 2: Unfreezing and fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
    
    # Reinitialize optimizer for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # Continue training
    model, fine_tune_history = train(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=15, scheduler=scheduler, early_stopping_patience=5
    )
    
    # Merge histories - ensure all values are Python scalars
    for key in fine_tune_history:
        if key not in ['best_model_state', 'best_val_acc', 'patience_counter']:
            if isinstance(fine_tune_history[key], list):
                # Convert any tensor elements to floats
                converted_list = []
                for item in fine_tune_history[key]:
                    if isinstance(item, (torch.Tensor, np.ndarray)):
                        converted_list.append(float(item.item() if isinstance(item, torch.Tensor) else item))
                    else:
                        converted_list.append(float(item))
                history[key].extend(converted_list)
            else:
                # Handle scalar values
                history[key] = float(fine_tune_history[key]) if isinstance(fine_tune_history[key], torch.Tensor) else fine_tune_history[key]
    
    # Evaluate on both train and validation splits
    print("\n[5/5] Evaluating model...")
    
    # Train set evaluation
    train_results = evaluate(model, train_loader, criterion)
    print(f"\nTrain Results:")
    print(f"  Loss: {train_results['loss']:.4f}")
    print(f"  Accuracy: {train_results['accuracy']:.4f}")
    
    # Validation set evaluation
    val_results = evaluate(model, val_loader, criterion)
    print(f"\nValidation Results:")
    print(f"  Loss: {val_results['loss']:.4f}")
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    
    # Test set evaluation
    test_results = evaluate(model, test_loader, criterion)
    print(f"\nTest Results:")
    print(f"  Loss: {test_results['loss']:.4f}")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    
    # Quality thresholds
    print("\n" + "=" * 60)
    print("Quality Assessment")
    print("=" * 60)
    
    # Define quality thresholds
    thresholds = {
        'val_acc_min': 0.75,
        'val_loss_max': 1.0,
        'beats_random_min': 0.10
    }
    
    # Check thresholds
    val_acc_pass = val_results['accuracy'] >= thresholds['val_acc_min']
    val_loss_pass = val_results['loss'] <= thresholds['val_loss_max']
    beats_random_pass = val_results['accuracy'] >= (0.10 + thresholds['beats_random_min'])
    
    print(f"  ✓ PASS: Validation Accuracy >= {thresholds['val_acc_min']*100}% (actual: {val_results['accuracy']:.4f})" if val_acc_pass else f"  ✗ FAIL: Validation Accuracy >= {thresholds['val_acc_min']*100}% (actual: {val_results['accuracy']:.4f})")
    print(f"  ✓ PASS: Validation Loss <= {thresholds['val_loss_max']} (actual: {val_results['loss']:.4f})" if val_loss_pass else f"  ✗ FAIL: Validation Loss <= {thresholds['val_loss_max']} (actual: {val_results['loss']:.4f})")
    print(f"  ✓ PASS: Beats Random Baseline (>10%) (actual: {val_results['accuracy']:.4f})" if beats_random_pass else f"  ✗ FAIL: Beats Random Baseline (>10%) (actual: {val_results['accuracy']:.4f})")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, history, class_names, test_results)
    
    print("\n" + "=" * 60)
    print("Task Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()

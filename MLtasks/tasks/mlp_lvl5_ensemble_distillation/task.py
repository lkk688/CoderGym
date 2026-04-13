"""
Knowledge Distillation: Ensemble Teacher -> Student

Mathematical Formulation:
- Distillation Loss: L = alpha*CE(student, target) + (1-alpha)*KL(p_s, p_t)
  where p_s = softmax(z_s/T), p_t = softmax(z_t/T)
  T is temperature (typically 3-20)
- Teacher Ensemble: Average logits from N independent trained teachers
- Student Model: Compact architecture with fewer parameters that learns to mimic ensemble

This task demonstrates:
1. Training multiple teacher models independently
2. Creating an ensemble predictor (average logits)
3. Knowledge distillation with temperature scaling
4. Measuring distillation efficiency and student compression
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import json
from pathlib import Path

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory for artifacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'mlp_lvl5_ensemble_distillation')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'knowledge_distillation_ensemble',
        'description': 'Train ensemble of teacher MLPs and distill into compact student',
        'n_teachers': 3,
        'teacher_architecture': '784->256->128->10',
        'student_architecture': '784->128->64->10',
        'temperature': 4,
        'alpha': 0.5,
        'task_type': 'image_classification',
        'dataset': 'MNIST'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=32, download=True):
    """
    Create MNIST dataloaders for train and validation.
    
    Args:
        batch_size: Batch size
        download: Whether to download MNIST
    
    Returns:
        train_loader, val_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=download, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=download, transform=transform
    )
    
    # Split training into 80/20 train/val
    n_train = int(0.8 * len(mnist_train))
    indices = torch.randperm(len(mnist_train))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = torch.utils.data.Subset(mnist_train, train_indices)
    val_subset = torch.utils.data.Subset(mnist_train, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class TeacherMLP(nn.Module):
    """Teacher MLP: 784 -> 256 -> 128 -> 10"""
    
    def __init__(self, dropout=0.2):
        super(TeacherMLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class StudentMLP(nn.Module):
    """Student MLP: 784 -> 128 -> 64 -> 10 (Compact)"""
    
    def __init__(self, dropout=0.2):
        super(StudentMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def build_model(model_type='teacher', **kwargs):
    """Build a model (teacher or student)."""
    if model_type == 'teacher':
        return TeacherMLP(**kwargs)
    elif model_type == 'student':
        return StudentMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_teacher(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    """
    Train a single teacher model to high accuracy.
    
    Args:
        model: Teacher model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        dict with training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Teacher Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return history


def train_with_distillation(student, teachers, train_loader, val_loader, device,
                            temperature=4, alpha=0.5, epochs=20, lr=0.001):
    """
    Train student model with knowledge distillation from ensemble of teachers.
    
    Distillation Loss: L = alpha*CE(student, target) + (1-alpha)*KL(p_s, p_t)
    
    Args:
        student: Student model
        teachers: List of teacher models (all in eval mode)
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        temperature: Temperature for softmax scaling
        alpha: Weight for distillation loss (1-alpha for CE loss)
        epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        dict with training history
    """
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student.parameters(), lr=lr)
    
    # Set teachers to eval mode
    for teacher in teachers:
        teacher.eval()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        student.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get teacher ensemble predictions (average logits)
            teacher_logits = []
            with torch.no_grad():
                for teacher in teachers:
                    logits = teacher(images)
                    teacher_logits.append(logits)
            teacher_logits_ensemble = torch.mean(torch.stack(teacher_logits), dim=0)
            
            # Student prediction
            student_logits = student(images)
            
            # Distillation loss
            ce_loss = criterion_ce(student_logits, labels)
            
            # KL divergence loss (using temperature)
            p_student = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
            p_teacher = torch.nn.functional.softmax(teacher_logits_ensemble / temperature, dim=1)
            kl_loss = criterion_kl(p_student, p_teacher)
            
            # Combined loss
            loss = alpha * ce_loss + (1 - alpha) * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        student.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Ensemble prediction
                teacher_logits = []
                for teacher in teachers:
                    logits = teacher(images)
                    teacher_logits.append(logits)
                teacher_logits_ensemble = torch.mean(torch.stack(teacher_logits), dim=0)
                
                # Student prediction
                student_logits = student(images)
                
                # Loss
                ce_loss = criterion_ce(student_logits, labels)
                p_student = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
                p_teacher = torch.nn.functional.softmax(teacher_logits_ensemble / temperature, dim=1)
                kl_loss = criterion_kl(p_student, p_teacher)
                loss = alpha * ce_loss + (1 - alpha) * kl_loss
                
                val_loss += loss.item()
                _, predicted = torch.max(student_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Distillation Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return history


def train(train_loader, val_loader, device, model_type='teacher', **kwargs):
    """
    Train wrapper function for compatibility with protocol.
    """
    model = build_model(model_type=model_type)
    history = train_teacher(model, train_loader, val_loader, device)
    return model, history


def evaluate(model, data_loader, device, teachers=None, temperature=4, return_dict=True):
    """
    Evaluate model on data loader.
    
    Computes: accuracy, CE loss, and distillation efficiency metrics.
    
    Args:
        model: Model to evaluate (student or teacher)
        data_loader: Data loader
        device: Device
        teachers: Optional list of teachers for ensemble comparison
        temperature: Temperature for distillation
        return_dict: Whether to return as dict
    
    Returns:
        dict with metrics or float (accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = loss / len(data_loader)
    
    if not return_dict:
        return accuracy
    
    metrics = {
        'accuracy': accuracy,
        'ce_loss': avg_loss,
        'correct': correct,
        'total': total
    }
    
    # If teachers provided, compute ensemble comparison
    if teachers is not None:
        ensemble_correct = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Ensemble prediction
                teacher_logits = []
                for teacher in teachers:
                    logits = teacher(images)
                    teacher_logits.append(logits)
                ensemble_logits = torch.mean(torch.stack(teacher_logits), dim=0)
                
                _, predicted = torch.max(ensemble_logits, 1)
                ensemble_correct += (predicted == labels).sum().item()
        
        ensemble_accuracy = ensemble_correct / total
        metrics['ensemble_accuracy'] = ensemble_accuracy
        metrics['accuracy_ratio'] = accuracy / ensemble_accuracy if ensemble_accuracy > 0 else 0
    
    return metrics


def predict(model, x, device):
    """
    Make predictions on input.
    
    Args:
        model: Model
        x: Input tensor or numpy array
        device: Device
    
    Returns:
        Predictions (class labels)
    """
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x)
    x = x.to(device)
    
    with torch.no_grad():
        outputs = model(x)
        _, predictions = torch.max(outputs, 1)
    
    return predictions.cpu().numpy()


def save_artifacts(teachers, student, histories, metrics, output_dir=OUTPUT_DIR):
    """
    Save all models and metrics.
    
    Args:
        teachers: List of teacher models
        student: Student model
        histories: Dict with training histories
        metrics: Dict with evaluation metrics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save teachers
    for i, teacher in enumerate(teachers):
        path = os.path.join(output_dir, f'teacher_{i}.pt')
        torch.save(teacher.state_dict(), path)
        print(f"Saved teacher {i} to {path}")
    
    # Save student
    student_path = os.path.join(output_dir, 'student.pt')
    torch.save(student.state_dict(), student_path)
    print(f"Saved student to {student_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save histories
    history_path = os.path.join(output_dir, 'histories.json')
    with open(history_path, 'w') as f:
        json.dump(histories, f, indent=2)
    print(f"Saved histories to {history_path}")


if __name__ == '__main__':
    """
    Main training and evaluation pipeline:
    1. Load MNIST data
    2. Train 3 independent teacher models
    3. Create ensemble predictor
    4. Train student model with distillation
    5. Evaluate both and compute efficiency metrics
    6. Assert quality thresholds
    7. Exit 0 on success, 1 on failure
    """
    
    try:
        device = get_device()
        print(f"Using device: {device}")
        
        # Load data
        print("\nLoading MNIST data...")
        train_loader, val_loader = make_dataloaders(batch_size=64, download=True)
        
        # Train teachers
        print("\n" + "="*60)
        print("Training Teacher Models")
        print("="*60)
        teachers = []
        teacher_histories = {}
        
        for i in range(3):
            print(f"\nTraining Teacher {i+1}/3...")
            teacher = build_model('teacher')
            history = train_teacher(teacher, train_loader, val_loader, device, epochs=10, lr=0.001)
            teachers.append(teacher)
            teacher_histories[f'teacher_{i}'] = history
            teacher_acc = history['val_acc'][-1]
            print(f"Teacher {i} final validation accuracy: {teacher_acc:.4f}")
        
        # Evaluate teachers individually
        print("\n" + "="*60)
        print("Evaluating Teachers")
        print("="*60)
        teacher_metrics = {}
        for i, teacher in enumerate(teachers):
            metrics = evaluate(teacher, val_loader, device, return_dict=True)
            teacher_metrics[f'teacher_{i}'] = metrics
            print(f"Teacher {i} - Accuracy: {metrics['accuracy']:.4f}")
        
        # Evaluate ensemble
        print("\nEvaluating Teacher Ensemble...")
        ensemble_correct = 0
        ensemble_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                teacher_logits = []
                for teacher in teachers:
                    teacher.eval()
                    logits = teacher(images)
                    teacher_logits.append(logits)
                ensemble_logits = torch.mean(torch.stack(teacher_logits), dim=0)
                _, predicted = torch.max(ensemble_logits, 1)
                ensemble_correct += (predicted == labels).sum().item()
                ensemble_total += labels.size(0)
        
        ensemble_accuracy = ensemble_correct / ensemble_total
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        # Train student with distillation
        print("\n" + "="*60)
        print("Training Student Model (Distillation)")
        print("="*60)
        student = build_model('student')
        distillation_history = train_with_distillation(
            student, teachers, train_loader, val_loader, device,
            temperature=4, alpha=0.5, epochs=20, lr=0.001
        )
        
        # Evaluate student
        print("\n" + "="*60)
        print("Evaluating Student Model")
        print("="*60)
        student_metrics = evaluate(student, val_loader, device, teachers=teachers, 
                                  temperature=4, return_dict=True)
        print(f"Student accuracy: {student_metrics['accuracy']:.4f}")
        print(f"Ensemble accuracy: {student_metrics['ensemble_accuracy']:.4f}")
        print(f"Student/Ensemble ratio: {student_metrics['accuracy_ratio']:.4f}")
        
        # Compute parameter counts
        teacher_params = count_parameters(teachers[0])
        student_params = count_parameters(student)
        compression_ratio = student_params / (teacher_params * 3)
        
        print(f"\nParameter efficiency:")
        print(f"Teacher params (single): {teacher_params}")
        print(f"Total teacher params (3x): {teacher_params * 3}")
        print(f"Student params: {student_params}")
        print(f"Compression ratio: {compression_ratio:.4f}")
        
        # Save artifacts
        all_histories = {
            'teachers': teacher_histories,
            'distillation': distillation_history
        }
        all_metrics = {
            'teachers': teacher_metrics,
            'ensemble_accuracy': ensemble_accuracy,
            'student': student_metrics,
            'compression_ratio': compression_ratio
        }
        
        save_artifacts(teachers, student, all_histories, all_metrics)
        
        # Assertions for quality
        print("\n" + "="*60)
        print("Quality Assertions")
        print("="*60)
        
        assert student_metrics['accuracy'] > 0.92, \
            f"Student accuracy {student_metrics['accuracy']:.4f} must be > 0.92"
        print("✓ Student accuracy > 0.92")
        
        assert student_metrics['accuracy'] > ensemble_accuracy * 0.85, \
            f"Student accuracy should be >= 85% of ensemble accuracy"
        print("✓ Student achieves >= 85% of ensemble performance")
        
        assert ensemble_accuracy > 0.97, \
            f"Ensemble accuracy {ensemble_accuracy:.4f} must be > 0.97"
        print("✓ Ensemble accuracy > 0.97")
        
        assert compression_ratio < 1.0, \
            f"Student should have fewer params than single teacher"
        print(f"✓ Student is compressed (ratio: {compression_ratio:.4f})")
        
        print("\n" + "="*60)
        print("SUCCESS: All assertions passed!")
        print("="*60)
        sys.exit(0)
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

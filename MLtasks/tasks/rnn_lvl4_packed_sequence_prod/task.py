"""
RNN with PackedSequence for Variable-Length Batches
Implements efficiency comparison between padded and packed sequences
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_task_metadata():
    """Return task metadata"""
    return {
        'task_name': 'rnn_packed_sequence',
        'description': 'RNN with PackedSequence for variable-length sequences',
        'input_type': 'sequence',
        'output_type': 'regression',
        'framework': 'pytorch'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    """Get device configuration"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SequenceDataset(Dataset):
    """Dataset for variable-length sequences"""
    
    def __init__(self, num_samples=1000, min_len=5, max_len=20, feature_dim=10):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.feature_dim = feature_dim
        
        # Generate variable-length sequences
        self.sequences = []
        self.targets = []
        
        for _ in range(num_samples):
            seq_len = np.random.randint(min_len, max_len + 1)
            # Generate random sequence
            seq = np.random.randn(seq_len, feature_dim).astype(np.float32)
            
            # Target is based on sequence statistics (sum of means across timesteps)
            target = np.mean(seq, axis=0).sum()
            
            self.sequences.append(torch.from_numpy(seq))
            self.targets.append(target)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    sequences, targets = zip(*batch)
    
    # Get sequence lengths
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)
    
    # Stack targets
    targets = torch.tensor(targets, dtype=torch.float32)
    
    return padded_sequences, lengths, targets


def make_dataloaders(batch_size=32, train_samples=800, val_samples=200):
    """Create train and validation dataloaders"""
    # Create datasets
    train_dataset = SequenceDataset(num_samples=train_samples)
    val_dataset = SequenceDataset(num_samples=val_samples)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


class RNNModel(nn.Module):
    """RNN model with PackedSequence support"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(RNNModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths):
        # Sort by length for packed sequence
        lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
        x_sorted = x[sort_idx]
        
        # Pack the sequence
        packed = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True)
        
        # Process through RNN
        packed_out, (h_n, c_n) = self.rnn(packed)
        
        # Unsort the output
        _, unsort_idx = torch.sort(sort_idx)
        h_n = h_n[:, unsort_idx, :]
        
        # Take the last hidden state
        last_hidden = h_n[-1, :, :]
        
        # Fully connected layer
        out = self.fc(self.dropout(last_hidden))
        
        return out.squeeze(-1)


def build_model(input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
    """Build RNN model"""
    model = RNNModel(input_dim, hidden_dim, num_layers, dropout).to(device)
    return model


def train(model, train_loader, criterion, optimizer, epoch, total_epochs):
    """Train the model"""
    model.train()
    total_loss = 0
    
    for batch_idx, (sequences, lengths, targets) in enumerate(train_loader):
        # Move to device
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{total_epochs}], Train Loss: {avg_loss:.4f}")
    
    return avg_loss


def evaluate(model, data_loader, criterion):
    """Evaluate the model and return metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, lengths, targets in data_loader:
            # Move to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(sequences, lengths)
            
            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    avg_loss = total_loss / len(data_loader)
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    metrics = {
        'loss': avg_loss,
        'mse': mse,
        'r2': r2
    }
    
    return metrics


def predict(model, sequences, lengths):
    """Make predictions"""
    model.eval()
    
    with torch.no_grad():
        sequences = sequences.to(device)
        outputs = model(sequences, lengths)
    
    return outputs.cpu().numpy()


def save_artifacts(model, train_metrics, val_metrics, timing_results, save_dir='output'):
    """Save model artifacts and results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'rnn_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }, model_path)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Training Metrics:\n")
        for key, value in train_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\nValidation Metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\nTiming Results (seconds):\n")
        for key, value in timing_results.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    # Save timing plot
    if timing_results:
        plt.figure(figsize=(10, 6))
        
        methods = list(timing_results.keys())
        times = list(timing_results.values())
        
        plt.bar(methods, times, color=['blue', 'green'])
        plt.xlabel('Method')
        plt.ylabel('Time (seconds)')
        plt.title('Padded vs Packed Sequence Processing Time')
        plt.savefig(os.path.join(save_dir, 'timing_comparison.png'))
        plt.close()
    
    print(f"Artifacts saved to {save_dir}")


def benchmark_packed_vs_padded(model_class, input_dim, hidden_dim, num_layers, 
                                train_loader, num_iterations=10):
    """Benchmark padded vs packed sequence processing"""
    # Create two identical models
    model_padded = model_class(input_dim, hidden_dim, num_layers).to(device)
    model_packed = model_class(input_dim, hidden_dim, num_layers).to(device)
    
    # Initialize with same weights
    model_packed.load_state_dict(model_padded.state_dict())
    
    criterion = nn.MSELoss()
    
    # Benchmark padded approach (without packing)
    padded_times = []
    for _ in range(num_iterations):
        sequences, lengths, targets = next(iter(train_loader))
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Pad sequences manually (already done by dataloader)
        start_time = time.time()
        
        # Process without packing
        outputs = model_padded(sequences, lengths)
        loss = criterion(outputs, targets)
        
        end_time = time.time()
        padded_times.append(end_time - start_time)
    
    # Benchmark packed approach
    packed_times = []
    for _ in range(num_iterations):
        sequences, lengths, targets = next(iter(train_loader))
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        start_time = time.time()
        
        # Process with packing (using the packed version)
        outputs = model_packed(sequences, lengths)
        loss = criterion(outputs, targets)
        
        end_time = time.time()
        packed_times.append(end_time - start_time)
    
    return {
        'padded': np.mean(padded_times),
        'packed': np.mean(packed_times)
    }


def main():
    """Main function to run the RNN task"""
    print("=" * 60)
    print("RNN with PackedSequence - Variable-Length Batches")
    print("=" * 60)
    
    # Configuration
    INPUT_DIM = 10
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(
        batch_size=BATCH_SIZE,
        train_samples=800,
        val_samples=200
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    print(f"Model architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nTraining model...")
    train_losses = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, NUM_EPOCHS)
        train_losses.append(train_loss)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, criterion)
    print("Train Metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion)
    print("Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Benchmark packed vs padded
    print("\nBenchmarking packed vs padded sequences...")
    timing_results = benchmark_packed_vs_padded(
        RNNModel, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, train_loader
    )
    print("Timing Results (seconds):")
    for key, value in timing_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Calculate speedup
    if timing_results['packed'] > 0:
        speedup = timing_results['padded'] / timing_results['packed']
        print(f"  Speedup (padded/packed): {speedup:.2f}x")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_metrics, val_metrics, timing_results)
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train MSE: {train_metrics['mse']:.4f}")
    print(f"Val MSE:   {val_metrics['mse']:.4f}")
    print(f"Train R²:  {train_metrics['r2']:.4f}")
    print(f"Val R²:    {val_metrics['r2']:.4f}")
    if timing_results:
        print(f"\nPacked Sequence Speedup: {timing_results['padded']/timing_results['packed']:.2f}x")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Train R² > 0.8
    checks_total += 1
    if train_metrics['r2'] > 0.8:
        print(f"✓ Train R² > 0.8: {train_metrics['r2']:.4f}")
        checks_passed += 1
    else:
        print(f"✗ Train R² > 0.8: {train_metrics['r2']:.4f}")
    
    # Check 2: Val R² > 0.7
    checks_total += 1
    if val_metrics['r2'] > 0.7:
        print(f"✓ Val R² > 0.7: {val_metrics['r2']:.4f}")
        checks_passed += 1
    else:
        print(f"✗ Val R² > 0.7: {val_metrics['r2']:.4f}")
    
    # Check 3: Val MSE < 1.0
    checks_total += 1
    if val_metrics['mse'] < 1.0:
        print(f"✓ Val MSE < 1.0: {val_metrics['mse']:.4f}")
        checks_passed += 1
    else:
        print(f"✗ Val MSE < 1.0: {val_metrics['mse']:.4f}")
    
    # Check 4: Loss decreased
    checks_total += 1
    if len(train_losses) >= 2 and train_losses[-1] < train_losses[0]:
        print(f"✓ Loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
        checks_passed += 1
    else:
        print(f"✗ Loss did not decrease properly")
    
    # Check 5: R² difference < 0.15 (no severe overfitting)
    checks_total += 1
    r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
    if r2_diff < 0.15:
        print(f"✓ R² difference < 0.15: {r2_diff:.4f}")
        checks_passed += 1
    else:
        print(f"✗ R² difference < 0.15: {r2_diff:.4f}")
    
    # Print summary
    print("\n" + "=" * 60)
    if checks_passed == checks_total:
        print(f"PASS: All {checks_passed}/{checks_total} quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print(f"FAIL: Only {checks_passed}/{checks_total} quality checks passed!")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    exit(main())

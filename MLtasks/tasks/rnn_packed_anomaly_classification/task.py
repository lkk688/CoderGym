"""
RNN with PackedSequence for Variable-Length Batches
Anomaly vs Normal Sequence Classification

Created new task based on rnn_lvl4_packed_sequence_prod by changing task from regression to anomaly-vs-normal binary classification.
Updated the synthetic dataset to generate anomalous sequences with shifted values and spikes, changed model output and loss binary classification, and replaced regression metrics with classification metrics."""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output directory
OUTPUT_DIR = "/Developer/AIserver/output/tasks/rnn_lvl2_packed_anomaly_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "rnn_packed_anomaly_classification",
        "description": "LSTM with PackedSequence for anomaly vs normal variable-length sequence classification",
        "input_type": "sequence",
        "output_type": "binary_classification",
        "framework": "pytorch",
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    """Dataset for variable-length normal vs anomalous sequences."""

    def __init__(
        self, num_samples=1000, min_len=5, max_len=20, feature_dim=10, anomaly_ratio=0.5
    ):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.feature_dim = feature_dim
        self.anomaly_ratio = anomaly_ratio

        # Generate variable-length sequences
        self.sequences = []
        self.targets = []

        num_anomalies = int(num_samples * anomaly_ratio)
        num_normals = num_samples - num_anomalies

        labels = [0] * num_normals + [1] * num_anomalies
        np.random.shuffle(labels)

        for label in labels:
            seq_len = np.random.randint(min_len, max_len + 1)

            if label == 0:
                # Normal sequence: values centered near 0 with moderate noise
                seq = np.random.normal(
                    loc=0.0, scale=1.0, size=(seq_len, feature_dim)
                ).astype(np.float32)
            else:
                # Anomalous sequence: shifted mean + one strong spike region
                seq = np.random.normal(
                    loc=1.5, scale=1.2, size=(seq_len, feature_dim)
                ).astype(np.float32)

                spike_start = np.random.randint(0, seq_len)
                spike_width = min(np.random.randint(1, 4), seq_len - spike_start)
                seq[spike_start : spike_start + spike_width] += np.random.normal(
                    loc=3.5, scale=0.8, size=(spike_width, feature_dim)
                ).astype(np.float32)

            self.sequences.append(torch.tensor(seq, dtype=torch.float32))
            self.targets.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    sequences, targets = zip(*batch)

    # Get sequence lengths
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)

    # Stack targets
    targets = torch.tensor(targets, dtype=torch.float32)

    return padded_sequences, lengths, targets


def make_dataloaders(batch_size=32, train_samples=800, val_samples=200):
    """Create train and validation dataloaders."""
    # Create datasets
    train_dataset = SequenceDataset(num_samples=train_samples, anomaly_ratio=0.5)
    val_dataset = SequenceDataset(num_samples=val_samples, anomaly_ratio=0.5)

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader


class PackedLSTMClassifier(nn.Module):
    """LSTM classifier with PackedSequence support."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(PackedLSTMClassifier, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
        x_sorted = x[sort_idx]

        packed = pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (h_n, _) = self.rnn(packed)

        # Because bidirectional=True:
        # last layer forward hidden state = h_n[-2]
        # last layer backward hidden state = h_n[-1]
        hidden_fwd = h_n[-2]
        hidden_bwd = h_n[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        _, unsort_idx = torch.sort(sort_idx)
        hidden_cat = hidden_cat[unsort_idx]

        logits = self.fc(self.dropout(hidden_cat)).squeeze(-1)
        return logits


class PaddedLSTMClassifier(nn.Module):
    """Baseline LSTM classifier without PackedSequence for timing comparison."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(PaddedLSTMClassifier, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        output, _ = self.rnn(x)

        # Gather the last valid timestep for each sequence
        batch_size = x.size(0)
        last_indices = (lengths - 1).to(x.device)
        forward_last = output[
            torch.arange(batch_size, device=x.device),
            last_indices,
            : self.rnn.hidden_size,
        ]
        backward_first = output[:, 0, self.rnn.hidden_size :]
        hidden_cat = torch.cat([forward_last, backward_first], dim=1)

        logits = self.fc(self.dropout(hidden_cat)).squeeze(-1)
        return logits


def build_model(input_dim, hidden_dim=64, num_layers=2, dropout=0.3, use_packed=True):
    """Build model."""
    if use_packed:
        model = PackedLSTMClassifier(input_dim, hidden_dim, num_layers, dropout).to(
            device
        )
    else:
        model = PaddedLSTMClassifier(input_dim, hidden_dim, num_layers, dropout).to(
            device
        )
    return model


def train(model, train_loader, criterion, optimizer, epoch, total_epochs):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for sequences, lengths, targets in train_loader:
        # Move to device
        sequences = sequences.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(sequences, lengths)

        # Compute loss
        loss = criterion(logits, targets)

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
    """Evaluate the model and return classification metrics."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sequences, lengths, targets in data_loader:
            # Move to device
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(sequences, lengths)

            # Compute Loss
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_loss += loss.item()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def predict(model, sequences, lengths):
    """Make predictions."""
    model.eval()
    with torch.no_grad():
        sequences = sequences.to(device)
        logits = model(sequences, lengths)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
    return probs.cpu().numpy(), preds.cpu().numpy()


def save_artifacts(
    model, train_metrics, val_metrics, timing_results, save_dir=OUTPUT_DIR
):
    """Save model artifacts and results."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, "rnn_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        model_path,
    )

    # Save metadata
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(get_task_metadata(), f, indent=2)

    # Save metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "timing_results": timing_results,
            },
            f,
            indent=2,
        )

    print(f"Artifacts saved to {save_dir}")


def benchmark_packed_vs_padded(
    input_dim, hidden_dim, num_layers, train_loader, num_iterations=20
):
    """Benchmark packed vs padded sequence processing with two real implementations."""
    model_padded = build_model(input_dim, hidden_dim, num_layers, use_packed=False)
    model_packed = build_model(input_dim, hidden_dim, num_layers, use_packed=True)

    # Match initial weights as closely as possible for fairer comparison
    padded_state = model_padded.state_dict()
    packed_state = model_packed.state_dict()
    compatible_keys = [
        k
        for k in packed_state.keys()
        if k in padded_state and packed_state[k].shape == padded_state[k].shape
    ]
    for key in compatible_keys:
        packed_state[key] = padded_state[key].clone()
    model_packed.load_state_dict(packed_state)

    model_padded.eval()
    model_packed.eval()

    padded_times = []
    packed_times = []

    with torch.no_grad():
        for _ in range(num_iterations):
            sequences, lengths, _ = next(iter(train_loader))
            sequences = sequences.to(device)

            start_time = time.time()
            _ = model_padded(sequences, lengths)
            end_time = time.time()
            padded_times.append(end_time - start_time)

        for _ in range(num_iterations):
            sequences, lengths, _ = next(iter(train_loader))
            sequences = sequences.to(device)

            start_time = time.time()
            _ = model_packed(sequences, lengths)
            end_time = time.time()
            packed_times.append(end_time - start_time)

    return {
        "padded": float(np.mean(padded_times)),
        "packed": float(np.mean(packed_times)),
    }


def main():
    """Main function to run the anomaly classification task."""
    print("=" * 60)
    print("RNN with PackedSequence - Anomaly vs Normal Classification")
    print("=" * 60)

    # Configuration
    INPUT_DIM = 10
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(
        batch_size=BATCH_SIZE, train_samples=800, val_samples=200
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, use_packed=True)
    print(f"Model architecture:\n{model}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nTraining model...")
    train_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, NUM_EPOCHS)
        train_losses.append(train_loss)

    # Evaluate on training set, then validation set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, criterion)
    for key, value in train_metrics.items():
        print(f"  {key}: {value}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion)
    for key, value in val_metrics.items():
        print(f"  {key}: {value}")

    # Benchmark packed vs padded
    print("\nBenchmarking packed vs padded sequences...")
    timing_results = benchmark_packed_vs_padded(
        INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, train_loader
    )
    for key, value in timing_results.items():
        print(f"  {key}: {value:.6f} sec")

    # Calculate speedup
    if timing_results["packed"] > 0:
        speedup = timing_results["padded"] / timing_results["packed"]
        print(f"  Speedup (padded/packed): {speedup:.2f}x")

    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_metrics, val_metrics, timing_results)

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = 0
    checks_total = 0

    checks_total += 1
    if train_metrics["accuracy"] >= 0.90:
        print(f"✓ Train accuracy >= 0.90: {train_metrics['accuracy']:.4f}")
        checks_passed += 1
    else:
        print(f"✗ Train accuracy >= 0.90: {train_metrics['accuracy']:.4f}")

    checks_total += 1
    if val_metrics["accuracy"] >= 0.85:
        print(f"✓ Val accuracy >= 0.85: {val_metrics['accuracy']:.4f}")
        checks_passed += 1
    else:
        print(f"✗ Val accuracy >= 0.85: {val_metrics['accuracy']:.4f}")

    checks_total += 1
    if val_metrics["f1"] >= 0.85:
        print(f"✓ Val F1 >= 0.85: {val_metrics['f1']:.4f}")
        checks_passed += 1
    else:
        print(f"✗ Val F1 >= 0.85: {val_metrics['f1']:.4f}")

    checks_total += 1
    if len(train_losses) >= 2 and train_losses[-1] < train_losses[0]:
        print(f"✓ Loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
        checks_passed += 1
    else:
        print("✗ Loss did not decrease properly")

    checks_total += 1
    if abs(train_metrics["accuracy"] - val_metrics["accuracy"]) < 0.10:
        gap = abs(train_metrics["accuracy"] - val_metrics["accuracy"])
        print(f"✓ Accuracy gap < 0.10: {gap:.4f}")
        checks_passed += 1
    else:
        gap = abs(train_metrics["accuracy"] - val_metrics["accuracy"])
        print(f"✗ Accuracy gap < 0.10: {gap:.4f}")

    print("\n" + "=" * 60)
    if checks_passed == checks_total:
        print(f"PASS: All {checks_passed}/{checks_total} quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print(f"FAIL: Only {checks_passed}/{checks_total} quality checks passed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

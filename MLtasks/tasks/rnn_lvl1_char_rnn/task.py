#!/usr/bin/env python3
"""
Char-RNN: Character-level RNN for next-character prediction.
A self-contained implementation for text generation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
OUTPUT_DIR = '/Developer/AIserver/output/tasks/rnn_lvl1_char_rnn'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample text for training (simple Shakespeare-like text)
SAMPLE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep—
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
""".strip()

# Character vocabulary
chars = sorted(list(set(SAMPLE_TEXT)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'char_rnn',
        'description': 'Character-level RNN for next-character prediction',
        'input_type': 'sequence of characters',
        'output_type': 'character probability distribution',
        'vocab_size': vocab_size,
        'sample_text_length': len(SAMPLE_TEXT)
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CharDataset(Dataset):
    """Dataset for character-level text data."""
    
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.data = [char_to_idx[c] for c in text]
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Input: sequence of characters
        x = self.data[idx:idx + self.seq_length]
        # Target: next character
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def make_dataloaders(seq_length=20, batch_size=32, train_ratio=0.8):
    """Create training and validation dataloaders."""
    # Create dataset
    dataset = CharDataset(SAMPLE_TEXT, seq_length)
    
    # Split into train and validation
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    
    return train_loader, val_loader


class CharRNN(nn.Module):
    """Character-level RNN model."""
    
    def __init__(self, vocab_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layers
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        embed = self.embedding(x)  # (batch, seq_len, hidden_size)
        
        if hidden is None:
            output, hidden = self.rnn(embed)
        else:
            output, hidden = self.rnn(embed, hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Project to vocab size
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


def build_model(device):
    """Build and return the CharRNN model."""
    model = CharRNN(vocab_size, hidden_size=128, num_layers=2, dropout=0.2)
    model = model.to(device)
    return model


def train(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            # Move to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            
            # Reshape for loss calculation
            # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
            # targets: (batch, seq_len) -> (batch*seq_len)
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = evaluate(model, val_loader, device, return_loss_only=True)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return model, train_losses, val_losses


def evaluate(model, data_loader, device, return_loss_only=False):
    """Evaluate the model and return metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits, _ = model(batch_x)
            
            # Calculate loss
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.numel()
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    accuracy = correct / total
    
    if return_loss_only:
        return avg_loss
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy
    }


def predict(model, device, seed_text, length=100, temperature=1.0):
    """Generate text using the model."""
    model.eval()
    
    # Convert seed text to indices
    seq = [char_to_idx[c] for c in seed_text if c in char_to_idx]
    
    # Initialize hidden state
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    # Process seed text
    if len(seq) > 0:
        x = torch.tensor([seq], dtype=torch.long).to(device)
        _, hidden = model(x, hidden)
    
    # Start with last character from seed
    if len(seq) > 0:
        x = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)
    else:
        x = torch.tensor([[np.random.randint(0, vocab_size)]], dtype=torch.long).to(device)
    
    generated = list(seed_text)
    
    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(x, hidden)
            
            # Apply temperature
            logits = logits / temperature
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from distribution
            next_idx = torch.multinomial(probs[:, -1, :], 1).item()
            
            # Convert to character
            next_char = idx_to_char[next_idx]
            generated.append(next_char)
            
            # Update input
            x = torch.tensor([[next_idx]], dtype=torch.long).to(device)
    
    return ''.join(generated)


def save_artifacts(model, train_losses, val_losses, device):
    """Save model and training artifacts."""
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'char_rnn.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()
    
    # Save metrics
    metrics = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'vocab_size': vocab_size,
        'sample_text_length': len(SAMPLE_TEXT)
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate and save sample text
    seed_texts = ["To ", "The ", "And "]
    samples = {}
    
    for seed in seed_texts:
        generated = predict(model, device, seed, length=100, temperature=0.8)
        samples[seed] = generated
    
    samples_path = os.path.join(OUTPUT_DIR, 'samples.json')
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run the Char-RNN training and evaluation."""
    print("=" * 60)
    print("Char-RNN: Character-level RNN for Next-Character Prediction")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print(f"Sample text length: {metadata['sample_text_length']}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(seq_length=20, batch_size=32)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\nTraining model...")
    model, train_losses, val_losses = train(
        model, train_loader, val_loader, device, 
        num_epochs=50, lr=0.001
    )
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Training Loss: {train_metrics['loss']:.4f}")
    print(f"Training Perplexity: {train_metrics['perplexity']:.4f}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Perplexity: {val_metrics['perplexity']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    
    # Quality thresholds
    print("\n" + "=" * 60)
    print("Quality Thresholds Check")
    print("=" * 60)
    
    # Check if validation perplexity is reasonable (< 50 for this simple task)
    perplexity_threshold = 50.0
    accuracy_threshold = 0.70
    
    try:
        assert val_metrics['perplexity'] < perplexity_threshold, \
            f"Validation perplexity {val_metrics['perplexity']:.2f} exceeds threshold {perplexity_threshold}"
        print(f"✓ Validation perplexity ({val_metrics['perplexity']:.2f}) < {perplexity_threshold}")
        
        assert val_metrics['accuracy'] > accuracy_threshold, \
            f"Validation accuracy {val_metrics['accuracy']:.2f} below threshold {accuracy_threshold}"
        print(f"✓ Validation accuracy ({val_metrics['accuracy']:.2f}) > {accuracy_threshold}")
        
        # Check that training loss decreased
        assert train_losses[-1] < train_losses[0], \
            "Training loss did not decrease during training"
        print(f"✓ Training loss decreased from {train_losses[0]:.4f} to {train_losses[-1]:.4f}")
        
        # Check that validation loss decreased
        assert val_losses[-1] < val_losses[0], \
            "Validation loss did not decrease during training"
        print(f"✓ Validation loss decreased from {val_losses[0]:.4f} to {val_losses[-1]:.4f}")
        
        print("\n" + "=" * 60)
        print("PASS: All quality thresholds met!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_losses, val_losses, device)
    
    # Generate sample text
    print("\nGenerating sample text...")
    seed_texts = ["To ", "The ", "And "]
    for seed in seed_texts:
        generated = predict(model, device, seed, length=50, temperature=0.8)
        print(f"Seed: '{seed}' -> Generated: '{generated[:50]}...'")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

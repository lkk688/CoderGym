"""
Seq2Seq + Attention for Toy Translation (Sequence Reversal)
Implements encoder-decoder with attention on a toy dataset (reversing sequences).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, List, Optional

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Task Metadata
# ============================================================================

def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        'task_name': 'seq2seq_attention_reverse',
        'description': 'Sequence reversal using encoder-decoder with attention',
        'input_type': 'sequence',
        'output_type': 'sequence',
        'input_vocab_size': 10,
        'output_vocab_size': 10,
        'max_seq_length': 10,
        'max_len': 10,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 1,
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'quality_thresholds': {
            'train_accuracy': 0.95,
            'val_accuracy': 0.95,
            'train_mse': 0.1,
            'val_mse': 0.1
        }
    }

# ============================================================================
# Data Generation
# ============================================================================

def generate_reverse_dataset(num_samples: int, max_len: int = 10, 
                             vocab_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset for sequence reversal task.
    
    Args:
        num_samples: Number of samples to generate
        max_len: Maximum sequence length
        vocab_size: Size of vocabulary (0 to vocab_size-1)
    
    Returns:
        Tuple of (inputs, targets) where targets are reversed inputs
    """
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        # Random sequence length (1 to max_len)
        seq_len = np.random.randint(1, max_len + 1)
        
        # Generate random sequence (excluding 0 which is used for padding)
        seq = np.random.randint(1, vocab_size, size=seq_len)
        
        # Pad to max_len
        padded_input = np.zeros(max_len, dtype=np.int64)
        padded_input[:seq_len] = seq
        
        # Target is the reversed sequence
        padded_target = np.zeros(max_len, dtype=np.int64)
        padded_target[:seq_len] = seq[::-1]
        
        inputs.append(padded_input)
        targets.append(padded_target)
    
    return np.array(inputs), np.array(targets)


class SequenceDataset(Dataset):
    """Dataset for sequence data."""
    
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.LongTensor(inputs)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def make_dataloaders(batch_size: int = 64, train_samples: int = 800, 
                     val_samples: int = 200, max_len: int = 10, 
                     vocab_size: int = 10) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        batch_size: Batch size
        train_samples: Number of training samples
        val_samples: Number of validation samples
        max_len: Maximum sequence length
        vocab_size: Size of vocabulary
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Generate data
    train_inputs, train_targets = generate_reverse_dataset(
        train_samples, max_len, vocab_size
    )
    val_inputs, val_targets = generate_reverse_dataset(
        val_samples, max_len, vocab_size
    )
    
    # Create datasets
    train_dataset = SequenceDataset(train_inputs, train_targets)
    val_dataset = SequenceDataset(val_inputs, val_targets)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    
    return train_loader, val_loader


# ============================================================================
# Model Architecture
# ============================================================================

class Encoder(nn.Module):
    """Encoder RNN with embedding layer."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int = 1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           num_layers=num_layers, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            outputs: LSTM outputs of shape (batch_size, seq_len, hidden_dim)
            hidden: (h_n, c_n) tuple of hidden and cell states
        """
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau-style attention mechanism."""
    
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim * 2, hidden_dim)
        self.va = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden: Decoder hidden state of shape (batch_size, hidden_dim)
            encoder_outputs: Encoder outputs of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            context: Context vector of shape (batch_size, hidden_dim)
            attention_weights: Attention weights of shape (batch_size, seq_len)
        """
        # Repeat decoder hidden state for each encoder output
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Compute attention scores
        # Concatenate decoder hidden with each encoder output
        decoder_hidden_expanded = decoder_hidden.expand(-1, encoder_outputs.size(1), -1)
        energy = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
        energy = torch.tanh(self.Wa(energy))
        scores = self.va(energy).squeeze(2)  # (batch_size, seq_len)
        
        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class Decoder(nn.Module):
    """Decoder RNN with attention mechanism."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int = 1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, 
                           num_layers=num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through decoder with attention.
        
        Args:
            x: Input token of shape (batch_size, 1)
            hidden: (h_n, c_n) tuple of hidden and cell states
            encoder_outputs: Encoder outputs of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            output: Output logits of shape (batch_size, vocab_size)
            hidden: Updated hidden state
            attention_weights: Attention weights of shape (batch_size, seq_len)
        """
        # Embed input
        embedded = self.embedding(x)  # (batch_size, 1, embedding_dim)
        
        # Compute attention
        # Use the last layer's hidden state
        h_t = hidden[0][-1]  # (batch_size, hidden_dim)
        context, attention_weights = self.attention(h_t, encoder_outputs)
        
        # Concatenate context with embedded input
        context = context.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        lstm_input = torch.cat([embedded, context], dim=2)  # (batch_size, 1, embedding_dim + hidden_dim)
        
        # Pass through LSTM
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Compute output
        output = output.squeeze(1)  # (batch_size, hidden_dim)
        output = torch.cat([output, context.squeeze(1)], dim=1)  # (batch_size, 2*hidden_dim)
        output = self.fc_out(output)  # (batch_size, vocab_size)
        
        return output, hidden, attention_weights


class Seq2SeqAttention(nn.Module):
    """Seq2Seq model with attention mechanism."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int = 1, max_len: int = 10):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        self.vocab_size = vocab_size
        self.max_len = max_len
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass through seq2seq model.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len)
            target: Target sequence for teacher forcing
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size = x.size(0)
        
        # Encode
        encoder_outputs, hidden = self.encoder(x)
        
        # Initialize decoder input with start token (use 0 as start token)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
        
        # Store outputs
        outputs = []
        attention_weights_list = []
        
        # Decode
        for t in range(self.max_len):
            output, hidden, attention_weights = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(1))
            attention_weights_list.append(attention_weights.unsqueeze(1))
            
            # Teacher forcing
            if target is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                _, topi = output.topk(1)
                decoder_input = topi.detach()
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict sequences without teacher forcing.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len)
        
        Returns:
            predictions: Predicted sequences of shape (batch_size, seq_len)
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            
            # Encode
            encoder_outputs, hidden = self.encoder(x)
            
            # Initialize decoder input
            decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
            
            # Store predictions
            predictions = []
            
            # Decode
            for _ in range(self.max_len):
                output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
                _, topi = output.topk(1)
                predictions.append(topi.squeeze(1))
                decoder_input = topi.detach()
            
            predictions = torch.stack(predictions, dim=1)
            return predictions


def build_model(vocab_size: int = 10, embedding_dim: int = 64, 
                hidden_dim: int = 128, num_layers: int = 1, 
                max_len: int = 10) -> Seq2SeqAttention:
    """
    Build the seq2seq model with attention.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden states
        num_layers: Number of LSTM layers
        max_len: Maximum sequence length
    
    Returns:
        Configured Seq2SeqAttention model
    """
    model = Seq2SeqAttention(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_len=max_len
    )
    model = model.to(device)
    return model


# ============================================================================
# Training
# ============================================================================

def train(model: nn.Module, train_loader: DataLoader, 
          criterion: nn.Module, optimizer: optim.Optimizer,
          num_epochs: int = 100, print_every: int = 20) -> List[float]:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        print_every: Print loss every N epochs
    
    Returns:
        List of training losses per epoch
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs, targets, teacher_forcing_ratio=0.5)
            
            # Compute loss
            # Reshape for loss computation
            outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, vocab_size)
            targets = targets.view(-1)  # (batch_size * seq_len)
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return losses


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    total_mse = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get predictions
            predictions = model.predict(inputs)
            
            # Compute accuracy (element-wise)
            mask = targets != 0  # Ignore padding
            correct += (predictions[mask] == targets[mask]).sum().item()
            total += mask.sum().item()
            
            # Compute MSE
            total_mse += ((predictions.float() - targets.float()) ** 2).mean().item()
    
    accuracy = correct / total if total > 0 else 0.0
    mse = total_mse / len(data_loader)
    
    return {
        'accuracy': accuracy,
        'mse': mse
    }


# ============================================================================
# Main Function
# ============================================================================

def main() -> int:
    """Main function to run the task."""
    print("=" * 60)
    print("Seq2Seq + Attention for Sequence Reversal")
    print("=" * 60)
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Description: {metadata['description']}")
    
    # Extract parameters from metadata
    vocab_size = metadata['input_vocab_size']
    embedding_dim = metadata['embedding_dim']
    hidden_dim = metadata['hidden_dim']
    num_layers = metadata['num_layers']
    max_len = metadata['max_len']
    batch_size = metadata['batch_size']
    num_epochs = metadata['num_epochs']
    learning_rate = metadata['learning_rate']
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Max sequence length: {max_len}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    
    # Create dataloaders
    print("\n" + "-" * 40)
    print("Creating dataloaders...")
    train_loader, val_loader = make_dataloaders(
        batch_size=batch_size,
        max_len=max_len,
        vocab_size=vocab_size
    )
    
    # Build model
    print("\n" + "-" * 40)
    print("Building model...")
    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_len=max_len
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    print("\n" + "-" * 40)
    print("Training...")
    losses = train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        print_every=10
    )
    
    # Evaluation
    print("\n" + "-" * 40)
    print("Evaluating...")
    train_metrics = evaluate(model, train_loader)
    val_metrics = evaluate(model, val_loader)
    
    print(f"\nTraining Metrics:")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  MSE: {val_metrics['mse']:.4f}")
    
    # Check quality thresholds
    thresholds = metadata['quality_thresholds']
    print("\n" + "-" * 40)
    print("Quality Check:")
    print(f"  Train accuracy >= {thresholds['train_accuracy']}: {'PASS' if train_metrics['accuracy'] >= thresholds['train_accuracy'] else 'FAIL'}")
    print(f"  Val accuracy >= {thresholds['val_accuracy']}: {'PASS' if val_metrics['accuracy'] >= thresholds['val_accuracy'] else 'FAIL'}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

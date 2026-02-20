"""
LSTM Sentiment Classification Task
Tokenization + embedding + LSTM classifier
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import Dict, List, Tuple, Any
import pickle
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
OUTPUT_DIR = '/Developer/AIserver/output/tasks/rnn_lvl2_lstm_sentiment'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Vocabulary:
    """Simple vocabulary class for tokenization."""
    
    def __init__(self, max_size: int = 10000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts."""
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        # Filter by minimum frequency
        filtered_words = [word for word, freq in counter.items() if freq >= self.min_freq]
        
        # Sort by frequency, then alphabetically for determinism
        sorted_words = sorted(filtered_words, key=lambda x: (-counter[x], x))
        
        # Add to vocabulary up to max_size
        for word in sorted_words[:self.max_size - 2]:  # -2 for <PAD> and <UNK>
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split by whitespace and lowercase."""
        return text.lower().split()
    
    def encode(self, text: str, max_len: int = 100) -> List[int]:
        """Encode text to indices with padding/truncation."""
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # Truncate if too long
        if len(indices) > max_len:
            indices = indices[:max_len]
        
        # Pad if too short
        if len(indices) < max_len:
            indices = indices + [self.word2idx['<PAD>']] * (max_len - len(indices))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.word2idx)


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification."""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.vocab.encode(text, self.max_len)
        text_tensor = torch.tensor(encoded, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return text_tensor, label_tensor


class LSTMClassifier(nn.Module):
    """LSTM-based sentiment classifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.5, num_classes: int = 1):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Concatenate the final forward and backward hidden states
        hidden_fwd = hidden[-2, :, :]  # (batch_size, hidden_dim)
        hidden_bwd = hidden[-1, :, :]  # (batch_size, hidden_dim)
        hidden_cat = torch.cat((hidden_fwd, hidden_bwd), dim=1)  # (batch_size, hidden_dim * 2)
        
        # Apply dropout and fully connected layer
        out = self.dropout(hidden_cat)
        out = self.fc(out)  # (batch_size, num_classes)
        out = self.sigmoid(out).squeeze(-1)  # (batch_size,)
        
        return out


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        'task_name': 'lstm_sentiment_classification',
        'task_type': 'binary_classification',
        'description': 'LSTM-based sentiment classification with tokenization and embedding',
        'input_type': 'text',
        'output_type': 'binary',
        'created_at': datetime.now().isoformat()
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size: int = 32, max_len: int = 100, 
                     train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """Create dataloaders for training and validation."""
    # Sample sentiment data (positive/negative reviews)
    positive_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Great acting and wonderful storyline. Highly recommended!",
        "The best film I've seen this year. Amazing visuals and plot.",
        "What a masterpiece! The director did an incredible job.",
        "I was blown away by the performances and cinematography.",
        "Truly an excellent movie with a powerful message.",
        "Brilliant direction and outstanding cast performances.",
        "A must-watch film that delivers on all fronts.",
        "Incredible storytelling and beautiful cinematography.",
        "One of the best movies I have ever seen."
    ]
    
    negative_texts = [
        "This movie was terrible. Complete waste of time.",
        "Awful acting and a boring storyline. Very disappointing.",
        "The worst film I've ever seen. Do not waste your money.",
        "Poor direction and weak character development.",
        "I fell asleep halfway through. So boring!",
        "A complete disaster with no redeeming qualities.",
        "Terrible script and awful performances throughout.",
        "Worst movie experience I've had in years.",
        "Boring from start to finish. Very poorly made.",
        "I regret watching this. Total garbage."
    ]
    
    # Combine and label data
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    # Shuffle data
    np.random.seed(42)
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Split into train and validation
    split_idx = int(len(texts) * train_ratio)
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    val_texts = texts[split_idx:]
    val_labels = labels[split_idx:]
    
    # Build vocabulary
    vocab = Vocabulary(max_size=1000, min_freq=1)
    vocab.build_vocab(texts)
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_len)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, max_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab


def build_model(vocab_size: int) -> LSTMClassifier:
    """Build the LSTM classifier model."""
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        num_classes=1
    )
    return model.to(device)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          num_epochs: int = 20, learning_rate: float = 0.001) -> Dict[str, List[float]]:
    """Train the model and return training history."""
    # Use weighted BCE loss for imbalanced data
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 5
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            texts = texts.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        val_metrics = evaluate(model, val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_metrics["loss"]:.4f}, '
              f'Val Acc: {val_metrics["accuracy"]:.4f}, Val F1: {val_metrics["f1"]:.4f}')
        
        # Early stopping based on F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt')))
    
    return history


def evaluate(model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    model.eval()
    criterion = nn.BCELoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for texts, labels in data_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Convert to binary predictions
            predictions = (outputs >= 0.5).float()
            
            # Calculate metrics
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # For F1 score calculation
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    # Calculate F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def predict(model: nn.Module, texts: List[str], vocab: Vocabulary, max_len: int = 100) -> List[float]:
    """Predict sentiment for a list of texts."""
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for text in texts:
            # Encode text
            encoded = vocab.encode(text, max_len)
            text_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
            
            # Get prediction
            output = model(text_tensor)
            pred = output.item()
            predictions.append(pred)
    
    return predictions


def save_artifacts(model: nn.Module, vocab: Vocabulary, history: Dict[str, List[float]]):
    """Save model, vocabulary, and training history."""
    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pt'))
    
    # Save vocabulary
    with open(os.path.join(OUTPUT_DIR, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    
    # Save training history
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save task metadata
    metadata = get_task_metadata()
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run the LSTM sentiment classification task."""
    print("=" * 60)
    print("LSTM Sentiment Classification Task")
    print("=" * 60)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, vocab = make_dataloaders(batch_size=4, max_len=50)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(vocab_size=len(vocab))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Train model
    print("\nTraining model...")
    history = train(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001)
    
    # Evaluate on training set
    print("\n" + "=" * 60)
    print("Evaluating on Training Set")
    print("=" * 60)
    train_metrics = evaluate(model, train_loader)
    print(f"Training Loss: {train_metrics['loss']:.4f}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Training Precision: {train_metrics['precision']:.4f}")
    print(f"Training Recall: {train_metrics['recall']:.4f}")
    print(f"Training F1 Score: {train_metrics['f1']:.4f}")
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Evaluating on Validation Set")
    print("=" * 60)
    val_metrics = evaluate(model, val_loader)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {val_metrics['precision']:.4f}")
    print(f"Validation Recall: {val_metrics['recall']:.4f}")
    print(f"Validation F1 Score: {val_metrics['f1']:.4f}")
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    save_artifacts(model, vocab, history)
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)
    test_texts = [
        "This moviewas great!",
        "This movie was terrible!"
    ]
    predictions = predict(model, test_texts, vocab, max_len=50)
    for i, (text, pred) in enumerate(zip(test_texts, predictions)):
        sentiment = "positive" if pred >= 0.5 else "negative"
        print(f"Text {i+1}: {text}")
        print(f"Prediction: {pred:.4f} ({sentiment})\n")
    
    print("Task completed successfully!")


if __name__ == "__main__":
    main()

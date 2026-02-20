import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter
import os
import json
from typing import Dict, List, Tuple, Any
import tempfile

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Task metadata
def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": "transformer_encoder_classifier",
        "task_type": "text_classification",
        "input_type": "text",
        "output_type": "classification",
        "num_classes": 2,
        "description": "Encoder-only transformer for text classification"
    }

# Simple vocabulary class
class Vocabulary:
    def __init__(self, max_size: int = 10000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.max_size = max_size
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())
        
        # Add words up to max_size
        for word, count in word_counts.most_common(self.max_size - 2):
            if count >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text: str, max_len: int = 50) -> List[int]:
        words = text.lower().split()
        encoded = [self.word2idx.get(w, 1) for w in words[:max_len]]
        # Pad or truncate
        if len(encoded) < max_len:
            encoded = encoded + [0] * (max_len - len(encoded))
        return encoded
    
    def __len__(self) -> int:
        return len(self.word2idx)

# Generate synthetic text classification data
def generate_synthetic_data(num_samples: int = 1000, seq_len: int = 50) -> Tuple[List[str], List[int]]:
    """Generate synthetic text data for binary classification."""
    # Positive keywords
    positive_words = ["great", "excellent", "amazing", "wonderful", "fantastic", 
                     "love", "best", "beautiful", "perfect", "happy"]
    # Negative keywords
    negative_words = ["bad", "terrible", "awful", "horrible", "worst",
                     "hate", "poor", "boring", "disappointing", "sad"]
    
    texts = []
    labels = []
    
    for i in range(num_samples):
        # Determine label (balanced)
        label = i % 2
        num_words = np.random.randint(10, seq_len)
        
        if label == 1:  # Positive
            words = [np.random.choice(positive_words) for _ in range(num_words // 2)]
            words += [np.random.choice(negative_words) for _ in range(num_words - len(words))]
        else:  # Negative
            words = [np.random.choice(negative_words) for _ in range(num_words // 2)]
            words += [np.random.choice(positive_words) for _ in range(num_words - len(words))]
        
        # Shuffle words
        np.random.shuffle(words)
        text = " ".join(words)
        texts.append(text)
        labels.append(label)
    
    return texts, labels

# Custom dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 50):
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
        input_ids = torch.tensor(encoded, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return input_ids, label_tensor

# Create dataloaders
def make_dataloaders(batch_size: int = 32, val_split: float = 0.2, max_len: int = 50) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """Create dataloaders for text classification."""
    # Generate data
    texts, labels = generate_synthetic_data(num_samples=1000, seq_len=max_len)
    
    # Build vocabulary
    vocab = Vocabulary(max_size=1000)
    vocab.build_vocab(texts, min_freq=1)
    
    # Split data
    split_idx = int(len(texts) * (1 - val_split))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Create datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, vocab, max_len)
    val_dataset = TextClassificationDataset(val_texts, val_labels, vocab, max_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        
        # Feedforward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

# Transformer Encoder Classifier
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, num_classes: int = 2,
                 max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # Embedding + positional encoding
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Transpose for transformer (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Take [CLS] token (first token) for classification
        x = x[0, :, :]
        
        # Classification
        x = self.dropout(x)
        out = self.fc(x)
        
        return out

# Build model
def build_model(vocab: Vocabulary, device: torch.device) -> TransformerEncoderClassifier:
    """Build the transformer encoder classifier."""
    model = TransformerEncoderClassifier(
        vocab_size=len(vocab),
        d_model=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=2,
        max_len=50,
        dropout=0.1
    ).to(device)
    
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Vocabulary size: {len(vocab)}")
    return model

# Train function
def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          device: torch.device, num_epochs: int = 20, lr: float = 0.001) -> Dict[str, List[float]]:
    """Train the transformer encoder classifier."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    return history

# Evaluate function
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    # Calculate F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1
    }

# Predict function
def predict(model: nn.Module, texts: List[str], vocab: Vocabulary, 
            device: torch.device, max_len: int = 50) -> np.ndarray:
    """Predict labels for input texts."""
    model.eval()
    
    # Encode texts
    encoded_texts = [vocab.encode(text, max_len) for text in texts]
    input_tensor = torch.tensor(encoded_texts, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.cpu().numpy()

# Save artifacts
def save_artifacts(model: nn.Module, vocab: Vocabulary, history: Dict, 
                   output_dir: str = "output"):
    """Save model artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'num_classes': 2,
        'max_len': model.max_len
    }, model_path)
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump({
            'word2idx': vocab.word2idx,
            'idx2word': {str(k): v for k, v in vocab.idx2word.items()}
        }, f)
    
    # Save history
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")

# Main function
def main():
    """Main function to run the transformer encoder classifier."""
    print("=" * 60)
    print("Transformer Encoder Classifier for Text Classification")
    print("=" * 60)
    
    # Set device
    device = get_device()
    set_seed(42)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, vocab = make_dataloaders(batch_size=32, val_split=0.2, max_len=50)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(vocab, device)
    
    # Train model
    print("\nTraining model...")
    history = train(model, train_loader, val_loader, device, num_epochs=30, lr=0.001)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Train Metrics: Loss={train_metrics['loss']:.4f}, "
          f"Accuracy={train_metrics['accuracy']:.4f}, "
          f"F1 Macro={train_metrics['f1_macro']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Val Metrics: Loss={val_metrics['loss']:.4f}, "
          f"Accuracy={val_metrics['accuracy']:.4f}, "
          f"F1 Macro={val_metrics['f1_macro']:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, vocab, history, output_dir="output")
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"Train F1 Macro:  {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:    {val_metrics['f1_macro']:.4f}")
    print(f"Train Loss:      {train_metrics['loss']:.4f}")
    print(f"Val Loss:        {val_metrics['loss']:.4f}")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)
    
    checks_passed = True
    
    # Check 1: Train accuracy > 0.85
    check1 = train_metrics['accuracy'] > 0.85
    status1 = "PASS" if check1 else "FAIL"
    print(f"{status1} Train Accuracy > 0.85: {train_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check1
    
    # Check 2: Val accuracy > 0.80
    check2 = val_metrics['accuracy'] > 0.80
    status2 = "PASS" if check2 else "FAIL"
    print(f"{status2} Val Accuracy > 0.80: {val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check2
    
    # Check 3: Val F1 > 0.85
    check3 = val_metrics['f1_macro'] > 0.85
    status3= "PASS" if check3 else "FAIL"
    print(f"{status3} Val F1 Macro > 0.85: {val_metrics['f1_macro']:.4f}")
    checks_passed = checks_passed and check3
    
    # Check 4: Loss decreasing
    check4 = history['train_loss'][-1] < history['train_loss'][0]
    status4 = "PASS" if check4 else "FAIL"
    print(f"{status4} Training loss decreased: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    checks_passed = checks_passed and check4
    
    # Final summary
    print("\n" + "=" * 60)
    if checks_passed:
        print("ALL QUALITY CHECKS PASSED")
    else:
        print("SOME QUALITY CHECKS FAILED")
    print("=" * 60)
    
    return checks_passed

if __name__ == "__main__":
    main()
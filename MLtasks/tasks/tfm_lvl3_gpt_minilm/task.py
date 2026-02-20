"""
Mini-GPT (Causal LM) Implementation
A tiny causal transformer language model with training, evaluation, and sampling.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device setup
def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()

# Task metadata
def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "mini_gpt_causal_lm",
        "description": "Tiny causal transformer language model",
        "input_type": "text",
        "output_type": "text_generation",
        "model_type": "causal_lm",
        "vocab_size": 50,
        "max_seq_len": 32,
        "embedding_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "num_epochs": 50,
        "train_samples": 500,
        "val_samples": 100,
        "sample_length": 20
    }

# Set seed function
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Simple text dataset for training
class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, texts: List[str], vocab: Dict[str, int], max_seq_len: int = 32):
        self.texts = texts
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.vocab_size = len(vocab)
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        # Convert text to tokens
        tokens = [self.vocab.get(c, self.vocab['<unk>']) for c in text[:self.max_seq_len-1]]
        # Add end token
        if len(tokens) < self.max_seq_len - 1:
            tokens.append(self.vocab['<eos>'])
        # Pad if necessary
        while len(tokens) < self.max_seq_len:
            tokens.append(self.vocab['<pad>'])
        
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long)
        )

# Create vocabulary from text
def create_vocab(texts: List[str]) -> Dict[str, int]:
    """Create vocabulary from text corpus."""
    vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

# Generate synthetic text data for training
def generate_synthetic_data(num_samples: int, vocab: Dict[str, int], max_len: int = 20) -> List[str]:
    """Generate synthetic text data for training."""
    chars = list(vocab.keys())
    chars = [c for c in chars if c not in ['<pad>', '<unk>', '<eos>', '<sos>']]
    
    # If no characters available, use a default set
    if not chars:
        chars = list('abcdefghijklmnopqrstuvwxyz0123456789')
    
    texts = []
    for _ in range(num_samples):
        length = random.randint(5, max_len)
        text = ''.join(random.choice(chars) for _ in range(length))
        texts.append(text)
    return texts

# Causal self-attention module
class CausalSelfAttention(nn.Module):
    """Causal self-attention implementation."""
    
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int = 32, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("causal_mask", 
                           torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch, SeqLen, EmbedDim
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :T, :T]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        
        return self.out_proj(out)

# Transformer block
class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x

# Mini GPT model
class MiniGPT(nn.Module):
    """Tiny causal transformer language model."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_layers: int = 2, 
                 num_heads: int = 4, max_seq_len: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        # Apply weight tying (optional but common in GPT)
        self.token_embedding.weight = self.head.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, T = x.size()
        
        # Token embeddings
        tok_emb = self.token_embedding(x)  # (B, T, embed_dim)
        
        # Position embeddings
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)  # (1, T, embed_dim)
        
        # Combine embeddings
        x = tok_emb + pos_emb
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits
    
    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, max_length: int = 20, temperature: float = 1.0) -> torch.Tensor:
        """Generate text given starting tokens."""
        self.eval()
        
        # Ensure start_tokens is on correct device
        start_tokens = start_tokens.to(self.token_embedding.weight.device)
        
        generated = start_tokens.clone()
        
        for _ in range(max_length):
            # Get predictions for the last token
            logits = self(generated)[:, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == 2:  # <eos> token
                break
        
        return generated

# Make dataloaders
def make_dataloaders(batch_size: int = 32, max_seq_len: int = 32) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create training and validation dataloaders."""
    # Generate synthetic text data
    vocab = create_vocab([''])  # Initialize with special tokens
    train_texts = generate_synthetic_data(500, vocab, max_len=max_seq_len)
    val_texts = generate_synthetic_data(100, vocab, max_len=max_seq_len)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, vocab, max_seq_len)
    val_dataset = TextDataset(val_texts, vocab, max_seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab

# Build model
def build_model(vocab_size: int, embed_dim: int = 64, num_layers: int = 2, 
                num_heads: int = 4, max_seq_len: int = 32, dropout: float = 0.1) -> MiniGPT:
    """Build the MiniGPT model."""
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    return model.to(device)

# Train function
def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          num_epochs: int = 50, learning_rate: float = 1e-3) -> Dict[str, List[float]]:
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_perplexity': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss (flatten for cross-entropy)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Evaluate on validation set
        val_loss, val_perplexity = evaluate(model, val_loader, return_perplexity=True)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_perplexity'].append(val_perplexity)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Perplexity: {val_perplexity:.2f}")
    
    return history

# Evaluate function
def evaluate(model: nn.Module, data_loader: DataLoader, return_perplexity: bool = False) -> Tuple[float, Dict[str, float]]:
    """Evaluate the model on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    avg_loss = total_loss / total_samples
    
    metrics = {'loss': avg_loss}
    
    if return_perplexity:
        perplexity = np.exp(avg_loss)
        metrics['perplexity'] = perplexity
    
    return avg_loss, metrics

# Predict function
def predict(model: nn.Module, vocab: Dict[str, int], start_text: str = "a", 
            max_length: int = 20, temperature: float = 1.0) -> str:
    """Generate text prediction."""
    model.eval()
    
    # Convert start text to tokens
    tokens = [vocab.get(c, vocab['<unk>']) for c in start_text]
    tokens = [vocab['<sos>']] + tokens  # Add start token
    
    # Convert to tensor
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(input_tensor, max_length=max_length, temperature=temperature)
    
    # Convert tokens back to text
    reverse_vocab = {v: k for k, v in vocab.items()}
    generated_tokens = output[0].cpu().numpy()
    
    # Remove padding and convert to text
    text = ''
    for token in generated_tokens[len(tokens):]:  # Skip input tokens
        if token == vocab['<eos>']:
            break
        if token in reverse_vocab:
            text += reverse_vocab[token]
    
    return text

# Save artifacts
def save_artifacts(model: nn.Module, vocab: Dict[str, int], history: Dict[str, List[float]], 
                   output_dir: str = 'output/tasks/tfm_lvl3_gpt_minilm') -> None:
    """Save model artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'embed_dim': model.embed_dim,
        'num_layers': len(model.transformer_blocks),
        'num_heads': model.transformer_blocks[0].attention.num_heads if hasattr(model.transformer_blocks[0], 'attention') else 4,
        'max_seq_len': model.max_seq_len,
    }, model_path)
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    
    # Save```python
 history
    history_path = os.path.join(output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

# Load artifacts
def load_artifacts(output_dir: str = 'output/tasks/tfm_lvl3_gpt_minilm') -> Tuple[MiniGPT, Dict[str, int], Dict[str, List[float]]]:
    """Load model artifacts."""
    # Load vocabulary
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Load model config
    model_path = os.path.join(output_dir, 'model.pt')
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Build model
    model = MiniGPT(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=checkpoint['embed_dim'],
        num_layers=checkpoint['num_layers'],
        num_heads=checkpoint['num_heads'],
        max_seq_len=checkpoint['max_seq_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load history
    history_path = os.path.join(output_dir, 'history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return model, vocab, history

# Main function
def main():
    """Main function to run the training."""
    # Get task metadata
    metadata = get_task_metadata()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create dataloaders
    train_loader, val_loader, vocab = make_dataloaders(
        batch_size=metadata['batch_size'],
        max_seq_len=metadata['max_seq_len']
    )
    
    # Build model
    model = build_model(
        vocab_size=len(vocab),
        embed_dim=metadata['embedding_dim'],
        num_layers=metadata['num_layers'],
        num_heads=metadata['num_heads'],
        max_seq_len=metadata['max_seq_len'],
        dropout=0.1
    )
    
    # Train model
    history = train(
        model,
        train_loader,
        val_loader,
        num_epochs=metadata['num_epochs'],
        learning_rate=metadata['learning_rate']
    )
    
    # Save artifacts
    save_artifacts(model, vocab, history)
    
    # Test prediction
    print("\nTesting prediction:")
    test_text = predict(model, vocab, start_text="a", max_length=10)
    print(f"Generated text: {test_text}")
    
    return model, vocab, history

if __name__ == "__main__":
    main()

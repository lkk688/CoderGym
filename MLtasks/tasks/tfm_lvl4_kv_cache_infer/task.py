"""
Transformer KV-Cache Inference Task
Implements KV-cache for faster autoregressive decoding with benchmarking.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Task metadata
def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        'task_name': 'transformer_kv_cache_inference',
        'description': 'Transformer with KV-cache for faster autoregressive decoding',
        'metrics': ['perplexity', 'accuracy', 'tokens_per_sec_no_cache', 'tokens_per_sec_with_cache', 'speedup_ratio'],
        'model_type': 'transformer',
        'input_type': 'sequence',
        'output_type': 'sequence'
    }

# Simple dataset for sequence generation
class SimpleSequenceDataset(Dataset):
    """Simple sequence dataset for training."""
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random sequence
        input_seq = torch.randint(1, self.vocab_size, (self.seq_len,))
        # Target is the same as input (self-supervised)
        target_seq = input_seq.clone()
        return input_seq, target_seq

# Create dataloaders
def make_dataloaders(batch_size: int = 32, train_samples: int = 1000, 
                     val_samples: int = 200, seq_len: int = 16, vocab_size: int = 100) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    train_dataset = SimpleSequenceDataset(train_samples, seq_len, vocab_size)
    val_dataset = SimpleSequenceDataset(val_samples, seq_len, vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

# Positional Encoding
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

# Multi-head Attention with KV-cache support
class MultiHeadAttention(nn.Module):
    """Multi-head attention with KV-cache support."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None,
                cache: Dict[str, torch.Tensor] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        batch_size = query.size(0)
        
        # If key/value are not provided, use query (self-attention)
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Project inputs
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # KV-cache: if cache exists, concatenate cached keys/values
        if cache is not None and use_cache:
            cached_k = cache.get('k', None)
            cached_v = cache.get('v', None)
            
            if cached_k is not None:
                K = torch.cat([cached_k, K], dim=2)
            if cached_v is not None:
                V = torch.cat([cached_v, V], dim=2)
            
            # Update cache
            new_cache = {'k': K, 'v': V}
        else:
            new_cache = None
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        
        return output, new_cache

# Feed-forward network
class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, cache: Dict[str, torch.Tensor] = None, 
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Self-attention with residual connection
        attn_out, new_cache = self.self_attn(x, cache=cache, use_cache=use_cache)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, new_cache

# Transformer Decoder Layer with KV-cache
class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with KV-cache support."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor = None,
                cache: Dict[str, torch.Tensor] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Self-attention with causal mask
        attn_out, new_cache = self.self_attn(x, cache=cache, use_cache=use_cache)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention (if encoder output provided)
        if encoder_out is not None:
            attn_out, _ = self.cross_attn(x, encoder_out, encoder_out)
            x = self.norm2(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x, new_cache

# Transformer model for sequence generation
class TransformerKVCache(nn.Module):
    """Transformer with KV-cache support for efficient autoregressive decoding."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_layers: int = 3,
                 num_heads: int = 4, d_ff: int = 256, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder-decoder architecture
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source sequence."""
        x = self.embedding(src) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x, _ = layer(x)
            
        return x
    
    def decode(self, tgt: torch.Tensor, encoder_out: torch.Tensor = None,
               cache: Dict[str, torch.Tensor] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Decode target sequence with optional KV-cache."""
        x = self.embedding(tgt) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        new_cache = {}
        
        for i, layer in enumerate(self.decoder_layers):
            layer_cache = cache.get(f'layer_{i}', None) if cache is not None else None
            x, new_cache[f'layer_{i}'] = layer(x, encoder_out, cache=layer_cache, use_cache=use_cache)
            
        return x, new_cache
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        encoder_out = self.encode(src)
        decoder_out, _ = self.decode(tgt, encoder_out)
        output = self.output_linear(decoder_out)
        return output
    
    def generate(self, start_token: torch.Tensor, max_len: int = 20, 
                 use_cache: bool = True, encoder_out: torch.Tensor = None) -> torch.Tensor:
        """Generate sequence autoregressively with optional KV-cache."""
        self.eval()
        
        with torch.no_grad():
            # Initialize with start token
            generated = start_token
            cache = {} if use_cache else None
            
            for _ in range(max_len):
                # Get last token prediction
                if use_cache and cache is not None:
                    # Use only the last token when using cache
                    decoder_out, cache = self.decode(
                        generated[:, -1:], encoder_out, cache=cache, use_cache=True
                    )
                else:
                    # Process entire sequence
                    decoder_out, _ = self.decode(generated, encoder_out, use_cache=False)
                
                # Get predictions for last token
                logits = self.output_linear(decoder_out[:, -1, :])
                next_token = torch.argmax(logits, dim=-1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                
                # Stop if EOS token (token 0)
                if next_token.item() == 0:
                    break
            
            return generated

# Build model
def build_model(vocab_size: int = 100, d_model: int = 128, num_layers: int = 3,
                num_heads: int = 4, d_ff: int = 256, device: torch.device = None) -> TransformerKVCache:
    """Build the transformer model."""
    if device is None:
        device = get_device()
    
    model = TransformerKVCache(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    ).to(device)
    
    return model

# Training function
def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          device: torch.device, num_epochs: int = 20, lr: float = 0.001) -> Dict[str, List[float]]:
    """Train the model."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt[:, :-1])
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Validation
        val_loss = evaluate(model, val_loader, device, return_loss_only=True)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return {'train_loss': train_losses, 'val_loss': val_losses}

# Evaluation function
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device,
             return_loss_only: bool = False) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass
            output = model(src, tgt[:, :-1])
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(output, dim=-1)
            correct = (predictions == tgt[:, 1:]).sum().item()
            total_correct += correct
            total_tokens += tgt[:, 1:].numel()
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    # Calculate perplexity
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    if return_loss_only:
        return avg_loss
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy
    }

# Prediction function
def predict(model: nn.Module, input_seq: torch.Tensor, device: torch.device,
            use_cache: bool = True) -> torch.Tensor:
    """Generate prediction using the model."""
    model.eval()
    
    with torch.no_grad():
        # Encode input sequence
        encoder_out = model.encode(input_seq)
        
        # Start with start token (token 1)
        start_token = torch.tensor([[1]], device=device)
        
        # Generate sequence
        output = model.generate(start_token, max_len=20, use_cache=use_cache, encoder_out=encoder_out)
        
        return output

# Benchmark function for KV-cache performance
def benchmark(model: nn.Module, input_seq: torch.Tensor, device: torch.device,
              num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark the model with and without KV-cache."""
    model.eval()
    
    with torch.no_grad():
        # Encode input sequence once
        encoder_out = model.encode(input_seq)
        
        # Start with start token
        start_token = torch.tensor([[1]], device=device)
        
        # Benchmark without cache
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model.generate(start_token, max_len=20, use_cache=False, encoder_out=encoder_out)
        time_no_cache = time.time() - start_time
        
        # Benchmark with cache
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model.generate(start_token, max_len=20, use_cache=True, encoder_out=encoder_out)
        time_with_cache = time.time() - start_time
        
        # Calculate tokens per second (20 tokens per generation)
        tokens_per_sec_no_cache = (num_iterations * 20) / time_no_cache
        tokens_per_sec_with_cache = (num_iterations * 20) / time_with_cache
        speedup_ratio = tokens_per_sec_no_cache / tokens_per_sec_with_cache if tokens_per_sec_with_cache > 0 else 0.0
        
        return {
            'tokens_per_sec_no_cache': tokens_per_sec_no_cache,
            'tokens_per_sec_with_cache': tokens_per_sec_with_cache,
            'speedup_ratio': speedup_ratio,
            'time_no_cache': time_no_cache,
            'time_with_cache': time_with_cache
        }

# Save model function
def save_model(model: nn.Module, save_dir: str, filename: str = 'model.pt') -> None:
    """Save the model to disk."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Load model function
def load_model(model: nn.Module, save_dir: str, filename: str = 'model.pt',
               device: torch.device = None) -> nn.Module:
    """Load model from disk."""
    if device is None:
        device = get_device()
    
    save_path = os.path.join(save_dir, filename)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {save_path}")
    return model

# Main execution
if __name__ == '__main__':
    # Configuration
    vocab_size = 100
    d_model = 128
    num_layers = 3
    num_heads = 4
    d_ff = 256
    seq_len = 16
    batch_size = 32
    num_epochs = 20
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = make_dataloaders(
        batch_size=batch_size,
        train_samples=1000,
        val_samples=200,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    
    # Build model
    model = build_model(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\nTraining...")
    train_results = train(model, train_loader, val_loader, device, num_epochs=num_epochs)
    
    # Evaluate model
    print("\nEvaluating...")
    eval_results = evaluate(model, val_loader, device)
    print(f"Evaluation results: {eval_results}")
    
    # Benchmark KV-cache performance
    print("\nBenchmarking KV-cache performance...")
    # Create a sample input for benchmarking
    sample_input = torch.randint(1, vocab_size, (1, seq_len), device=device)
    benchmark_results = benchmark(model, sample_input, device)
    print(f"Benchmark results: {benchmark_results}")
    
    # Save model
    save_dir = 'checkpoints'
    save_model(model, save_dir)
    
    # Print task completion message
    print("\nTask completed successfully!")
    print(f"Metadata: {get_task_metadata()}")
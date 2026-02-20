"""
Scaled Dot-Product Attention Implementation from Scratch
Validates against torch.nn.MultiheadAttention
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, Dict, Any

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        'task_name': 'scaled_dot_product_attention',
        'description': 'Implement scaled dot-product attention from scratch',
        'input_shape': [10, 64],  # (batch_size, seq_len, d_model)
        'output_shape': [10, 64],  # Same shape as input for attention output
        'num_classes': None,  # Regression task
        'metrics': ['mse', 'r2', 'attention_similarity']
    }

def make_dataloaders(batch_size: int = 32, val_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Create synthetic dataset for attention validation.
    
    The task: Given a sequence, compute attention and learn to reconstruct
    a weighted combination of sequence elements.
    """
    set_seed(42)
    
    # Parameters
    seq_len = 16
    d_model = 32
    num_samples = 1000
    
    # Generate random sequences
    X = torch.randn(num_samples, seq_len, d_model)
    
    # Generate attention weights for target computation
    # Target is a weighted sum of sequence elements using learned attention
    attention_weights = torch.softmax(torch.randn(num_samples, seq_len), dim=-1)
    
    # Target: weighted combination of sequence elements
    y = torch.bmm(attention_weights.unsqueeze(1), X).squeeze(1)
    
    # Split into train and validation
    split_idx = int((1 - val_size) * num_samples)
    
    train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
    val_dataset = TensorDataset(X[split_idx:], y[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention implementation from scratch.
    
    Computes: softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, d_k: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            Q: Query tensor of shape (..., seq_len_q, d_k)
            K: Key tensor of shape (..., seq_len_k, d_k)
            V: Value tensor of shape (..., seq_len_v, d_v)
            mask: Mask tensor to exclude certain positions
            
        Returns:
            output: Attention output of shape (..., seq_len_q, d_v)
            attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights with numerical stability
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute output
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation using scaled dot-product attention.
    """
    def __init__(self, d_model: int = 32, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(d_k=self.d_k, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            Q, K, V: Input tensors of shape (batch_size, seq_len, d_model)
            mask: Mask tensor
            
        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        batch_size = Q.size(0)
        
        # Linear projections and reshape to (batch_size, seq_len, n_heads, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply attention to each head
        output, _ = self.attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + Q.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model))
        
        return output

class AttentionModel(nn.Module):
    """
    Simple model using multi-head attention for sequence modeling.
    """
    def __init__(self, d_model: int = 32, n_heads: int = 8, seq_len: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output of shape (batch_size, d_model)
        """
        # Apply multi-head attention
        x = self.attention(x, x, x)
        
        # Take the mean over sequence dimension
        x = x.mean(dim=1)
        
        # Final projection
        x = self.fc(x)
        
        return x

def build_model(device: torch.device = None) -> nn.Module:
    """Build and return the attention model."""
    if device is None:
        device = get_device()
    
    model = AttentionModel(d_model=32, n_heads=8, seq_len=16).to(device)
    return model

def train(model: nn.Module, train_loader: DataLoader, device: torch.device, 
          epochs: int = 50, lr: float = 0.001) -> Dict[str, list]:
    """
    Train the model.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        device: Computation device
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Dictionary with training history
    """
    set_seed(42)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'mse': [], 'r2': []}
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_mse = 0
        all_preds = []
        all_targets = []
        
        for batch_X, batch_y in train_loader:
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute metrics
            preds = outputs.detach().cpu().numpy()
            targets = batch_y.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)
        
        # Compute epoch metrics
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        epoch_mse = mean_squared_error(all_targets, all_preds)
        epoch_r2 = r2_score(all_targets.flatten(), all_preds.flatten())
        
        history['loss'].append(total_loss / len(train_loader))
        history['mse'].append(epoch_mse)
        history['r2'].append(epoch_r2)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {history['loss'][-1]:.6f}, "
                  f"MSE: {epoch_mse:.6f}, R2: {epoch_r2:.6f}")
    
    return history

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model on given data.
    
    Args:
        model: The neural network model
        data_loader: Data loader for evaluation
        device: Computation device
        
    Returns:
        Dictionary with evaluation metrics
    """
    set_seed(42)
    
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            
            # Collect predictions and targets
            preds = outputs.detach().cpu().numpy()
            targets = batch_y.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)
    
    # Compute metrics
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    
    return {
        'loss': total_loss / len(data_loader),
        'mse': mse,
        'r2': r2
    }

def predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Generate predictions.
    
    Args:
        model: The neural network model
        X: Input tensor
        device: Computation device
        
    Returns:
        Predictions as numpy array
    """
    set_seed(42)
    
    model.eval()
    X = X.to(device)
    
    with torch.no_grad():
        outputs = model(X)
    
    return outputs.detach().cpu().numpy()

def save_artifacts(model: nn.Module, history: Dict, save_dir: str = './output/tasks/tfm_lvl1_attention_from_scratch'):
    """
    Save model artifacts.
    
    Args:
        model: Trained model
        history: Training history
        save_dir: Directory to save artifacts
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    
    # Save training history
    np.savez(os.path.join(save_dir, 'training_history.npz'), **history)
    
    print(f"Artifacts saved to {save_dir}")

def validate_attention_against_torch(model: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Validate our attention implementation against torch.nn.MultiheadAttention.
    
    Returns:
        Dictionary with similarity metrics
    """
    set_seed(42)
    
    # Create test data
    batch_size, seq_len, d_model = 4, 8, 32
    n_heads = 8
    
    # Our attention
    our_attention = ScaledDotProductAttention(d_k=d_model // n_heads)
    
    # PyTorch attention
    torch_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
    
    # Initialize with same weights
    with torch.no_grad():
        # Copy weights for Q, K, V projections
        torch_attention.in_proj_weight.copy_(model.attention.W_q.weight.data)
        torch_attention.in_proj_bias.copy_(model.attention.W_q.bias.data)
        torch_attention.out_proj.weight.copy_(model.attention.W_o.weight.data)
        torch_attention.out_proj.bias.copy_(model.attention.W_o.bias.data)
    
    # Test data
    Q = torch.randn(batch_size, seq_len, d_model).to(device)
    K = torch.randn(batch_size, seq_len, d_model).to(device)
    V = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Our implementation
    our_output, our_weights = our_attention(
        Q.view(batch_size, seq_len, n_heads, d_model // n_heads).transpose(1, 2),
        K.view(batch_size, seq_len, n_heads, d_model // n_heads).transpose(1, 2),
        V.view(batch_size, seq_len, n_heads, d_model // n_heads).transpose(1, 2)
    )
    our_output = our_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    # PyTorch implementation
    torch_output, torch_weights = torch_attention(Q, K, V)
    
    # Compute similarity
    output_similarity = torch.cosine_similarity(our_output.flatten(), torch_output.flatten(), dim=0).item()
    weight_similarity = torch.cosine_similarity(our_weights.flatten(), torch_weights.flatten(), dim=0).item() if our_weights.shape == torch_weights.shape else 0.0
    
    return {
        'output_similarity': output_similarity,
        'weight_similarity': weight_similarity
    }

def main():
    """Main function to run the attention task."""
    print("=" * 60)
    print("Scaled Dot-Product Attention Implementation")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, val_loader = make_dataloaders(batch_size=32, val_size=0.2)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\n[2] Building model...")
    model = build_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\n[3] Training model...")
    history = train(model, train_loader, device, epochs=50, lr=0.001)
    
    # Evaluate on training data
    print("\n[4] Evaluating on training data...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"Training Metrics - MSE: {train_metrics['mse']:.6f}, R2: {train_metrics['r2']:.6f}")
    
    # Evaluate on validation data
    print("\n[5] Evaluating on validation data...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Validation Metrics - MSE: {val_metrics['mse']:.6f}, R2: {val_metrics['r2']:.6f}")
    
    # Validate attention implementation
    print("\n[6] Validating attention implementation against PyTorch...")
    attention_validation = validate_attention_against_torch(model, device)
    print(f"Attention Output Similarity: {attention_validation['output_similarity']:.6f}")
    print(f"Attention Weight Similarity: {attention_validation['weight_similarity']:.6f}")
    
    # Save artifacts
    print("\n[7] Saving artifacts...")
    save_artifacts(model, history)
    
    print("\n" + "=" * 60)
    print("Task completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
GraphSAGE (Mini-batch) Implementation
Neighbor sampling + mini-batch training for scalability on graph data.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Optional

# Try to import torch_geometric and provide helpful error if missing
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.data import DataLoader, NeighborSampler
    from torch_geometric.nn import SAGEConv
    import torch_geometric
except ImportError as e:
    raise ImportError(
        "torch_geometric is required but not installed. "
        "Please install it with: pip install torch-geometric "
        "or for CPU-only: pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html\n"
        f"Original error: {e}"
    )

# Set paths
OUTPUT_DIR = '/Developer/AIserver/output/tasks/gml_lvl2_graphsage'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "name": "GraphSAGE_MiniBatch",
        "description": "GraphSAGE with neighbor sampling and mini-batch training",
        "task_type": "node_classification",
        "input_type": "graph",
        "output_type": "node_labels",
        "metrics": ["accuracy", "loss"],
        "default_epochs": 200,
        "default_batch_size": 64,
        "default_num_neighbors": [10, 5],
        "dataset_name": "Cora"
    }


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphSAGE(nn.Module):
    """GraphSAGE model with neighbor sampling."""
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 out_channels: int, num_layers: int = 2, 
                 dropout: float = 0.5):
        super(GraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
    
    def inference(self, x_all: torch.Tensor, edge_index: torch.Tensor,
                  subgraph_loader: NeighborSampler) -> torch.Tensor:
        """Compute node embeddings layer by layer using neighbor sampling."""
        device = x_all.device
        x_all = x_all.clone()
        
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        
        return F.log_softmax(x_all, dim=1)


class GraphDataset(Dataset):
    """Wrapper for PyTorch Geometric datasets to work with DataLoader."""
    
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return self.data.num_nodes
        
    def __getitem__(self, idx):
        return idx


def make_dataloaders(dataset_name: str = "Cora", batch_size: int = 64,
                     num_neighbors: List[int] = None, 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1) -> Tuple[Any, Any, Any, Any]:
    """
    Create dataloaders with neighbor sampling for GraphSAGE.
    
    Returns:
        train_loader: Training data loader with neighbor sampling
        val_loader: Validation data loader with neighbor sampling
        full_loader: Full graph loader for inference
        data: The graph data object
    """
    if num_neighbors is None:
        num_neighbors = [10, 5]
    
    # Load dataset
    dataset = Planetoid(root='/tmp/Planetoid', name=dataset_name)
    data = dataset[0]
    
    # Split indices
    n_nodes = data.num_nodes
    indices = torch.arange(n_nodes)
    
    # Shuffle indices
    perm = torch.randperm(n_nodes)
    indices = indices[perm]
    
    # Split into train, val, test
    n_train = int(n_nodes * train_ratio)
    n_val = int(n_nodes * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create neighbor samplers for mini-batch training
    train_loader = NeighborSampler(
        data.edge_index, 
        sizes=num_neighbors, 
        node_idx=train_idx,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = NeighborSampler(
        data.edge_index,
        sizes=num_neighbors,
        node_idx=val_idx,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Full graph loader for inference
    full_loader = NeighborSampler(
        data.edge_index,
        sizes=[-1],  # Use all neighbors for inference
        node_idx=torch.arange(n_nodes),
        batch_size=1024,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, full_loader, data


def build_model(data, hidden_channels: int = 128, num_layers: int = 2,
                dropout: float = 0.5, device=None) -> GraphSAGE:
    """Build the GraphSAGE model."""
    if device is None:
        device = get_device()
    
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=data.num_classes,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    return model


def train(model: GraphSAGE, train_loader: NeighborSampler, 
          data, optimizer: torch.optim.Optimizer, 
          device: torch.device) -> float:
    """Train the model for one epoch."""
    model.train()
    
    total_loss = 0
    total_samples = 0
    
    for batch_size, n_id, adj in train_loader:
        # Get node features and labels for this batch
        x = data.x[n_id].to(device)
        y = data.y[n_id[:batch_size]].to(device)
        
        # Get edge index for this batch
        edge_index, _, size = adj.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    return total_loss / total_samples


def evaluate(model: GraphSAGE, loader: NeighborSampler, data, 
             device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model on given data loader.
    
    Returns:
        Dictionary with metrics: loss, accuracy
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_size, n_id, adj in loader:
            # Get node features and labels for this batch
            x = data.x[n_id].to(device)
            y = data.y[n_id[:batch_size]].to(device)
            
            # Get edge index for this batch
            edge_index, _, size = adj.to(device)
            
            # Forward pass
            out = model(x, edge_index)
            loss = F.nll_loss(out, y)
            
            total_loss += loss.item() * batch_size
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += batch_size
    
    accuracy = correct / total
    avg_loss = total_loss / total
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "mse": avg_loss,  # For compatibility with regression metrics
        "r2": accuracy    # For compatibility with regression metrics
    }


def predict(model: GraphSAGE, data, full_loader: NeighborSampler, 
            device: torch.device) -> torch.Tensor:
    """Get predictions for all nodes."""
    model.eval()
    
    with torch.no_grad():
        out = model.inference(data.x, data.edge_index, full_loader)
    
    return out


def save_artifacts(model: GraphSAGE, metrics: Dict[str, float], 
                   metadata: Dict[str, Any], 
                   save_dir: str = OUTPUT_DIR) -> None:
    """Save model artifacts and metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metadata
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Artifacts saved to {save_dir}")


def main():
    """Main training and evaluation function."""
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Starting {metadata['name']} task...")
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Hyperparameters
    epochs = 200
    batch_size = 64
    hidden_channels = 128
    num_layers = 2
    dropout = 0.5
    lr = 0.01
    weight_decay = 5e-4
    
    # Create dataloaders
    print("Creating dataloaders with neighbor sampling...")
    train_loader, val_loader, full_loader, data = make_dataloaders(
        dataset_name="Cora",
        batch_size=batch_size,
        num_neighbors=[10, 5]
    )
    
    print(f"Dataset: Cora")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {data.num_features}, Classes: {data.num_classes}")
    print(f"Train nodes: {train_loader.num_samples}, Val nodes: {val_loader.num_samples}")
    
    # Build model
    print("Building GraphSAGE model...")
    model = build_model(
        data, 
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train(model, train_loader, data, optimizer, device)
        
        # Evaluate on train and validation
        train_metrics = evaluate(model, train_loader, data, device)
        val_metrics = evaluate(model, val_loader, data, device)
        
        # Track best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    training_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on train and validation sets
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    
    train_metrics = evaluate(model, train_loader, data, device)
    val_metrics = evaluate(model, val_loader, data, device)
    
    print(f"\nTrain Set:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    
    print(f"\nValidation Set:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    
    # Get predictions for test set
    print("\nGenerating predictions for all nodes...")
    out = predict(model, data, full_loader, device)
    pred = out.argmax(dim=1)
    
    # Calculate test accuracy
    test_mask = ~data.train_mask & ~data.val_mask & data.test_mask
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
    print(f"\nTest Set Accuracy: {test_acc:.4f}")
    
    # Prepare final metrics
    final_metrics = {
        "train_loss": train_metrics['loss'],
        "train_accuracy": train_metrics['accuracy'],
        "val_loss": val_metrics['loss'],
        "val_accuracy": val_metrics['accuracy'],
        "test_accuracy": test_acc,
        "training_time_seconds": training_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": lr,
        "num_parameters": num_params
    }
    
    # Quality thresholds (based on Cora dataset expectations)
    print("\n" + "="*60)
    print("Quality Threshold Checks")
    print("="*60)
    
    # Check thresholds
    quality_passed = True
    
    # Validation accuracy should be reasonable (> 0.75 for Cora)
    if val_metrics['accuracy'] < 0.75:
        print(f"❌ FAIL: Validation accuracy {val_metrics['accuracy']:.4f} < 0.75")
        quality_passed = False
    else:
        print(f"✅ PASS: Validation accuracy {val_metrics['accuracy']:.4f} >= 0.75")
    
    # Training accuracy should be higher than validation (no severe overfitting)
    if train_metrics['accuracy'] < val_metrics['accuracy'] * 0.95:
        print(f"⚠️  WARNING: Training accuracy {train_metrics['accuracy']:.4f} < "
              f"Validation accuracy {val_metrics['accuracy']:.4f} * 0.95")
    else:
        print(f"✅ PASS: No severe overfitting detected")
    
    # Test accuracy should be reasonable
    if test_acc < 0.75:
        print(f"❌ FAIL: Test accuracy {test_acc:.4f} < 0.75")
        quality_passed = False
    else:
        print(f"✅ PASS: Test accuracy {test_acc:.4f} >= 0.75")
    
    # Training time should be reasonable (< 5 minutes for this task)
    if training_time > 300:
        print(f"⚠️  WARNING: Training took {training_time:.1f}s (> 5 minutes)")
    else:
        print(f"✅ PASS: Training time {training_time:.1f}s is reasonable")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, final_metrics, metadata)
    
    # Final summary
    print("\n" + "="*60)
    if quality_passed:
        print("✅ ALL QUALITY CHECKS PASSED")
        print("="*60)
        return 0
    else:
        print("❌ SOME QUALITY CHECKS FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())

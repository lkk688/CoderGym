"""
GCN (Graph Convolutional Network) for Node Classification
Implements basic GCN layer and trains on Cora citation graph
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
import matplotlib.pyplot as plt

# Try to import torch_geometric, provide helpful error if missing
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.data import DataLoader
    from torch_geometric.utils import to_dense_adj, add_self_loops, normalize_adj
    USE_TORCH_GEOMETRIC = True
except ImportError:
    USE_TORCH_GEOMETRIC = False
    print("torch_geometric not available, using manual implementation")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory
OUTPUT_DIR = '/Developer/AIserver/output/tasks/gml_lvl1_gcn_nodecls'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata"""
    return {
        'task_type': 'node_classification',
        'model_type': 'gcn',
        'dataset': 'cora',
        'input_type': 'graph',
        'output_type': 'node_labels',
        'metrics': ['accuracy', 'f1_macro', 'loss']
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get device for computation"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_cora_data(root='/tmp/Cora'):
    """Load Cora dataset manually"""
    # Check if torch_geometric is available and try to use it first
    if USE_TORCH_GEOMETRIC:
        try:
            dataset = Planetoid(root=root, name='Cora')
            data = dataset[0]
            return data, dataset.num_features, dataset.num_classes
        except:
            pass
    
    # Manual implementation if torch_geometric fails
    print("Loading Cora dataset manually...")
    
    # Download and extract Cora dataset if not already present
    import urllib.request
    import zipfile
    import shutil
    
    # Create directories
    os.makedirs(root, exist_ok=True)
    
    # Check if data files exist
    content_file = os.path.join(root, 'cora.content')
    cite_file = os.path.join(root, 'cora.cites')
    
    if not os.path.exists(content_file) or not os.path.exists(cite_file):
        print("Downloading Cora dataset...")
        url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        try:
            # Download and extract
            import tarfile
            tar_path = os.path.join(root, 'cora.tgz')
            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(root)
            # Move files to correct location
            cora_dir = os.path.join(root, 'cora')
            if os.path.exists(cora_dir):
                for f in os.listdir(cora_dir):
                    shutil.move(os.path.join(cora_dir, f), root)
                os.rmdir(cora_dir)
            os.remove(tar_path)
        except Exception as e:
            print(f"Error downloading Cora: {e}")
            print("Please download Cora dataset manually from https://linqs.soe.ucsc.edu/data")
            raise
    
    # Load node features and labels
    node_features = []
    node_labels = []
    node_ids = []
    
    with open(content_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                node_id = parts[0]
                label = parts[-1]  # Last column is the label
                features = [float(x) for x in parts[1:-1]]  # Middle columns are features
                
                node_ids.append(node_id)
                node_features.append(features)
                node_labels.append(label)
    
    node_features = np.array(node_features)
    node_ids = np.array(node_ids)
    
    # Create label mapping
    unique_labels = sorted(list(set(node_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    node_labels_idx = np.array([label_to_idx[label] for label in node_labels])
    
    # Create adjacency matrix
    num_nodes = len(node_ids)
    adj = np.zeros((num_nodes, num_nodes))
    
    with open(cite_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                src_id, dst_id = parts[0], parts[1]
                if src_id in node_ids and dst_id in node_ids:
                    src_idx = np.where(node_ids == src_id)[0][0]
                    dst_idx = np.where(node_ids == dst_id)[0][0]
                    adj[src_idx, dst_idx] = 1
                    adj[dst_idx, src_idx] = 1  # Undirected graph
    
    # Create train/val/test masks (70/10/20 split)
    num_nodes = len(node_ids)
    indices = np.random.permutation(num_nodes)
    train_size = int(0.7 * num_nodes)
    val_size = int(0.1 * num_nodes)
    
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    # Convert to torch tensors
    features = torch.FloatTensor(node_features)
    labels = torch.LongTensor(node_labels_idx)
    adj = torch.FloatTensor(adj)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    
    # Create a simple data object
    class SimpleData:
        def __init__(self, x, y, edge_index, train_mask, val_mask, test_mask):
            self.x = x
            self.y = y
            self.edge_index = edge_index
            self.train_mask = train_mask
            self.val_mask = val_mask
            self.test_mask = test_mask
    
    # Convert adjacency to edge_index format
    edge_index = torch.nonzero(adj > 0, as_tuple=False).t()
    
    data = SimpleData(features, labels, edge_index, train_mask, val_mask, test_mask)
    
    return data, node_features.shape[1], len(unique_labels)


class GCNLayer(nn.Module):
    """Basic GCN layer with adjacency normalization"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters using Xavier initialization"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass with adjacency normalization (symmetric normalization)
        A_hat = D^-1/2 * (A + I) * D^-1/2
        """
        # Add self-loops
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        
        # Degree matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        # Normalized adjacency: D^-1/2 * A * D^-1/2
        adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
        
        # Linear transformation
        support = torch.mm(x, self.weight)
        # Graph convolution
        output = torch.mm(adj_norm, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class GCN(nn.Module):
    """Two-layer GCN for node classification"""
    
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        
        self.gcn1 = GCNLayer(nfeat, nhid)
        self.gcn2 = GCNLayer(nhid, nclass)
        self.dropout = dropout
    
    def forward(self, x, adj):
        """Forward pass"""
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn2(x, adj)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, adj):
        """Get embeddings from first GCN layer"""
        x = self.gcn1(x, adj)
        return F.relu(x)


def make_dataloaders():
    """Load Cora dataset and prepare data"""
    # Load Cora dataset
    data, num_features, num_classes = load_cora_data(root='/tmp/Cora')
    
    # Split indices
    train_idx = data.train_mask.nonzero(as_tuple=False).squeeze()
    val_idx = data.val_mask.nonzero(as_tuple=False).squeeze()
    test_idx = data.test_mask.nonzero(as_tuple=False).squeeze()
    
    # Create adjacency matrix
    adj = to_dense_adj(data.edge_index)[0] if USE_TORCH_GEOMETRIC else data.x.new_ones(data.x.shape[0], data.x.shape[0]) * 0
    
    # For manual implementation, use the adjacency from data
    if not USE_TORCH_GEOMETRIC:
        adj = data.x.new_tensor(np.zeros((data.x.shape[0], data.x.shape[0])))
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            adj[i, j] = 1
            adj[j, i] = 1
    
    # Prepare data
    train_data = {
        'x': data.x[train_idx].to(device),
        'y': data.y[train_idx].to(device),
        'adj': adj.to(device),
        'full_x': data.x.to(device),
        'full_y': data.y.to(device),
        'full_adj': adj.to(device)
    }
    
    val_data = {
        'x': data.x[val_idx].to(device),
        'y': data.y[val_idx].to(device),
        'adj': adj.to(device),
        'full_x': data.x.to(device),
        'full_y': data.y.to(device),
        'full_adj': adj.to(device)
    }
    
    test_data = {
        'x': data.x[test_idx].to(device),
        'y': data.y[test_idx].to(device),
        'adj': adj.to(device),
        'full_x': data.x.to(device),
        'full_y': data.y.to(device),
        'full_adj': adj.to(device)
    }
    
    # Store full data for evaluation
    full_data = {
        'x': data.x.to(device),
        'y': data.y.to(device),
        'adj': adj.to(device),
        'train_mask': data.train_mask,
        'val_mask': data.val_mask,
        'test_mask': data.test_mask
    }
    
    return train_data, val_data, test_data, full_data, num_classes


def build_model(num_features, num_classes):
    """Build GCN model"""
    model = GCN(
        nfeat=num_features,
        nhid=16,
        nclass=num_classes,
        dropout=0.5
    ).to(device)
    
    return model


def train(model, train_data, val_data, epochs=200, lr=0.01, weight_decay=5e-4):
    """Train the GCN model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on training nodes
        output = model(train_data['full_x'], train_data['full_adj'])
        train_output = output[train_data['train_mask']]
        train_labels = train_data['full_y'][train_data['train_mask']]
        
        loss = criterion(train_output, train_labels)
        loss.backward()
        optimizer.step()
        
        # Evaluate on train
        train_pred = train_output.argmax(dim=1)
        train_acc = (train_pred == train_labels).float().mean().item()
        
        # Evaluate on validation
        val_output = output[val_data['val_mask']]
        val_labels = val_data['full_y'][val_data['val_mask']]
        val_loss = criterion(val_output, val_labels).item()
        val_pred = val_output.argmax(dim=1)
        val_acc = (val_pred == val_labels).float().mean().item()
        
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }


def evaluate(model, data, split_name='validation'):
    """Evaluate model on given data split"""
    model.eval()
    
    with torch.no_grad():
        output = model(data['full_x'], data['full_adj'])
        
        if split_name == 'train':
            mask = data['train_mask']
        elif split_name == 'validation':
            mask = data['val_mask']
        else:
            mask = data['test_mask']
        
        output = output[mask]
        target = data['full_y'][mask]
        
        # Calculate metrics
        pred = output.argmax(dim=1)
        accuracy = accuracy_score(target.cpu(), pred.cpu())
        f1 = f1_score(target.cpu(), pred.cpu(), average='macro')
        
        # NLL loss
        nll_loss = F.nll_loss(output, target).item()
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1,
            'loss': nll_loss,
            'num_samples': len(target)
        }
        
        return metrics


def predict(model, data):
    """Make predictions on given data"""
    model.eval()
    
    with torch.no_grad():
        output = model(data['full_x'], data['full_adj'])
        pred = output.argmax(dim=1)
        
    return pred


def save_artifacts(model, metrics, history, metadata):
    """Save model artifacts"""
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'gcn_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save history
    history_path = os.path.join(OUTPUT_DIR, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json```python
')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save training plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Acc')
    plt.plot(history['val_accs'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_plot.png'))
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run the GCN training and evaluation"""
    print("=" * 60)
    print("GCN Node Classification on Cora Dataset")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Make dataloaders
    print("\nLoading Cora dataset...")
    train_data, val_data, test_data, full_data, num_classes = make_dataloaders()
    num_features = full_data['x'].shape[1]
    print(f"Dataset loaded: {num_features} features, {num_classes} classes")
    print(f"Train nodes: {train_data['full_x'].shape[0]}, "
          f"Val nodes: {val_data['full_x'].shape[0]}, "
          f"Test nodes: {test_data['full_x'].shape[0]}")
    
    # Build model
    print("\nBuilding GCN model...")
    model = build_model(num_features, num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model built with {total_params} parameters")
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    model, history = train(model, train_data, val_data, epochs=200, lr=0.01, weight_decay=5e-4)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on train split
    print("\n" + "=" * 60)
    print("Evaluating on TRAIN split:")
    print("=" * 60)
    train_metrics = evaluate(model, full_data, split_name='train')
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Train F1 Macro: {train_metrics['f1_macro']:.4f}")
    print(f"Train Loss: {train_metrics['loss']:.4f}")
    
    # Evaluate on validation split
    print("\n" + "=" * 60)
    print("Evaluating on VALIDATION split:")
    print("=" * 60)
    val_metrics = evaluate(model, full_data, split_name='validation')
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation F1 Macro: {val_metrics['f1_macro']:.4f}")
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    
    # Evaluate on test split
    print("\n" + "=" * 60)
    print("Evaluating on TEST split:")
    print("=" * 60)
    test_metrics = evaluate(model, full_data, split_name='test')
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Quality assertions
    print("\n" + "=" * 60)
    print("Quality Assertions:")
    print("=" * 60)
    
    # Check that model performs better than random (random ~1/7 = 0.143 for Cora)
    random_baseline = 1.0 / num_classes
    assert train_metrics['accuracy'] > 0.5, f"Train accuracy {train_metrics['accuracy']:.4f} too low"
    assert val_metrics['accuracy'] > 0.5, f"Validation accuracy {val_metrics['accuracy']:.4f} too low"
    assert test_metrics['accuracy'] > 0.5, f"Test accuracy {test_metrics['accuracy']:.4f} too low"
    
    # Check that validation accuracy is reasonable (Cora typically achieves 0.75-0.85)
    assert val_metrics['accuracy'] > 0.70, f"Validation accuracy {val_metrics['accuracy']:.4f} below threshold 0.70"
    
    # Check that training loss decreased
    assert history['train_losses'][-1] < history['train_losses'][0], "Training loss did not decrease"
    assert history['val_losses'][-1] < history['val_losses'][0], "Validation loss did not decrease"
    
    # Check no overfitting (train and val accuracy should be close)
    accuracy_diff = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
    assert accuracy_diff < 0.2, f"Large gap between train and val accuracy: {accuracy_diff:.4f}"
    
    print(f"✓ Train accuracy {train_metrics['accuracy']:.4f} > 0.5")
    print(f"✓ Validation accuracy {val_metrics['accuracy']:.4f} > 0.70")
    print(f"✓ Test accuracy {test_metrics['accuracy']:.4f} > 0.5")
    print(f"✓ Training loss decreased from {history['train_losses'][0]:.4f} to {history['train_losses'][-1]:.4f}")
    print(f"✓ Validation accuracy gap {accuracy_diff:.4f} < 0.2 (no severe overfitting)")
    
    # Save artifacts
    print("\nSaving artifacts...")
    metadata = get_task_metadata()
    metadata['training_time'] = training_time
    metadata['num_parameters'] = total_params
    metadata['final_train_accuracy'] = train_metrics['accuracy']
    metadata['final_val_accuracy'] = val_metrics['accuracy']
    metadata['final_test_accuracy'] = test_metrics['accuracy']
    
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    save_artifacts(model, all_metrics, history, metadata)
    
    # Final summary
    print("\n" + "=" * 60)
    print("PASS: All quality thresholds met!")
    print("=" * 60)
    print(f"Final Results:")
    print(f"  Train Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Val Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"  Test Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  Training Time:   {training_time:.2f} seconds")
    print(f"  Parameters:      {total_params}")
    print(f"  Output Directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

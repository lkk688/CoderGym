"""
Graph Attention Network (GAT) Implementation
Compares GAT to GCN on citation network classification task.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from collections import defaultdict
import time

# Set paths
OUTPUT_DIR = '/Developer/AIserver/output/tasks/gml_lvl3_gat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata():
    return {
        'task_name': 'gml_lvl3_gat',
        'description': 'Graph Attention Network vs GCN comparison',
        'input_type': 'graph',
        'output_type': 'classification',
        'metrics': ['accuracy', 'loss', 'mse', 'r2'],
        'model_types': ['gat', 'gcn']
    }

def make_dataloaders(device, batch_size=1):
    """Load Cora dataset (citation network)"""
    set_seed(42)
    
    # Load Cora dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    
    # Cora has one large graph
    data = dataset[0]
    data = data.to(device)
    
    # Split indices
    n_nodes = data.x.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(n_nodes)
    
    train_idx = indices[:int(0.6 * n_nodes)]
    val_idx = indices[int(0.6 * n_nodes):int(0.8 * n_nodes)]
    test_idx = indices[int(0.8 * n_nodes):]
    
    # Create masks
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    # Create dataloaders (for single graph, we return the graph itself)
    train_loader = [(data, train_idx)]
    val_loader = [(data, val_idx)]
    test_loader = [(data, test_idx)]
    
    return train_loader, val_loader, test_loader, dataset.num_features, dataset.num_classes

class GATLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        # h: (N, in_features)
        # adj: adjacency matrix (N, N)
        
        # Linear transformation
        Wh = torch.mm(h, self.W)  # (N, out_features)
        
        # Compute attention scores
        # For each pair of nodes (i, j), compute attention score
        N = Wh.size()[0]
        
        # Concatenate node features for attention computation
        # a^T [Wh_i || Wh_j]
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), 
                            Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        
        # Compute attention
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Mask attention scores (only for connected nodes)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregate features
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ])
        
        # Output layer
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        
        self.device = get_device()
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Multi-head attention
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        
        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.device = get_device()
    
    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)

def build_model(model_type, nfeat, nhid, nclass, device):
    """Build GAT or GCN model"""
    if model_type == 'gat':
        model = GAT(nfeat=nfeat, nhid=nhid, nclass=nclass, nheads=8)
    elif model_type == 'gcn':
        model = GCN(nfeat=nfeat, nhid=nhid, nclass=nclass)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    return model

def train(model, data, train_idx, optimizer, criterion):
    """Train the model for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Get predictions for training nodes
    out = model(data.x, data.edge_index)
    
    if train_idx is not None:
        loss = criterion(out[train_idx], data.y[train_idx])
    else:
        loss = criterion(out, data.y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, data, idx, criterion):
    """Evaluate the model on given indices"""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        if idx is not None:
            loss = criterion(out[idx], data.y[idx]).item()
            pred = out[idx].argmax(dim=1)
            target = data.y[idx]
        else:
            loss = criterion(out, data.y).item()
            pred = out.argmax(dim=1)
            target = data.y
        
        # Compute metrics
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        accuracy = accuracy_score(target_np, pred_np)
        
        # For MSE and R2, treat as multi-class classification
        # Convert to one-hot for MSE computation
        n_classes = out.shape[1]
        target_onehot = np.zeros((len(target_np), n_classes))
        target_onehot[np.arange(len(target_np)), target_np] = 1
        
        pred_probs = F.softmax(out[idx], dim=1).detach().cpu().numpy() if idx is not None else F.softmax(out, dim=1).detach().cpu().numpy()
        mse = mean_squared_error(target_onehot, pred_probs)
        r2 = r2_score(target_onehot, pred_probs)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'mse': mse,
            'r2': r2
        }
    
    return metrics

def predict(model, data, idx=None):
    """Get predictions"""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        if idx is not None:
            pred = out[idx].argmax(dim=1)
        else:
            pred = out.argmax(dim=1)
    
    return pred.detach().cpu().numpy()

def save_artifacts(model, model_type, metrics, history, data):
    """Save model and artifacts"""
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'{model_type}_model.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, f'{model_type}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, f'{model_type}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training history
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title(f'{model_type.upper()} Training History')
    
    # Plot metrics
    axes[1].bar(['Accuracy', 'MSE', 'R2'], [metrics['accuracy'], metrics['mse'], metrics['r2']])
    axes[1].set_ylabel('Value')
    axes[1].set_title(f'{model_type.upper()} Final Metrics')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'{model_type}_results.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Artifacts saved to {OUTPUT_DIR}")

def main():
    """Main training and evaluation pipeline"""
    print("=" * 60)
    print("GAT vs GCN Comparison on Cora Dataset")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Make dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, nfeat, nclass = make_dataloaders(device)
    data = train_loader[0][0]  # Get the graph
    
    print(f"Dataset: Cora")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {nfeat}, Classes: {nclass}")
    print(f"Train nodes: {data.train_mask.sum().item()}")
    print(f"Val nodes: {data.val_mask.sum().item()}")
    
    # Parameters
    nhid = 64
    lr = 0.005
    weight_decay = 5e-4
    epochs = 200
    
    results = {}
    models = {}
    
    # Train GAT
    print("\n" + "=" * 60)
    print("Training GAT Model")
    print("=" * 60)
    
    set_seed(42)
    gat_model = build_model('gat', nfeat, nhid, nclass, device)
    gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    gat_history = defaultdict(list)
    
    for epoch in range(1, epochs + 1):
        loss = train(gat_model, data, data.train_mask, gat_optimizer, criterion)
        
        # Evaluate
        train_metrics = evaluate(gat_model, data, data.train_mask, criterion)
        val_metrics = evaluate(gat_model, data, data.val_mask, criterion)
        
        gat_history['train_loss'].append(loss)
        gat_history['train_acc'].append(train_metrics['accuracy'])
        gat_history['val_loss'].append(val_metrics['loss'])
        gat_history['val_acc'].append(val_metrics['accuracy'])
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    
    # Final evaluation
    gat_train_metrics = evaluate(gat_model, data, data.train_mask, criterion)
    gat_val_metrics = evaluate(gat_model, data, data.val_mask, criterion)
    gat_test_metrics = evaluate(gat_model, data, data.test_mask, criterion)
    
    print(f"\nGAT Final Results:")
    print(f"  Train - Loss: {gat_train_metrics['loss']:.4f}, Acc: {gat_train_metrics['accuracy']:.4f}")
    print(f"  Val   - Loss: {gat_val_metrics['loss']:.4f}, Acc: {gat_val_metrics['accuracy']:.4f}")
    print(f"  Test  - Loss: {gat_test_metrics['loss']:.4f}, Acc: {gat_test_metrics['accuracy']:.4f}")
    
    results['gat'] = {
        'train': gat_train_metrics,
        'val': gat_val_metrics,
        'test': gat_test_metrics
    }
    models['gat'] = gat_model
    
    # Train GCN
    print("\n" + "=" * 60)
    print("Training GCN Model")
    print("=" * 60)
    
    set_seed(42)
    gcn_model = build_model('gcn', nfeat, nhid, nclass, device)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    gcn_history = defaultdict(list)
    
    for epoch in range(1, epochs + 1):
        loss = train(gcn_model, data, data.train_mask, gcn_optimizer, criterion)
        
        # Evaluate
        train_metrics = evaluate(gcn_model, data, data.train_mask, criterion)
        val_metrics = evaluate(gcn_model, data, data.val_mask, criterion)
        
        gcn_history['train_loss'].append(loss)
        gcn_history['train_acc'].append(train_metrics['accuracy'])
        gcn_history['val_loss'].append(val_metrics['loss'])
        gcn_history['val_acc'].append(val_metrics['accuracy'])
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    
    # Final evaluation
    gcn_train_metrics = evaluate(gcn_model, data, data.train_mask, criterion)
    gcn_val_metrics = evaluate(gcn_model, data, data.val_mask, criterion)
    gcn_test_metrics = evaluate(gcn_model, data, data.test_mask, criterion)
    
    print(f"\nGCN Final Results:")
    print(f"  Train - Loss: {gcn_train_metrics['loss']:.4f}, Acc: {gcn_train_metrics['accuracy']:.4f}")
    print(f"  Val   - Loss: {gcn_val_metrics['loss']:.4f}, Acc: {gcn_val_metrics['accuracy']:.4f}")
    print(f"  Test  - Loss: {gcn_test_metrics['loss']:.4f}, Acc: {gcn_test_metrics['accuracy']:.4f}")
    
    results['gcn'] = {
        'train': gcn_train_metrics,
        'val': gcn_val_metrics,
        'test': gcn_test_metrics
    }
    models['gcn'] = gcn_model
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    
    save_artifacts(gat_model, 'gat', gat_val_metrics, gat_history, data)
    save_artifacts(gcn_model, 'gcn', gcn_val_metrics, gcn_history, data)
    
    # Compare models
    print("\n" + "=" * 60)
    print("Model Comparison (Validation Set)")
    print("=" * 60)
    
    gat_acc = gat_val_metrics['accuracyaccuracy']
    gcn_acc = gcn_val_metrics['accuracy']
    
    print(f"GAT Accuracy:  {gat_acc:.4f}")
    print(f"GCN Accuracy:  {gcn_acc:.4f}")
    print(f"Difference:    {gat_acc - gcn_acc:.4f}")
    
    # Quality assertions
    print("\n" + "=" * 60)
    print("Quality Assertions")
    print("=" * 60)
    
    all_passed = True
    
    # Check GAT quality
    if gat_val_metrics['accuracy'] < 0.75:
        print(f"FAIL: GAT validation accuracy ({gat_val_metrics['accuracy']:.4f}) < 0.75")
        all_passed = False
    else:
        print(f"PASS: GAT validation accuracy ({gat_val_metrics['accuracy']:.4f}) >= 0.75")
    
    # Check GCN quality
    if gcn_val_metrics['accuracy'] < 0.75:
        print(f"FAIL: GCN validation accuracy ({gcn_val_metrics['accuracy']:.4f}) < 0.75")
        all_passed = False
    else:
        print(f"PASS: GCN validation accuracy ({gcn_val_metrics['accuracy']:.4f}) >= 0.75")
    
    # Check GAT >= GCN (with tolerance)
    tolerance = 0.02
    if gat_acc >= gcn_acc - tolerance:
        print(f"PASS: GAT accuracy ({gat_acc:.4f}) >= GCN accuracy ({gcn_acc:.4f}) - tolerance ({tolerance})")
    else:
        print(f"FAIL: GAT accuracy ({gat_acc:.4f}) < GCN accuracy ({gcn_acc:.4f}) + tolerance ({tolerance})")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"All quality checks passed: {all_passed}")
    
    # Save comparison results
    comparison_path = os.path.join(OUTPUT_DIR, 'comparison_results.json')
    comparison_data = {
        'gat': results['gat']['val'],
        'gcn': results['gcn']['val'],
        'gat_vs_gcn': gat_acc - gcn_acc,
        'all_passed': all_passed
    }
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Comparison results saved to {comparison_path}")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

"""
Graph Link Prediction Task
Implements ML Task: Graph Link Prediction with embeddings + decoder
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

# Task metadata
def get_task_metadata():
    return {
        'task_name': 'graph_link_prediction',
        'input_type': 'graph',
        'output_type': 'edge_prediction',
        'metrics': ['auc', 'ap', 'mse', 'r2'],
        'num_nodes': 100,
        'embedding_dim': 16,
        'num_edges': 300
    }

# Generate synthetic graph data
def generate_synthetic_graph(num_nodes=100, embedding_dim=16, num_edges=300, noise_level=0.1):
    """Generate synthetic graph with node features and edges"""
    # Generate node embeddings as features
    node_features = torch.randn(num_nodes, embedding_dim)
    
    # Generate true edge probabilities based on node similarity
    edge_probs = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            sim = torch.dot(node_features[i], node_features[j])
            edge_probs[i, j] = torch.sigmoid(sim)
            edge_probs[j, i] = edge_probs[i, j]
    
    # Sample edges more carefully to ensure balanced dataset
    edges = []
    prob_matrix = edge_probs.cpu().numpy()
    
    # Sample positive edges based on probabilities
    all_pairs = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
    
    # Sort pairs by probability
    pair_probs = [(pair, prob_matrix[pair[0], pair[1]]) for pair in all_pairs]
    pair_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Select top pairs for positive edges
    selected_pairs = [pair for pair, prob in pair_probs[:num_edges]]
    edges.extend(selected_pairs)
    
    # Create edge list
    edge_list = torch.tensor(edges, dtype=torch.long)
    
    # Generate features for edges (concatenation of node features)
    edge_features = []
    labels = []
    
    # Positive edges
    for i, j in edges:
        edge_features.append(torch.cat([node_features[i], node_features[j]]))
        labels.append(1.0)
    
    # Generate negative edges with similar probability distribution
    negative_edges = []
    attempts = 0
    max_attempts = 10000
    
    while len(negative_edges) < len(edges) and attempts < max_attempts:
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j:
            pair = (min(i, j), max(i, j))
            if pair not in edges and pair not in negative_edges:
                # Accept with probability based on inverse of edge probability
                prob = prob_matrix[i, j]
                if np.random.random() > prob:  # Low probability edges more likely to be negative
                    negative_edges.append(pair)
        attempts += 1
    
    # If we couldn't get enough negative edges, fill with random non-edges
    while len(negative_edges) < len(edges):
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j:
            pair = (min(i, j), max(i, j))
            if pair not in edges and pair not in negative_edges:
                negative_edges.append(pair)
        if attempts > max_attempts:
            break
        attempts += 1
    
    for i, j in negative_edges:
        edge_features.append(torch.cat([node_features[i], node_features[j]]))
        labels.append(0.0)
    
    # Shuffle data
    all_features = torch.stack(edge_features)
    all_labels = torch.tensor(labels, dtype=torch.float32)
    
    # Add some noise to features
    all_features += noise_level * torch.randn_like(all_features)
    
    return all_features, all_labels, node_features, edge_list

# Custom dataset for link prediction
class LinkPredictionDataset(Dataset):
    def __init__(self, edge_features, labels):
        self.edge_features = edge_features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.edge_features[idx], self.labels[idx]

# Build dataloaders with stratified splitting
def make_dataloaders(edge_features, labels, batch_size=32, val_ratio=0.2):
    """Split data and create dataloaders with stratification"""
    # Convert to numpy for sklearn
    features_np = edge_features.numpy()
    labels_np = labels.numpy()
    
    # Stratified split to ensure balanced classes
    try:
        train_features, val_features, train_labels, val_labels = train_test_split(
            features_np, labels_np, 
            test_size=val_ratio, 
            random_state=42, 
            stratify=labels_np
        )
    except ValueError:
        # If stratification fails (e.g., class too small), use regular split
        print("Warning: Stratified split failed, using regular split")
        n = len(labels_np)
        val_size = int(n * val_ratio)
        train_features = features_np[val_size:]
        val_features = features_np[:val_size]
        train_labels = labels_np[val_size:]
        val_labels = labels_np[:val_size]
    
    # Create datasets
    train_dataset = LinkPredictionDataset(
        torch.FloatTensor(train_features), 
        torch.FloatTensor(train_labels)
    )
    val_dataset = LinkPredictionDataset(
        torch.FloatTensor(val_features), 
        torch.FloatTensor(val_labels)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Print class distribution
    train_pos = np.sum(train_labels == 1)
    train_neg = np.sum(train_labels == 0)
    val_pos = np.sum(val_labels == 1)
    val_neg = np.sum(val_labels == 0)
    
    print(f"Training samples: {len(train_dataset)} (Pos: {train_pos}, Neg: {train_neg})")
    print(f"Validation samples: {len(val_dataset)} (Pos: {val_pos}, Neg: {val_neg})")
    
    return train_loader, val_loader, train_features, val_features, train_labels, val_labels

# Build model (embeddings + decoder) - simplified architecture
class LinkPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(LinkPredictionModel, self).__init__()
        
        # Simpler decoder network with batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.decoder(x)

def build_model(input_dim, device):
    """Build the link prediction model"""
    model = LinkPredictionModel(input_dim)
    model = model.to(device)
    print(f"Model architecture: {model}")
    return model

# Training function with early stopping
def train(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, patience=15):
    """Train the link prediction model with early stopping"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    print("Training model...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            # Move to device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_labels).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# Evaluation function
def evaluate(model, data_loader, device):
    """Evaluate the model and return metrics"""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            
            # Get probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    # Concatenate
    all_preds = np.vstack(all_preds).flatten()
    all_labels = np.vstack(all_labels).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    # Check if we have both classes for AUC
    unique_labels = np.unique(all_labels)
    
    # AUC and AP
    try:
        if len(unique_labels) > 1:
            auc = roc_auc_score(all_labels, all_preds)
            ap = average_precision_score(all_labels, all_preds)
        else:
            # If only one class, set to 0.5 (random)
            auc = 0.5
            ap = 0.5
            print(f"Warning: Only one class in evaluation, setting AUC/AP to 0.5")
    except ValueError:
        auc = 0.5
        ap = 0.5
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'mse': mse,
        'r2': r2,
        'auc': auc,
        'ap': ap
    }
    
    return metrics

# Predict function
def predict(model, features, device):
    """Make predictions on given features"""
    model.eval()
    features = torch.FloatTensor(features).to(device)
    
    with torch.no_grad():
        outputs = model(features)
        probs = torch.sigmoid(outputs)
    
    return probs.cpu().numpy()

# Save artifacts
def save_artifacts(model, metrics, output_dir='output'):
    """Save model and metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'link_prediction_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.npy')
    np.save(metrics_path, metrics)
    
    print(f"Artifacts saved to {output_dir}")

# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("Graph Link Prediction Task")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Generate data
    print("\nGenerating synthetic graph data...")
    edge_features, labels, node_features, edge_list = generate_synthetic_graph(
        num_nodes=100, 
        embedding_dim=16, 
        num_edges=300,
        noise_level=0.05
    )
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, train_features, val_features, train_labels, val_labels = \
        make_dataloaders(edge_features, labels, batch_size=32, val_ratio=0.2)
    
    # Build model
    print("\nBuilding model...")
    input_dim = edge_features.shape[1]
    model = build_model(input_dim, device)
    
    # Train model
    print("\n" + "=" * 60)
    model = train(model, train_loader, val_loader, device, num_epochs=150, lr=0.001, patience=20)
    print("=" * 60)
    
    # Evaluate on train and validation sets
    print("\nEvaluating model...")
    
    # Train metrics
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print("Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Validation metrics
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print("Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, val_metrics, output_dir='output')
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train AUC:  {train_metrics['auc']:.4f}")
    print(f"Val AUC:    {val_metrics['auc']:.4f}")
    print(f"Train AP:   {train_metrics['ap']:.4f}")
    print(f"Val AP:     {val_metrics['ap']:.4f}")
    print(f"Train MSE:  {train_metrics['mse']:.4f}")
    print(f"Val MSE:    {val_metrics['mse']:.4f}")
    print(f"Train R2:   {train_metrics['r2']:.4f}")
    print(f"Val R2:     {val_metrics['r2']:.4f}")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)
    
    checks_passed = []
    
    # Check 1: Train AUC > 0.7
    check1 = train_metrics['auc'] > 0.7
    checks_passed.append(check1)
    status1 = "✓" if check1 else "✗"
    print(f"{status1} Train AUC > 0.7: {train_metrics['auc']:.4f}")
    
    # Check 2: Val AUC > 0.65
    check2 = val_metrics['auc'] > 0.65
    checks_passed.append(check2)
    status2 = "✓" if check2 else "✗"
    print(f"{status2} Val AUC > 0.65: {val_metrics['auc']:.4f}")
    
    # Check 3: Train AP > 0.7
    check3 = train_metrics['ap'] > 0.7
    checks_passed.append(check3)
    status3 = "✓" if check3 else "✗"
    print(f"{status3} Train AP > 0.7: {train_metrics['ap']:.4f}")
    
    # Check 4: Val AP > 0.65
    check4 = val_metrics['ap'] > 0.65
    checks_passed.append(check4)
    status4 = "✓" if check4 else "✗"
    print(f"{status4} Val AP > 0.65: {val_metrics['ap']:.4f}")
    
    # Check 5: Train R2 > 0.5
    check5 = train_metrics['r2'] > 0.5
    checks_passed.append(check5)
    status5 = "✓" if check5 else "✗"
    print(f"{status5} Train R2 > 0.5: {train_metrics['r2']:.4f}")
    
    # Check 6: Val R2 >0.4
    check6 = val_metrics['r2'] > 0.4
    checks_passed.append(check6)
    status6 = "✓" if check6 else "✗"
    print(f"{status6} Val R2 > 0.4: {val_metrics['r2']:.4f}")
    
    # Check 7: Val MSE < 0.3
    check7 = val_metrics['mse'] < 0.3
    checks_passed.append(check7)
    status7 = "✓" if check7 else "✗"
    print(f"{status7} Val MSE < 0.3: {val_metrics['mse']:.4f}")
    
    # Check 8: AUC gap < 0.2
    auc_gap = abs(train_metrics['auc'] - val_metrics['auc'])
    check8 = auc_gap < 0.2
    checks_passed.append(check8)
    status8 = "✓" if check8 else "✗"
    print(f"{status8} AUC gap < 0.2: {auc_gap:.4f}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all(checks_passed):
        print("PASS: All quality checks passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        sys.exit(1)

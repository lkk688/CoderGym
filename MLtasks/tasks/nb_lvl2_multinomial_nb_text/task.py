"""
Multinomial Naive Bayes for Text Classification
Bag-of-words + Multinomial NB for text classification
"""

import os
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_task_metadata():
    """Return task metadata"""
    return {
        "task_name": "nb_lvl2_multinomial_nb_text",
        "task_type": "text_classification",
        "model_type": "multinomial_nb",
        "input_type": "text",
        "output_type": "classification"
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Get computation device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(test_size=0.2, max_features=1000):
    """
    Create dataloaders for text classification
    
    Args:
        test_size: Fraction of data to use for validation
        max_features: Maximum number of features for vectorizer
    
    Returns:
        train_loader, val_loader, vocabulary
    """
    # Load 20 newsgroups dataset
    categories = ['sci.space', 'comp.graphics', 'rec.sport.baseball', 'talk.politics.mideast']
    
    try:
        dataset = fetch_20newsgroups(
            subset='all',
            categories=categories,
            remove=('headers', 'footers', 'quotes'),
            shuffle=True,
            random_state=SEED
        )
    except Exception as e:
        print(f"Warning: Could not fetch 20newsgroups: {e}")
        print("Using small toy corpus instead")
        # Fallback to toy corpus
        dataset = {
            'data': [
                "space shuttle orbit earth moon mars",
                "computer graphics video display screen",
                "baseball game score run win",
                "politics war peace middle east",
                "space exploration NASA mission spacecraft",
                "computer programming code software",
                "baseball player home run game",
                "politics government law justice",
                "space station orbit satellite planet",
                "computer network internet web",
                "baseball pitcher strike out win",
                "politics election vote campaign",
                "space telescope star galaxy universe",
                "computer processor memory disk",
                "baseball bat hit run score",
                "politics president cabinet minister",
                "space rocket launch mission orbit",
                "computer database query table",
                "baseball field inning play",
                "politics debate speech address"
            ] * 20,  # Repeat to get more samples
            'target': [0, 1, 2, 3] * 15,  # 4 categories
            'target_names': ['sci.space', 'comp.graphics', 'rec.sport.baseball', 'talk.politics.mideast']
        }
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        dataset['data'],
        dataset['target'],
        test_size=test_size,
        random_state=SEED,
        stratify=dataset['target']
    )
    
    # Create vocabulary and transform to bag-of-words
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        strip_accents='unicode'
    )
    
    X_train_bow = vectorizer.fit_transform(X_train).toarray()
    X_val_bow = vectorizer.transform(X_val).toarray()
    
    # Convert to torch tensors
    X_train_tensor = torch.FloatTensor(X_train_bow)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_bow)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets (simplified - no actual DataLoader needed for NB)
    train_data = (X_train_tensor, y_train_tensor)
    val_data = (X_val_tensor, y_val_tensor)
    
    return train_data, val_data, vectorizer.get_feature_names_out()

def build_model(alpha=1.0):
    """
    Build Multinomial Naive Bayes model
    
    Args:
        alpha: Laplace smoothing parameter
    
    Returns:
        model (MultinomialNB)
    """
    model = MultinomialNB(alpha=alpha)
    return model

def train(model, train_data, device):
    """
    Train the Multinomial Naive Bayes model
    
    Args:
        model: MultinomialNB model
        train_data: Tuple of (X_train, y_train)
        device: Computation device
    
    Returns:
        Trained model
    """
    X_train, y_train = train_data
    
    # Move to device if needed
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.detach().cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.detach().cpu().numpy()
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate(model, data, device):
    """
    Evaluate the model and return metrics
    
    Args:
        model: Trained MultinomialNB model
        data: Tuple of (X, y)
        device: Computation device
    
    Returns:
        dict with metrics
    """
    X, y = data
    
    # Move to device if needed
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    # Predict
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'loss': float(1.0 - accuracy)  # Cross-entropy-like loss approximation
    }
    
    return metrics

def predict(model, X, device):
    """
    Make predictions
    
    Args:
        model: Trained MultinomialNB model
        X: Input features
        device: Computation device
    
    Returns:
        Predictions
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    
    predictions = model.predict(X)
    return predictions

def save_artifacts(model, vectorizer, metadata, output_dir="output"):
    """
    Save model artifacts
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        metadata: Task metadata
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    with open(os.path.join(output_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    
    # Save vectorizer
    with open(os.path.join(output_dir, "vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Artifacts saved to {output_dir}")

def main():
    """Main function to run the task"""
    print("=" * 60)
    print("Multinomial Naive Bayes for Text Classification")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_data, val_data, vocabulary = make_dataloaders(test_size=0.2, max_features=1000)
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(alpha=1.0)  # Laplace smoothing
    print(f"Model: MultinomialNB with alpha=1.0")
    
    # Train model
    print("\nTraining model...")
    model = train(model, train_data, device)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_data, device)
    print(f"Train Metrics:")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {train_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {train_metrics['f1_weighted']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_data, device)
    print(f"Val Metrics:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {val_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {val_metrics['f1_weighted']:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    metadata = get_task_metadata()
    metadata.update({
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'vocabulary_size': len(vocabulary),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    })
    save_artifacts(model, None, metadata)
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:   {val_metrics['accuracy']:.4f}")
    print(f"Train F1 Macro: {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:   {val_metrics['f1_macro']:.4f}")
    print("=" * 60)
    
    # Quality checks
    print("\nQUALITY CHECKS")
    print("=" * 60)
    
    checks_passed = True
    
    # Check 1: Train accuracy > 0.7
    check1 = train_metrics['accuracy'] > 0.7
    print(f"{'✓' if check1 else '✗'} Train Accuracy > 0.7: {train_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check1
    
    # Check 2: Val accuracy > 0.65
    check2 = val_metrics['accuracy'] > 0.65
    print(f"{'✓' if check2 else '✗'} Val Accuracy > 0.65: {val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check2
    
    # Check 3: Val F1 Macro > 0.65
    check3 = val_metrics['f1_macro'] > 0.65
    print(f"{'✓' if check3 else '✗'} Val F1 Macro > 0.65: {val_metrics['f1_macro']:.4f}")
    checks_passed = checks_passed and check3
    
    # Check 4: Loss decreased (train loss < 0.5)
    check4 = train_metrics['loss'] < 0.5
    print(f"{'✓' if check4 else '✗'} Train Loss < 0.5: {train_metrics['loss']:.4f}")
    checks_passed = checks_passed and check4
    
    # Check 5: Accuracy gap < 0.2
    accuracy_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
    check5 = accuracy_gap < 0.2
    print(f"{'✓' if check5 else '✗'} Accuracy gap < 0.2: {accuracy_gap:.4f}")
    checks_passed = checks_passed and check5
    
    print("=" * 60)
    
    if checks_passed:
        print("PASS: All quality checks passed!")
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        return 1

if __name__ == '__main__':
    exit(main())

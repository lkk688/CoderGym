"""
Word Embeddings with Skip-gram and Sentiment Classification

Mathematical Formulation:
- Skip-gram Objective: maximize sum_w log p(w_context | w_target)
- Negative Sampling: log(sigmoid(v_c · v_t)) + sum_i E_neg[log(sigmoid(-v_i · v_t))]
- Similarity: cosine similarity between word vectors
- Classification: Logistic regression on sentence embeddings

This task demonstrates:
1. Training word embeddings from scratch using Skip-gram with negative sampling
2. Evaluating embedding quality through word similarity
3. Building a sentiment classifier on top of embeddings
4. Handling text processing and vocabulary management
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
from collections import defaultdict, Counter
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Output directory for artifacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'nlp_lvl1_text_embedding')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'word_embeddings_sentiment',
        'description': 'Skip-gram word embeddings + sentiment classification',
        'embedding_dim': 100,
        'context_window': 5,
        'negative_samples': 15,
        'task_type': 'nlp_sentiment',
        'vocab_size_target': 5000
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    """Get device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocabulary:
    """Build and manage vocabulary."""
    
    def __init__(self, min_freq=2):
        self.word2id = {}
        self.id2word = {}
        self.word_freq = Counter()
        self.min_freq = min_freq
        self.size = 0
    
    def add_word(self, word):
        """Add word to vocabulary."""
        self.word_freq[word] += 1
    
    def build_vocab(self, max_vocab=5000):
        """Build vocabulary from word frequencies."""
        # Add special tokens
        self.word2id['<UNK>'] = 0
        self.word2id['<PAD>'] = 1
        self.id2word[0] = '<UNK>'
        self.id2word[1] = '<PAD>'
        
        current_id = 2
        for word, freq in self.word_freq.most_common(max_vocab):
            if freq >= self.min_freq and current_id < max_vocab:
                self.word2id[word] = current_id
                self.id2word[current_id] = word
                current_id += 1
        
        self.size = len(self.word2id)
    
    def get_id(self, word):
        """Get word ID."""
        return self.word2id.get(word, 0)  # 0 is <UNK>
    
    def get_word(self, word_id):
        """Get word from ID."""
        return self.id2word.get(word_id, '<UNK>')


class SkipGramDataset(Dataset):
    """Skip-gram dataset for word embeddings."""
    
    def __init__(self, sentences, vocab, window_size=5, neg_samples=15):
        self.pairs = []
        self.neg_samples = neg_samples
        self.vocab = vocab
        
        # Build (target, context) pairs
        for sentence in sentences:
            tokens = [vocab.get_id(w) for w in sentence.split()]
            # Remove <UNK> tokens (0) that exceed threshold
            tokens = [t for t in tokens if t != 0 or random.random() < 0.1]
            
            for i, target in enumerate(tokens):
                # Context window
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context = tokens[j]
                        self.pairs.append((target, context))
        
        # Pre-compute negative sampling distribution (unigram distribution)
        freq_sum = sum(vocab.word_freq.values())
        self.neg_samples_dist = np.array([
            vocab.word_freq.get(vocab.get_word(i), 1) ** 0.75 / freq_sum
            for i in range(vocab.size)
        ])
        self.neg_samples_dist = self.neg_samples_dist / self.neg_samples_dist.sum()
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        
        # Sample negative examples
        negatives = np.random.choice(
            self.vocab.size, size=self.neg_samples, p=self.neg_samples_dist
        )
        
        return target, context, torch.LongTensor(negatives)


def make_dataloaders(batch_size=64):
    """
    Create toy sentiment corpus and dataloaders for embedding training and classification.
    
    Returns:
        embedding_loader, sentiment_data (for final classification evaluation)
    """
    # Toy sentiment dataset
    positive_reviews = [
        "this movie is great and amazing",
        "i loved this film very much",
        "fantastic performance by actors",
        "excellent direction and cinematography",
        "absolutely wonderful movie experience",
        "best film i have seen this year",
        "incredible story and great actors",
        "loved every minute of this film",
        "amazing plot and perfect ending",
        "outstanding performance throughout"
    ]
    
    negative_reviews = [
        "this movie is terrible and boring",
        "i hated this film very much",
        "awful performance by actors",
        "terrible direction and cinematography",
        "absolutely horrible movie experience",
        "worst film i have seen this year",
        "terrible story and bad actors",
        "hated every minute of this film",
        "boring plot and disappointing ending",
        "awful performance throughout"
    ]
    
    # Build vocabulary
    vocab = Vocabulary(min_freq=1)
    
    all_texts = positive_reviews + negative_reviews
    for text in all_texts:
        for word in text.split():
            vocab.add_word(word)
    
    vocab.build_vocab(max_vocab=5000)
    
    # Create embedding training dataset
    embedding_dataset = SkipGramDataset(all_texts, vocab, window_size=5, neg_samples=15)
    embedding_loader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=True)
    
    # Create sentiment classification data
    sentiment_data = {
        'positive': positive_reviews,
        'negative': negative_reviews,
        'labels': [1] * len(positive_reviews) + [0] * len(negative_reviews)
    }
    
    return embedding_loader, sentiment_data, vocab


class SkipGramModel(nn.Module):
    """Skip-gram embedding model with negative sampling."""
    
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embedding matrices
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize
        nn.init.uniform_(self.target_embedding.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.context_embedding.weight, -0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, target, context, negatives):
        """
        Compute negative sampling loss.
        
        Args:
            target: Target word IDs (B,)
            context: Context word IDs (B,)
            negatives: Negative sample IDs (B, neg_samples)
        
        Returns:
            Loss (scalar)
        """
        # Positive pair
        target_emb = self.target_embedding(target)  # (B, D)
        context_emb = self.context_embedding(context)  # (B, D)
        
        pos_score = torch.sum(target_emb * context_emb, dim=1)  # (B,)
        pos_loss = -torch.log(torch.sigmoid(pos_score))  # (B,)
        
        # Negative pairs
        neg_emb = self.context_embedding(negatives)  # (B, neg_samples, D)
        neg_score = torch.matmul(target_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1)  # (B, neg_samples)
        neg_loss = -torch.log(torch.sigmoid(-neg_score))  # (B, neg_samples)
        
        loss = pos_loss + torch.sum(neg_loss, dim=1)
        return loss.mean()
    
    def get_embeddings(self):
        """Get learned word embeddings."""
        return self.target_embedding.weight.data


def build_model(vocab_size, embedding_dim=100, **kwargs):
    """Build Skip-gram model."""
    return SkipGramModel(vocab_size, embedding_dim)


def train_embeddings(model, embedding_loader, device, epochs=10, lr=0.025):
    """
    Train word embeddings using Skip-gram with negative sampling.
    
    Args:
        model: SkipGramModel
        embedding_loader: DataLoader for embedding training
        device: Device
        epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        Training history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for target, context, negatives in embedding_loader:
            target, context = target.to(device), context.to(device)
            negatives = negatives.to(device)
            
            optimizer.zero_grad()
            loss = model(target, context, negatives)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(embedding_loader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f"Embedding Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return history


def compute_word_similarity(embeddings, vocab, word1, word2):
    """Compute cosine similarity between two words."""
    id1 = vocab.get_id(word1)
    id2 = vocab.get_id(word2)
    
    emb1 = embeddings[id1]
    emb2 = embeddings[id2]
    
    similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return similarity


def get_similar_words(embeddings, vocab, word, top_k=5):
    """Get top-k similar words using embedding similarity."""
    word_id = vocab.get_id(word)
    word_emb = embeddings[word_id]  # (D,)
    
    # Compute similarities with all words
    similarities = torch.nn.functional.cosine_similarity(
        word_emb.unsqueeze(0), embeddings, dim=1
    )  # (vocab_size,)
    
    # Get top-k (excluding the word itself)
    top_ids = torch.argsort(similarities, descending=True)[:top_k+1]
    top_words = []
    for wid in top_ids:
        if wid != word_id:
            top_words.append((vocab.get_word(wid.item()), similarities[wid].item()))
    
    return top_words[:top_k]


class SentimentClassifier(nn.Module):
    """Sentiment classifier on top of embeddings."""
    
    def __init__(self, embedding_dim, frozen_embeddings=None):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def evaluate_sentiment(embeddings, vocab, sentiment_data, device, epochs=20, lr=0.01):
    """
    Train and evaluate sentiment classifier on top of embeddings.
    
    Args:
        embeddings: Word embeddings (vocab_size, embedding_dim)
        vocab: Vocabulary
        sentiment_data: Dict with reviews and labels
        device: Device
        epochs: Training epochs
        lr: Learning rate
    
    Returns:
        dict with metrics
    """
    classifier = SentimentClassifier(embeddings.size(1))
    classifier.to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Prepare data
    all_reviews = sentiment_data['positive'] + sentiment_data['negative']
    labels = torch.tensor(sentiment_data['labels'], dtype=torch.float32).to(device)
    
    # Convert reviews to embeddings (average word embeddings)
    review_embeddings = []
    for review in all_reviews:
        tokens = [vocab.get_id(w) for w in review.split()]
        if tokens:
            review_emb = embeddings[tokens].mean(dim=0)
        else:
            review_emb = torch.zeros(embeddings.size(1), device=device)
        review_embeddings.append(review_emb)
    
    review_embeddings = torch.stack(review_embeddings).to(device)
    
    # Train
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        
        logits = classifier(review_embeddings)
        loss = criterion(logits, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Sentiment Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate
    classifier.eval()
    with torch.no_grad():
        logits = classifier(review_embeddings)
        predictions = (logits > 0.5).long().squeeze()
        accuracy = (predictions == labels.long()).float().mean().item()
    
    return {
        'accuracy': accuracy,
        'loss': loss.item()
    }


def train(embedding_loader, device, epochs=10, lr=0.025, **kwargs):
    """Train wrapper for protocol."""
    model = build_model(5000, embedding_dim=100)
    model.to(device)
    history = train_embeddings(model, embedding_loader, device, epochs=epochs, lr=lr)
    return model, history


def predict(embeddings, vocab, word_list):
    """Predict similar words for a list of words."""
    results = {}
    for word in word_list:
        similar = get_similar_words(embeddings, vocab, word, top_k=5)
        results[word] = similar
    return results


def evaluate(model, embedding_loader, device, vocab=None, return_dict=True):
    """Evaluate model (embedding quality metrics)."""
    embeddings = model.get_embeddings()
    
    # Test word similarities
    test_pairs = [
        ('great', 'good'),
        ('bad', 'terrible'),
        ('movie', 'film')
    ]
    
    similarities = []
    for w1, w2 in test_pairs:
        sim = compute_word_similarity(embeddings, vocab, w1, w2)
        similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    
    if not return_dict:
        return avg_similarity
    
    return {
        'avg_similarity': avg_similarity,
        'embedding_norm': embeddings.norm().item()
    }


def save_artifacts(model, embeddings, vocab, histories, metrics, output_dir=OUTPUT_DIR):
    """Save models and artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'embedding_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, 'embeddings.pt')
    torch.save(embeddings, embeddings_path)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump({
            'word2id': vocab.word2id,
            'id2word': {str(k): v for k, v in vocab.id2word.items()}
        }, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    # Convert tensors to python types for json serialization
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            metrics_serializable[k] = {k2: float(v2) if isinstance(v2, (torch.Tensor, np.ndarray)) else v2 
                                      for k2, v2 in v.items()}
        else:
            metrics_serializable[k] = float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    """
    Main pipeline:
    1. Build vocabulary and embedding dataset
    2. Train Skip-gram embeddings
    3. Evaluate embedding quality (word similarity)
    4. Train sentiment classifier on embeddings
    5. Assert quality thresholds
    """
    
    try:
        device = get_device()
        print(f"Using device: {device}")
        
        # Load data
        print("\nPreparing embedding and sentiment data...")
        embedding_loader, sentiment_data, vocab = make_dataloaders(batch_size=16)
        
        print(f"Vocabulary size: {vocab.size}")
        print(f"Embedding training pairs: {len(embedding_loader.dataset)}")
        
        # Train embeddings
        print("\n" + "="*60)
        print("Training Word Embeddings (Skip-gram with Negative Sampling)")
        print("="*60)
        
        model = build_model(vocab.size, embedding_dim=100)
        model.to(device)
        embedding_history = train_embeddings(model, embedding_loader, device, epochs=10, lr=0.025)
        
        # Get learned embeddings
        embeddings = model.get_embeddings()
        
        # Evaluate embedding quality
        print("\n" + "="*60)
        print("Evaluating Embedding Quality")
        print("="*60)
        
        print("\nSimilar words to 'great':")
        similar_great = get_similar_words(embeddings, vocab, 'great', top_k=5)
        for word, sim in similar_great:
            print(f"  {word}: {sim:.4f}")
        
        print("\nSimilar words to 'bad':")
        similar_bad = get_similar_words(embeddings, vocab, 'bad', top_k=5)
        for word, sim in similar_bad:
            print(f"  {word}: {sim:.4f}")
        
        print("\nSimilar words to 'movie':")
        similar_movie = get_similar_words(embeddings, vocab, 'movie', top_k=5)
        for word, sim in similar_movie:
            print(f"  {word}: {sim:.4f}")
        
        # Compute word pair similarities
        print("\nWord pair similarities:")
        pairs = [('great', 'good'), ('bad', 'terrible'), ('movie', 'film')]
        pair_similarities = []
        for w1, w2 in pairs:
            sim = compute_word_similarity(embeddings, vocab, w1, w2)
            pair_similarities.append(sim)
            print(f"  '{w1}' <-> '{w2}': {sim:.4f}")
        
        avg_pair_sim = np.mean(pair_similarities)
        
        # Train sentiment classifier
        print("\n" + "="*60)
        print("Training Sentiment Classifier on Embeddings")
        print("="*60)
        
        sentiment_metrics = evaluate_sentiment(embeddings, vocab, sentiment_data, device, epochs=20, lr=0.01)
        print(f"\nSentiment classification accuracy: {sentiment_metrics['accuracy']:.4f}")
        
        # Collect metrics
        all_metrics = {
            'embedding_training_loss': embedding_history['loss'][-1],
            'avg_word_pair_similarity': float(avg_pair_sim),
            'embedding_norm': float(embeddings.norm()),
            'sentiment_accuracy': sentiment_metrics['accuracy'],
            'vocab_size': vocab.size
        }
        
        save_artifacts(model, embeddings, vocab, embedding_history, all_metrics)
        
        # Assertions for quality
        print("\n" + "="*60)
        print("Quality Assertions")
        print("="*60)
        
        assert avg_pair_sim > 0.3, \
            f"Average word pair similarity {avg_pair_sim:.4f} should be > 0.3"
        print("✓ Word embeddings show reasonable semantic similarity")
        
        assert sentiment_metrics['accuracy'] > 0.7, \
            f"Sentiment accuracy {sentiment_metrics['accuracy']:.4f} should be > 0.7"
        print("✓ Sentiment classifier achieves > 70% accuracy on embeddings")
        
        assert embeddings.norm() > 0, \
            "Embeddings should have non-zero magnitude"
        print("✓ Embeddings are properly initialized")
        
        print("\n" + "="*60)
        print("SUCCESS: All assertions passed!")
        print("="*60)
        sys.exit(0)
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

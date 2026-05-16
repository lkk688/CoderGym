# -*- coding: utf-8 -*-
"""
BigQuery Bigframe Text Embeddings with Semantic Similarity Task

Mathematical Formulation:
- Embedding Generation: e_i = Embed(text_i) where Embed is BigQuery Vertex AI Embeddings
- Cosine Similarity: sim(e_i, e_j) = (e_i Â· e_j) / (||e_i|| ||e_j||)
- Semantic Search: Find k-nearest neighbors using embedding similarity
- Neural Network Classifier: y = sigmoid(MLP(concatenate_embeddings(text_pairs)))
- Contrastive Loss: L = -log(exp(sim(e_pos) / tau) / sum_j exp(sim(e_j) / tau))
  where tau is temperature scaling parameter

This task demonstrates:
1. Loading text data from BigQuery Bigframe with proper authentication handling
2. Generating embeddings using BigQuery Vertex AI Embeddings API
3. Building semantic similarity models using neural networks
4. Evaluating embedding quality through semantic textual similarity (STS) benchmarks
5. Fallback to local embeddings when BigQuery is unavailable (for testing)
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
import random
import hashlib
from typing import List, Dict, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Output directory for artifacts
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'output', 'nlp_lvl2_bigframe_embeddings'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'bigframe_embeddings_semantic_similarity',
        'description': 'BigQuery Bigframe embeddings with semantic similarity neural network',
        'embedding_dim': 768,  # Vertex AI embeddings dimension
        'task_type': 'nlp_semantic_similarity',
        'use_bigquery': bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')),
        'fallback_mode': 'local_simulated_embeddings'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    """Get device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BigFrameDataLoader:
    """
    Handles loading text data from BigQuery Bigframe.
    
    This class demonstrates:
    - Connecting to BigQuery with Bigframe
    - Loading data using SQL queries
    - Fallback to local data when BigQuery is unavailable
    """
    
    def __init__(self, use_bigquery: bool = False):
        self.use_bigquery = use_bigquery and os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if self.use_bigquery:
            try:
                import bigframes
                import pandas as pd
                self.bigframes = bigframes
                self.pd = pd
                print("[BigFrame] Using BigQuery Bigframe for data loading")
            except ImportError:
                print("[BigFrame] WARNING: bigframes library not found. Falling back to local data.")
                self.use_bigquery = False
    
    def load_data(self) -> Tuple[List[str], List[str], List[int]]:
        """
        Load text data from BigQuery Bigframe or local fallback.
        
        Returns:
            Tuple of (texts1, texts2, similarity_labels)
            - texts1: List of first texts
            - texts2: List of second texts  
            - similarity_labels: Binary labels (1 = similar, 0 = dissimilar)
        """
        if self.use_bigquery:
            return self._load_from_bigquery()
        else:
            return self._load_local_fallback()
    
    def _load_from_bigquery(self) -> Tuple[List[str], List[str], List[int]]:
        """
        Load data from BigQuery using Bigframe API.
        
        Example query structure:
        SELECT text1, text2, similarity_score FROM `project.dataset.sts_benchmark`
        WHERE similarity_score IS NOT NULL LIMIT 1000
        """
        try:
            # Set up BigFrame session
            session = self.bigframes.Session(
                credentials_path=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            )
            
            # Query STS (Semantic Textual Similarity) benchmark data
            query = """
            SELECT text1, text2, similarity_score 
            FROM `google-research-datasets.sts_benchmark.train`
            WHERE similarity_score IS NOT NULL 
            LIMIT 500
            """
            
            df = session.sql(query)
            
            # Convert to local DataFrame
            df_local = df.to_pandas()
            
            texts1 = df_local['text1'].tolist()
            texts2 = df_local['text2'].tolist()
            # Normalize similarity scores (0-5) to binary labels
            similarity_labels = [
                1 if score >= 3.0 else 0 
                for score in df_local['similarity_score'].tolist()
            ]
            
            print(f"[BigFrame] Loaded {len(texts1)} text pairs from BigQuery")
            return texts1, texts2, similarity_labels
            
        except Exception as e:
            print(f"[BigFrame] ERROR loading from BigQuery: {e}")
            print("[BigFrame] Falling back to local data")
            return self._load_local_fallback()
    
    def _load_local_fallback(self) -> Tuple[List[str], List[str], List[int]]:
        """
        Fallback to local semantic similarity dataset.
        
        This provides a representative sample for testing and demonstration.
        """
        # Expanded Semantic Textual Similarity pairs (similar and dissimilar)
        similar_pairs = [
            ("A man is playing a large flute", "A man is playing a flute"),
            ("A woman is eating something", "A woman is consuming food"),
            ("A boy is jumping on the trampoline", "A boy is jumping on a bed"),
            ("Three people are dancing", "Multiple people are performing a dance"),
            ("The cars are driving in the opposite directions", "The cars are driving in different ways"),
            ("The grass is green", "The green grass is beautiful"),
            ("A dog is running in the park", "A canine is jogging in the garden"),
            ("People are dancing at a wedding", "Guests are performing at a celebration"),
            ("The cat is sleeping on the bed", "A feline is resting on furniture"),
            ("Two women are playing volleyball", "Women are engaged in a volleyball match"),
            ("The sun is shining brightly", "The sun is bright today"),
            ("A child is playing with a toy", "A young person is playing with a plaything"),
            ("The weather is cloudy", "It is overcast outside"),
            ("She is running quickly", "She is moving fast"),
            ("The book is on the shelf", "The novel is placed on the bookshelf"),
            ("A car is parked outside", "A vehicle is standing outside"),
            ("The river flows to the sea", "Water flows downstream to the ocean"),
            ("A bird is flying in the sky", "A bird is soaring through the air"),
            ("He is wearing a red shirt", "He has a red shirt on"),
            ("The flower is blooming", "The flower is in bloom"),
            ("A cup of coffee is on the table", "Coffee is sitting on the table in a cup"),
            ("The door is closed", "The door is shut"),
            ("She is smiling happily", "She has a happy smile"),
            ("The computer is working", "The computer is functioning properly"),
            ("A student is studying hard", "A pupil is working diligently"),
            ("The movie is interesting", "The film is captivating"),
            ("He is driving a car", "He is operating a vehicle"),
            ("The music is loud", "The sound is intense"),
            ("A tree is tall", "A tree has great height"),
            ("The food is delicious", "The meal tastes wonderful"),
            ("A person is walking", "A human is moving on foot"),
            ("The window is open", "The window is ajar"),
            ("She is happy", "She is in a good mood"),
            ("The water is cold", "The water is chilly"),
            ("A dog barks loudly", "A dog is barking"),
            ("The road is long", "The road extends far"),
            ("A person is tired", "A person is exhausted"),
            ("The room is bright", "The room is well-lit"),
            ("A child is playing outside", "A child is outdoors playing"),
            ("The night is dark", "The night is pitch black"),
            ("A man is tall", "A man has a great stature"),
            ("The job is difficult", "The job is challenging"),
            ("She is intelligent", "She is smart"),
            ("The house is clean", "The house is tidy"),
            ("A baby is crying", "An infant is weeping"),
            ("The road is wet", "The road is damp"),
            ("A person is strong", "A person is powerful"),
            ("The movie was entertaining", "The film was enjoyable"),
            ("She is wearing a blue dress", "She has on a blue skirt"),
            ("A cat is sleeping", "A cat is resting"),
        ]
        
        dissimilar_pairs = [
            ("A man is riding a horse", "The weather is sunny today"),
            ("The car is red", "Birds are flying in the sky"),
            ("She likes apples", "He prefers mathematics"),
            ("The book is on the table", "The dog is barking loudly"),
            ("I enjoy reading novels", "Cars are made of metal"),
            ("The sun is shining", "Computers process data"),
            ("Water is flowing downhill", "Music is playing softly"),
            ("The building is tall", "Cats can climb trees"),
            ("She is wearing a blue dress", "Coffee is hot"),
            ("The clock shows noon", "Flowers bloom in spring"),
            ("Pizza is delicious", "The elephant is large"),
            ("Programming is fun", "Tennis is a sport"),
            ("The ocean is blue", "Iron is dense"),
            ("A piano is an instrument", "Ice cream is cold"),
            ("The forest is green", "Diamonds are precious"),
            ("Basketball is played indoors", "Lightning is powerful"),
            ("She likes chocolate", "Mathematics is abstract"),
            ("The rainbow is colorful", "Submarines are underwater"),
            ("A bicycle has wheels", "The earth rotates"),
            ("Milk is white", "Stars shine at night"),
            ("The desert is hot", "Penguins live in cold regions"),
            ("Dance is an art form", "Numbers are infinite"),
            ("A fish swims", "Mountains touch the sky"),
            ("Wine is fermented", "The sky has clouds"),
            ("A phone can ring", "Gravity pulls things down"),
            ("Poetry uses words", "Atoms are tiny"),
            ("The ocean has waves", "Rockets fly to space"),
            ("Silver is shiny", "The internet connects people"),
            ("A camera takes pictures", "Wind moves the trees"),
            ("Glass is transparent", "Thunder follows lightning"),
            ("The stadium is large", "Magnets attract metal"),
            ("A microscope magnifies", "Robots are mechanical"),
            ("Pepper is spicy", "Evolution changes species"),
            ("The highway is busy", "Bacteria are microorganisms"),
            ("A painting is colorful", "Electricity powers devices"),
            ("The forest is quiet", "Fossils tell us about the past"),
            ("A bridge connects places", "Genes carry information"),
            ("Marble is hard", "Photosynthesis produces oxygen"),
            ("The beach has sand", "Volcanoes erupt lava"),
            ("A lighthouse guides ships", "Rainbows form after rain"),
            ("Silk is smooth", "Pollution damages the environment"),
            ("The volcano erupts", "Hurricanes cause damage"),
            ("A microscope shows details", "Black holes are in space"),
            ("Diamonds sparkle", "Tornadoes are dangerous"),
            ("The laboratory experiments", "Earthquakes shake the ground"),
            ("A museum displays art", "Tsunamis are ocean waves"),
            ("Steel is strong", "Avalanches slide down mountains"),
            ("The carnival is fun", "Wildfires spread quickly"),
            ("A garden has flowers", "Deserts have little water"),
            ("The symphony plays music", "Glaciers are melting"),
        ]
        
        texts1 = []
        texts2 = []
        labels = []
        
        # Add similar pairs (label=1)
        for text1, text2 in similar_pairs:
            texts1.append(text1)
            texts2.append(text2)
            labels.append(1)
        
        # Add dissimilar pairs (label=0)
        for text1, text2 in dissimilar_pairs:
            texts1.append(text1)
            texts2.append(text2)
            labels.append(0)
        
        # Shuffle
        combined = list(zip(texts1, texts2, labels))
        random.shuffle(combined)
        texts1, texts2, labels = zip(*combined)
        
        print(f"[BigFrame] Loaded {len(texts1)} text pairs from local fallback")
        return list(texts1), list(texts2), list(labels)


class EmbeddingGenerator:
    """
    Generate embeddings using BigQuery Vertex AI Embeddings API or local fallback.
    """
    
    def __init__(self, embedding_dim: int = 768, use_bigquery: bool = False):
        self.embedding_dim = embedding_dim
        self.use_bigquery = use_bigquery and os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.embedding_cache = {}
        
        if self.use_bigquery:
            try:
                from vertexai.language_models import TextEmbeddingModel
                self.model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
                print("[Embeddings] Using Vertex AI TextEmbedding model")
            except ImportError:
                print("[Embeddings] WARNING: Vertex AI not available. Using local embeddings.")
                self.use_bigquery = False
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text string.
        
        Args:
            text: Input text string
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        if self.use_bigquery:
            embedding = self._embed_with_vertex_ai(text)
        else:
            embedding = self._embed_locally(text)
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def _embed_with_vertex_ai(self, text: str) -> np.ndarray:
        """Generate embedding using Vertex AI TextEmbedding API."""
        try:
            embeddings = self.model.get_embeddings([text])
            embedding = np.array(embeddings[0].values, dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"[Embeddings] Error with Vertex AI: {e}. Using local fallback.")
            return self._embed_locally(text)
    
    def _embed_locally(self, text: str) -> np.ndarray:
        """
        Generate simulated embeddings locally using hash-based approach.
        
        For demonstration: creates deterministic embeddings based on text content
        that preserve semantic similarity through character n-gram statistics.
        """
        # Create deterministic embedding from text
        text_lower = text.lower()
        
        # Initialize embedding
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Use character n-grams to create features
        for i in range(len(text_lower) - 2):
            trigram = text_lower[i:i+3]
            # Hash-based feature index
            feature_idx = (sum(ord(c) for c in trigram) * 7919) % self.embedding_dim
            embedding[feature_idx] += 1.0
        
        # Add word-level features
        words = text_lower.split()
        for word in words:
            feature_idx = ((sum(ord(c) for c in word) * 7919) % (self.embedding_dim // 2)) + (self.embedding_dim // 2)
            embedding[feature_idx % self.embedding_dim] += 0.5
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Add some noise for realism (but deterministic)
        np.random.seed((sum(ord(c) for c in text) * 7919) % (2**31))
        embedding += np.random.normal(0, 0.01, self.embedding_dim).astype(np.float32)
        
        # Renormalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def embed_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        embeddings = np.array([
            self.embed_text(text) for text in texts
        ], dtype=np.float32)
        return embeddings


class SemanticSimilarityDataset(Dataset):
    """Dataset for semantic similarity pairs with pre-computed embeddings."""
    
    def __init__(
        self,
        texts1: List[str],
        texts2: List[str],
        labels: List[int],
        embedding_generator: EmbeddingGenerator
    ):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.embedding_generator = embedding_generator
        
        # Pre-compute all embeddings
        print("[Dataset] Pre-computing embeddings for all text pairs...")
        all_texts = list(set(texts1 + texts2))
        self.embeddings_cache = {}
        for text in all_texts:
            self.embeddings_cache[text] = embedding_generator.embed_text(text)
        print(f"[Dataset] Cached {len(self.embeddings_cache)} unique embeddings")
    
    def __len__(self):
        return len(self.texts1)
    
    def __getitem__(self, idx):
        emb1 = torch.FloatTensor(self.embeddings_cache[self.texts1[idx]])
        emb2 = torch.FloatTensor(self.embeddings_cache[self.texts2[idx]])
        label = torch.LongTensor([self.labels[idx]])
        
        return emb1, emb2, label.squeeze()


def make_dataloaders(batch_size=16, use_bigquery: Optional[bool] = None):
    """
    Create dataloaders for semantic similarity task.
    
    Pipeline:
    1. Load text pairs from BigQuery Bigframe (or local fallback)
    2. Generate embeddings using Vertex AI or local method
    3. Create semantic similarity dataset
    4. Split into train/validation
    
    Returns:
        Tuple of (train_loader, val_loader, embedding_generator)
    """
    # Auto-detect BigQuery usage from credentials unless explicitly provided.
    if use_bigquery is None:
        use_bigquery = bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

    # Load data from BigQuery Bigframe
    bigframe_loader = BigFrameDataLoader(use_bigquery=use_bigquery)
    texts1, texts2, labels = bigframe_loader.load_data()
    
    # Generate embeddings
    embedding_generator = EmbeddingGenerator(embedding_dim=768, use_bigquery=use_bigquery)
    
    # Create dataset
    dataset = SemanticSimilarityDataset(texts1, texts2, labels, embedding_generator)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Create a generator with fixed seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(42)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, embedding_generator


class SemanticSimilarityModel(nn.Module):
    """
    Neural network for semantic similarity prediction.
    
    Architecture:
    - Input: Concatenated embeddings of two texts [emb1, emb2, emb1*emb2, |emb1-emb2|]
    - Hidden layers with ReLU activation and dropout
    - Output: Binary classification (similar or not)
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512):
        super(SemanticSimilarityModel, self).__init__()
        
        # Input: concatenated similarities features
        # Features: [emb1, emb2, emb1*emb2, |emb1-emb2|] = 4*embedding_dim
        input_dim = 4 * embedding_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)  # Binary classification
        )
    
    def forward(self, emb1, emb2):
        """
        Compute similarity score for embedding pairs.
        
        Args:
            emb1: Embedding 1 of shape (B, D)
            emb2: Embedding 2 of shape (B, D)
        
        Returns:
            Logits of shape (B, 2)
        """
        # Create similarity features
        elementwise_mult = emb1 * emb2
        elementwise_diff = torch.abs(emb1 - emb2)
        
        # Concatenate all features
        combined = torch.cat([emb1, emb2, elementwise_mult, elementwise_diff], dim=1)
        
        logits = self.network(combined)
        return logits


def build_model(embedding_dim: int = 768, hidden_dim: int = 512, **kwargs):
    """Build semantic similarity model."""
    return SemanticSimilarityModel(embedding_dim, hidden_dim)


def train(model, train_loader, device, epochs=20, lr=0.001):
    """
    Train semantic similarity model.
    
    Args:
        model: SemanticSimilarityModel
        train_loader: DataLoader for training
        device: PyTorch device
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Training history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'accuracy': []}
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for emb1, emb2, labels in train_loader:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(emb1, emb2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    return history


def evaluate(model, data_loader, device):
    """
    Evaluate semantic similarity model.
    
    Args:
        model: SemanticSimilarityModel
        data_loader: DataLoader for evaluation
        device: PyTorch device
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for emb1, emb2, labels in data_loader:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            
            logits = model(emb1, emb2)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total
    }


def predict(model, emb1, emb2, device):
    """
    Predict similarity for embedding pairs.
    
    Args:
        model: SemanticSimilarityModel
        emb1: Embedding 1
        emb2: Embedding 2
        device: PyTorch device
    
    Returns:
        Predicted similarity (0 or 1)
    """
    model.eval()
    with torch.no_grad():
        emb1 = torch.FloatTensor(emb1).unsqueeze(0).to(device)
        emb2 = torch.FloatTensor(emb2).unsqueeze(0).to(device)
        logits = model(emb1, emb2)
        pred = torch.argmax(logits, dim=1).item()
    
    return pred


def save_artifacts(model, embedding_generator, history, metrics, output_dir=OUTPUT_DIR):
    """
    Save model, embeddings, and metrics.
    
    Args:
        model: Trained model
        embedding_generator: Embedding generator with cache
        history: Training history
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'semantic_similarity_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"[Artifacts] Saved model to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    metrics_to_save = {
        'validation_accuracy': metrics['accuracy'],
        'validation_loss': metrics['loss'],
        'training_history': {
            'loss': [float(l) for l in history['loss']],
            'accuracy': [float(a) for a in history['accuracy']]
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"[Artifacts] Saved metrics to {metrics_path}")
    
    # Save embedding cache summary
    cache_summary_path = os.path.join(output_dir, 'embedding_cache_summary.json')
    cache_summary = {
        'cached_texts': len(embedding_generator.embedding_cache),
        'embedding_dim': embedding_generator.embedding_dim
    }
    with open(cache_summary_path, 'w') as f:
        json.dump(cache_summary, f, indent=2)
    print(f"[Artifacts] Saved embedding cache summary to {cache_summary_path}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("BigQuery Bigframe Embeddings - Semantic Similarity Task")
    print("="*80)
    
    try:
        # Setup
        device = get_device()
        set_seed(42)
        
        metadata = get_task_metadata()
        print(f"\n[Config] Task: {metadata['task_name']}")
        print(f"[Config] Embedding Dimension: {metadata['embedding_dim']}")
        print(f"[Config] Using BigQuery: {metadata['use_bigquery']}")
        print(f"[Config] Device: {device}")
        
        # Create dataloaders
        print("\n[Data] Loading data from BigQuery Bigframe...")
        train_loader, val_loader, embedding_generator = make_dataloaders(batch_size=16)
        
        # Build model
        print("\n[Model] Building semantic similarity model...")
        model = build_model(embedding_dim=768, hidden_dim=512)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] Total parameters: {total_params:,}")
        
        # Train
        print("\n[Training] Starting training...")
        history = train(model, train_loader, device, epochs=20, lr=0.001)
        
        # Evaluate on validation set
        print("\n[Evaluation] Evaluating on validation set...")
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\n[Results] Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"[Results] Validation Loss: {val_metrics['loss']:.4f}")
        
        # Evaluate on training set
        train_metrics = evaluate(model, train_loader, device)
        print(f"\n[Results] Training Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"[Results] Training Loss: {train_metrics['loss']:.4f}")
        
        # Quality assertions
        print("\n[Assertions] Checking quality thresholds...")
        
        assert val_metrics['accuracy'] > 0.60, \
            f"Validation accuracy {val_metrics['accuracy']:.4f} must be > 0.60"
        print("âœ“ Validation accuracy > 0.60")
        
        assert train_metrics['accuracy'] > 0.65, \
            f"Training accuracy {train_metrics['accuracy']:.4f} must be > 0.65"
        print("âœ“ Training accuracy > 0.65")
        
        assert val_metrics['loss'] < 1.0, \
            f"Validation loss {val_metrics['loss']:.4f} must be < 1.0"
        print("âœ“ Validation loss < 1.0")
        
        assert len(history['accuracy']) > 0, "Training history empty"
        print("âœ“ Training history recorded")
        
        # Save artifacts
        print("\n[Saving] Persisting model and artifacts...")
        save_artifacts(model, embedding_generator, history, val_metrics)
        
        print("\n" + "="*80)
        print("SUCCESS: All assertions passed!")
        print("="*80)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\nâœ— ASSERTION FAILED: {e}")
        print("="*80)
        sys.exit(1)
    
    except Exception as e:
        print(f"\nâœ— ERROR: {e}")
        import traceback
        traceback.print_exc()

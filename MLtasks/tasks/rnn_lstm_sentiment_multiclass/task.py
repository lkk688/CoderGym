"""
LSTM Sentiment Classification Task (3-Class)
Tokenization + embedding + LSTM classifier for negative, neutral, and positive sentiment

This task is based on rnn_lvl2_lstm_sentiment, creating a new task by replacing BCELoss + sigmoid with CrossEntropyLoss using multi-class logits.
The new task is a three class sentiment (negative, neutral, positive) instead of a binary sentiment classificater.
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import Dict, List, Tuple, Any
import pickle
from datetime import datetime

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
OUTPUT_DIR = "/Developer/AIserver/output/tasks/rnn_lstm_sentiment_multiclass"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Vocabulary:
    """Simple vocabulary class for tokenization."""

    def __init__(self, max_size: int = 10000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts."""
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Filter by minimum frequency
        filtered_words = [
            word for word, freq in counter.items() if freq >= self.min_freq
        ]

        # Sort by frequency, then alphabetically for determinism
        sorted_words = sorted(filtered_words, key=lambda x: (-counter[x], x))

        # Add to vocabulary up to max_size
        for word in sorted_words[: self.max_size - 2]:  # -2 for <PAD> and <UNK>
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split by whitespace and lowercase."""
        return text.lower().split()

    def encode(self, text: str, max_len: int = 100) -> List[int]:
        """Encode text to indices with padding/truncation."""
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

        # Truncate if too long
        if len(indices) > max_len:
            indices = indices[:max_len]

        # Pad if too short
        if len(indices) < max_len:
            indices = indices + [self.word2idx["<PAD>"]] * (max_len - len(indices))

        return indices

    def __len__(self) -> int:
        return len(self.word2idx)


class SentimentDataset(Dataset):
    """PyTorch Dataset for 3-class sentiment classification."""

    def __init__(
        self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 100
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Encode text
        encoded = self.vocab.encode(text, self.max_len)
        text_tensor = torch.tensor(encoded, dtype=torch.long)

        # CrossEntropyLoss expects class indices as LongTensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return text_tensor, label_tensor


class LSTMClassifier(nn.Module):
    """LSTM-based 3-class sentiment classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 3,
    ):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)

        # Final forward and backward hidden states
        hidden_fwd = hidden[-2, :, :]  # (batch_size, hidden_dim)
        hidden_bwd = hidden[-1, :, :]  # (batch_size, hidden_dim)
        hidden_cat = torch.cat(
            (hidden_fwd, hidden_bwd), dim=1
        )  # (batch_size, hidden_dim * 2)

        # Apply dropout and fully connected layer
        out = self.dropout(hidden_cat)
        logits = self.fc(out)  # raw logits for CrossEntropyLoss

        return logits


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "lstm_sentiment_multiclass_classification",
        "task_type": "multiclass_classification",
        "description": "LSTM-based sentiment classification with tokenization and embedding for negative, neutral, and positive sentiment",
        "input_type": "text",
        "output_type": "3_class_sentiment",
        "num_classes": 3,
        "class_names": ["negative", "neutral", "positive"],
        "created_at": datetime.now().isoformat(),
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(
    batch_size: int = 32, max_len: int = 100, train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """Create dataloaders for training and validation."""
    # Sample sentiment data (positive/negative/neutral reviews)

    positive_texts = [
        "This movie was absolutely fantastic and I loved it.",
        "Great acting and an amazing storyline.",
        "The film was wonderful and very enjoyable.",
        "What a beautiful and inspiring movie.",
        "Excellent performances and strong direction.",
        "I really enjoyed this movie from start to finish.",
        "The story was engaging and the acting was excellent.",
        "A very entertaining and satisfying film.",
        "This was a brilliant movie with a strong message.",
        "One of the best films I have seen recently.",
    ]

    neutral_texts = [
        "This movie was okay and nothing special.",
        "The film was average and had some decent moments.",
        "It was neither good nor bad, just fine.",
        "The movie was watchable but not memorable.",
        "Some parts were interesting, others were dull.",
        "It was a fairly standard film with mixed results.",
        "The acting was acceptable and the plot was simple.",
        "This movie was just average overall.",
        "I did not love it, but I did not hate it either.",
        "The film was decent enough for a casual watch.",
    ]

    negative_texts = [
        "This movie was terrible and a complete waste of time.",
        "Awful acting and a very boring storyline.",
        "The worst film I have seen in a long time.",
        "Poor direction and weak performances throughout.",
        "I regret watching this movie, it was bad.",
        "A very disappointing and poorly made film.",
        "The script was terrible and the pacing was awful.",
        "This movie was boring from start to finish.",
        "One of the worst movie experiences I have had.",
        "It was badly written and not enjoyable at all.",
    ]

    texts = positive_texts + neutral_texts + negative_texts

    # Label mapping:
    # 0 = negative, 1 = neutral, 2 = positive
    labels = (
        [2] * len(positive_texts) + [1] * len(neutral_texts) + [0] * len(negative_texts)
    )

    # Shuffle data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Split ito train and validation sets
    split_idx = int(len(texts) * train_ratio)
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    val_texts = texts[split_idx:]
    val_labels = labels[split_idx:]

    # Build Vocabulary
    vocab = Vocabulary(max_size=1000, min_freq=1)
    vocab.build_vocab(texts)

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_len)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, max_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocab


def build_model(vocab_size: int) -> LSTMClassifier:
    """Build the LSTM classifier model."""
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        num_classes=3,
    )
    return model.to(device)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
) -> Dict[str, List[float]]:
    """Train the model and return training history."""
    # Use Cross Entropy Loss for multiclass
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        val_metrics = evaluate(model, val_loader)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        # Early stopping base on F1 score
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            # Save best model
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(OUTPUT_DIR, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def evaluate(model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    """Evaluate the model and return multiclass metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    num_classes = 3
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes

    with torch.no_grad():
        for texts, labels in data_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            logits = model(texts)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            # Calculate metrics
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for cls in range(num_classes):
                true_positives[cls] += (
                    ((predictions == cls) & (labels == cls)).sum().item()
                )
                false_positives[cls] += (
                    ((predictions == cls) & (labels != cls)).sum().item()
                )
                false_negatives[cls] += (
                    ((predictions != cls) & (labels == cls)).sum().item()
                )

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for cls in range(num_classes):
        precision = (
            true_positives[cls] / (true_positives[cls] + false_positives[cls])
            if (true_positives[cls] + false_positives[cls]) > 0
            else 0.0
        )
        recall = (
            true_positives[cls] / (true_positives[cls] + false_negatives[cls])
            if (true_positives[cls] + false_negatives[cls]) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    macro_precision = sum(per_class_precision) / num_classes
    macro_recall = sum(per_class_recall) / num_classes
    macro_f1 = sum(per_class_f1) / num_classes

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "negative_f1": per_class_f1[0],
        "neutral_f1": per_class_f1[1],
        "positive_f1": per_class_f1[2],
    }


def predict(
    model: nn.Module, texts: List[str], vocab: Vocabulary, max_len: int = 100
) -> List[Dict[str, Any]]:
    """Predict class probabilities and sentiment labels for a list of texts."""
    model.eval()
    class_names = ["negative", "neutral", "positive"]
    predictions = []

    with torch.no_grad():
        for text in texts:
            # Encode text
            encoded = vocab.encode(text, max_len)
            text_tensor = (
                torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
            )

            logits = model(text_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_class = torch.argmax(probs).item()

            predictions.append(
                {
                    "text": text,
                    "predicted_class_index": pred_class,
                    "predicted_label": class_names[pred_class],
                    "probabilities": {
                        "negative": float(probs[0].item()),
                        "neutral": float(probs[1].item()),
                        "positive": float(probs[2].item()),
                    },
                }
            )

    return predictions


def save_artifacts(
    model: nn.Module, vocab: Vocabulary, history: Dict[str, List[float]]
):
    """Save model, vocabulary, training history, and metrics."""
    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))

    # Save vocabulary
    with open(os.path.join(OUTPUT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    # Save training history
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save task metadata
    metadata = get_task_metadata()
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to {OUTPUT_DIR}")


def main():
    """Main function to run the 3-class LSTM sentiment classification task."""
    print("=" * 60)
    print("LSTM Sentiment Classification Task (3-Class)")
    print("=" * 60)

    set_seed(42)

    print(f"Using device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, vocab = make_dataloaders(batch_size=4, max_len=50)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(vocab_size=len(vocab))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Train model
    print("\nTraining model...")
    history = train(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001)

    # Evaluate on training set
    print("\n" + "=" * 60)
    print("Evaluating on Training Set")
    print("=" * 60)
    train_metrics = evaluate(model, train_loader)
    print(f"Training Loss: {train_metrics['loss']:.4f}")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Training Macro Precision: {train_metrics['macro_precision']:.4f}")
    print(f"Training Macro Recall: {train_metrics['macro_recall']:.4f}")
    print(f"Training Macro F1: {train_metrics['macro_f1']:.4f}")

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Evaluating on Validation Set")
    print("=" * 60)
    val_metrics = evaluate(model, val_loader)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Macro Precision: {val_metrics['macro_precision']:.4f}")
    print(f"Validation Macro Recall: {val_metrics['macro_recall']:.4f}")
    print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Negative Class F1: {val_metrics['negative_f1']:.4f}")
    print(f"Neutral Class F1: {val_metrics['neutral_f1']:.4f}")
    print(f"Positive Class F1: {val_metrics['positive_f1']:.4f}")

    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    save_artifacts(model, vocab, history)

    # Test predictions
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)
    test_texts = [
        "This movie was amazing and very enjoyable.",
        "This movie was okay, nothing really stood out.",
        "This movie was awful and boring.",
    ]
    predictions = predict(model, test_texts, vocab, max_len=50)

    for i, pred in enumerate(predictions, start=1):
        print(f"Text {i}: {pred['text']}")
        print(f"Predicted Label: {pred['predicted_label']}")
        print(f"Probabilities: {pred['probabilities']}\n")

    print("\n" + "=" * 60)
    print("Quality Thresholds Check")
    print("=" * 60)

    min_val_accuracy = 0.60
    min_val_macro_f1 = 0.55

    accuracy_pass = val_metrics["accuracy"] >= min_val_accuracy
    f1_pass = val_metrics["macro_f1"] >= min_val_macro_f1

    print(
        f"Validation accuracy >= {min_val_accuracy}: "
        f"{'PASS' if accuracy_pass else 'FAIL'} ({val_metrics['accuracy']:.4f})"
    )
    print(
        f"Validation macro F1 >= {min_val_macro_f1}: "
        f"{'PASS' if f1_pass else 'FAIL'} ({val_metrics['macro_f1']:.4f})"
    )

    all_pass = accuracy_pass and f1_pass

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if all_pass:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

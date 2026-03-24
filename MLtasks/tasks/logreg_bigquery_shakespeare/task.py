#!/usr/bin/env python3
"""
Binary Logistic Regression on BigQuery Shakespeare Data using PyTorch that classifies Hamlet vs Romeo and Juliet.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from google.cloud import bigquery
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "./output/tasks/logreg_lvl4_bigquery_shakespeare_binary"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata"""
    return {
        "task_name": "logistic_regression_bigquery_shakespeare",
        "task_type": "binary_classification",
        "input_dim": 5,
        "output_dim": 1,
        "description": "Binary logistic regression on BigQuery Shakespeare data",
    }


# ensure google cloud project already is set
def get_bigquery_client():
    """Create a BigQuery Client"""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set.")
    return bigquery.Client(project=project_id)


def load_data_from_bigquery(limit_rows=20000):
    """
    Load data from the public Shakespeare BigQuery dataset.

    Label:
        1 if corpus = 'hamlet'
        0 if corpus = 'romeoandjuliet'
    """
    client = get_bigquery_client()

    query = f"""
    SELECT
      CAST(word_count AS FLOAT64) AS word_count,
      CAST(LENGTH(word) AS FLOAT64) AS word_length,
      CAST(LENGTH(REGEXP_REPLACE(LOWER(word), r'[^aeiou]', '')) AS FLOAT64) AS vowel_count,
      CAST(IF(REGEXP_CONTAINS(LOWER(word), r'^[aeiou]'), 1, 0) AS FLOAT64) AS starts_with_vowel,
      CAST(IF(REGEXP_CONTAINS(LOWER(word), r'e$'), 1, 0) AS FLOAT64) AS ends_with_e,
      CAST(IF(corpus = 'hamlet', 1, 0) AS INT64) AS label
    FROM `bigquery-public-data.samples.shakespeare`
    WHERE corpus IN ('hamlet', 'romeoandjuliet')
      AND word IS NOT NULL
      AND word_count IS NOT NULL
      AND LENGTH(word) >= 2
    LIMIT {limit_rows}
    """

    print("Running BigQuery Shakespeare query...")
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} rows from BigQuery")

    if len(df) < 500:
        raise ValueError("Too few rows returned from BigQuery.")

    return df


def preprocess_dataframe(df):
    """Convert DataFrame to numpy arrays including feature matrix X and target vector y."""
    feature_cols = [
        "word_count",
        "word_length",
        "vowel_count",
        "starts_with_vowel",
        "ends_with_e",
    ]
    target_col = "label"

    X = df[feature_cols].astype(np.float32).values
    y = df[target_col].astype(np.float32).values.reshape(-1, 1)

    # standardizing input features manually
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-7
    X = (X - X_mean) / X_std

    return X, y, feature_cols


def make_dataloaders(batch_size=64, test_size=0.2):
    """Create logistic regression datasets and dataloaders."""
    df = load_data_from_bigquery(limit_rows=20000)
    X, y, feature_cols = preprocess_dataframe(df)

    # split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # create tensors and dataloaders
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val, feature_cols


class LogisticRegressionModel(nn.Module):
    """Simple Binary logistic regression model."""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def build_model(input_dim, device):
    """Build and return logreg model."""
    return LogisticRegressionModel(input_dim).to(device)


def train(model, train_loader, val_loader, device, epochs=80, lr=0.01, verbose=True):
    """Train model with Adam optimizer and BCEWithLogitsLoss for binary classification."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # validation loss
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f}"
            )

    return history


def evaluate(model, data_loader, device):
    """
    Evaluate classification performance.

    Returns metrics: loss, accuracy, precision, recall, f1
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()

            all_logits.append(logits.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_targets = np.vstack(all_targets)

    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(np.float32)

    accuracy = accuracy_score(all_targets, preds)
    precision = precision_score(all_targets, preds, zero_division=0)
    recall = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    avg_loss = total_loss / len(data_loader)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_artifacts(model, history, save_dir="output"):
    """Save model artifacts and visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, "linreg_model.pth")
    torch.save(model.state_dict(), model_path)

    # Save history
    history_path = os.path.join(save_dir, "linreg_history.npz")
    np.savez(
        history_path, train_loss=history["train_loss"], val_loss=history["val_loss"]
    )

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_dir, "logreg_bigquery_shakespeare_loss.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Artifacts saved to {save_dir}")

    return save_dir


def main():
    print("=" * 60)
    print("Logistic Regression on BigQuery Shakespeare Data")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data and create dataloaders
    print("\nLoading data from BigQuery...")
    train_loader, val_loader, X_train, X_val, y_train, y_val, feature_cols = (
        make_dataloaders(batch_size=64, test_size=0.2)
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {feature_cols}")

    print("\nBuilding model...")
    model = build_model(input_dim=len(feature_cols), device=device)
    print(model)

    print("\nTraining model...")
    history = train(
        model, train_loader, val_loader, device, epochs=80, lr=0.01, verbose=True
    )

    # Evaluate on train set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    # evaluate of validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, history, save_dir="output")
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    quality_passed = True

    check1 = val_metrics["accuracy"] > 0.60
    print(
        f"{'✓' if check1 else '✗'} Validation accuracy > 0.60: {val_metrics['accuracy']:.4f}"
    )
    quality_passed = quality_passed and check1

    check2 = val_metrics["f1"] > 0.60
    print(f"{'✓' if check2 else '✗'} Validation F1 > 0.60: {val_metrics['f1']:.4f}")
    quality_passed = quality_passed and check2

    check3 = history["train_loss"][-1] < history["train_loss"][0]
    print(
        f"{'✓' if check3 else '✗'} Training loss decreased: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}"
    )
    quality_passed = quality_passed and check3

    gap = abs(train_metrics["accuracy"] - val_metrics["accuracy"])
    check4 = gap < 0.20
    print(f"{'✓' if check4 else '✗'} Accuracy gap < 0.20: {gap:.4f}")
    quality_passed = quality_passed and check4

    print("\n" + "=" * 60)
    print(
        "PASS: All quality checks passed!"
        if quality_passed
        else "FAIL: Some quality checks failed!"
    )
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())

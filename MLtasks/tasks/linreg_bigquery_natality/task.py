"""
Linear Regression on BigQuery Natality Dataset to predict newborn birth weight using PyTorch."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from google.cloud import bigquery
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "./output/tasks/linreg_bigquery_natality"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return task metadata."""
    return {
        "name": "linear_regression_bigquery_natality",
        "description": "Predict birth weight from BigQuery natality data using Linear Regression with torch.autograd",
        "input_shape": [6],
        "output_shape": [1],
        "task_type": "regression",
    }


class LinearRegressionModel(nn.Module):
    """Simple linear regression model."""

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# ensure google cloud project already is set
def get_bigquery_client():
    """Create a BigQuery Client"""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set.")
    return bigquery.Client(project=project_id)


def load_data_from_bigquery(sample_frac=0.02, limit_rows=20000):
    """Query natality data from BigQuery public dataset."""
    client = get_bigquery_client()

    query = f"""
    SELECT
      CAST(mother_age AS FLOAT64) AS mother_age,
      CAST(father_age AS FLOAT64) AS father_age,
      CAST(gestation_weeks AS FLOAT64) AS gestation_weeks,
      CAST(weight_gain_pounds AS FLOAT64) AS weight_gain_pounds,
      CAST(plurality AS FLOAT64) AS plurality,
      CAST(IF(is_male, 1, 0) AS FLOAT64) AS is_male,
      CAST(weight_pounds AS FLOAT64) AS birth_weight_pounds
    FROM `bigquery-public-data.samples.natality`
    WHERE mother_age IS NOT NULL
      AND father_age IS NOT NULL
      AND gestation_weeks IS NOT NULL
      AND weight_gain_pounds IS NOT NULL
      AND plurality IS NOT NULL
      AND is_male IS NOT NULL
      AND weight_pounds IS NOT NULL
      AND RAND() < {sample_frac}
    LIMIT {limit_rows}
    """

    print("Running BigQuery query...")
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} rows from BigQuery")

    if len(df) < 500:
        raise ValueError(
            "Too few rows returned from BigQuery. Increase sample_frac or limit_rows."
        )

    return df


def preprocess_dataframe(df):
    """Preprocess DataFrame into a feature matrix X and target vector y."""
    feature_cols = [
        "mother_age",
        "father_age",
        "gestation_weeks",
        "weight_gain_pounds",
        "plurality",
        "is_male",
    ]
    target_col = "birth_weight_pounds"

    X = df[feature_cols].astype(np.float32).values
    y = df[target_col].astype(np.float32).values.reshape(-1, 1)

    # standardizing input features manually
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-7
    X = (X - X_mean) / X_std

    return X, y, feature_cols


def make_dataloaders(test_size=0.8, batch_size=32):
    """
    Create linear regression dataset and dataloaders by splitting preprocessed data
    into training and validation sets and then wrapping the tensors.
    """
    df = load_data_from_bigquery(sample_frac=0.02, limit_rows=20000)
    X, y, feature_cols = preprocess_dataframe(df)

    # split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
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


def build_model(input_dim, output_dim, device):
    """Build and return the model."""
    return LinearRegressionModel(input_dim, output_dim).to(device)


def train(model, train_loader, val_loader, device, epochs=100, lr=0.01, verbose=True):
    """
    Train the linear regression model using autograd.

    Loss function: J(θ) = (1/2m) * Σ(hθ(x^(i)) - y^(i))^2
    Gradient: ∇J(θ) = (1/m) * Σ(hθ(x^(i)) - y^(i)) * x^(i)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    return history


def evaluate(model, data_loader, device, return_loss_only=False):
    """
    Evaluate the model on the given data loader.

    Returns metrics: loss, MSE, MAE, R2
    """
    model.eval()
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    avg_loss = total_loss / len(data_loader)

    metrics = {"loss": avg_loss, "mse": mse, "mae": mae, "r2": r2}

    if return_loss_only:
        return avg_loss

    return metrics


def predict(model, X, device):
    """Generate predictions for input X."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        return outputs.cpu().numpy()


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
    loss_plot_path = os.path.join(save_dir, "linreg_bigquery_natality_loss.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Artifacts saved to {save_dir}")

    return save_dir


def main():
    """Main function to run the linear regression task."""
    print("=" * 60)
    print("Starting BigQuery Natality Data Linear Regression Task...")
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

    # Build model
    print("\nBuilding model...")
    model = build_model(input_dim=len(feature_cols), output_dim=1, device=device)

    # Train model
    print("\nTraining model...")
    history = train(
        model, train_loader, val_loader, device, epochs=100, lr=0.01, verbose=True
    )

    # Evaluate on train set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print("Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print("Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, history, save_dir="output")

    # Print final results
    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    print(f"MSE (Train): {train_metrics['mse']:.4f}")
    print(f"MSE (Val):   {val_metrics['mse']:.4f}")
    print(f"MAE (Train): {train_metrics['mae']:.4f}")
    print(f"MAE (Val):   {val_metrics['mae']:.4f}")
    print(f"R² (Train):  {train_metrics['r2']:.4f}")
    print(f"R² (Val):    {val_metrics['r2']:.4f}")
    print("=" * 60)

    # Quality checks
    print("\nQuality Checks:")
    quality_passed = True

    # Real-world data is noisier, so use realistic thresholds.
    check1 = val_metrics["r2"] > 0.20
    print(f"{'✓' if check1 else '✗'} Validation R2 > 0.20: {val_metrics['r2']:.4f}")
    quality_passed = quality_passed and check1

    check2 = val_metrics["mae"] < 1.5
    print(f"{'✓' if check2 else '✗'} Validation MAE < 1.5: {val_metrics['mae']:.4f}")
    quality_passed = quality_passed and check2

    check3 = history["train_loss"][-1] < history["train_loss"][0]
    print(
        f"{'✓' if check3 else '✗'} Training loss decreased: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}"
    )
    quality_passed = quality_passed and check3

    r2_gap = abs(train_metrics["r2"] - val_metrics["r2"])
    check4 = r2_gap < 0.25
    print(f"{'✓' if check4 else '✗'} R2 gap < 0.25: {r2_gap:.4f}")
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

# text_classification_pipelines.py

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from gensim.models import Word2Vec


# load dataset
df = pd.read_csv("train_with_task_type.csv")
# preview first few rows
print(df.head())
# see what columns the dataset has
print(df.columns)

# Use prompt as input text and task_type as label, so drop those rows missing those values
df = df.dropna(subset=["prompt", "task_type"])

X = df["prompt"].astype(str)
y = df["task_type"].astype(str)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

results = []


# function to evaluate the model
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    # train the model and time how long it takes.
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # have the model predict results on test dataset and time how long it takes
    start_infer = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_infer

    # get metrics, including accuracy, precision, recall, and F1 Score
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # save the results to to a list
    results.append(
        {
            "Pipeline": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Training Time (s)": train_time,
            "Inference Time (s)": inference_time,
        }
    )

    # display evaluation results, including all metrics, timings, and classification results
    print("\n", "-" * 100)
    print(name, "\n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
        )
    )

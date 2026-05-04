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

# Use prompt as input text and task_type as label, so drop those columns and save them separately
df = df.dropna(subset=["prompt", "task_type"])

X = df["prompt"].astype(str)
y = df["task_type"].astype(str)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

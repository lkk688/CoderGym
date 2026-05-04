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


### Pipeline 1: TF-IDF + Naive Bayes
pipeline_nb = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000, stop_words="english", ngram_range=(1, 2)
            ),
        ),
        ("classifier", MultinomialNB()),
    ]
)  # create pipeline using TF-IDF Vectorization and Naive Bayes Classifier and evaluate the model
evaluate_model("TF-IDF + Naive Bayes", pipeline_nb, X_train, X_test, y_train, y_test)


### Pipeline 2: TF-IDF + SVM
pipeline_svm = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000, stop_words="english", ngram_range=(1, 2)
            ),
        ),
        ("classifier", LinearSVC()),
    ]
)  # create pipeline using TF-IDF Vectorization and Linear SVM Classifier and evaluate the model
evaluate_model("TF-IDF + Linear SVM", pipeline_svm, X_train, X_test, y_train, y_test)


### Pipeline 3: TF-IDF + Random Forest
pipeline_rf = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000, stop_words="english", ngram_range=(1, 2)
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        ),
    ]
)  # create pipeline using TF-IDF Vectorization and a Random Forest Classifier and evaluate the model
evaluate_model("TF-IDF + Random Forest", pipeline_rf, X_train, X_test, y_train, y_test)


### Pipeline 4: Word Embeddings + Neural Network
def tokenize(text):
    # helper function to turn text into tokens (words)
    return str(text).lower().split()


# tokenize training and testing datasets
tokenized_train = [tokenize(text) for text in X_train]
tokenized_test = [tokenize(text) for text in X_test]

# Create word embeddings and get vectors for all the tokens
start_w2v = time.time()
w2v_model = Word2Vec(
    sentences=tokenized_train, vector_size=100, window=5, min_count=1, workers=4, sg=1
)
w2v_train_time = time.time() - start_w2v


# helper function to average all the vectors into one to represent the entire text
def document_vector(tokens, model, vector_size=100):
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if len(vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(vectors, axis=0)


# create training and testings datasets of the averaged vector saved as a matrix
X_train_w2v = np.array(
    [document_vector(tokens, w2v_model, 100) for tokens in tokenized_train]
)
X_test_w2v = np.array(
    [document_vector(tokens, w2v_model, 100) for tokens in tokenized_test]
)

# create MLP classifier model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)

# time how long it takes to train MLP classifier
start_train = time.time()
mlp.fit(X_train_w2v, y_train)
train_time = time.time() - start_train + w2v_train_time

# time how long it takes for the MLP classifier to predict the test dataset
start_infer = time.time()
y_pred = mlp.predict(X_test_w2v)
inference_time = time.time() - start_infer

# get evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)

# save evaluation metrics in results list
results.append(
    {
        "Pipeline": "Word2Vec + Neural Network",
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
print("Word2Vec + Neural Network", "\n")
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


# display the results table sorted by F1 Score
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score", ascending=False)
print("\nFinal Comparison:")
print(results_df)

# save results to a file
results_df.to_csv("results/model_comparison_results.csv", index=False)


### Create Visualizations

# compare accuracy metric of each pipeline
plt.figure(figsize=(10, 6))
plt.bar(results_df["Pipeline"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Pipeline")
plt.ylabel("Accuracy")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("graphics/accuracy_comparison.png")
plt.show()

# compare F1 Score of each pipeline
plt.figure(figsize=(10, 6))
plt.bar(results_df["Pipeline"], results_df["F1-Score"])
plt.title("Model F1-Score Comparison")
plt.xlabel("Pipeline")
plt.ylabel("Weighted F1-Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("graphics/f1_comparison.png")
plt.show()

# compare the time it took to train the model of each pipeline
plt.figure(figsize=(10, 6))
plt.bar(results_df["Pipeline"], results_df["Training Time (s)"])
plt.title("Training Time Comparison")
plt.xlabel("Pipeline")
plt.ylabel("Training Time in Seconds")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("graphics/training_time_comparison.png")
plt.show()

# compare training time and f1-score for each pipeline
plt.figure(figsize=(10, 6))
plt.scatter(results_df["Training Time (s)"], results_df["F1-Score"], s=120)

for i, row in results_df.iterrows():
    plt.annotate(
        row["Pipeline"],
        (row["Training Time (s)"], row["F1-Score"]),
        textcoords="offset points",
        xytext=(5, 5),
    )

plt.title("Performance vs Computational Cost")
plt.xlabel("Training Time in Seconds")
plt.ylabel("Weighted F1-Score")
plt.tight_layout()
plt.savefig("graphics/performance_vs_speed.png")
plt.show()

print("\nDone\n")

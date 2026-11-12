
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load training and validation data
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

# Keep malicious training prompts as reference examples
malicious_prompts = train_df[
    train_df["label"] == 1
]["text"].tolist()

X_val = val_df["text"].tolist()
y_val = val_df["label"]

# Load sentence embedding model
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("Generating malicious reference embeddings...")
malicious_embeddings = embedding_model.encode(
    malicious_prompts,
    show_progress_bar=True
)

print("Generating validation embeddings...")
validation_embeddings = embedding_model.encode(
    X_val,
    show_progress_bar=True
)

# Calculate similarity against malicious references
print("Calculating semantic similarities...")

similarity_matrix = cosine_similarity(
    validation_embeddings,
    malicious_embeddings
)

# Highest similarity to any known malicious prompt
similarity_scores = similarity_matrix.max(axis=1)

# Initial experimental threshold
threshold = 0.50

predictions = (
    similarity_scores >= threshold
).astype(int)

# Evaluate
print("\n=== SEMANTIC MODEL RESULTS ===")

print(f"Threshold: {threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_val, predictions):.4f}")
print(f"Precision: {precision_score(y_val, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_val, predictions, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, similarity_scores):.4f}")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_val, predictions))

# Compare different semantic similarity thresholds
print("\n=== SEMANTIC THRESHOLD ANALYSIS ===")

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:

    threshold_predictions = (
        similarity_scores >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, threshold_predictions, zero_division=0
    )

    recall = recall_score(
        y_val, threshold_predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, threshold_predictions, zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

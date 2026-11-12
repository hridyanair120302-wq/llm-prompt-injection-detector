
import pandas as pd
from pathlib import Path
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity

# Load training and validation data
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

X_train = train_df["text"]
y_train = train_df["label"]

X_val = val_df["text"]
y_val = val_df["label"]

# 1. Train word + character TF-IDF model
print("Training word + character model...")

classifier = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000
        )),
        ("character", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000
        ))
    ])),
    ("classifier", LogisticRegression(max_iter=1000))
])

classifier.fit(X_train, y_train)

# Get malicious probabilities
classifier_scores = classifier.predict_proba(X_val)[:, 1]

# 2. Calculate semantic similarity
print("Loading sentence embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

malicious_prompts = train_df[
    train_df["label"] == 1
]["text"].tolist()

print("Generating embeddings...")

malicious_embeddings = embedding_model.encode(
    malicious_prompts,
    show_progress_bar=True
)

validation_embeddings = embedding_model.encode(
    X_val.tolist(),
    show_progress_bar=True
)

similarity_matrix = cosine_similarity(
    validation_embeddings,
    malicious_embeddings
)

semantic_scores = similarity_matrix.max(axis=1)

# 3. Combine both signals
print("\n=== HYBRID MODEL EXPERIMENTS ===")

best_result = None

for semantic_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:

    classifier_weight = 1 - semantic_weight

    hybrid_scores = (
        classifier_weight * classifier_scores
        + semantic_weight * semantic_scores
    )

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:

        predictions = (
            hybrid_scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        if best_result is None or f1 > best_result["f1"]:
            best_result = {
                "semantic_weight": semantic_weight,
                "classifier_weight": classifier_weight,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "scores": hybrid_scores,
                "predictions": predictions
            }

# 4. Display best validation configuration
print("\n=== BEST HYBRID CONFIGURATION ===")

print(f"Classifier weight: {best_result['classifier_weight']:.2f}")
print(f"Semantic weight:   {best_result['semantic_weight']:.2f}")
print(f"Threshold:         {best_result['threshold']:.2f}")

print("\n=== VALIDATION RESULTS ===")

print(f"Accuracy:  {accuracy_score(y_val, best_result['predictions']):.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall:    {best_result['recall']:.4f}")
print(f"F1-score:  {best_result['f1']:.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, best_result['scores']):.4f}")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_val, best_result["predictions"]))

# 5. Final evaluation on the untouched test set
print("\n=== FINAL TEST SET EVALUATION ===")

test_df = pd.read_csv("data/processed/test.csv")

X_test = test_df["text"]
y_test = test_df["label"]

# Use the already-trained classifier
test_classifier_scores = classifier.predict_proba(X_test)[:, 1]

# Generate test embeddings
print("Generating test embeddings...")

test_embeddings = embedding_model.encode(
    X_test.tolist(),
    show_progress_bar=True
)

# Compare test prompts against malicious training examples
test_similarity_matrix = cosine_similarity(
    test_embeddings,
    malicious_embeddings
)

test_semantic_scores = test_similarity_matrix.max(axis=1)

# Apply the configuration selected on validation
final_scores = (
    0.50 * test_classifier_scores
    + 0.50 * test_semantic_scores
)

final_predictions = (
    final_scores >= 0.55
).astype(int)

print("\n=== FINAL TEST RESULTS ===")

print(f"Accuracy:  {accuracy_score(y_test, final_predictions):.4f}")
print(f"Precision: {precision_score(y_test, final_predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, final_predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, final_predictions, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, final_scores):.4f}")

print("\n=== TEST CONFUSION MATRIX ===")
print(confusion_matrix(y_test, final_predictions))

# 6. Save the trained hybrid detector
print("\n=== SAVING HYBRID MODEL ===")

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_artifact = {
    "classifier": classifier,
    "malicious_embeddings": malicious_embeddings,
    "malicious_prompts": malicious_prompts,
    "embedding_model_name": "all-MiniLM-L6-v2",
    "classifier_weight": 0.50,
    "semantic_weight": 0.50,
    "threshold": 0.55
}

model_path = model_dir / "hybrid_detector.joblib"

joblib.dump(model_artifact, model_path, compress=3)

print(f"Model saved successfully: {model_path}")

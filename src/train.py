
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load the prepared training and validation datasets
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

X_train = train_df["text"]
y_train = train_df["label"]

X_val = val_df["text"]
y_val = val_df["label"]

# Build the machine-learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train using only the training split
print("Training baseline model...")
model.fit(X_train, y_train)

# Evaluate on the validation split
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

print("\n=== VALIDATION RESULTS ===")

print(f"Accuracy:  {accuracy_score(y_val, predictions):.4f}")
print(f"Precision: {precision_score(y_val, predictions):.4f}")
print(f"Recall:    {recall_score(y_val, predictions):.4f}")
print(f"F1-score:  {f1_score(y_val, predictions):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, probabilities):.4f}")

print("\n=== CONFUSION MATRIX ===")
print("Rows = actual labels; columns = predicted labels")
print(confusion_matrix(y_val, predictions))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_val,
    predictions,
    target_names=["Benign", "Malicious"],
    zero_division=0
))

# Analyze incorrect predictions
val_results = val_df.copy()

val_results["predicted_label"] = predictions
val_results["malicious_score"] = probabilities

# False positives: benign prompts incorrectly flagged
false_positives = val_results[
    (val_results["label"] == 0) &
    (val_results["predicted_label"] == 1)
]

# False negatives: malicious prompts incorrectly missed
false_negatives = val_results[
    (val_results["label"] == 1) &
    (val_results["predicted_label"] == 0)
]

pd.set_option("display.max_colwidth", 180)

print("\n=== FALSE POSITIVES (TOP 10) ===")
print(
    false_positives
    .sort_values("malicious_score", ascending=False)
    [["text", "category", "malicious_score"]]
    .head(10)
    .to_string(index=False)
)

print("\n=== FALSE NEGATIVES ===")
print(
    false_negatives
    .sort_values("malicious_score")
    [["text", "category", "malicious_score"]]
    .to_string(index=False)
)

# Compare different classification thresholds
print("\n=== THRESHOLD ANALYSIS ===")

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    threshold_predictions = (
        probabilities >= threshold
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

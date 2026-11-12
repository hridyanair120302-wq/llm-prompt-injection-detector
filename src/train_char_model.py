
import pandas as pd

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

# Load training and validation data
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

X_train = train_df["text"]
y_train = train_df["label"]

X_val = val_df["text"]
y_val = val_df["label"]

# Combine word-level and character-level features
features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000
        )
    ),
    (
        "character",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000
        )
    )
])

# Build the model
model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
print("Training word + character TF-IDF model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

print("\n=== WORD + CHARACTER MODEL ===")
print(f"Accuracy:  {accuracy_score(y_val, predictions):.4f}")
print(f"Precision: {precision_score(y_val, predictions):.4f}")
print(f"Recall:    {recall_score(y_val, predictions):.4f}")
print(f"F1-score:  {f1_score(y_val, predictions):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, probabilities):.4f}")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_val, predictions))

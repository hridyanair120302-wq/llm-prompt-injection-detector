import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

# Load dataset
df = pd.read_csv("data/raw/prompts.csv")

X = df["text"]
y = df["label"]

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model
scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"]
)

print("Cross-validation results:")
print(f"Accuracy:  {scores['test_accuracy'].mean():.3f}")
print(f"Precision: {scores['test_precision'].mean():.3f}")
print(f"Recall:    {scores['test_recall'].mean():.3f}")
print(f"F1-score:  {scores['test_f1'].mean():.3f}")

from pathlib import Path

import pandas as pd
from datasets import load_dataset

print("Loading public dataset...")

dataset = load_dataset(
    "NeurAlchemy/prompt-injection-dataset",
    "core"
)

# Preserve all useful dataset metadata.
columns = [
    "text",
    "label",
    "category",
    "source",
    "severity",
    "group_id",
    "augmented",
    "tags"
]

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

for split_name in ["train", "validation", "test"]:
    df = pd.DataFrame(dataset[split_name])

    # Keep only the columns needed for this project.
    df = df[columns]

    # Validate the data before saving.
    if df["text"].isna().any() or df["text"].str.strip().eq("").any():
        raise ValueError(f"Missing or empty prompts in {split_name}")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {split_name}")

    # Save each official split separately.
    output_path = output_dir / f"{split_name}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Examples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

print("\nDataset preparation complete!")

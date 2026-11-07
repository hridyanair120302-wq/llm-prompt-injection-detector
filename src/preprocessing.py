import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/prompts.csv")

# Display the first few rows
print("Dataset preview:")
print(df.head())

# Check dataset size
print("\nDataset shape:")
print(df.shape)

# Check class distribution
print("\nLabel distribution:")
print(df["label"].value_counts())
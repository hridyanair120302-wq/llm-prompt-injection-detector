from datasets import load_dataset

# Load the public dataset
print("Loading dataset...")

dataset = load_dataset(
    "NeurAlchemy/prompt-injection-dataset",
    "core"
)

# Display available splits
print("\nAvailable splits:")
print(dataset)

# Inspect training data
train_data = dataset["train"]

print("\nTraining dataset size:")
print(len(train_data))

# Inspect one example
print("\nFirst training example:")
print(train_data[0])

# Display column names
print("\nDataset columns:")
print(train_data.column_names)

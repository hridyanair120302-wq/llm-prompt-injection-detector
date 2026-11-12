
from datasets import load_dataset
from collections import Counter
import re

print("Loading dataset...")

dataset = load_dataset(
    "NeurAlchemy/prompt-injection-dataset",
    "core"
)

# Display split sizes
print("\n=== SPLIT SIZES ===")

for split_name, split_data in dataset.items():
    print(f"{split_name}: {len(split_data)} examples")


# Inspect label and category distributions
print("\n=== LABEL DISTRIBUTION ===")

for split_name, split_data in dataset.items():
    labels = Counter(split_data["label"])
    print(f"\n{split_name}:")
    print(labels)


print("\n=== CATEGORY DISTRIBUTION ===")

for split_name, split_data in dataset.items():
    categories = Counter(split_data["category"])
    print(f"\n{split_name}:")
    for category, count in categories.most_common():
        print(f"{category}: {count}")


# Check missing and empty prompts
print("\n=== MISSING TEXT CHECK ===")

for split_name, split_data in dataset.items():
    missing = sum(
        1 for text in split_data["text"]
        if text is None or not str(text).strip()
    )
    print(f"{split_name}: {missing} missing or empty prompts")


# Normalize text for duplicate detection
def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


print("\n=== DUPLICATE CHECK ===")

split_texts = {}

for split_name, split_data in dataset.items():
    texts = [
        normalize_text(text)
        for text in split_data["text"]
        if text is not None
    ]

    split_texts[split_name] = set(texts)

    duplicates = len(texts) - len(set(texts))

    print(f"{split_name}: {duplicates} duplicate prompts within split")


# Check exact normalized prompt overlap between splits
print("\n=== CROSS-SPLIT PROMPT OVERLAP ===")

split_names = list(split_texts.keys())

for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        first = split_names[i]
        second = split_names[j]

        overlap = split_texts[first] & split_texts[second]

        print(f"{first} vs {second}: {len(overlap)} shared prompts")


# Check group ID overlap between splits
print("\n=== GROUP ID OVERLAP ===")

split_groups = {}

for split_name, split_data in dataset.items():
    split_groups[split_name] = {
        group_id
        for group_id in split_data["group_id"]
        if group_id is not None and str(group_id).strip()
    }

for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        first = split_names[i]
        second = split_names[j]

        overlap = split_groups[first] & split_groups[second]

        print(f"{first} vs {second}: {len(overlap)} shared group IDs")


# Inspect augmentation distribution
print("\n=== AUGMENTATION DISTRIBUTION ===")

for split_name, split_data in dataset.items():
    augmented = Counter(split_data["augmented"])
    print(f"{split_name}: {dict(augmented)}")


print("\nDataset inspection complete.")

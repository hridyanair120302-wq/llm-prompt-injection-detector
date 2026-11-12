# LLM Prompt Injection & Jailbreak Detection

A hybrid machine learning pipeline that detects potentially malicious prompts using **Word + Character TF-IDF classification** and **semantic similarity analysis**.

## Overview

Large Language Models (LLMs) can be targeted by prompt injection and jailbreak attempts that try to override instructions, manipulate model behavior, or expose sensitive information.

This project explores a hybrid detection approach that combines traditional machine learning with sentence embeddings to identify potentially malicious prompts.

## Key Features

- Binary classification of benign and potentially malicious prompts.
- Word-level and character-level TF-IDF feature extraction.
- Logistic Regression classifier.
- Semantic similarity using Sentence Transformers (`all-MiniLM-L6-v2`).
- Weighted hybrid scoring mechanism.
- Configurable detection threshold.
- Dataset integrity checks and error analysis.
- Saved model artifact for inference.
- FastAPI integration (in development).

## Architecture

```text
Input Prompt
     |
     v
Word + Character TF-IDF
     |
     v
Logistic Regression --------+
                            |
                            v
                     Hybrid Scoring
                            ^
                            |
Sentence Embeddings         |
     |                      |
     v                      |
Cosine Similarity ----------+
                            |
                            v
                    Detection Threshold
                            |
                            v
              BENIGN / POTENTIALLY MALICIOUS
```

## Dataset

The project uses the `core` configuration of the Hugging Face dataset:

[NeurAlchemy/prompt-injection-dataset](https://huggingface.co/datasets/NeurAlchemy/prompt-injection-dataset)

| Split | Samples |
|---|---:|
| Training | 4,391 |
| Validation | 941 |
| Testing | 942 |
| **Total** | **6,274** |

The dataset includes benign prompts and multiple categories of potentially malicious prompts.

Dataset integrity checks include:
- Missing and empty prompt checks.
- Duplicate prompt checks.
- Cross-split exact prompt overlap checks.
- Group ID overlap checks.
- Label distribution analysis.

The original training, validation, and test splits are preserved.

## Model Performance

The final hybrid configuration was selected using validation data and evaluated on the held-out test set.

**Final test results:**

| Metric | Score |
|---|---:|
| Accuracy | 95.86% |
| Precision | 95.40% |
| Recall | 97.64% |
| F1-score | 96.51% |
| ROC-AUC | 99.33% |

### Test Confusion Matrix

| | Predicted Benign | Predicted Malicious |
|---|---:|---:|
| Actual Benign | 364 | 26 |
| Actual Malicious | 13 | 539 |

### Hybrid Configuration

| Component | Weight |
|---|---:|
| Word + Character Classifier | 0.50 |
| Semantic Similarity | 0.50 |
| Decision Threshold | 0.55 |

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Sentence Transformers
- Hugging Face Datasets
- FastAPI
- Uvicorn
- Joblib
- Pytest

## Project Structure

```text
llm-prompt-injection-detector/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── project-log.md
│
├── models/
│   └── hybrid_detector.joblib
│
├── src/
│   ├── detector.py
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── train_char_model.py
│   ├── train_semantic_model.py
│   └── train_hybrid_model.py
│
├── tests/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/hridyanair120302-wq/llm-prompt-injection-detector.git

cd llm-prompt-injection-detector
```

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv

source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Prepare the Dataset

```bash
python src/inspect_dataset.py

python src/prepare_dataset.py
```

## Train the Models

```bash
python src/train.py

python src/train_char_model.py

python src/train_semantic_model.py

python src/train_hybrid_model.py
```

The hybrid training script evaluates the selected configuration and saves the model artifact under `models/`.

## Run the Detector

```bash
python src/detector.py
```

The detector returns:
- Classifier score.
- Semantic similarity score.
- Combined hybrid score.
- Final prediction.

## Limitations

- Performance depends on the dataset's coverage and labeling quality.
- Semantic similarity does not establish malicious intent.
- The binary classification setup combines multiple types of potentially unsafe prompts.
- The model may produce false positives and false negatives.
- High benchmark performance does not guarantee detection of previously unseen real-world attacks.

## Future Improvements

- Build and test a FastAPI inference service.
- Add automated unit and API tests.
- Evaluate against additional prompt injection benchmarks.
- Improve detection of obfuscated and indirect prompt injection.
- Explore transformer-based classification models.
- Add Docker support and deployment.

## Author

**Hridya Nair**

M.S. Cybersecurity, New York University

[GitHub](https://github.com/hridyanair120302-wq)

[LinkedIn](https://www.linkedin.com/in/hridya-nair-598ab5336/)
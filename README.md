# Precision–Recall Introduction

This small Python notebook/script introduces **precision**, **recall**, and **F1-score**
for **binary classification** tasks. It includes:

- Manual metric calculations
- scikit-learn metric functions
- Precision–Recall curve visualization

## Key Concepts

- **Precision** = TP / (TP + FP): correctness among predicted positives  
- **Recall** = TP / (TP + FN): coverage among actual positives  
- **F1-score** = harmonic mean of precision & recall

## Run Examples

```bash
pip install matplotlib scikit-learn numpy
python precision_recall_intro.py
```

The script will:

1. Print manual precision/recall calculations
2. Compute metrics with scikit-learn
3. Plot a confusion matrix and a precision–recall curve

## Learning Goals

- Understand the intuition behind precision and recall
- Observe the tradeoff between the two using a PR curve
- Prepare for multi-class metrics (macro vs micro F1)

---

© 2025 — 5350 Applied Deep Learning Class Example (University of Colorado Boulder)

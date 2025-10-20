
# precision_recall_intro.py
"""
Demonstration of Precision, Recall, and F1-score for binary classification.
Includes manual calculations, scikit-learn metrics, and a precision–recall curve.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, ConfusionMatrixDisplay, precision_recall_curve
)

# Example 1: Manual computation
TP, FP, FN, TN = 80, 10, 20, 890
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (TP + TN) / (TP + FP + FN + TN)
print("Example 1 — Manual metrics")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}\n")

# Example 2: scikit-learn metrics
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]

prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("Example 2 — scikit-learn metrics")
print("Confusion Matrix:\n", cm)
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1-score:  {f1:.2f}\n")

ConfusionMatrixDisplay(cm, display_labels=["Not Spam (0)", "Spam (1)"]).plot(cmap="Blues")
plt.title("Binary Classification Confusion Matrix")
plt.show()

# Example 3: Precision–Recall curve
y_scores = np.array([0.9, 0.8, 0.7, 0.55, 0.52, 0.45, 0.4, 0.35, 0.2, 0.1])
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

plt.plot(recalls, precisions, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Tradeoff)")
plt.grid(True)
plt.show()

print("✅ Precision–Recall examples completed successfully.")

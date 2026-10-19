"""
Project 040: Model evaluation dashboard for classification metrics

Overview:
  This project builds a production-minded model evaluation dashboard for classification metrics workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_probs = np.array([0.9, 0.1, 0.8, 0.65, 0.3, 0.75, 0.2, 0.4, 0.85, 0.15])
y_pred = (y_probs >= 0.5).astype(int)

print("=== Comprehensive Model Evaluation Dashboard ===")
print(f"Accuracy:         {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision:        {precision_score(y_true, y_pred):.4f}")
print(f"Recall:           {recall_score(y_true, y_pred):.4f}")
print(f"F1-Score:         {f1_score(y_true, y_pred):.4f}")
print(f"ROC-AUC:          {roc_auc_score(y_true, y_probs):.4f}")
print(f"Brier Score Loss: {brier_score_loss(y_true, y_probs):.4f}")

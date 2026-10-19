"""
Project 022: Learning curve visualizations

Overview:
  This project builds a production-minded learning curve visualizations workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

digits = load_digits()
train_sizes, train_scores, test_scores = learning_curve(
    SVC(gamma=0.001), digits.data, digits.target, cv=3, n_jobs=-1,
    train_sizes=np.linspace(0.2, 1.0, 5)
)

print("=== Learning Curve Visualizations ===")
print("Training Set Sample Sizes:", train_sizes.tolist())
print("Mean Train Accuracy (%):", np.round(train_scores.mean(axis=1) * 100, 2).tolist())
print("Mean Validation Accuracy (%):", np.round(test_scores.mean(axis=1) * 100, 2).tolist())

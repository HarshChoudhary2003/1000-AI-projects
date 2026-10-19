"""
Project 015: Cross-validation implementation

Overview:
  This project builds a production-minded cross-validation implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

X, y = make_classification(n_samples=500, n_features=8, weights=[0.85, 0.15], random_state=42)
model = LogisticRegression(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

kf_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
skf_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("=== Cross-Validation Implementation ===")
print("5-Fold CV Scores:", np.round(kf_scores, 4).tolist(), f"| Mean: {kf_scores.mean():.4f}")
print("5-Fold Stratified CV Scores:", np.round(skf_scores, 4).tolist(), f"| Mean: {skf_scores.mean():.4f}")

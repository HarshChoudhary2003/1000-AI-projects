"""
Project 014: Feature selection techniques

Overview:
  This project builds a production-minded feature selection techniques workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=400, n_features=12, n_informative=4, n_redundant=2, random_state=42)

# 1. Univariate Selection (ANOVA F-test)
k_best = SelectKBest(score_func=f_classif, k=4)
k_best.fit(X, y)
selected_kbest = np.where(k_best.get_support())[0]

# 2. Recursive Feature Elimination (RFE)
rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=4)
rfe.fit(X, y)
selected_rfe = np.where(rfe.support_)[0]

print("=== Feature Selection Techniques ===")
print(f"Original Features: 12")
print(f"SelectKBest (F-Score) selected indices: {selected_kbest.tolist()}")
print(f"RFE (Recursive Elimination) selected indices: {selected_rfe.tolist()}")

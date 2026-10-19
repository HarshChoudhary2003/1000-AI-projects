"""
Project 009: Random forest classifier

Overview:
  This project builds a production-minded random forest classifier workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=800, n_features=12, n_informative=7, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=6, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("=== Random Forest Classifier ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Out-of-Bag (OOB) Score: {rf.oob_score_ * 100:.2f}%")
print("\nTop 5 Feature Importances:")
indices = np.argsort(rf.feature_importances_)[::-1][:5]
for idx in indices:
    print(f"  Feature {idx}: {rf.feature_importances_[idx]:.4f}")

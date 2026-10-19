"""
Project 756: Stakeholder-friendly AI explanation generator
Category: Explainable AI and Interpretability Projects
Difficulty: Advanced
Domain: explainability and trustworthy AI

Description:
  This project builds a production-minded stakeholder-friendly AI explanation generator workflow inside the broader area of explainability and trustworthy AI. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

PROJECT_NAME = "Stakeholder-friendly AI explanation generator"

X, y = make_classification(n_samples=700, n_features=8, n_informative=5, random_state=42)
columns = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
report = pd.DataFrame({"feature": columns, "importance": importance.importances_mean})
report = report.sort_values("importance", ascending=False)

print(PROJECT_NAME)
print("Model score:", round(model.score(X_test, y_test), 3))
print(report.head())

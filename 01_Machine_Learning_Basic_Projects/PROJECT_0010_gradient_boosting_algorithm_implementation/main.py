"""
Project 010: Gradient boosting algorithm implementation

Overview:
  This project builds a production-minded gradient boosting algorithm implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

gbc = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
probs = gbc.predict_proba(X_test)[:, 1]

print("=== Gradient Boosting Machine (GBM) Implementation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

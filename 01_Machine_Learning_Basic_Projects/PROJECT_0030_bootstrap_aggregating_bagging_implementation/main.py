"""
Project 030: Bootstrap aggregating (bagging) implementation

Overview:
  This project builds a production-minded bootstrap aggregating (bagging) implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=42)

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    oob_score=True,
    random_state=42
)
bagging.fit(X_train, y_train)

print("=== Bootstrap Aggregating (Bagging) Classifier ===")
print(f"Out-of-Bag (OOB) Score: {bagging.oob_score_ * 100:.2f}%")
print(f"Test Accuracy: {bagging.score(X_test, y_test) * 100:.2f}%")

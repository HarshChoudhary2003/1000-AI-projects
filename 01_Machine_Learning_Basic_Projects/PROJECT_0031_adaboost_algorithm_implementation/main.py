"""
Project 031: AdaBoost algorithm implementation

Overview:
  This project builds a production-minded adaBoost algorithm implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

adaboost = AdaBoostClassifier(n_estimators=80, learning_rate=0.8, random_state=42)
adaboost.fit(X_train, y_train)

print("=== AdaBoost Algorithm Implementation ===")
print(f"Accuracy: {accuracy_score(y_test, adaboost.predict(X_test)) * 100:.2f}%")
print(f"Number of Estimator Trees: {len(adaboost.estimators_)}")

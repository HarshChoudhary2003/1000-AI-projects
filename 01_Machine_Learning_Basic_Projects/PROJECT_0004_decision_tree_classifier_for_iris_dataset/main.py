"""
Project 004: Decision tree classifier for Iris dataset

Overview:
  This project builds a production-minded decision tree classifier for Iris dataset workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)

dt = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
print("=== Decision Tree Classifier: Iris Dataset ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Tree Structure Rules:")
print(export_text(dt, feature_names=list(iris.feature_names)))
print("\nFeature Importances:")
for name, imp in zip(iris.feature_names, dt.feature_importances_):
    print(f"  {name}: {imp:.4f}")

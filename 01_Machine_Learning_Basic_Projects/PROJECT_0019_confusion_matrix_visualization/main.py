"""
Project 019: Confusion matrix visualization

Overview:
  This project builds a production-minded confusion matrix visualization workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=300, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

cm = confusion_matrix(y_test, preds)
print("=== Confusion Matrix Visualization ===")
print("Raw Confusion Matrix:")
print(cm)
print("\nPer-Class Accuracy:")
for i, name in enumerate(iris.target_names):
    class_acc = cm[i, i] / cm[i].sum()
    print(f"  {name}: {class_acc * 100:.1f}%")

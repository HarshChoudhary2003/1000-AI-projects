"""
Project 007: KNN algorithm implementation

Overview:
  This project builds a production-minded kNN algorithm implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=600, n_features=8, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=== K-Nearest Neighbors (KNN) Implementation ===")
print("Evaluating K from 1 to 15:")

best_k, best_acc = 1, 0
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"  k = {k:2d} -> Validation Accuracy: {acc * 100:.2f}%")
    if acc > best_acc:
        best_k, best_acc = k, acc

print(f"\nOptimal K found: {best_k} with {best_acc * 100:.2f}% accuracy")

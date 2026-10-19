"""
Project 008: Support vector machine for classification

Overview:
  This project builds a production-minded support vector machine for classification workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=700, n_features=6, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)

y_pred = svm_rbf.predict(X_test)
print("=== Support Vector Machine (SVM) Classification ===")
print(f"Support Vectors count: {len(svm_rbf.support_vectors_)}")
print(classification_report(y_test, y_pred))

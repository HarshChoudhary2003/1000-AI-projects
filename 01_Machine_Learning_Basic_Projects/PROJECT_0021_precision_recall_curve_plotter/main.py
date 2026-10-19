"""
Project 021: Precision-recall curve plotter

Overview:
  This project builds a production-minded precision-recall curve plotter workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

X, y = make_classification(n_samples=1200, n_features=10, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_scores = clf.predict_proba(X_test)[:, 1]

prec, rec, _ = precision_recall_curve(y_test, y_scores)
ap = average_precision_score(y_test, y_scores)

print("=== Precision-Recall Curve Plotter ===")
print(f"Average Precision (AP): {ap:.4f}")
print(f"Total Evaluation Cutoffs: {len(prec)}")
print(f"Baseline Random AP: {y.mean():.4f}")

"""
Project 020: ROC curve generator

Overview:
  This project builds a production-minded rOC curve generator workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_scores = clf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

print("=== ROC Curve Generator ===")
print(f"Computed Area Under Curve (AUC): {roc_auc:.4f}")
print(f"Sample Threshold Check:")
for idx in [5, len(thresholds)//2, -5]:
    if idx < len(thresholds):
        print(f"  Threshold: {thresholds[idx]:.3f} | FPR: {fpr[idx]:.3f} | TPR: {tpr[idx]:.3f}")

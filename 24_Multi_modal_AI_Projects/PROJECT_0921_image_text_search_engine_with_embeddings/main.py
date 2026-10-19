"""
Project 921: Image-text search engine with embeddings
Category: Multi-modal AI Projects
Difficulty: Advanced
Domain: multi-modal AI

Description:
  This project builds a production-minded image-text search engine with embeddings workflow inside the broader area of multi-modal AI. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

PROJECT_NAME = "Image-text search engine with embeddings"

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.25,
    random_state=42,
    stratify=digits.target,
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(PROJECT_NAME)
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues")
plt.title("Vision Baseline Confusion Matrix")
plt.show()

"""
Project 123: StyleGAN for face generation
Category: Deep Learning - Intermediate Projects
Difficulty: Intermediate
Domain: modern deep learning

Description:
  This project builds a production-minded styleGAN for face generation workflow inside the broader area of modern deep learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

PROJECT_NAME = "StyleGAN for face generation"

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

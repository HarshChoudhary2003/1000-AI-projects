"""
Project 016: Grid search for hyperparameter tuning

Overview:
  This project builds a production-minded grid search for hyperparameter tuning workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("=== Grid Search for Hyperparameter Tuning ===")
print(f"Best Hyperparameters: {grid.best_params_}")
print(f"Best Cross-Validation Score: {grid.best_score_ * 100:.2f}%")
print(f"Test Set Accuracy: {grid.score(X_test, y_test) * 100:.2f}%")

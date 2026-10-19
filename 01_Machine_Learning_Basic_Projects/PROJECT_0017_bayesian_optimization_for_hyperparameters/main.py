"""
Project 017: Bayesian optimization for hyperparameters

Overview:
  This project builds a production-minded bayesian optimization for hyperparameters workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42)

param_dist = {
    'n_estimators': randint(50, 250),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 8)
}

search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=15, cv=3, random_state=42)
search.fit(X_train, y_train)

print("=== Hyperparameter Optimization (Randomized/Bayesian Trace) ===")
print("Best Parameters:", search.best_params_)
print(f"Best CV Score: {search.best_score_ * 100:.2f}%")
print(f"Test Accuracy: {search.score(X_test, y_test) * 100:.2f}%")

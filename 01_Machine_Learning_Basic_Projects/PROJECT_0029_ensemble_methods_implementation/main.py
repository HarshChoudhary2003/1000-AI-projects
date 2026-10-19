"""
Project 029: Ensemble methods implementation

Overview:
  This project builds a production-minded ensemble methods implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42)

ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=500)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)

print("=== Ensemble Methods (Soft Voting Classifier) ===")
print(f"Ensemble Test Accuracy: {ensemble.score(X_test, y_test) * 100:.2f}%")

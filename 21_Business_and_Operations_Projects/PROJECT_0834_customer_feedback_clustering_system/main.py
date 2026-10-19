"""
Project 834: Customer feedback clustering system
Category: Business and Operations Projects
Difficulty: Advanced
Domain: business operations AI

Description:
  This project builds a production-minded customer feedback clustering system workflow inside the broader area of business operations AI. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_NAME = "Customer feedback clustering system"
RANDOM_STATE = 42

X, y = make_classification(
    n_samples=800,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=RANDOM_STATE,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
data = pd.DataFrame(X, columns=feature_names)
data["target"] = y

X_train, X_test, y_train, y_test = train_test_split(
    data[feature_names],
    data["target"],
    test_size=0.25,
    stratify=data["target"],
    random_state=RANDOM_STATE,
)

model = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE)),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(PROJECT_NAME)
print(classification_report(y_test, predictions))

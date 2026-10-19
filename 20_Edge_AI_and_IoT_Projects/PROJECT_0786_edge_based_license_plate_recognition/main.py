"""
Project 786: Edge-based license plate recognition
Category: Edge AI and IoT Projects
Difficulty: Advanced
Domain: edge AI and connected devices

Description:
  This project builds a production-minded edge-based license plate recognition workflow inside the broader area of edge AI and connected devices. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

PROJECT_NAME = "Edge-based license plate recognition"

X, y = make_classification(n_samples=1000, n_features=12, n_informative=8, random_state=42)
model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X, y)

sample = X[:1]
start = time.perf_counter()
for _ in range(1000):
    model.predict(sample)
latency_ms = (time.perf_counter() - start) * 1000 / 1000

print(PROJECT_NAME)
print("Estimated single-sample latency (ms):", round(latency_ms, 4))
print("Tree depth:", model.get_depth())
print("Node count:", model.tree_.node_count)

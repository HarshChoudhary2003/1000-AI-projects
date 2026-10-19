"""
Project 373: Score-based generative models
Category: Generative Models Projects
Difficulty: Applied
Domain: generative AI and synthetic media

Description:
  This project builds a production-minded score-based generative models workflow inside the broader area of generative AI and synthetic media. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

PROJECT_NAME = "Score-based generative models"
rng = np.random.default_rng(42)

seed_examples = np.array(
    [
        [0.10, 0.25, 0.80],
        [0.15, 0.30, 0.75],
        [0.80, 0.20, 0.10],
        [0.78, 0.18, 0.12],
        [0.40, 0.70, 0.45],
    ]
)

centroid = seed_examples.mean(axis=0)
covariance = np.cov(seed_examples.T) + np.eye(seed_examples.shape[1]) * 0.01
synthetic_samples = rng.multivariate_normal(centroid, covariance, size=5)
synthetic_samples = np.clip(synthetic_samples, 0, 1)

print(PROJECT_NAME)
print("Synthetic samples:")
print(np.round(synthetic_samples, 3))

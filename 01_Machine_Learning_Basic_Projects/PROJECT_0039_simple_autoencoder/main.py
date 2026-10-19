"""
Project 039: Simple autoencoder

Overview:
  This project builds a production-minded simple autoencoder workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Bottleneck Autoencoder (8 -> 2 -> 8)
X = np.eye(8)

autoencoder = MLPRegressor(hidden_layer_sizes=(2,), activation='logistic', max_iter=2000, random_state=42)
autoencoder.fit(X, X)

reconstructed = autoencoder.predict(X)
mse = mean_squared_error(X, reconstructed)

print("=== Simple Bottleneck Autoencoder ===")
print(f"Input Dimension: {X.shape[1]} -> Latent Bottleneck: 2 -> Reconstructed: {reconstructed.shape[1]}")
print(f"Reconstruction MSE: {mse:.6f}")

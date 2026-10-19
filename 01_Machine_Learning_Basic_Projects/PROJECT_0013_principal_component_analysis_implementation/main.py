"""
Project 013: Principal component analysis implementation

Overview:
  This project builds a production-minded principal component analysis implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X, y = digits.data, digits.target

pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

print("=== Principal Component Analysis (PCA) ===")
print(f"Original Dimension: {X.shape[1]} features (8x8 pixel images)")
print(f"Reduced Dimension: {X_projected.shape[1]} principal components")
print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total variance retained in 2D: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

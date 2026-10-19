"""
Project 037: Dimensionality reduction techniques

Overview:
  This project builds a production-minded dimensionality reduction techniques workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np

X, _ = load_digits(return_X_y=True)
pca = PCA(n_components=2)
svd = TruncatedSVD(n_components=2)

X_pca = pca.fit_transform(X)
X_svd = svd.fit_transform(X)

print("=== Dimensionality Reduction Comparison ===")
print("PCA Explained Variance (2D):", round(np.sum(pca.explained_variance_ratio_)*100, 2), "%")
print("SVD Explained Variance (2D):", round(np.sum(svd.explained_variance_ratio_)*100, 2), "%")

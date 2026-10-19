"""
Project 003: K-means clustering for customer segmentation

Overview:
  This project builds a production-minded k-means clustering for customer segmentation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Simulate customer spending score and annual income
X, _ = make_blobs(n_samples=500, centers=4, n_features=2, cluster_std=1.2, random_state=42)

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X)
sil_score = silhouette_score(X, cluster_labels)

print("=== K-Means Clustering: Customer Segmentation ===")
print(f"Number of Customer Segments: {len(np.unique(cluster_labels))}")
print(f"Silhouette Score: {sil_score:.4f}")
print("Cluster Centroids (Coordinates):")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  Segment {i+1}: Income={center[0]:.2f}, Spending={center[1]:.2f}")

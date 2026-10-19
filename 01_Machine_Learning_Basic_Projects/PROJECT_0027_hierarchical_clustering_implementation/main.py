"""
Project 027: Hierarchical clustering implementation

Overview:
  This project builds a production-minded hierarchical clustering implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=250, centers=3, n_features=2, random_state=42)

model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = model.fit_predict(X)

sil = silhouette_score(X, labels)
print("=== Hierarchical (Agglomerative) Clustering ===")
print(f"Fitted Clusters: {len(set(labels))}")
print(f"Silhouette Coefficient: {sil:.4f}")

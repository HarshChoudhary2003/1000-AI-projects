"""
Project 038: t-SNE visualization tool

Overview:
  This project builds a production-minded t-SNE visualization tool workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

digits = load_digits()
X, y = digits.data[:400], digits.target[:400]

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

print("=== t-SNE High-Dimensional Visualization Tool ===")
print(f"Input Shape: {X.shape}")
print(f"Embedded 2D Coordinates Shape: {X_embedded.shape}")
print(f"KL-Divergence error: {tsne.kl_divergence_:.4f}")

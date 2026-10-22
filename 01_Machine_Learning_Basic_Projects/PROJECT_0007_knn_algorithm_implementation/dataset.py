"""
Project 007: Dataset Provider & Benchmark Generator for KNN Studio
===================================================================
Provides authentic real-world datasets and synthetic non-linear 2D manifolds
to benchmark K-Nearest Neighbors distance metrics, boundary geometry,
and the curse of dimensionality.

Datasets included:
1. Breast Cancer Wisconsin Diagnostic (Real, High-dimensional Binary Classification, 569 samples, 30 features)
2. Wine Cultivar Chemical Recognition (Real, Multi-class Chemical Profiling, 178 samples, 13 features)
3. Synthetic 2D Non-Linear Manifolds (Two Moons, Concentric Circles, Intertwined Spirals, Gaussian Clusters)
4. High-Dimensional Curse of Dimensionality Probe Generator
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.datasets import load_breast_cancer, load_wine, make_moons, make_circles, make_blobs


def load_real_dataset(name: str = "breast_cancer") -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Load an authentic real-world benchmark dataset with full feature and target metadata.

    Returns:
        (df, feature_names, target_col, target_names)
    """
    name_clean = name.lower()
    
    if "wine" in name_clean:
        raw = load_wine()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        target_names = [str(n) for n in raw.target_names]
        feature_names = list(raw.feature_names)
        target_col = "target"
        df["target_label"] = df["target"].map(lambda idx: target_names[idx])
        return df, feature_names, target_col, target_names
        
    else:  # Default: Breast Cancer Wisconsin Diagnostic
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        # raw.target_names: 0 = 'malignant', 1 = 'benign'
        target_names = [str(n).capitalize() for n in raw.target_names]
        feature_names = list(raw.feature_names)
        target_col = "target"
        df["target_label"] = df["target"].map(lambda idx: target_names[idx])
        return df, feature_names, target_col, target_names


def generate_2d_manifold(
    manifold_type: str = "Two Moons",
    n_samples: int = 400,
    noise: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Generate non-linear 2D classification datasets optimal for continuous
    decision boundary visualization and Voronoi neighborhood mapping.
    """
    np.random.seed(random_state)
    
    if manifold_type == "Concentric Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        target_names = ["Outer Ring", "Inner Core"]
    elif manifold_type == "Intertwined Spirals":
        # Two-arm Archimedean spiral generator
        n = n_samples // 2
        theta = np.sqrt(np.random.rand(n)) * 2.8 * np.pi
        r_a = 2 * theta + np.pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T + np.random.randn(n, 2) * (noise * 5)
        
        r_b = -2 * theta - np.pi
        data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T + np.random.randn(n, 2) * (noise * 5)
        
        X = np.vstack([data_a, data_b])
        # Standardize spiral coordinates to [-3, 3] range
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        target_names = ["Spiral Alpha", "Spiral Beta"]
    elif manifold_type == "Gaussian Blobs (Multi-Class)":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.5]],
            cluster_std=1.0 + (noise * 2),
            random_state=random_state
        )
        target_names = ["Cluster 0", "Cluster 1", "Cluster 2"]
    else:  # Two Moons default
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        target_names = ["Crescent Upper", "Crescent Lower"]

    df = pd.DataFrame(X, columns=["Feature_X1", "Feature_X2"])
    df["target"] = y
    df["target_label"] = df["target"].map(lambda idx: target_names[idx])
    return df, ["Feature_X1", "Feature_X2"], "target", target_names


def generate_curse_of_dimensionality_experiment(
    n_samples: int = 300,
    dimensions: List[int] = [2, 5, 10, 25, 50, 100, 200],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Demonstrates the mathematical Curse of Dimensionality:
    As dimension D increases, the contrast between the distance to the nearest
    and farthest neighbor converges to 0:
    (d_max - d_min) / d_min -> 0
    """
    np.random.seed(random_state)
    results = []

    for d in dimensions:
        # Uniform sampling inside a D-dimensional hypercube [0, 1]^D
        points = np.random.uniform(0.0, 1.0, size=(n_samples, d))
        # Compute all pairwise Euclidean distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Mask out self-distance (diagonal = 0)
        np.fill_diagonal(dist_matrix, np.nan)
        
        d_min = np.nanmin(dist_matrix)
        d_max = np.nanmax(dist_matrix)
        d_mean = np.nanmean(dist_matrix)
        relative_contrast = (d_max - d_min) / (d_min + 1e-9)

        results.append({
            "Dimension (D)": d,
            "Min Distance": round(float(d_min), 3),
            "Max Distance": round(float(d_max), 3),
            "Mean Distance": round(float(d_mean), 3),
            "Distance Contrast ((Max-Min)/Min)": round(float(relative_contrast), 3)
        })

    return pd.DataFrame(results)

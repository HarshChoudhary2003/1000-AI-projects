"""
Project 013: Principal Component Analysis (PCA) - Dataset Module
================================================================
Curated benchmark datasets and synthetic manifold generators for dimensionality reduction,
eigen-analysis, biplot interpretation, and image compression experiments.

Part of the 1,000 AI Projects Implementation Library.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris


def load_benchmark_dataset(dataset_name: str = "digits", n_samples: Optional[int] = None, seed: int = 42) -> Dict[str, Any]:
    """
    Load and curate benchmark datasets tailored for PCA exploration.
    
    Supported Datasets:
      - 'digits': 8x8 pixel grayscale digits (64 features, 10 classes)
      - 'breast_cancer': Diagnostic features computed from digitized FNA images (30 features, 2 classes)
      - 'wine': Chemical constituents of Italian wines (13 features, 3 cultivars)
      - 'iris': Classic botanical measurements (4 features, 3 species)
      - 'swiss_roll': 3D nonlinear manifold embedded in 3D space
      - 'correlated_gaussian': High-dimensional Gaussian ellipsoid with controlled covariance rank
    """
    dataset_name = dataset_name.lower().strip()
    np.random.seed(seed)
    
    if dataset_name == "digits":
        raw = load_digits()
        X = raw.data
        y = raw.target
        target_names = [f"Digit {i}" for i in range(10)]
        feature_names = [f"pixel_{r}_{c}" for r in range(8) for c in range(8)]
        description = "8x8 grayscale handwritten digit images (64 dimensions) - ideal for compression & reconstruction."
        is_image_data = True
        image_shape = (8, 8)
        
    elif dataset_name == "breast_cancer":
        raw = load_breast_cancer()
        X = raw.data
        y = raw.target
        target_names = list(raw.target_names)
        feature_names = list(raw.feature_names)
        description = "30 numerical cell nucleus morphological characteristics from digitized FNA biopsy slides."
        is_image_data = False
        image_shape = None
        
    elif dataset_name == "wine":
        raw = load_wine()
        X = raw.data
        y = raw.target
        target_names = list(raw.target_names)
        feature_names = list(raw.feature_names)
        description = "13 chemical and physical constituents analyzing 3 distinct Italian wine cultivars."
        is_image_data = False
        image_shape = None
        
    elif dataset_name == "iris":
        raw = load_iris()
        X = raw.data
        y = raw.target
        target_names = list(raw.target_names)
        feature_names = list(raw.feature_names)
        description = "4 sepal and petal dimension measurements for 3 species of Iris flowers."
        is_image_data = False
        image_shape = None

    elif dataset_name == "swiss_roll":
        n = n_samples if n_samples else 1200
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
        x_coord = t * np.cos(t)
        y_coord = 21 * np.random.rand(n)
        z_coord = t * np.sin(t)
        X = np.column_stack([x_coord, y_coord, z_coord])
        # Add 3 orthogonal noise dimensions to simulate higher dimension
        noise = np.random.randn(n, 3) * 0.8
        X = np.hstack([X, noise])
        y = (t // (0.5 * np.pi)).astype(int)
        target_names = [f"Roll Segment {i}" for i in np.unique(y)]
        feature_names = ["manifold_x", "manifold_height", "manifold_z", "noise_dim_1", "noise_dim_2", "noise_dim_3"]
        description = "Synthetic 3D Swiss Roll embedded into 6D with 3 noisy orthogonal dimensions."
        is_image_data = False
        image_shape = None

    elif dataset_name in ["correlated_gaussian", "gaussian_ellipsoid"]:
        n = n_samples if n_samples else 1000
        n_features = 12
        # Latent true rank is 3
        z = np.random.randn(n, 3)
        # Random mixing matrix
        mix = np.random.RandomState(seed).randn(3, n_features)
        X = np.dot(z, mix) + np.random.randn(n, n_features) * 0.25
        # Target assigned by dominant latent dimension
        y = (z[:, 0] > 0).astype(int) * 2 + (z[:, 1] > 0).astype(int)
        target_names = ["Quadrant 1", "Quadrant 2", "Quadrant 3", "Quadrant 4"]
        feature_names = [f"Sensor_{i+1:02d}" for i in range(n_features)]
        description = "12-dimensional Gaussian sensor array with intrinsic rank 3 and low additive noise."
        is_image_data = False
        image_shape = None
        
    else:
        raise ValueError(f"Unknown benchmark dataset: '{dataset_name}'. Supported: digits, breast_cancer, wine, iris, swiss_roll, correlated_gaussian")

    if n_samples is not None and n_samples < len(X):
        indices = np.random.choice(len(X), size=n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df["target_label"] = [target_names[val] if val < len(target_names) else str(val) for val in y]

    return {
        "name": dataset_name,
        "description": description,
        "X": X,
        "y": y,
        "df": df,
        "feature_names": feature_names,
        "target_names": target_names,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "is_image_data": is_image_data,
        "image_shape": image_shape
    }


def get_archetype_presets() -> Dict[str, Dict[str, Any]]:
    """
    Returns curated preset profiles for interactive demonstrations.
    """
    return {
        "Handwritten Digits (64D -> 2D)": {
            "dataset": "digits",
            "n_components": 2,
            "standardize": False,
            "recommended_view": "reconstruction",
            "notes": "Grayscale values naturally on [0, 16] scale; unstandardized retains pixel brightness contrast."
        },
        "Breast Cancer Biopsy (30D -> 3D)": {
            "dataset": "breast_cancer",
            "n_components": 3,
            "standardize": True,
            "recommended_view": "biplot",
            "notes": "Vastly different feature scales (area vs smoothness) require strict z-score standardization."
        },
        "Wine Cultivars (13D -> 2D)": {
            "dataset": "wine",
            "n_components": 2,
            "standardize": True,
            "recommended_view": "loadings",
            "notes": "Flavanoids and proline dominate PC1 separation across all three distinct grape cultivars."
        },
        "Botanical Iris (4D -> 2D)": {
            "dataset": "iris",
            "n_components": 2,
            "standardize": True,
            "recommended_view": "projection",
            "notes": "PC1 alone accounts for >72% of total variance, cleanly separating Setosa from Versicolor/Virginica."
        },
        "Synthetic Manifold (6D -> 3D)": {
            "dataset": "swiss_roll",
            "n_components": 3,
            "standardize": True,
            "recommended_view": "projection",
            "notes": "Tests linear PCA limits on nonlinear rolled geometry and noise suppression."
        }
    }

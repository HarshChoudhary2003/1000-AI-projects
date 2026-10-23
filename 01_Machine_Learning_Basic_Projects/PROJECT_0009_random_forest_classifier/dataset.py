"""
Project 009: Dataset Provider & Benchmark Generator for Random Forest Studio
=============================================================================
Provides authentic real-world datasets and synthetic geometric manifolds
to benchmark Random Forest bagging ensembles, feature importances,
Out-of-Bag (OOB) error convergence, and tree voting diversity.

Datasets included:
1. Wine Cultivars Chemical Recognition (Real, Multi-class Chemical Profiling, 178 samples, 13 features)
2. Breast Cancer Wisconsin Diagnostic (Real, High-dimensional Binary Classification, 569 samples, 30 features)
3. Iris Botanical Benchmark (Real, Multi-class Geometric, 150 samples, 4 features)
4. Synthetic 2D Geometric Manifolds (Two Moons, Concentric Circles, Overlapping Blobs, Complex Multimodal)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.datasets import load_wine, load_breast_cancer, load_iris, make_blobs, make_moons, make_circles


def load_real_dataset(name: str = "wine") -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Load an authentic real-world benchmark dataset with full feature and target metadata.

    Returns:
        (df, feature_names, target_col, target_names)
    """
    name_clean = name.lower()

    if "cancer" in name_clean or "breast" in name_clean:
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        target_names = [str(n).capitalize() for n in raw.target_names]  # Malignant (0), Benign (1)
        feature_names = list(raw.feature_names)
        target_col = "target"
        df["target_label"] = df["target"].map(lambda idx: target_names[idx])
        return df, feature_names, target_col, target_names

    elif "iris" in name_clean:
        raw = load_iris()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        target_names = [str(n).capitalize() for n in raw.target_names]
        feature_names = list(raw.feature_names)
        target_col = "target"
        df["target_label"] = df["target"].map(lambda idx: target_names[idx])
        return df, feature_names, target_col, target_names

    else:  # Default: Wine Cultivars Chemical Recognition
        raw = load_wine()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = raw.target
        target_names = [f"Cultivar {n.split('_')[-1].upper()}" if "_" in n else f"Cultivar {n.capitalize()}" for n in raw.target_names]
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
    Generate non-linear 2D classification datasets optimal for testing
    Random Forest orthogonal partition boundaries and ensemble voting consensus.
    """
    np.random.seed(random_state)

    if manifold_type == "Concentric Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.45, random_state=random_state)
        target_names = ["Outer Cluster (0)", "Inner Core (1)"]
    elif manifold_type == "Overlapping Blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[[-1.8, -1.2], [1.8, 1.2], [0.0, 2.0]],
            cluster_std=1.0 + (noise * 2.5),
            random_state=random_state
        )
        target_names = ["Cluster Alpha (0)", "Cluster Beta (1)", "Cluster Gamma (2)"]
    elif manifold_type == "Complex Multimodal":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[[-2, -2], [2, 2], [-2, 2], [2, -2]],
            cluster_std=0.85 + (noise * 1.5),
            random_state=random_state
        )
        # Combine into XOR-like alternating 2 classes
        y = np.where((y == 0) | (y == 1), 0, 1)
        target_names = ["Quadrant Set A (0)", "Quadrant Set B (1)"]
    else:  # Two Moons default
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        target_names = ["Crescent Upper (0)", "Crescent Lower (1)"]

    df = pd.DataFrame(X, columns=["Feature_X1", "Feature_X2"])
    df["target"] = y
    df["target_label"] = df["target"].map(lambda idx: target_names[idx])
    return df, ["Feature_X1", "Feature_X2"], "target", target_names


def get_archetype_presets() -> Dict[str, Dict[str, float]]:
    """
    Representative sample profiles based on the Wine Cultivars chemical dataset
    for interactive real-time prediction and voting consensus inspection.
    """
    return {
        "Classic High-Alcohol Cultivar A": {
            "alcohol": 14.23,
            "malic_acid": 1.71,
            "ash": 2.43,
            "alcalinity_of_ash": 15.6,
            "magnesium": 127.0,
            "total_phenols": 2.80,
            "flavanoids": 3.06,
            "nonflavanoid_phenols": 0.28,
            "proanthocyanins": 2.29,
            "color_intensity": 5.64,
            "hue": 1.04,
            "od280/od315_of_diluted_wines": 3.92,
            "proline": 1065.0
        },
        "Light & Crisp Cultivar B": {
            "alcohol": 12.37,
            "malic_acid": 0.94,
            "ash": 1.36,
            "alcalinity_of_ash": 10.6,
            "magnesium": 88.0,
            "total_phenols": 1.98,
            "flavanoids": 0.57,
            "nonflavanoid_phenols": 0.28,
            "proanthocyanins": 0.42,
            "color_intensity": 1.95,
            "hue": 1.05,
            "od280/od315_of_diluted_wines": 1.82,
            "proline": 520.0
        },
        "Astringent Tannin-Rich Cultivar C": {
            "alcohol": 13.11,
            "malic_acid": 3.83,
            "ash": 2.54,
            "alcalinity_of_ash": 21.0,
            "magnesium": 98.0,
            "total_phenols": 1.48,
            "flavanoids": 0.60,
            "nonflavanoid_phenols": 0.45,
            "proanthocyanins": 1.03,
            "color_intensity": 7.10,
            "hue": 0.65,
            "od280/od315_of_diluted_wines": 1.68,
            "proline": 615.0
        }
    }

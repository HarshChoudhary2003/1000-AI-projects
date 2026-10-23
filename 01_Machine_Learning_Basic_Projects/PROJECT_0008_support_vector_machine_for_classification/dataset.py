"""
Project 008: Dataset Provider & Benchmark Generator for SVM Studio
===================================================================
Provides authentic real-world datasets and synthetic geometric manifolds
to benchmark Support Vector Machines (SVM), kernel functions, margins,
and support vector distributions.

Datasets included:
1. Breast Cancer Wisconsin Diagnostic (Real, High-dimensional Binary Classification, 569 samples, 30 features)
2. Wine Cultivar Chemical Recognition (Real, Multi-class Chemical Profiling, 178 samples, 13 features)
3. Iris Botanical Benchmark (Real, Multi-class Geometric, 150 samples, 4 features)
4. Synthetic 2D Geometric Manifolds (Linearly Separable, Two Moons, Concentric Circles, Overlapping Blobs)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
        target_names = [f"Class {n}" for n in raw.target_names]
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
    manifold_type: str = "Linearly Separable",
    n_samples: int = 300,
    noise: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Generate 2D classification datasets tailored for continuous SVM
    decision boundary visualization, margin gutters (+1 / -1), and support vector identification.
    """
    np.random.seed(random_state)

    if manifold_type == "Concentric Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.45, random_state=random_state)
        target_names = ["Outer Class (0)", "Inner Core (1)"]
    elif manifold_type == "Two Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        target_names = ["Upper Crescent (0)", "Lower Crescent (1)"]
    elif manifold_type == "Overlapping Blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[[-1.5, -1.0], [1.5, 1.0]],
            cluster_std=1.0 + (noise * 2.5),
            random_state=random_state
        )
        target_names = ["Cluster Alpha (0)", "Cluster Beta (1)"]
    else:  # Linearly Separable (default)
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[[-2.0, -1.5], [2.0, 1.5]],
            cluster_std=0.85 + (noise * 1.5),
            random_state=random_state
        )
        target_names = ["Class Negative (0)", "Class Positive (1)"]

    df = pd.DataFrame(X, columns=["Feature_X1", "Feature_X2"])
    df["target"] = y
    df["target_label"] = df["target"].map(lambda idx: target_names[idx])
    return df, ["Feature_X1", "Feature_X2"], "target", target_names


def get_clinical_patient_presets() -> Dict[str, Dict[str, float]]:
    """
    Realistic clinical biopsy parameter profiles based on the Wisconsin Breast Cancer dataset
    for interactive real-time inference demonstration.
    """
    return {
        "High-Risk Malignant Case (Biopsy #401)": {
            "mean radius": 19.69,
            "mean texture": 21.25,
            "mean perimeter": 130.0,
            "mean area": 1203.0,
            "mean smoothness": 0.1096,
            "mean compactness": 0.1599,
            "mean concavity": 0.1974,
            "mean concave points": 0.1279,
            "mean symmetry": 0.2069,
            "mean fractal dimension": 0.05999,
            "radius error": 0.7456,
            "texture error": 0.7869,
            "perimeter error": 4.585,
            "area error": 94.03,
            "smoothness error": 0.00615,
            "compactness error": 0.04006,
            "concavity error": 0.03832,
            "concave points error": 0.02058,
            "symmetry error": 0.02250,
            "fractal dimension error": 0.004571,
            "worst radius": 23.57,
            "worst texture": 25.53,
            "worst perimeter": 152.5,
            "worst area": 1709.0,
            "worst smoothness": 0.1444,
            "worst compactness": 0.4245,
            "worst concavity": 0.4504,
            "worst concave points": 0.2430,
            "worst symmetry": 0.3613,
            "worst fractal dimension": 0.08758
        },
        "Typical Benign Fibroadenoma (Biopsy #112)": {
            "mean radius": 11.42,
            "mean texture": 15.34,
            "mean perimeter": 73.34,
            "mean area": 399.8,
            "mean smoothness": 0.0865,
            "mean compactness": 0.0573,
            "mean concavity": 0.0234,
            "mean concave points": 0.0184,
            "mean symmetry": 0.1654,
            "mean fractal dimension": 0.0612,
            "radius error": 0.2415,
            "texture error": 0.9125,
            "perimeter error": 1.632,
            "area error": 17.52,
            "smoothness error": 0.0058,
            "compactness error": 0.0124,
            "concavity error": 0.0142,
            "concave points error": 0.0076,
            "symmetry error": 0.0182,
            "fractal dimension error": 0.0028,
            "worst radius": 12.84,
            "worst texture": 19.45,
            "worst perimeter": 83.12,
            "worst area": 502.5,
            "worst smoothness": 0.1142,
            "worst compactness": 0.1124,
            "worst concavity": 0.0886,
            "worst concave points": 0.0582,
            "worst symmetry": 0.2415,
            "worst fractal dimension": 0.0712
        },
        "Borderline Indeterminate Case (Biopsy #289)": {
            "mean radius": 14.25,
            "mean texture": 18.80,
            "mean perimeter": 92.40,
            "mean area": 632.0,
            "mean smoothness": 0.0980,
            "mean compactness": 0.1040,
            "mean concavity": 0.0850,
            "mean concave points": 0.0550,
            "mean symmetry": 0.1820,
            "mean fractal dimension": 0.0620,
            "radius error": 0.3800,
            "texture error": 1.0500,
            "perimeter error": 2.500,
            "area error": 33.00,
            "smoothness error": 0.0062,
            "compactness error": 0.0210,
            "concavity error": 0.0230,
            "concave points error": 0.0120,
            "symmetry error": 0.0190,
            "fractal dimension error": 0.0034,
            "worst radius": 16.20,
            "worst texture": 24.10,
            "worst perimeter": 106.5,
            "worst area": 815.0,
            "worst smoothness": 0.1310,
            "worst compactness": 0.2200,
            "worst concavity": 0.2100,
            "worst concave points": 0.1250,
            "worst symmetry": 0.2850,
            "worst fractal dimension": 0.0790
        }
    }

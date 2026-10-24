"""
Project 010: Dataset Provider & Benchmark Generator for Gradient Boosting Studio
================================================================================
Provides authentic real-world datasets and synthetic geometric manifolds
to benchmark Gradient Tree Boosting, stage-by-stage residual reduction,
learning rate shrinkage, stochastic subsampling, and decision surfaces.

Datasets included:
1. Breast Cancer Wisconsin Diagnostic (Real, High-dimensional Binary Classification, 569 samples, 30 features)
2. Wine Cultivar Chemical Recognition (Real, Multi-class Chemical Profiling, 178 samples, 13 features)
3. Credit Default & Risk Scoring (Realistic Tabular Financial Dataset, 1,000 samples, 10 features)
4. Synthetic 2D Geometric Manifolds (Two Moons, Concentric Circles, Overlapping Blobs, Linearly Separable, Spiral)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.datasets import load_breast_cancer, load_wine, make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler


def load_real_dataset(name: str = "breast_cancer") -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Load an authentic real-world or tabular benchmark dataset with full feature and target metadata.

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

    elif "credit" in name_clean or "default" in name_clean or "risk" in name_clean:
        return generate_credit_default_dataset()

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


def generate_credit_default_dataset(
    n_samples: int = 1000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Generate a realistic, production-inspired tabular Credit Default dataset.
    Gradient Boosting excels on tabular data with non-linear thresholds and interaction terms.
    """
    np.random.seed(random_state)

    credit_score = np.random.normal(680, 75, n_samples).clip(350, 850)
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).round(-2)
    debt_to_income = np.random.beta(a=2, b=5, size=n_samples) * 0.6
    loan_amount = (annual_income * np.random.uniform(0.1, 0.45, n_samples)).round(-2)
    employment_years = np.random.exponential(scale=5.0, size=n_samples).clip(0, 35)
    num_credit_lines = np.random.poisson(lam=7, size=n_samples).clip(1, 25)
    late_payments_2yr = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.70, 0.15, 0.08, 0.04, 0.03])
    revolving_utilization = np.random.beta(a=3, b=4, size=n_samples).clip(0.01, 1.0)
    inquiries_6mo = np.random.poisson(lam=1.2, size=n_samples).clip(0, 10)
    bankruptcy_record = np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08])

    # Non-linear log-odds of default with interaction terms that tree ensembles capture cleanly
    log_odds = (
        - 0.015 * (credit_score - 650)
        + 3.5 * debt_to_income
        + 1.8 * revolving_utilization
        + 0.55 * late_payments_2yr
        + 0.35 * inquiries_6mo
        + 1.2 * bankruptcy_record
        - 0.08 * employment_years
        + 0.000015 * loan_amount
        # Interaction effect: low credit + high DTI amplifies risk exponentially
        + 2.2 * ((credit_score < 620) & (debt_to_income > 0.35)).astype(float)
        - 2.8
    )

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    target = (np.random.uniform(0, 1, n_samples) < prob).astype(int)

    feature_cols = [
        "credit_score",
        "annual_income",
        "debt_to_income",
        "loan_amount",
        "employment_years",
        "num_credit_lines",
        "late_payments_2yr",
        "revolving_utilization",
        "inquiries_6mo",
        "bankruptcy_record"
    ]

    data = {
        "credit_score": np.round(credit_score, 1),
        "annual_income": np.round(annual_income, 0),
        "debt_to_income": np.round(debt_to_income, 4),
        "loan_amount": np.round(loan_amount, 0),
        "employment_years": np.round(employment_years, 1),
        "num_credit_lines": num_credit_lines,
        "late_payments_2yr": late_payments_2yr,
        "revolving_utilization": np.round(revolving_utilization, 4),
        "inquiries_6mo": inquiries_6mo,
        "bankruptcy_record": bankruptcy_record,
        "target": target
    }

    df = pd.DataFrame(data)
    target_names = ["Non-Default", "Default"]
    df["target_label"] = df["target"].map(lambda t: target_names[t])

    return df, feature_cols, "target", target_names


def generate_2d_manifold(
    manifold_type: str = "Two Moons",
    n_samples: int = 400,
    noise: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """
    Generate 2D classification datasets tailored for continuous Gradient Boosting
    decision boundary visualization and probability landscape exploration.
    """
    manifold_clean = manifold_type.lower()

    if "moon" in manifold_clean:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        target_names = ["Moon A", "Moon B"]
    elif "circle" in manifold_clean:
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        target_names = ["Inner Ring", "Outer Ring"]
    elif "blob" in manifold_clean:
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.4 + noise * 2, random_state=random_state)
        target_names = ["Cluster 0", "Cluster 1"]
    elif "spiral" in manifold_clean:
        # Generate 2-arm interlocking spirals
        n_arm = n_samples // 2
        theta = np.sqrt(np.random.rand(n_arm)) * 2.8 * np.pi
        r_a = 2 * theta + np.pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T + np.random.randn(n_arm, 2) * (noise * 5)
        
        theta_b = np.sqrt(np.random.rand(n_arm)) * 2.8 * np.pi
        r_b = -2 * theta_b - np.pi
        data_b = np.array([np.cos(theta_b) * r_b, np.sin(theta_b) * r_b]).T + np.random.randn(n_arm, 2) * (noise * 5)
        
        X = np.vstack([data_a, data_b])
        y = np.hstack([np.zeros(n_arm, dtype=int), np.ones(n_arm, dtype=int)])
        scaler = StandardScaler()
        X = scaler.fit_transform(X) * 2.5
        target_names = ["Spiral Arm 1", "Spiral Arm 2"]
    else:  # Linearly Separable / Slightly Overlapping
        X, y = make_blobs(n_samples=n_samples, centers=[[-1.5, -1.5], [1.5, 1.5]], cluster_std=0.9 + noise, random_state=random_state)
        target_names = ["Group Alpha", "Group Beta"]

    df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
    df["target"] = y
    df["target_label"] = df["target"].map(lambda idx: target_names[idx])

    return df, ["Feature_1", "Feature_2"], "target", target_names


def get_archetype_presets() -> Dict[str, Dict[str, Any]]:
    """
    Curated hyperparameter presets representing different operational strategies.
    """
    return {
        "⚡ Fast Stochastic Shrinkage": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "description": "Fast convergence with stochastic sample sub-sampling (80%) for variance reduction and speed."
        },
        "🎯 High Precision / Deep Trees": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "description": "Captures intricate non-linear interactions with smaller learning rate and deeper trees."
        },
        "🛡️ Conservative Regularized": {
            "n_estimators": 80,
            "learning_rate": 0.03,
            "max_depth": 2,
            "subsample": 1.0,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "description": "Stumps/shallow trees with strong shrinkage to prevent overfitting on noisy data."
        },
        "🔬 Baseline Reference": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "description": "Standard Scikit-Learn defaults for quick benchmarking and baseline comparisons."
        }
    }

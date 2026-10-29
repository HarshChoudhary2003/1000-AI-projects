"""
Project 014: Feature Selection Techniques - Dataset Module
===========================================================
Curated benchmark datasets and synthetic generators with ground-truth
knowledge of informative, redundant, and pure noise features for rigorous
evaluation of feature selection algorithms.

Part of the 1,000 AI Projects Implementation Library.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification,
    make_regression,
    load_breast_cancer,
    load_wine,
    load_diabetes,
    fetch_california_housing
)


def load_feature_dataset(
    dataset_name: str = "synthetic_classification",
    n_samples: int = 500,
    n_features: int = 20,
    n_informative: int = 6,
    n_redundant: int = 4,
    noise_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Load or generate benchmark datasets for feature selection.
    
    Supported datasets:
      - 'synthetic_classification': Controlled classification with known informative, redundant, and noise features.
      - 'synthetic_regression': Controlled regression with known informative features.
      - 'breast_cancer': 30 diagnostic cell morphology features (binary classification).
      - 'wine': 13 chemical and physical constituents (3 classes).
      - 'diabetes': 10 clinical metrics for disease progression (regression).
      - 'california_housing': 8 macroeconomic and geographic features (regression).
    """
    dataset_name = dataset_name.lower().strip()
    np.random.seed(seed)
    
    if dataset_name in ["synthetic_classification", "synthetic"]:
        # Ensure parameters are valid
        n_features = max(6, n_features)
        n_informative = min(n_features - 2, max(2, n_informative))
        n_redundant = min(n_features - n_informative, max(0, n_redundant))
        
        X_raw, y_raw = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=0,
            n_classes=2,
            flip_y=noise_ratio,
            random_state=seed
        )
        
        # Ground-truth categorization
        # In scikit-learn make_classification: informative features are first, then redundant, then noise
        true_informative_indices = list(range(n_informative))
        true_redundant_indices = list(range(n_informative, n_informative + n_redundant))
        true_noise_indices = list(range(n_informative + n_redundant, n_features))
        
        feature_names = [f"F_{i:02d}_info" for i in true_informative_indices] + \
                        [f"F_{i:02d}_redundant" for i in true_redundant_indices] + \
                        [f"F_{i:02d}_noise" for i in true_noise_indices]
        
        # Shuffle features so they aren't ordered strictly to ensure unbiased testing
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_features)
        
        X = X_raw[:, perm]
        feature_names = [feature_names[i] for i in perm]
        
        # Track updated ground truth indices
        ground_truth_informative = [i for i, orig_idx in enumerate(perm) if orig_idx in true_informative_indices]
        ground_truth_redundant = [i for i, orig_idx in enumerate(perm) if orig_idx in true_redundant_indices]
        ground_truth_noise = [i for i, orig_idx in enumerate(perm) if orig_idx in true_noise_indices]
        
        task_type = "classification"
        target_names = ["Class 0", "Class 1"]
        description = (
            f"Controlled Synthetic Classification: {n_samples} samples, {n_features} features "
            f"({len(ground_truth_informative)} Informative, {len(ground_truth_redundant)} Redundant collinear, "
            f"{len(ground_truth_noise)} Pure Noise)."
        )
        has_ground_truth = True

    elif dataset_name == "synthetic_regression":
        n_features = max(6, n_features)
        n_informative = min(n_features - 2, max(2, n_informative))
        
        X_raw, y_raw, coef = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=15.0 * noise_ratio,
            coef=True,
            random_state=seed
        )
        
        true_informative_indices = [i for i, c in enumerate(coef) if abs(c) > 1e-4]
        true_noise_indices = [i for i, c in enumerate(coef) if abs(c) <= 1e-4]
        
        feature_names = [f"feat_{i:02d}_{'signal' if i in true_informative_indices else 'noise'}" for i in range(n_features)]
        
        X = X_raw
        ground_truth_informative = true_informative_indices
        ground_truth_redundant = []
        ground_truth_noise = true_noise_indices
        task_type = "regression"
        target_names = ["Continuous Target"]
        description = (
            f"Controlled Synthetic Regression: {n_samples} samples, {n_features} features "
            f"({len(ground_truth_informative)} Non-zero Informative, {len(ground_truth_noise)} Zero-coefficient Noise)."
        )
        has_ground_truth = True

    elif dataset_name == "breast_cancer":
        raw = load_breast_cancer()
        X = raw.data
        y_raw = raw.target
        target_names = list(raw.target_names)
        feature_names = list(raw.feature_names)
        task_type = "classification"
        ground_truth_informative = []
        ground_truth_redundant = []
        ground_truth_noise = []
        has_ground_truth = False
        description = "Wisconsin Diagnostic Breast Cancer: 30 cell nucleus morphometric features (Malignant vs Benign)."

    elif dataset_name == "wine":
        raw = load_wine()
        X = raw.data
        y_raw = raw.target
        target_names = list(raw.target_names)
        feature_names = list(raw.feature_names)
        task_type = "classification"
        ground_truth_informative = []
        ground_truth_redundant = []
        ground_truth_noise = []
        has_ground_truth = False
        description = "Wine Cultivars: 13 chemical & spectrographic properties across 3 Italian cultivars."

    elif dataset_name == "diabetes":
        raw = load_diabetes()
        X = raw.data
        y_raw = raw.target
        target_names = ["Disease Progression Score"]
        feature_names = list(raw.feature_names)
        task_type = "regression"
        ground_truth_informative = []
        ground_truth_redundant = []
        ground_truth_noise = []
        has_ground_truth = False
        description = "Diabetes Progression: 10 baseline physiological metrics (age, sex, bmi, bp, s1-s6)."

    elif dataset_name == "california_housing":
        raw = fetch_california_housing()
        # Subsample for interactive responsiveness
        if n_samples is None or n_samples > 1200:
            n_sub = 1000
        else:
            n_sub = n_samples
        indices = np.random.RandomState(seed).choice(len(raw.data), size=n_sub, replace=False)
        X = raw.data[indices]
        y_raw = raw.target[indices]
        target_names = ["Median House Value ($100k)"]
        feature_names = list(raw.feature_names)
        task_type = "regression"
        ground_truth_informative = []
        ground_truth_redundant = []
        ground_truth_noise = []
        has_ground_truth = False
        description = f"California Housing Subsample: {len(X)} districts, 8 census and geographic features."

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: synthetic_classification, synthetic_regression, breast_cancer, wine, diabetes, california_housing")

    # Construct DataFrame representation
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y_raw

    return {
        "dataset_name": dataset_name,
        "X": X,
        "y": y_raw,
        "df": df,
        "feature_names": feature_names,
        "target_names": target_names,
        "task_type": task_type,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "description": description,
        "has_ground_truth": has_ground_truth,
        "ground_truth_informative": ground_truth_informative,
        "ground_truth_redundant": ground_truth_redundant,
        "ground_truth_noise": ground_truth_noise
    }


def add_synthetic_noise_features(
    dataset_dict: Dict[str, Any],
    n_noise_features: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Inject pure noise columns into an existing dataset to benchmark
    whether feature selection techniques successfully eliminate them.
    """
    if n_noise_features <= 0:
        return dataset_dict

    rng = np.random.RandomState(seed)
    n_samples = dataset_dict["n_samples"]
    
    # Generate Gaussian and Uniform noise features
    noise_data = []
    noise_names = []
    for i in range(n_noise_features):
        if i % 2 == 0:
            col = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 1))
            name = f"noise_gaussian_{i+1:02d}"
        else:
            col = rng.uniform(low=-2.0, high=2.0, size=(n_samples, 1))
            name = f"noise_uniform_{i+1:02d}"
        noise_data.append(col)
        noise_names.append(name)
        
    noise_matrix = np.hstack(noise_data)
    new_X = np.hstack([dataset_dict["X"], noise_matrix])
    new_feature_names = dataset_dict["feature_names"] + noise_names
    
    orig_n_feat = dataset_dict["n_features"]
    new_noise_indices = list(range(orig_n_feat, orig_n_feat + n_noise_features))
    
    new_ground_truth_noise = list(dataset_dict.get("ground_truth_noise", [])) + new_noise_indices
    
    new_df = pd.DataFrame(new_X, columns=new_feature_names)
    new_df["target"] = dataset_dict["y"]
    
    updated = dict(dataset_dict)
    updated["X"] = new_X
    updated["df"] = new_df
    updated["feature_names"] = new_feature_names
    updated["n_features"] = len(new_feature_names)
    updated["ground_truth_noise"] = new_noise_indices
    updated["description"] = dataset_dict["description"] + f" [Injected +{n_noise_features} Synthetic Noise Features]"
    return updated


def load_custom_csv(
    file_or_path: Any,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a user-supplied CSV file for feature selection analysis.
    """
    if isinstance(file_or_path, str):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    # Impute or drop nulls
    df = df.dropna(how="all")
    
    # Separate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        raise ValueError(f"Uploaded CSV must contain at least 3 numerical columns. Found: {numeric_cols}")

    if target_column is None or target_column not in df.columns:
        # Default to the last column
        target_column = numeric_cols[-1]
    
    feature_cols = [c for c in numeric_cols if c != target_column]
    
    # Fill remaining NaNs with column median
    df_clean = df[feature_cols + [target_column]].fillna(df[feature_cols + [target_column]].median())
    
    X = df_clean[feature_cols].values
    y = df_clean[target_column].values
    
    # Determine classification vs regression
    unique_vals = len(np.unique(y))
    if unique_vals <= 10 or (np.issubdtype(y.dtype, np.integer) and unique_vals < 20):
        task_type = "classification"
        target_names = [f"Class {val}" for val in np.unique(y)]
    else:
        task_type = "regression"
        target_names = [target_column]

    return {
        "dataset_name": "custom_uploaded_csv",
        "X": X,
        "y": y,
        "df": df_clean,
        "feature_names": feature_cols,
        "target_names": target_names,
        "task_type": task_type,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "description": f"Custom CSV Upload: {X.shape[0]} samples, {X.shape[1]} features, Target: '{target_column}'.",
        "has_ground_truth": False,
        "ground_truth_informative": [],
        "ground_truth_redundant": [],
        "ground_truth_noise": []
    }

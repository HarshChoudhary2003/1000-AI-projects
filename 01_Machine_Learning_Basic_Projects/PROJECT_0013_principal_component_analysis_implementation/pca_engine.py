"""
Project 013: Principal Component Analysis (PCA) - Analytical Engine
===================================================================
Production-grade computational core for PCA, including mathematical scratch
decompositions (Covariance Eigendecomposition vs SVD), scree plot metrics,
biplot vectors, image/sample reconstruction compression, and downstream ML benchmarking.

Part of the 1,000 AI Projects Implementation Library.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import time
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib


class CustomPCAScratch:
    """
    Pedagogical and mathematically rigorous PCA implementation from first principles.
    Computes both Covariance Matrix Eigendecomposition and Direct SVD.
    """
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.covariance_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def fit(self, X: np.ndarray) -> "CustomPCAScratch":
        n_samples, n_features = X.shape
        # 1. Mean centering
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Covariance matrix C = 1/(n - 1) * X_c^T X_c
        self.covariance_matrix_ = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # 3. Eigendecomposition of symmetric covariance matrix
        eigvals, eigvecs = np.linalg.eigh(self.covariance_matrix_)
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = eigvals[idx]
        self.eigenvectors_ = eigvecs[:, idx]

        # 4. Direct Singular Value Decomposition (SVD): X_c = U * S * V^T
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.singular_values_ = S

        # Scikit-learn convention: components are V^T (rows are eigenvectors)
        # Fix sign ambiguity so dominant entry of each component is positive
        V = Vt.T
        for i in range(V.shape[1]):
            max_abs_idx = np.argmax(np.abs(V[:, i]))
            if V[max_abs_idx, i] < 0:
                V[:, i] *= -1

        k = self.n_components if self.n_components is not None else n_features
        self.components_ = V[:, :k].T  # shape: (k, n_features)

        # Explained variance: lambda_i = S_i^2 / (n - 1)
        self.explained_variance_ = (S[:k] ** 2) / (n_samples - 1)
        total_var = np.sum((S ** 2) / (n_samples - 1))
        self.explained_variance_ratio_ = self.explained_variance_ / (total_var if total_var > 0 else 1.0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        return np.dot(X_proj, self.components_) + self.mean_


def fit_pca_pipeline(
    X: np.ndarray,
    n_components: Optional[Union[int, float]] = 2,
    scaling: str = "standard",
    pca_type: str = "standard",
    kernel: str = "rbf",
    gamma: Optional[float] = None
) -> Dict[str, Any]:
    """
    Fits a PCA transformation pipeline with optional scaling and variant selection.
    """
    # Preprocessing
    if scaling == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling == "robust":
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X.copy()

    n_samples, n_features = X.shape

    # Model instantiation
    if pca_type == "standard":
        model = PCA(n_components=n_components, svd_solver="auto")
        X_proj = model.fit_transform(X_scaled)
    elif pca_type == "incremental":
        k = int(n_components) if isinstance(n_components, (int, float)) and n_components >= 1 else 2
        model = IncrementalPCA(n_components=min(k, n_features), batch_size=max(10, n_samples // 5))
        X_proj = model.fit_transform(X_scaled)
    elif pca_type == "kernel":
        k = int(n_components) if isinstance(n_components, (int, float)) and n_components >= 1 else 2
        model = KernelPCA(n_components=k, kernel=kernel, gamma=gamma, fit_inverse_transform=True)
        X_proj = model.fit_transform(X_scaled)
    elif pca_type == "truncated_svd":
        k = int(n_components) if isinstance(n_components, (int, float)) and n_components >= 1 else 2
        model = TruncatedSVD(n_components=min(k, n_features - 1), algorithm="randomized")
        X_proj = model.fit_transform(X_scaled)
    else:
        raise ValueError(f"Unknown PCA type: {pca_type}")

    # Extract explained variance if available
    if hasattr(model, "explained_variance_ratio_"):
        exp_var_ratio = model.explained_variance_ratio_
        exp_var = model.explained_variance_
        cum_var_ratio = np.cumsum(exp_var_ratio)
    elif hasattr(model, "singular_values_"):
        exp_var = (model.singular_values_ ** 2) / (n_samples - 1)
        exp_var_ratio = exp_var / np.sum(exp_var)
        cum_var_ratio = np.cumsum(exp_var_ratio)
    else:
        # Approximate for KernelPCA by projected variance
        proj_vars = np.var(X_proj, axis=0)
        tot_proj_var = np.sum(proj_vars)
        exp_var_ratio = proj_vars / (tot_proj_var if tot_proj_var > 0 else 1.0)
        exp_var = proj_vars
        cum_var_ratio = np.cumsum(exp_var_ratio)

    return {
        "scaler": scaler,
        "model": model,
        "X_scaled": X_scaled,
        "X_proj": X_proj,
        "explained_variance_": exp_var,
        "explained_variance_ratio_": exp_var_ratio,
        "cumulative_variance_ratio_": cum_var_ratio,
        "n_components_actual": X_proj.shape[1],
        "pca_type": pca_type,
        "scaling": scaling
    }


def compute_scree_analysis(X_scaled: np.ndarray, max_components: Optional[int] = None) -> Dict[str, Any]:
    """
    Performs full dimensional decomposition to evaluate Scree plots, Kaiser criterion, and variance cutoffs.
    """
    n_samples, n_features = X_scaled.shape
    max_k = min(n_samples, n_features)
    if max_components is not None:
        max_k = min(max_k, max_components)

    full_pca = PCA(n_components=max_k)
    full_pca.fit(X_scaled)

    eigenvalues = full_pca.explained_variance_
    var_ratio = full_pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    # Kaiser Criterion: Eigenvalues > 1.0 (on standardized data)
    kaiser_k = int(np.sum(eigenvalues >= 1.0))
    if kaiser_k == 0:
        kaiser_k = 1

    # Variance threshold targets
    k_80 = int(np.argmax(cum_var_ratio >= 0.80) + 1) if np.any(cum_var_ratio >= 0.80) else max_k
    k_90 = int(np.argmax(cum_var_ratio >= 0.90) + 1) if np.any(cum_var_ratio >= 0.90) else max_k
    k_95 = int(np.argmax(cum_var_ratio >= 0.95) + 1) if np.any(cum_var_ratio >= 0.95) else max_k
    k_99 = int(np.argmax(cum_var_ratio >= 0.99) + 1) if np.any(cum_var_ratio >= 0.99) else max_k

    # Scree Elbow Detection via maximum distance to chord connecting point 1 and max_k
    n_pts = len(var_ratio)
    if n_pts >= 3:
        p1 = np.array([0, var_ratio[0]])
        p2 = np.array([n_pts - 1, var_ratio[-1]])
        vec = p2 - p1
        vec_norm = vec / np.linalg.norm(vec)
        distances = []
        for i in range(n_pts):
            pt = np.array([i, var_ratio[i]])
            proj = p1 + np.dot(pt - p1, vec_norm) * vec_norm
            dist = np.linalg.norm(pt - proj)
            distances.append(dist)
        elbow_k = int(np.argmax(distances) + 1)
    else:
        elbow_k = 1

    component_indices = np.arange(1, len(var_ratio) + 1)
    scree_df = pd.DataFrame({
        "Component": [f"PC{i}" for i in component_indices],
        "Component_Index": component_indices,
        "Eigenvalue": eigenvalues,
        "Explained_Variance_Ratio": var_ratio,
        "Explained_Variance_Pct": var_ratio * 100,
        "Cumulative_Variance_Ratio": cum_var_ratio,
        "Cumulative_Variance_Pct": cum_var_ratio * 100
    })

    return {
        "scree_df": scree_df,
        "eigenvalues": eigenvalues,
        "var_ratio": var_ratio,
        "cum_var_ratio": cum_var_ratio,
        "kaiser_k": kaiser_k,
        "elbow_k": elbow_k,
        "k_80": k_80,
        "k_90": k_90,
        "k_95": k_95,
        "k_99": k_99,
        "max_k": max_k
    }


def compute_loadings_and_biplot(
    model: Any,
    feature_names: List[str],
    top_n_features: int = 10
) -> Dict[str, Any]:
    """
    Computes component loadings matrix, feature contributions, and scaled biplot vectors.
    """
    if not hasattr(model, "components_"):
        return {"error": "Selected model does not expose explicit linear components_ matrix."}

    components = model.components_  # shape: (n_components, n_features)
    n_comps, n_feats = components.shape

    if hasattr(model, "explained_variance_"):
        # Loadings = eigenvectors * sqrt(eigenvalues)
        loadings = components.T * np.sqrt(model.explained_variance_)
    else:
        loadings = components.T

    # Loadings DataFrame: features as rows, PC1..PCk as columns
    cols = [f"PC{i+1}" for i in range(n_comps)]
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=cols)

    # Squared loadings / contributions
    sq_loadings = loadings ** 2
    feature_contributions = sq_loadings / np.sum(sq_loadings, axis=0, keepdims=True)
    contrib_df = pd.DataFrame(feature_contributions * 100, index=feature_names, columns=[f"{c}_Contrib_%" for c in cols])

    # Top features for PC1 and PC2
    top_pc1 = loadings_df["PC1"].abs().sort_values(ascending=False).head(top_n_features)
    top_pc2 = loadings_df["PC2"].abs().sort_values(ascending=False).head(top_n_features) if n_comps > 1 else top_pc1

    # Scaled vectors for biplot display
    # Scale feature arrows to fit comfortably within the observation scatter range
    biplot_vectors = []
    if n_comps >= 2:
        for idx, feat in enumerate(feature_names):
            biplot_vectors.append({
                "feature": feat,
                "vx": float(components[0, idx]),
                "vy": float(components[1, idx]),
                "loading_pc1": float(loadings[idx, 0]),
                "loading_pc2": float(loadings[idx, 1]),
                "length": float(np.sqrt(components[0, idx] ** 2 + components[1, idx] ** 2))
            })
        biplot_vectors = sorted(biplot_vectors, key=lambda x: x["length"], reverse=True)

    return {
        "loadings_df": loadings_df,
        "contrib_df": contrib_df,
        "top_pc1": top_pc1,
        "top_pc2": top_pc2,
        "biplot_vectors": biplot_vectors
    }


def compute_reconstruction_analysis(
    X_raw: np.ndarray,
    scaler: Optional[Any],
    k_components: int = 2,
    sample_index: int = 0
) -> Dict[str, Any]:
    """
    Computes compression reconstruction and pixel/feature residual error maps.
    """
    n_samples, n_features = X_raw.shape
    X_scaled = scaler.transform(X_raw) if scaler is not None else X_raw.copy()

    k = min(max(1, k_components), n_features)
    pca = PCA(n_components=k)
    X_proj = pca.fit_transform(X_scaled)
    X_rec_scaled = pca.inverse_transform(X_proj)

    X_reconstructed = scaler.inverse_transform(X_rec_scaled) if scaler is not None else X_rec_scaled

    # Mean squared error
    mse_total = float(np.mean((X_raw - X_reconstructed) ** 2))
    sample_raw = X_raw[sample_index]
    sample_rec = X_reconstructed[sample_index]
    sample_diff = sample_raw - sample_rec
    sample_mse = float(np.mean(sample_diff ** 2))

    return {
        "k": k,
        "mse_total": mse_total,
        "sample_raw": sample_raw,
        "sample_rec": sample_rec,
        "sample_diff": sample_diff,
        "sample_mse": sample_mse,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))
    }


def compute_reconstruction_curve(
    X_raw: np.ndarray,
    scaler: Optional[Any],
    max_k: int = 30,
    step: int = 1
) -> pd.DataFrame:
    """
    Evaluates reconstruction MSE vs number of components k across the dataset.
    """
    n_samples, n_features = X_raw.shape
    X_scaled = scaler.transform(X_raw) if scaler is not None else X_raw.copy()

    k_values = list(range(1, min(n_features + 1, max_k + 1), step))
    if k_values[-1] != min(n_features, max_k):
        k_values.append(min(n_features, max_k))

    records = []
    for k in k_values:
        pca = PCA(n_components=k)
        X_proj = pca.fit_transform(X_scaled)
        X_rec_scaled = pca.inverse_transform(X_proj)
        X_rec = scaler.inverse_transform(X_rec_scaled) if scaler is not None else X_rec_scaled
        
        mse = float(np.mean((X_raw - X_rec) ** 2))
        var_retained = float(np.sum(pca.explained_variance_ratio_)) * 100
        compression_ratio = float((1.0 - (k / n_features)) * 100)

        records.append({
            "k_components": k,
            "reconstruction_mse": mse,
            "variance_retained_pct": var_retained,
            "compression_ratio_pct": compression_ratio
        })

    return pd.DataFrame(records)


def run_downstream_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    scaling: str = "standard",
    classifier_name: str = "logistic_regression",
    k_candidates: Optional[List[int]] = None,
    test_size: float = 0.25,
    seed: int = 42
) -> pd.DataFrame:
    """
    Evaluates the downstream ML performance (Accuracy, F1, training latency)
    comparing original high-dimensional features against varying PCA projections.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )

    if scaling == "standard":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

    n_features = X.shape[1]
    if k_candidates is None:
        candidates = [1, 2, 3, 5, 10, 20, 30]
        k_candidates = [k for k in candidates if k < n_features]

    results = []

    def get_clf():
        if classifier_name == "logistic_regression":
            return LogisticRegression(max_iter=500, random_state=seed)
        return RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)

    # 1. Baseline: Raw Features
    clf_raw = get_clf()
    t0 = time.perf_counter()
    clf_raw.fit(X_train_scaled, y_train)
    t_train_raw = (time.perf_counter() - t0) * 1000

    preds_raw = clf_raw.predict(X_test_scaled)
    acc_raw = accuracy_score(y_test, preds_raw)
    f1_raw = f1_score(y_test, preds_raw, average="weighted")

    results.append({
        "Model_Type": "Raw High-Dim Baseline",
        "n_features": n_features,
        "variance_retained_pct": 100.0,
        "test_accuracy": acc_raw,
        "test_f1": f1_raw,
        "train_latency_ms": t_train_raw,
        "dimension_savings_pct": 0.0
    })

    # 2. PCA with varying k
    for k in k_candidates:
        pca = PCA(n_components=k)
        X_tr_proj = pca.fit_transform(X_train_scaled)
        X_te_proj = pca.transform(X_test_scaled)

        clf = get_clf()
        t0 = time.perf_counter()
        clf.fit(X_tr_proj, y_train)
        t_train_pca = (time.perf_counter() - t0) * 1000

        preds = clf.predict(X_te_proj)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        var_retained = float(np.sum(pca.explained_variance_ratio_)) * 100
        savings = float((1 - (k / n_features)) * 100)

        results.append({
            "Model_Type": f"PCA (k={k})",
            "n_features": k,
            "variance_retained_pct": var_retained,
            "test_accuracy": acc,
            "test_f1": f1,
            "train_latency_ms": t_train_pca,
            "dimension_savings_pct": savings
        })

    return pd.DataFrame(results)


def compare_scratch_vs_sklearn(X: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
    """
    Performs rigorous numerical validation comparing CustomPCAScratch vs Scikit-Learn PCA.
    """
    scaler = StandardScaler(with_std=True)
    X_scaled = scaler.fit_transform(X)

    scratch = CustomPCAScratch(n_components=n_components)
    X_scratch = scratch.fit_transform(X_scaled)

    sk_pca = PCA(n_components=n_components)
    X_sk = sk_pca.fit_transform(X_scaled)

    # Align signs for identical basis vectors
    for c in range(n_components):
        if np.dot(scratch.components_[c], sk_pca.components_[c]) < 0:
            X_scratch[:, c] *= -1

    diff_projection = np.max(np.abs(X_scratch - X_sk))
    diff_variance_ratio = np.max(np.abs(scratch.explained_variance_ratio_ - sk_pca.explained_variance_ratio_))
    diff_singular_vals = np.max(np.abs(scratch.singular_values_[:n_components] - sk_pca.singular_values_[:n_components]))

    return {
        "diff_projection_max": float(diff_projection),
        "diff_variance_ratio_max": float(diff_variance_ratio),
        "diff_singular_vals_max": float(diff_singular_vals),
        "scratch_eigenvalues": scratch.eigenvalues_[:n_components].tolist(),
        "sklearn_explained_variance": sk_pca.explained_variance_.tolist(),
        "is_numerically_identical": bool(diff_projection < 1e-10)
    }


def save_pca_pipeline(pipeline_bundle: Dict[str, Any], filepath: str) -> None:
    """Save fitted PCA model bundle."""
    joblib.dump(pipeline_bundle, filepath)


def load_pca_pipeline(filepath: str) -> Dict[str, Any]:
    """Load persisted PCA model bundle."""
    return joblib.load(filepath)

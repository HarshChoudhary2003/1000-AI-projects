"""
Project 008: Advanced Support Vector Machine (SVM) Classification Engine
========================================================================
Modular, production-grade SVM pipeline offering multi-kernel optimization,
support vector diagnostics, margin geometry analytics, cross-validation,
grid search hyperparameter tuning, and real-time clinical inference.
"""

import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)


def prepare_dataset_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.25,
    scale_method: str = "standard",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Split and preprocess feature matrix and labels with feature scaling.
    Standardization is essential for SVM because large scale features would dominate
    the distance calculations in the kernel space.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale_method == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    elif scale_method == "minmax":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        scaler = None
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_col": target_col
    }


def train_svm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: Any = "scale",
    degree: int = 3,
    coef0: float = 0.0,
    class_weight: Optional[str] = None,
    probability: bool = True,
    random_state: int = 42
) -> Tuple[SVC, float]:
    """
    Train a Support Vector Classifier with exact hyperparameter telemetry.
    Returns the trained model and execution latency in milliseconds.
    """
    start_time = time.perf_counter()

    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state
    )
    svm.fit(X_train, y_train)

    train_latency_ms = (time.perf_counter() - start_time) * 1000.0
    return svm, train_latency_ms


def extract_support_vector_diagnostics(
    model: SVC,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Dict[str, Any]:
    """
    Deep diagnostic inspection of the trained SVM's dual representation,
    margin boundaries, support vectors, and slack violations.
    """
    n_samples = len(X_train)
    n_sv = len(model.support_)
    sv_ratio = (n_sv / n_samples) * 100.0 if n_samples > 0 else 0.0

    diag: Dict[str, Any] = {
        "n_support_vectors": n_sv,
        "support_vector_ratio_pct": sv_ratio,
        "n_support_per_class": model.n_support_.tolist(),
        "support_indices": model.support_,
        "support_vectors": model.support_vectors_,
        "dual_coef": model.dual_coef_,
        "intercept": model.intercept_.tolist(),
        "kernel": model.kernel
    }

    # For linear kernel, analyze the separating hyperplane w and margin width 2 / ||w||
    if model.kernel == "linear" and hasattr(model, "coef_"):
        w = model.coef_[0]
        norm_w = float(np.linalg.norm(w))
        margin_width = (2.0 / norm_w) if norm_w > 0 else 0.0
        diag["w_norm"] = norm_w
        diag["margin_width"] = margin_width
        diag["weights"] = w.tolist()

    # Calculate margin violations (points inside the margin or misclassified)
    # Decision function f(x): if y_i * f(x_i) < 1, point is within margin or misclassified
    try:
        dec_train = model.decision_function(X_train)
        if dec_train.ndim == 1:
            # Binary classification: y in {0, 1} mapped to {-1, +1}
            y_binary = np.where(y_train == 0, -1, 1)
            margin_prod = y_binary * dec_train
            margin_violations = np.sum(margin_prod < 1.0)
            slack_positive = np.sum(margin_prod < 0.0)  # outright misclassified
            diag["margin_violations_count"] = int(margin_violations)
            diag["margin_violations_pct"] = float((margin_violations / n_samples) * 100.0)
            diag["slack_misclassifications"] = int(slack_positive)
    except Exception:
        pass

    return diag


def evaluate_svm_model(
    model: SVC,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for classification performance.
    """
    y_pred = model.predict(X_test)
    is_binary = len(np.unique(y_test)) <= 2

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    mcc = float(matthews_corrcoef(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    results: Dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "is_binary": is_binary
    }

    # Specificity for binary classification
    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        results["specificity"] = specificity
        results["tn"] = int(tn)
        results["fp"] = int(fp)
        results["fn"] = int(fn)
        results["tp"] = int(tp)

    # Probabilities and Decision Function
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            results["y_proba"] = y_proba
            if is_binary:
                roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                pr_auc = float(average_precision_score(y_test, y_proba[:, 1]))
                results["roc_auc"] = roc_auc
                results["pr_auc"] = pr_auc
            else:
                roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr"))
                results["roc_auc"] = roc_auc
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            results["decision_function"] = model.decision_function(X_test)
        except Exception:
            pass

    results["classification_report"] = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    return results


def run_cross_validation(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Perform stratified K-Fold cross validation to assess generalization reliability.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    return {
        "fold_scores": scores.tolist(),
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "min_accuracy": float(np.min(scores)),
        "max_accuracy": float(np.max(scores)),
        "n_folds": cv_folds
    }


def compare_kernel_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Train and benchmark the primary 4 SVM kernels side-by-side:
    Linear, Radial Basis Function (RBF), Polynomial (deg 3), and Sigmoid.
    """
    kernels = [
        ("Linear", {"kernel": "linear", "C": C}),
        ("RBF (Gaussian)", {"kernel": "rbf", "C": C, "gamma": "scale"}),
        ("Polynomial (d=3)", {"kernel": "poly", "C": C, "degree": 3, "gamma": "scale"}),
        ("Sigmoid", {"kernel": "sigmoid", "C": C, "gamma": "scale"})
    ]

    records = []
    is_binary = len(np.unique(y_test)) <= 2

    for name, params in kernels:
        t0 = time.perf_counter()
        clf = SVC(**params, probability=True, random_state=random_state)
        clf.fit(X_train, y_train)
        latency = (time.perf_counter() - t0) * 1000.0

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        n_sv = len(clf.support_)
        sv_ratio = (n_sv / len(X_train)) * 100.0

        roc = np.nan
        if is_binary and hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, proba)
            except Exception:
                pass

        records.append({
            "Kernel": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": roc,
            "Support Vectors": n_sv,
            "SV Ratio (%)": sv_ratio,
            "Train Latency (ms)": latency
        })

    return pd.DataFrame(records)


def run_grid_search_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    c_values: Optional[List[float]] = None,
    gamma_values: Optional[List[float]] = None,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Systematic grid search across C and gamma hyperparameter space with cross-validation.
    Returns best configuration and 2D heatmap matrix.
    """
    if c_values is None:
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    if gamma_values is None:
        gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    grid_matrix = np.zeros((len(c_values), len(gamma_values)))
    best_score = -1.0
    best_params = {"C": 1.0, "gamma": 0.1}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for i, c in enumerate(c_values):
        for j, g in enumerate(gamma_values):
            clf = SVC(kernel=kernel, C=c, gamma=g, random_state=42)
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
            mean_score = float(np.mean(scores))
            grid_matrix[i, j] = mean_score

            if mean_score > best_score:
                best_score = mean_score
                best_params = {"C": c, "gamma": g}

    return {
        "grid_matrix": grid_matrix,
        "c_values": c_values,
        "gamma_values": gamma_values,
        "best_score": best_score,
        "best_params": best_params
    }


def compute_2d_decision_mesh(
    model: SVC,
    X_2d: np.ndarray,
    resolution: int = 200,
    margin_padding: float = 0.75
) -> Dict[str, Any]:
    """
    Construct high-resolution 2D evaluation grid for continuous contour visualization
    of separating hyperplane and margin boundary contours:
    f(x) = 0 (Separating Hyperplane)
    f(x) = +1 (Positive Margin Boundary)
    f(x) = -1 (Negative Margin Boundary)
    """
    x_min, x_max = X_2d[:, 0].min() - margin_padding, X_2d[:, 0].max() + margin_padding
    y_min, y_max = X_2d[:, 1].min() - margin_padding, X_2d[:, 1].max() + margin_padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict class labels
    Z_label = model.predict(grid_points).reshape(xx.shape)

    # Continuous decision function values
    Z_dec = None
    if hasattr(model, "decision_function"):
        try:
            raw_dec = model.decision_function(grid_points)
            if raw_dec.ndim == 1:
                Z_dec = raw_dec.reshape(xx.shape)
            else:
                # Multi-class: use maximum confidence margin
                Z_dec = np.max(raw_dec, axis=1).reshape(xx.shape)
        except Exception:
            pass

    return {
        "xx": xx,
        "yy": yy,
        "Z_label": Z_label,
        "Z_dec": Z_dec,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max)
    }


def predict_single_instance(
    model: SVC,
    scaler: Optional[Any],
    raw_feature_dict: Dict[str, float],
    feature_order: List[str],
    target_names: List[str]
) -> Dict[str, Any]:
    """
    Real-time inference pipeline for single query input.
    Computes predicted class, probability distribution, and geometric distance to decision boundary.
    """
    feature_values = [raw_feature_dict.get(col, 0.0) for col in feature_order]
    x_arr = np.array(feature_values).reshape(1, -1)

    if scaler is not None:
        x_scaled = scaler.transform(x_arr)
    else:
        x_scaled = x_arr

    pred_idx = int(model.predict(x_scaled)[0])
    pred_label = target_names[pred_idx] if pred_idx < len(target_names) else str(pred_idx)

    # Probabilities
    proba_dict = {}
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(x_scaled)[0]
            for i, p in enumerate(probabilities):
                lbl = target_names[i] if i < len(target_names) else f"Class {i}"
                proba_dict[lbl] = float(p)
        except Exception:
            pass

    # Decision function distance f(x)
    dec_val = None
    if hasattr(model, "decision_function"):
        try:
            df_val = model.decision_function(x_scaled)[0]
            dec_val = float(df_val) if isinstance(df_val, (int, float, np.number)) else float(df_val[0])
        except Exception:
            pass

    return {
        "prediction_index": pred_idx,
        "prediction_label": pred_label,
        "probabilities": proba_dict,
        "decision_value": dec_val,
        "scaled_features": x_scaled[0].tolist()
    }

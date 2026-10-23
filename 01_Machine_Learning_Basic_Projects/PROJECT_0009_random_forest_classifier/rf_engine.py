"""
Project 009: Random Forest Classifier Engine & Ensemble Analytics
==================================================================
Modular, production-grade Random Forest pipeline featuring Out-of-Bag (OOB)
convergence profiling, Mean Decrease in Impurity (MDI) feature importances,
forest structural diagnostics, tree voting consensus telemetry, and 2D decision surfaces.
"""

import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
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
    scale_method: str = "none",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Split feature matrix and labels with optional scaling.
    While Random Forests are scale-invariant, optional scaling assists with
    comparisons and 2D grid stability.
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


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Any = "sqrt",
    criterion: str = "gini",
    bootstrap: bool = True,
    oob_score: bool = True,
    class_weight: Optional[str] = None,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, float]:
    """
    Train an ensemble Random Forest with exact hyperparameter telemetry.
    Returns the trained classifier and execution latency in milliseconds.
    """
    # OOB score requires bootstrap=True
    actual_oob = oob_score if bootstrap else False

    start_time = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        bootstrap=bootstrap,
        oob_score=actual_oob,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_latency_ms = (time.perf_counter() - start_time) * 1000.0

    return rf, train_latency_ms


def evaluate_rf_model(
    model: RandomForestClassifier,
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

    # Out-of-Bag (OOB) score if available
    if hasattr(model, "oob_score_") and model.oob_score_:
        results["oob_score"] = float(model.oob_score_)
        results["oob_error"] = float(1.0 - model.oob_score_)

    # Specificity for binary classification
    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        results["specificity"] = specificity
        results["tn"] = int(tn)
        results["fp"] = int(fp)
        results["fn"] = int(fn)
        results["tp"] = int(tp)

    # Probabilities & Curves
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

    results["classification_report"] = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    return results


def extract_feature_importances(
    model: RandomForestClassifier,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract Mean Decrease in Impurity (Gini importance) along with
    inter-tree standard deviations and cumulative percentage.
    """
    importances = model.feature_importances_
    tree_importances = np.array([tree.feature_importances_ for tree in model.estimators_])
    std_importances = np.std(tree_importances, axis=0)

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
        "Std_Dev": std_importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    df_imp["Cumulative"] = df_imp["Importance"].cumsum()
    df_imp["Importance_Pct"] = df_imp["Importance"] * 100.0
    return df_imp


def compute_oob_convergence(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_trees: int = 150,
    step: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute Out-of-Bag (OOB) error rate across an expanding number of trees
    to verify convergence and demonstrate asymptotic stabilization.
    """
    n_trees_range = list(range(10, max_trees + 1, step))
    records = []

    # Initialize warm start Random Forest to train incrementally
    rf = RandomForestClassifier(
        n_estimators=10,
        warm_start=True,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )

    for n_trees in n_trees_range:
        rf.set_params(n_estimators=n_trees)
        rf.fit(X_train, y_train)
        oob_err = 1.0 - rf.oob_score_
        records.append({
            "n_estimators": n_trees,
            "oob_score": rf.oob_score_,
            "oob_error": oob_err
        })

    return pd.DataFrame(records)


def extract_forest_diagnostics(model: RandomForestClassifier) -> Dict[str, Any]:
    """
    Structural statistics across all individual decision trees in the forest.
    """
    depths = [tree.get_depth() for tree in model.estimators_]
    leaf_counts = [tree.get_n_leaves() for tree in model.estimators_]
    node_counts = [tree.tree_.node_count for tree in model.estimators_]

    return {
        "n_trees": len(model.estimators_),
        "avg_depth": float(np.mean(depths)),
        "max_depth": int(np.max(depths)),
        "min_depth": int(np.min(depths)),
        "avg_leaves": float(np.mean(leaf_counts)),
        "avg_nodes": float(np.mean(node_counts)),
        "total_nodes_forest": int(np.sum(node_counts))
    }


def inspect_tree_votes(
    model: RandomForestClassifier,
    scaler: Optional[Any],
    raw_feature_dict: Dict[str, float],
    feature_order: List[str],
    target_names: List[str]
) -> Dict[str, Any]:
    """
    Extract individual predictions and voting consensus from every tree
    in the forest for a single query instance.
    """
    feature_values = [raw_feature_dict.get(col, 0.0) for col in feature_order]
    x_arr = np.array(feature_values).reshape(1, -1)

    if scaler is not None:
        x_scaled = scaler.transform(x_arr)
    else:
        x_scaled = x_arr

    # Predict ensemble class & probabilities
    pred_idx = int(model.predict(x_scaled)[0])
    pred_label = target_names[pred_idx] if pred_idx < len(target_names) else str(pred_idx)
    probabilities = model.predict_proba(x_scaled)[0]

    proba_dict = {
        (target_names[i] if i < len(target_names) else f"Class {i}"): float(p)
        for i, p in enumerate(probabilities)
    }

    # Collect individual tree votes
    tree_votes = []
    vote_counts = {t_name: 0 for t_name in target_names}

    for tree in model.estimators_:
        t_pred = int(tree.predict(x_scaled)[0])
        t_name = target_names[t_pred] if t_pred < len(target_names) else str(t_pred)
        tree_votes.append(t_name)
        vote_counts[t_name] = vote_counts.get(t_name, 0) + 1

    total_trees = len(model.estimators_)
    vote_percentages = {k: (v / total_trees) * 100.0 for k, v in vote_counts.items()}
    consensus_margin = max(vote_percentages.values())

    return {
        "prediction_index": pred_idx,
        "prediction_label": pred_label,
        "probabilities": proba_dict,
        "vote_counts": vote_counts,
        "vote_percentages": vote_percentages,
        "consensus_confidence": consensus_margin,
        "total_trees": total_trees
    }


def compute_2d_ensemble_mesh(
    model: RandomForestClassifier,
    X_2d: np.ndarray,
    resolution: int = 180,
    margin_padding: float = 0.75
) -> Dict[str, Any]:
    """
    Generate high-density 2D evaluation grid for continuous probability
    voting consensus and step-like decision partition boundaries.
    """
    x_min, x_max = X_2d[:, 0].min() - margin_padding, X_2d[:, 0].max() + margin_padding
    y_min, y_max = X_2d[:, 1].min() - margin_padding, X_2d[:, 1].max() + margin_padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict class labels and maximum probability
    Z_label = model.predict(grid_points).reshape(xx.shape)
    probas = model.predict_proba(grid_points)
    
    # Probability confidence surface: maximum class probability
    Z_conf = np.max(probas, axis=1).reshape(xx.shape)

    return {
        "xx": xx,
        "yy": yy,
        "Z_label": Z_label,
        "Z_conf": Z_conf,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max)
    }


def run_cross_validation(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Perform stratified K-Fold cross validation to assess ensemble generalization.
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

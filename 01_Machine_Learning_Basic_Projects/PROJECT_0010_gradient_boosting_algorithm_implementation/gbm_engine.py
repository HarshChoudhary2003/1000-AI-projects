"""
Project 010: Gradient Boosting Machine (GBM) Engine
===================================================
Robust core engine executing Gradient Tree Boosting workflows:
- Sequential tree boosting with shrinkage (learning rate)
- Staged residual deviance tracking (train vs validation loss convergence)
- Early stopping detection and overfitting boundary analysis
- Mean Decrease in Impurity (MDI) feature ranking
- Individual stage tree inspection
- 2D continuous probability decision surface generation
- Stratified cross-validation and real-time live inference telemetry
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)


def prepare_dataset_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing partitions using stratification.
    """
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> GradientBoostingClassifier:
    """
    Instantiate and fit a GradientBoostingClassifier with specified hyperparameters.
    """
    model = GradientBoostingClassifier(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def compute_staged_metrics(
    model: GradientBoostingClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Compute stage-by-stage (tree-by-tree) metrics for both training and testing sets.
    Tracks deviance (loss function) convergence and classification accuracy over boosting iterations.
    Identifies the optimal stopping stage where test loss is minimized.
    """
    n_stages = model.n_estimators_
    iterations = list(range(1, n_stages + 1))

    # Scikit-learn models store train_score_ (deviance of train set at each stage)
    train_deviance = [float(v) for v in model.train_score_]

    test_deviance: List[float] = []
    test_accuracy: List[float] = []

    classes = model.classes_

    # Iterate through staged predictions on test set
    for proba in model.staged_predict_proba(X_test):
        # Deviance is equivalent to log loss (or multiclass log loss)
        loss = log_loss(y_test, proba, labels=classes)
        test_deviance.append(float(loss))
        
        preds = classes[np.argmax(proba, axis=1)]
        acc = accuracy_score(y_test, preds)
        test_accuracy.append(float(acc))

    best_stage_idx = int(np.argmin(test_deviance))
    best_stage = best_stage_idx + 1
    min_test_deviance = test_deviance[best_stage_idx]
    best_stage_acc = test_accuracy[best_stage_idx]

    return {
        "iterations": iterations,
        "train_deviance": train_deviance,
        "test_deviance": test_deviance,
        "test_accuracy": test_accuracy,
        "best_stage": best_stage,
        "min_test_deviance": float(min_test_deviance),
        "best_stage_accuracy": float(best_stage_acc)
    }


def evaluate_model(
    model: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for model evaluation.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    n_classes = len(model.classes_)
    is_binary = n_classes == 2

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="binary" if is_binary else "weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="binary" if is_binary else "weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="binary" if is_binary else "weighted", zero_division=0))
    loss = float(log_loss(y_test, y_proba, labels=model.classes_))

    if is_binary:
        auc = float(roc_auc_score(y_test, y_proba[:, 1]))
    else:
        try:
            auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            auc = 0.0

    cm = confusion_matrix(y_test, y_pred)
    class_labels = target_names if target_names is not None else [f"Class {c}" for c in model.classes_]

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_labels,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "log_loss": loss,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "class_labels": class_labels,
        "is_binary": is_binary
    }


def extract_feature_importances(
    model: GradientBoostingClassifier,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract and rank Mean Decrease in Impurity (MDI / Friedman MSE improvement) feature importances.
    """
    raw_importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": raw_importances,
        "Percentage": (raw_importances / (np.sum(raw_importances) + 1e-12)) * 100.0
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    df_imp["Cumulative"] = df_imp["Percentage"].cumsum()
    return df_imp


def extract_forest_diagnostics(
    model: GradientBoostingClassifier
) -> Dict[str, Any]:
    """
    Extract structural metrics and architectural characteristics of the boosted ensemble.
    """
    # model.estimators_ is an array of shape (n_estimators, n_classes) or (n_estimators, 1)
    trees = model.estimators_.ravel()
    depths = [t.tree_.max_depth for t in trees]
    node_counts = [t.tree_.node_count for t in trees]
    leaf_counts = [t.tree_.n_leaves for t in trees]

    return {
        "n_estimators": int(model.n_estimators_),
        "learning_rate": float(model.learning_rate),
        "max_depth_configured": int(model.max_depth),
        "subsample": float(model.subsample),
        "total_trees": len(trees),
        "avg_depth": float(np.mean(depths)),
        "max_depth_observed": int(np.max(depths)),
        "avg_leaves": float(np.mean(leaf_counts)),
        "total_nodes": int(np.sum(node_counts)),
        "loss_function": str(model.loss)
    }


def inspect_stage_tree(
    model: GradientBoostingClassifier,
    stage_idx: int,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Inspect the underlying regression tree fitted at a specific boosting stage.
    """
    stage_idx = max(0, min(stage_idx, model.n_estimators_ - 1))
    estimator_row = model.estimators_[stage_idx]
    
    # Grab primary tree (first class tree in binary, or first class in multi-class)
    tree_regressor = estimator_row[0]
    tree_obj = tree_regressor.tree_

    # Count splits by feature in this tree
    split_features = tree_obj.feature[tree_obj.feature >= 0]
    feature_usage = {}
    for f_idx in split_features:
        name = feature_names[f_idx] if f_idx < len(feature_names) else f"Feature {f_idx}"
        feature_usage[name] = feature_usage.get(name, 0) + 1

    return {
        "stage": stage_idx + 1,
        "max_depth": int(tree_obj.max_depth),
        "node_count": int(tree_obj.node_count),
        "n_leaves": int(tree_obj.n_leaves),
        "feature_split_counts": feature_usage,
        "leaf_values_min": float(np.min(tree_obj.value)),
        "leaf_values_max": float(np.max(tree_obj.value))
    }


def compute_2d_decision_mesh(
    model: GradientBoostingClassifier,
    X: np.ndarray,
    resolution: int = 120
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D mesh grid for continuous decision boundary and probability surface visualization.
    Only applicable when X has 2 features.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(mesh_points).reshape(xx.shape)
    probas = model.predict_proba(mesh_points)

    if probas.shape[1] == 2:
        prob_surface = probas[:, 1].reshape(xx.shape)
    else:
        # For multiclass, confidence is the max probability
        prob_surface = np.max(probas, axis=1).reshape(xx.shape)

    return xx, yy, preds, prob_surface


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform Stratified K-Fold Cross-Validation on Gradient Boosting.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    # Include roc_auc if binary or well-supported
    is_binary = len(np.unique(y)) == 2
    if is_binary:
        scoring["roc_auc"] = "roc_auc"

    clf = GradientBoostingClassifier(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        random_state=random_state
    )

    results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False)

    summary = {
        "folds": cv_folds,
        "accuracy_mean": float(np.mean(results["test_accuracy"])),
        "accuracy_std": float(np.std(results["test_accuracy"])),
        "f1_mean": float(np.mean(results["test_f1"])),
        "f1_std": float(np.std(results["test_f1"])),
        "precision_mean": float(np.mean(results["test_precision"])),
        "precision_std": float(np.std(results["test_precision"])),
        "recall_mean": float(np.mean(results["test_recall"])),
        "recall_std": float(np.std(results["test_recall"])),
        "fold_accuracies": [float(v) for v in results["test_accuracy"]]
    }

    if "test_roc_auc" in results:
        summary["roc_auc_mean"] = float(np.mean(results["test_roc_auc"]))
        summary["roc_auc_std"] = float(np.std(results["test_roc_auc"]))

    return summary


def predict_single_sample(
    model: GradientBoostingClassifier,
    feature_vector: List[float],
    feature_names: List[str],
    target_names: List[str]
) -> Dict[str, Any]:
    """
    Perform real-time live inference on a single instance and compute confidence breakdown.
    """
    X_single = np.array(feature_vector).reshape(1, -1)
    predicted_idx = int(model.predict(X_single)[0])
    probabilities = model.predict_proba(X_single)[0]

    predicted_label = target_names[predicted_idx] if predicted_idx < len(target_names) else f"Class {predicted_idx}"
    confidence = float(probabilities[predicted_idx])

    prob_dist = {
        target_names[i] if i < len(target_names) else f"Class {i}": float(probabilities[i])
        for i in range(len(probabilities))
    }

    return {
        "predicted_index": predicted_idx,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100.0, 2),
        "probability_distribution": prob_dist
    }

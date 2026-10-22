"""
Project 007: K-Nearest Neighbors (KNN) Implementation & Spatial Benchmark
=========================================================================
An industrial-strength K-Nearest Neighbors implementation and spatial geometry
pipeline featuring real-world high-dimensional classification (Wisconsin Diagnostic
Breast Cancer), multi-metric distance benchmarking, scaling sensitivity analysis,
and optimal K selection via cross-validation.

Part of the 1,000 AI Projects Implementation Library.
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

try:
    from dataset import load_real_dataset, generate_2d_manifold
except ImportError:
    from .dataset import load_real_dataset, generate_2d_manifold


def load_dataset(dataset_name: str = "breast_cancer") -> Tuple[pd.DataFrame, List[str], str, List[str]]:
    """Load benchmark dataset (Wisconsin Breast Cancer or Wine chemical cultivar)."""
    return load_real_dataset(dataset_name)


def train_knn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    algorithm: str = "auto",
    p: float = 2.0
) -> KNeighborsClassifier:
    """Train a scikit-learn K-Nearest Neighbors classifier."""
    kwargs = {
        'n_neighbors': n_neighbors,
        'metric': metric,
        'weights': weights,
        'algorithm': algorithm
    }
    if metric == "minkowski":
        kwargs['p'] = p

    model = KNeighborsClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_knn(
    model: KNeighborsClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str] = None
) -> Dict[str, Any]:
    """Compute comprehensive holdout evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": rep,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def find_optimal_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_range: range = range(1, 26, 2),
    cv: int = 5
) -> Tuple[int, float, List[Dict[str, Any]]]:
    """Perform 5-fold cross-validation to discover the optimal K hyperparameter."""
    records = []
    best_k, best_score = 1, 0.0

    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        mean_score = float(scores.mean())
        std_score = float(scores.std())

        records.append({
            "K": k,
            "CV_Accuracy": mean_score,
            "CV_Std": std_score
        })

        if mean_score > best_score:
            best_k = k
            best_score = mean_score

    return best_k, best_score, records


def benchmark_distance_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5
) -> pd.DataFrame:
    """Benchmark classifier performance across diverse geometric distance metrics."""
    metrics_to_test = [
        ("Euclidean (L2)", "euclidean", 2),
        ("Manhattan (L1)", "manhattan", 1),
        ("Chebyshev (L-inf)", "chebyshev", None),
        ("Minkowski (p=3)", "minkowski", 3),
        ("Cosine", "cosine", None)
    ]

    results = []
    for display_name, m_key, p_val in metrics_to_test:
        kwargs = {'n_neighbors': k, 'metric': m_key}
        if p_val is not None and m_key == "minkowski":
            kwargs['p'] = p_val
        clf = KNeighborsClassifier(**kwargs)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        results.append({
            "Distance Metric": display_name,
            "Holdout Accuracy": f"{acc * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%"
        })

    return pd.DataFrame(results)


def main():
    print("=" * 74)
    print("  PROJECT 007: K-NEAREST NEIGHBORS (KNN) SPATIAL GEOMETRY PIPELINE")
    print("=" * 74)

    # 1. Load Real Dataset
    print("\n[1/5] Loading authentic Wisconsin Breast Cancer diagnostic dataset...")
    df, feature_names, target_col, target_names = load_dataset("breast_cancer")
    print(f"      Instances: {len(df):,} | Features: {len(feature_names)} clinical measurements")
    print(f"      Classes: {', '.join([f'{i}: {name}' for i, name in enumerate(target_names)])}")

    # 2. Preprocess & Split
    print("\n[2/5] Partitioning into stratified train (75%) and holdout test (25%) splits...")
    X = df[feature_names].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"      Training instances: {len(X_train):,} | Holdout test instances: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Demonstrating the Critical Need for Feature Scaling
    raw_knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    raw_knn.fit(X_train, y_train)
    raw_acc = accuracy_score(y_test, raw_knn.predict(X_test))

    scaled_knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    scaled_knn.fit(X_train_scaled, y_train)
    scaled_acc = accuracy_score(y_test, scaled_knn.predict(X_test_scaled))

    print(f"\n      Scaling Impact Check (K=5):")
    print(f"        - Raw Unscaled Features:   {raw_acc * 100:.2f}% Accuracy")
    print(f"        - StandardScaler Z-Score:  {scaled_acc * 100:.2f}% Accuracy (+{(scaled_acc - raw_acc)*100:.2f}% lift)")

    # 3. K Hyperparameter Optimization
    print("\n[3/5] Performing 5-fold cross-validation to determine optimal K neighbors:")
    best_k, best_cv, k_records = find_optimal_k(X_train_scaled, y_train, k_range=range(1, 21, 2))
    k_df = pd.DataFrame(k_records)
    print(k_df.to_string(index=False))
    print(f"\n      >>> Optimal K selected: K = {best_k} (Peak CV Accuracy: {best_cv * 100:.2f}%)")

    # 4. Distance Metric Benchmark
    print("\n[4/5] Benchmarking distance metrics on holdout test set (K=optimal):")
    metric_df = benchmark_distance_metrics(X_train_scaled, y_train, X_test_scaled, y_test, k=best_k)
    print(metric_df.to_string(index=False))

    # 5. Fit Champion Model & Holdout Evaluation
    print(f"\n[5/5] Training Champion Pipeline: StandardScaler + KNN(K={best_k}, metric='euclidean')...")
    champion_model = train_knn_classifier(X_train_scaled, y_train, n_neighbors=best_k, metric='euclidean')
    eval_res = evaluate_knn(champion_model, X_test_scaled, y_test, target_names=target_names)

    print("\n--- Holdout Evaluation Telemetry ---")
    print(f"Accuracy:  {eval_res['accuracy'] * 100:.2f}%")
    print(f"Precision: {eval_res['precision'] * 100:.2f}%")
    print(f"Recall:    {eval_res['recall'] * 100:.2f}%")
    print(f"F1-Score:  {eval_res['f1'] * 100:.2f}%")

    cm = eval_res['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"  [True Negative  ({target_names[0]}): {cm[0][0]:3d}  |  False Positive: {cm[0][1]:3d}]")
    print(f"  [False Negative:                  {cm[1][0]:3d}  |  True Positive  ({target_names[1]}): {cm[1][1]:3d}]")

    # Sample Neighbor Trace Inference
    sample_idx = 0
    test_instance = X_test_scaled[sample_idx:sample_idx+1]
    true_label = target_names[y_test[sample_idx]]
    pred_label = target_names[champion_model.predict(test_instance)[0]]
    dists, n_indices = champion_model.kneighbors(test_instance, n_neighbors=best_k)

    print(f"\n--- Live Neighbor Trace Demonstration (Test Sample #0) ---")
    print(f"  True Class:      {true_label}")
    print(f"  Predicted Class: {pred_label}")
    print(f"  Nearest {best_k} Training Neighbors:")
    for rank, (idx, dist) in enumerate(zip(n_indices[0], dists[0]), 1):
        n_class = target_names[y_train[idx]]
        print(f"    Neighbor #{rank}: Distance = {dist:.3f} | True Class = {n_class}")

    print("\n" + "=" * 74)
    print("  PIPELINE EXECUTION COMPLETE. Launch Streamlit Studio: streamlit run app.py")
    print("=" * 74)


if __name__ == "__main__":
    main()

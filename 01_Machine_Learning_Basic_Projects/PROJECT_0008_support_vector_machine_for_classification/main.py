"""
Project 008: Support Vector Machine (SVM) for Classification
============================================================
An industrial-strength Support Vector Machine workflow featuring real-world
biomedical diagnostic classification (Wisconsin Breast Cancer Diagnostic),
multi-kernel comparative benchmarks, support vector distribution telemetry,
margin boundary geometry, and cross-validated generalization assessment.

Part of the 1,000 AI Projects Implementation Library.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataset import load_real_dataset, generate_2d_manifold
from svm_engine import (
    prepare_dataset_split,
    train_svm_model,
    extract_support_vector_diagnostics,
    evaluate_svm_model,
    run_cross_validation,
    compare_kernel_suite
)


def run_pipeline(
    dataset_name: str = "breast_cancer",
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    degree: int = 3,
    test_size: float = 0.25,
    cv_folds: int = 5,
    compare_all_kernels: bool = True
) -> Dict[str, Any]:
    """
    Execute full end-to-end SVM classification pipeline.
    """
    print("=" * 75)
    print("🚀 PROJECT 008: SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION ENGINE")
    print("=" * 75)

    # 1. Load Dataset
    print(f"\n[1/6] Loading Dataset: '{dataset_name}'...")
    if dataset_name.lower() in ["moons", "circles", "blobs", "linear"]:
        manifold_map = {
            "moons": "Two Moons",
            "circles": "Concentric Circles",
            "blobs": "Overlapping Blobs",
            "linear": "Linearly Separable"
        }
        df, feature_cols, target_col, target_names = generate_2d_manifold(manifold_map[dataset_name.lower()])
    else:
        df, feature_cols, target_col, target_names = load_real_dataset(dataset_name)

    print(f"      Dataset Shape: {df.shape[0]} samples × {len(feature_cols)} features")
    print(f"      Classes ({len(target_names)}): {', '.join(target_names)}")
    class_dist = df[target_col].value_counts().to_dict()
    print(f"      Class Distribution: {class_dist}")

    # 2. Data Preparation & Standardization
    print(f"\n[2/6] Standardizing Features and Splitting (Test Ratio: {test_size:.2%})...")
    data_bundle = prepare_dataset_split(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=test_size,
        scale_method="standard",
        random_state=42
    )
    X_train, X_test = data_bundle["X_train"], data_bundle["X_test"]
    y_train, y_test = data_bundle["y_train"], data_bundle["y_test"]
    print(f"      Training Instances: {X_train.shape[0]} | Holdout Instances: {X_test.shape[0]}")

    # 3. Model Training
    # Convert gamma if numeric string
    parsed_gamma = gamma
    try:
        parsed_gamma = float(gamma)
    except ValueError:
        pass

    print(f"\n[3/6] Training Primary SVM Classifier:")
    print(f"      Kernel: {kernel.upper()} | C: {C} | Gamma: {parsed_gamma} | Degree: {degree}")
    model, train_latency = train_svm_model(
        X_train=X_train,
        y_train=y_train,
        kernel=kernel,
        C=C,
        gamma=parsed_gamma,
        degree=degree,
        probability=True,
        random_state=42
    )
    print(f"      Training Latency: {train_latency:.2f} ms")

    # 4. Support Vector Diagnostics
    print(f"\n[4/6] Support Vector Geometry & Dual Representation:")
    diagnostics = extract_support_vector_diagnostics(model, X_train, y_train)
    n_sv = diagnostics["n_support_vectors"]
    sv_pct = diagnostics["support_vector_ratio_pct"]
    print(f"      Total Support Vectors: {n_sv} / {len(X_train)} ({sv_pct:.1f}% of training set)")
    print(f"      Support Vectors per Class: {diagnostics['n_support_per_class']}")
    
    if "margin_width" in diagnostics:
        print(f"      Linear Hyperplane Norm ||w||: {diagnostics['w_norm']:.4f}")
        print(f"      Geometric Margin Width (2/||w||): {diagnostics['margin_width']:.4f}")

    if "margin_violations_count" in diagnostics:
        viol_cnt = diagnostics["margin_violations_count"]
        viol_pct = diagnostics["margin_violations_pct"]
        print(f"      Margin Violations (Slack xi > 0): {viol_cnt} ({viol_pct:.1f}%)")

    # 5. Holdout Evaluation
    print(f"\n[5/6] Model Evaluation on Unseen Test Partition:")
    eval_metrics = evaluate_svm_model(model, X_test, y_test, target_names=target_names)
    print(f"      Accuracy:          {eval_metrics['accuracy']:.4f}")
    print(f"      Precision (wt):    {eval_metrics['precision']:.4f}")
    print(f"      Recall (wt):       {eval_metrics['recall']:.4f}")
    print(f"      F1-Score (wt):     {eval_metrics['f1']:.4f}")
    print(f"      Balanced Accuracy: {eval_metrics['balanced_accuracy']:.4f}")
    print(f"      Matthews Corr (MCC):{eval_metrics['mcc']:.4f}")
    if "roc_auc" in eval_metrics:
        print(f"      ROC-AUC Score:     {eval_metrics['roc_auc']:.4f}")
    if "specificity" in eval_metrics:
        print(f"      Specificity:       {eval_metrics['specificity']:.4f}")

    print("\n      Confusion Matrix:")
    cm = eval_metrics["confusion_matrix"]
    for i, row in enumerate(cm):
        label = target_names[i] if i < len(target_names) else f"Class {i}"
        print(f"        {label[:12]:<12}: {row.tolist()}")

    # 6. Stratified Cross-Validation
    print(f"\n[6/6] Stratified {cv_folds}-Fold Cross-Validation:")
    cv_res = run_cross_validation(model, X_train, y_train, cv_folds=cv_folds)
    print(f"      Mean CV Accuracy: {cv_res['mean_accuracy']:.4f} ± {cv_res['std_accuracy']:.4f}")
    print(f"      Fold Scores: {[round(s, 4) for s in cv_res['fold_scores']]}")

    # Optional Multi-Kernel Benchmark
    if compare_all_kernels:
        print("\n" + "=" * 75)
        print("⚔️ MULTI-KERNEL BENCHMARK TOURNAMENT (Linear vs RBF vs Poly vs Sigmoid)")
        print("=" * 75)
        benchmark_df = compare_kernel_suite(X_train, y_train, X_test, y_test, C=C)
        print(benchmark_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("✅ SVM PIPELINE RUN COMPLETE")
    print("=" * 75)

    return {
        "model": model,
        "diagnostics": diagnostics,
        "eval_metrics": eval_metrics,
        "cv_results": cv_res
    }


def main():
    parser = argparse.ArgumentParser(
        description="Production Support Vector Machine (SVM) Classification Studio"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "wine", "iris", "moons", "circles", "blobs", "linear"],
        help="Dataset selection (default: breast_cancer)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid"],
        help="SVM Kernel function (default: rbf)"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization penalty C (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Kernel coefficient gamma (default: 'scale')"
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial kernel degree (default: 3)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split fraction (default: 0.25)"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip multi-kernel comparison table"
    )

    args = parser.parse_args()

    run_pipeline(
        dataset_name=args.dataset,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        degree=args.degree,
        test_size=args.test_size,
        cv_folds=args.cv,
        compare_all_kernels=not args.no_compare
    )


if __name__ == "__main__":
    main()

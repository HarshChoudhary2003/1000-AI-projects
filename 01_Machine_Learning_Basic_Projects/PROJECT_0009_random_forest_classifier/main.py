"""
Project 009: Random Forest Classifier
=====================================
An industrial-strength Random Forest classification pipeline featuring real-world
multi-class chemical profiling (Wine Cultivars), Out-of-Bag (OOB) error estimation,
Mean Decrease in Impurity (Gini) feature ranking, forest structural diagnostics,
and cross-validated generalization assessment.

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
from rf_engine import (
    prepare_dataset_split,
    train_random_forest,
    evaluate_rf_model,
    extract_feature_importances,
    extract_forest_diagnostics,
    run_cross_validation
)


def run_pipeline(
    dataset_name: str = "wine",
    n_estimators: int = 100,
    max_depth: int = None,
    criterion: str = "gini",
    max_features: str = "sqrt",
    test_size: float = 0.25,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Execute full end-to-end Random Forest classification pipeline.
    """
    print("=" * 75)
    print("🌲 PROJECT 009: RANDOM FOREST CLASSIFIER & ENSEMBLE STUDIO")
    print("=" * 75)

    # 1. Load Dataset
    print(f"\n[1/5] Loading Dataset: '{dataset_name}'...")
    if dataset_name.lower() in ["moons", "circles", "blobs", "multimodal"]:
        manifold_map = {
            "moons": "Two Moons",
            "circles": "Concentric Circles",
            "blobs": "Overlapping Blobs",
            "multimodal": "Complex Multimodal"
        }
        df, feature_cols, target_col, target_names = generate_2d_manifold(manifold_map[dataset_name.lower()])
    else:
        df, feature_cols, target_col, target_names = load_real_dataset(dataset_name)

    print(f"      Dataset Shape: {df.shape[0]} samples × {len(feature_cols)} features")
    print(f"      Classes ({len(target_names)}): {', '.join(target_names)}")
    class_dist = df[target_col].value_counts().to_dict()
    print(f"      Class Distribution: {class_dist}")

    # 2. Data Preparation
    print(f"\n[2/5] Splitting Partition (Test Ratio: {test_size:.2%})...")
    data_bundle = prepare_dataset_split(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=test_size,
        scale_method="none",
        random_state=42
    )
    X_train, X_test = data_bundle["X_train"], data_bundle["X_test"]
    y_train, y_test = data_bundle["y_train"], data_bundle["y_test"]
    print(f"      Training Instances: {X_train.shape[0]} | Holdout Instances: {X_test.shape[0]}")

    # 3. Model Training
    print(f"\n[3/5] Training Random Forest Ensemble:")
    print(f"      Trees (n_estimators): {n_estimators} | Max Depth: {max_depth} | Criterion: {criterion} | Max Features: {max_features}")
    model, train_latency = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        max_features=max_features,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    print(f"      Training Latency: {train_latency:.2f} ms")

    # Forest Diagnostics
    diag = extract_forest_diagnostics(model)
    print(f"      Average Tree Depth: {diag['avg_depth']:.1f} (Min: {diag['min_depth']}, Max: {diag['max_depth']})")
    print(f"      Average Leaves per Tree: {diag['avg_leaves']:.1f} | Total Forest Nodes: {diag['total_nodes_forest']}")

    # 4. Out-of-Bag & Holdout Evaluation
    print(f"\n[4/5] Model Evaluation on Unseen Test Partition & OOB Validation:")
    eval_metrics = evaluate_rf_model(model, X_test, y_test, target_names=target_names)
    
    if "oob_score" in eval_metrics:
        print(f"      Out-of-Bag (OOB) Accuracy: {eval_metrics['oob_score']:.4f} (OOB Error: {eval_metrics['oob_error']:.4f})")
    print(f"      Holdout Test Accuracy:     {eval_metrics['accuracy']:.4f}")
    print(f"      Precision (Weighted):      {eval_metrics['precision']:.4f}")
    print(f"      Recall (Weighted):         {eval_metrics['recall']:.4f}")
    print(f"      F1-Score (Weighted):       {eval_metrics['f1']:.4f}")
    print(f"      Balanced Accuracy:         {eval_metrics['balanced_accuracy']:.4f}")
    print(f"      Matthews Corr (MCC):       {eval_metrics['mcc']:.4f}")
    if "roc_auc" in eval_metrics:
        print(f"      ROC-AUC Score:             {eval_metrics['roc_auc']:.4f}")

    print("\n      Confusion Matrix:")
    cm = eval_metrics["confusion_matrix"]
    for i, row in enumerate(cm):
        label = target_names[i] if i < len(target_names) else f"Class {i}"
        print(f"        {label[:14]:<14}: {row.tolist()}")

    # 5. Top Feature Importances (Gini Impurity)
    print(f"\n[5/5] Mean Decrease in Impurity (Gini Feature Importance Ranking):")
    imp_df = extract_feature_importances(model, feature_cols)
    top_n = min(8, len(imp_df))
    for idx, row in imp_df.head(top_n).iterrows():
        bar = "█" * int(row['Importance_Pct'] * 0.8)
        print(f"      {row['Feature'][:22]:<22} : {row['Importance_Pct']:5.2f}% (±{row['Std_Dev']*100:4.1f}%) {bar}")

    # Stratified Cross-Validation
    print(f"\n[+] Stratified {cv_folds}-Fold Cross-Validation:")
    cv_res = run_cross_validation(model, X_train, y_train, cv_folds=cv_folds)
    print(f"      Mean CV Accuracy: {cv_res['mean_accuracy']:.4f} ± {cv_res['std_accuracy']:.4f}")
    print(f"      Fold Scores: {[round(s, 4) for s in cv_res['fold_scores']]}")

    print("\n" + "=" * 75)
    print("✅ RANDOM FOREST PIPELINE RUN COMPLETE")
    print("=" * 75)

    return {
        "model": model,
        "diagnostics": diag,
        "eval_metrics": eval_metrics,
        "feature_importances": imp_df,
        "cv_results": cv_res
    }


def main():
    parser = argparse.ArgumentParser(
        description="Production Random Forest Classifier & Ensemble Studio"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wine",
        choices=["wine", "cancer", "iris", "moons", "circles", "blobs", "multimodal"],
        help="Dataset selection (default: wine)"
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=100,
        help="Number of decision trees (default: 100)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of trees (default: None/unlimited)"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="gini",
        choices=["gini", "entropy", "log_loss"],
        help="Splitting criterion (default: gini)"
    )
    parser.add_argument(
        "--max-features",
        type=str,
        default="sqrt",
        choices=["sqrt", "log2"],
        help="Max features per split (default: sqrt)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Holdout test split ratio (default: 0.25)"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )

    args = parser.parse_args()

    run_pipeline(
        dataset_name=args.dataset,
        n_estimators=args.trees,
        max_depth=args.max_depth,
        criterion=args.criterion,
        max_features=args.max_features,
        test_size=args.test_size,
        cv_folds=args.cv
    )


if __name__ == "__main__":
    main()

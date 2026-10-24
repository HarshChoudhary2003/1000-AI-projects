"""
Project 010: Gradient Boosting Algorithm Implementation & Studio
================================================================
An industrial-strength Gradient Tree Boosting pipeline featuring:
- Sequential pseudo-residual gradient minimization (Bernoulli deviance)
- Stage-by-stage train vs test loss trajectory tracking
- Optimal early-stopping stage identification
- Mean Decrease in Impurity (MDI) feature importances
- Stratified K-Fold cross-validation generalization scoring
- Optional model serialization via Joblib

Part of the 1,000 AI Projects Implementation Library.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataset import load_real_dataset, generate_2d_manifold
from gbm_engine import (
    prepare_dataset_split,
    train_gradient_boosting,
    compute_staged_metrics,
    evaluate_model,
    extract_feature_importances,
    extract_forest_diagnostics,
    run_cross_validation
)


def run_pipeline(
    dataset_name: str = "credit",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    test_size: float = 0.25,
    cv_folds: int = 5,
    save_model_path: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute full end-to-end Gradient Tree Boosting classification pipeline.
    """
    print("=" * 78)
    print("⚡ PROJECT 010: GRADIENT BOOSTING ALGORITHM IMPLEMENTATION & STUDIO")
    print("=" * 78)

    # 1. Load Dataset
    print(f"\n[1/6] Loading Benchmark Dataset: '{dataset_name}'...")
    if dataset_name.lower() in ["moons", "circles", "blobs", "spiral", "linear"]:
        manifold_map = {
            "moons": "Two Moons",
            "circles": "Concentric Circles",
            "blobs": "Overlapping Blobs",
            "spiral": "Interlocking Spirals",
            "linear": "Linearly Separable"
        }
        df, feature_cols, target_col, target_names = generate_2d_manifold(
            manifold_type=manifold_map[dataset_name.lower()],
            n_samples=500,
            noise=0.20,
            random_state=random_state
        )
    else:
        df, feature_cols, target_col, target_names = load_real_dataset(dataset_name)

    print(f"      ✓ Dataset Loaded: {len(df)} samples, {len(feature_cols)} features, classes={target_names}")

    # 2. Split Dataset
    print(f"\n[2/6] Splitting Dataset (test_size={test_size:.2f}, stratified)...")
    X_train, X_test, y_train, y_test = prepare_dataset_split(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state
    )
    print(f"      ✓ Partitioned: Train={len(X_train)} samples | Test={len(X_test)} samples")

    # 3. Train Gradient Boosting Model
    print(f"\n[3/6] Fitting Gradient Boosting Classifier...")
    print(f"      • n_estimators (Stages): {n_estimators}")
    print(f"      • learning_rate (Shrinkage η): {learning_rate}")
    print(f"      • max_depth: {max_depth}")
    print(f"      • subsample: {subsample * 100:.0f}% (Stochastic)")

    model = train_gradient_boosting(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state
    )
    print("      ✓ Ensemble training complete.")

    # 4. Stage-by-Stage Deviance & Early Stopping Detection
    print("\n[4/6] Computing Stage-by-Stage Loss & Early Stopping Boundary...")
    staged = compute_staged_metrics(model, X_train, y_train, X_test, y_test)
    best_stage = staged["best_stage"]
    min_loss = staged["min_test_deviance"]
    final_loss = staged["test_deviance"][-1]

    print(f"      • Initial Train Deviance (Stage 1): {staged['train_deviance'][0]:.4f}")
    print(f"      • Final Train Deviance (Stage {n_estimators}): {staged['train_deviance'][-1]:.4f}")
    print(f"      • Optimal Early-Stopping Stage: Tree #{best_stage} (Loss: {min_loss:.4f})")
    print(f"      • Final Test Deviance (Stage {n_estimators}): {final_loss:.4f}")
    if best_stage < n_estimators:
        print(f"      ⚠️ Note: Test loss reaches minimum {n_estimators - best_stage} stages before stopping.")
    else:
        print("      ✓ Healthy convergence: Test loss strictly improves through all stages.")

    # 5. Evaluate Metrics
    print("\n[5/6] Evaluating Model Performance on Held-Out Test Set...")
    eval_res = evaluate_model(model, X_test, y_test, target_names=target_names)
    diagnostics = extract_forest_diagnostics(model)

    print(f"      • Test Accuracy:   {eval_res['accuracy'] * 100:.2f}%")
    if eval_res["roc_auc"] > 0:
        print(f"      • ROC-AUC Score:   {eval_res['roc_auc']:.4f}")
    print(f"      • F1-Score:        {eval_res['f1_score'] * 100:.2f}%")
    print(f"      • Precision:       {eval_res['precision'] * 100:.2f}%")
    print(f"      • Recall:          {eval_res['recall'] * 100:.2f}%")
    print(f"      • Log-Loss:        {eval_res['log_loss']:.4f}")
    print(f"      • Average Depth:   {diagnostics['avg_depth']:.2f}")
    print(f"      • Total Nodes:     {diagnostics['total_nodes']}")

    # Feature Importances
    imp_df = extract_feature_importances(model, feature_cols)
    print("\nTop 5 Feature Importances (MDI):")
    for _, row in imp_df.head(5).iterrows():
        print(f"      - {row['Feature']:<25}: {row['Percentage']:>6.2f}% (Cumulative: {row['Cumulative']:>6.2f}%)")

    # 6. Stratified Cross-Validation
    print(f"\n[6/6] Executing {cv_folds}-Fold Stratified Cross-Validation...")
    cv_res = run_cross_validation(
        X=df[feature_cols].to_numpy(),
        y=df[target_col].to_numpy(),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        cv_folds=cv_folds,
        random_state=random_state
    )
    print(f"      • Mean CV Accuracy: {cv_res['accuracy_mean'] * 100:.2f}% (± {cv_res['accuracy_std'] * 100:.2f}%)")
    print(f"      • Mean CV F1-Score: {cv_res['f1_mean'] * 100:.2f}% (± {cv_res['f1_std'] * 100:.2f}%)")
    if "roc_auc_mean" in cv_res:
        print(f"      • Mean CV ROC-AUC:  {cv_res['roc_auc_mean']:.4f} (± {cv_res['roc_auc_std']:.4f})")

    # Optional Model Serialization
    if save_model_path:
        joblib.dump(model, save_model_path)
        print(f"\n💾 Model successfully saved to '{save_model_path}'.")

    print("\n" + "=" * 78)
    print("✅ GRADIENT BOOSTING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 78)

    return {
        "model": model,
        "evaluation": eval_res,
        "staged_metrics": staged,
        "feature_importances": imp_df,
        "cross_validation": cv_res
    }


def main():
    parser = argparse.ArgumentParser(description="Execute Gradient Boosting Algorithm Pipeline")
    parser.add_argument("--dataset", type=str, default="credit", help="Dataset name: credit, breast_cancer, wine, moons, circles, blobs, spiral")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosting stages (trees)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Shrinkage factor (learning rate eta)")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum tree depth for weak learners")
    parser.add_argument("--subsample", type=float, default=1.0, help="Subsample ratio for stochastic gradient boosting")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test partition split ratio")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--save-model", type=str, default=None, help="File path to save the fitted model (.joblib)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")

    args = parser.parse_args()

    run_pipeline(
        dataset_name=args.dataset,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        test_size=args.test_size,
        cv_folds=args.cv,
        save_model_path=args.save_model,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()

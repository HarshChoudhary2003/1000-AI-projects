"""
Project 013: Principal Component Analysis (PCA) Implementation Pipeline & CLI
=============================================================================
An industrial-grade command-line interface and reproducible evaluation pipeline
for PCA across spectral analysis, mathematical first-principles verification,
lossy compression reconstruction, and downstream machine learning benchmarking.

Usage Examples:
    # Run default PCA on 64D Handwritten Digits dataset
    python main.py

    # Run on Breast Cancer dataset with 3 components and biplot loadings
    python main.py --dataset breast_cancer --components 3 --standardize

    # Run mathematical verification comparing scratch SVD vs Scikit-Learn
    python main.py --verify-math

    # Run downstream classifier benchmark (Raw vs PCA dimensions)
    python main.py --benchmark

    # Save fitted PCA pipeline artifact
    python main.py --save-model pca_digits_baseline.joblib

Part of the 1,000 AI Projects Implementation Library.
"""

import sys
import argparse
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataset import (
    load_benchmark_dataset,
    get_archetype_presets
)
from pca_engine import (
    CustomPCAScratch,
    fit_pca_pipeline,
    compute_scree_analysis,
    compute_loadings_and_biplot,
    compute_reconstruction_analysis,
    compute_reconstruction_curve,
    run_downstream_benchmark,
    compare_scratch_vs_sklearn,
    save_pca_pipeline,
    load_pca_pipeline
)


def run_pca_pipeline(
    dataset_name: str = "digits",
    n_components: int = 2,
    scaling: str = "standard",
    pca_type: str = "standard",
    verify_math: bool = False,
    run_benchmark: bool = False,
    save_model_path: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Execute full end-to-end Principal Component Analysis pipeline.
    """
    print("=" * 78)
    print("🌌  PROJECT 013: PRINCIPAL COMPONENT ANALYSIS (PCA) ENGINE & PIPELINE")
    print("=" * 78)

    # 1. Load benchmark dataset
    print(f"\n[1/5] Loading benchmark dataset: '{dataset_name}'...")
    data = load_benchmark_dataset(dataset_name, seed=seed)
    X, y = data["X"], data["y"]
    feature_names = data["feature_names"]
    print(f"      - Samples: {X.shape[0]}")
    print(f"      - Input Dimensions: {X.shape[1]} features")
    print(f"      - Classes: {len(np.unique(y))} distinct groups")
    print(f"      - Context: {data['description']}")

    # 2. Fit PCA Transformation
    print(f"\n[2/5] Fitting PCA ({pca_type.upper()}) with k={n_components} components & scaling='{scaling}'...")
    fit_res = fit_pca_pipeline(
        X,
        n_components=n_components,
        scaling=scaling,
        pca_type=pca_type
    )
    X_proj = fit_res["X_proj"]
    exp_var_ratio = fit_res["explained_variance_ratio_"]
    cum_var_ratio = fit_res["cumulative_variance_ratio_"]

    for idx, var in enumerate(exp_var_ratio[:min(5, len(exp_var_ratio))]):
        print(f"      - PC{idx+1} Explained Variance: {var*100:6.2f}% (Cumulative: {cum_var_ratio[idx]*100:6.2f}%)")
    total_var = np.sum(exp_var_ratio) * 100
    print(f"      - Total Variance Preserved in {n_components}D: {total_var:.2f}%")

    # 3. Scree & Dimensionality Heuristics
    print("\n[3/5] Computing Spectral Energy Distribution & Scree Analysis...")
    scree = compute_scree_analysis(fit_res["X_scaled"], max_components=min(X.shape[1], 30))
    print(f"      - Kaiser Criterion (Eigenvalues >= 1.0): {scree['kaiser_k']} components")
    print(f"      - Geometric Scree Elbow Point:          {scree['elbow_k']} components")
    print(f"      - 90% Variance Retention Threshold:     {scree['k_90']} components")
    print(f"      - 95% Variance Retention Threshold:     {scree['k_95']} components")

    # 4. Reconstruction Analysis
    print("\n[4/5] Evaluating Lossy Compression & Signal Reconstruction...")
    rec = compute_reconstruction_analysis(X, fit_res["scaler"], k_components=n_components, sample_index=0)
    print(f"      - Global Dataset Reconstruction MSE: {rec['mse_total']:.4f}")
    print(f"      - Dimension Savings / Compression:   {(1.0 - (n_components / X.shape[1])) * 100:.1f}%")

    # 5. Math verification if requested
    if verify_math:
        print("\n[EXTRA] Rigorous First-Principles Verification (Scratch SVD vs Scikit-Learn)...")
        math_cmp = compare_scratch_vs_sklearn(X, n_components=n_components)
        print(f"      - Max Projection Delta:      {math_cmp['diff_projection_max']:.2e}")
        print(f"      - Max Variance Ratio Delta:  {math_cmp['diff_variance_ratio_max']:.2e}")
        print(f"      - Numerically Identical:     {math_cmp['is_numerically_identical']}")

    # 6. Downstream ML Benchmark if requested
    if run_benchmark:
        print("\n[EXTRA] Benchmarking Downstream Classifier (Curse of Dimensionality)...")
        bench_df = run_downstream_benchmark(
            X, y,
            scaling=scaling,
            classifier_name="logistic_regression",
            k_candidates=[1, 2, 3, 5, min(10, X.shape[1]-1)]
        )
        print(bench_df.to_string(index=False))

    # 7. Model Persistence
    if save_model_path:
        print(f"\n[PERSISTENCE] Saving fitted pipeline bundle to '{save_model_path}'...")
        save_pca_pipeline(fit_res, save_model_path)
        print("      - Model successfully serialized!")

    print("\n" + "=" * 78)
    print("✅  PCA PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 78)

    return {
        "data": data,
        "fit_res": fit_res,
        "scree": scree,
        "rec": rec
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Principal Component Analysis Pipeline CLI")
    parser.add_argument("--dataset", type=str, default="digits", choices=["digits", "breast_cancer", "wine", "iris", "swiss_roll", "correlated_gaussian"], help="Benchmark dataset")
    parser.add_argument("--components", "-k", type=int, default=2, help="Number of principal components")
    parser.add_argument("--standardize", action="store_true", default=True, help="Apply StandardScaler z-scoring")
    parser.add_argument("--pca-type", type=str, default="standard", choices=["standard", "incremental", "kernel", "truncated_svd"], help="PCA variant")
    parser.add_argument("--verify-math", action="store_true", help="Run first-principles SVD math verification")
    parser.add_argument("--benchmark", action="store_true", help="Run downstream classifier benchmark")
    parser.add_argument("--save-model", type=str, default=None, help="Filepath to save fitted model artifact")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    scaling = "standard" if args.standardize else "none"
    run_pca_pipeline(
        dataset_name=args.dataset,
        n_components=args.components,
        scaling=scaling,
        pca_type=args.pca_type,
        verify_math=args.verify_math,
        run_benchmark=args.benchmark,
        save_model_path=args.save_model
    )

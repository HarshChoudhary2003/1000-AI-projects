"""
Project 014: Feature Selection Techniques Pipeline & CLI
=========================================================
An industrial-grade command-line interface and reproducible evaluation pipeline
for Feature Selection across Filter, Wrapper, Embedded, and Consensus paradigms,
with ground-truth signal recovery, stability metrics, and downstream ML benchmarking.

Usage Examples:
    # Run default consensus selection on synthetic classification benchmark
    python main.py

    # Run on Wisconsin Breast Cancer dataset selecting top 6 features
    python main.py --dataset breast_cancer --k 6

    # Run downstream ML benchmark (Full vs Filter vs RFE vs Consensus)
    python main.py --benchmark

    # Evaluate feature selection stability via bootstrap subsampling
    python main.py --stability

    # Launch interactive Web UI Studio
    python main.py --ui

Part of the 1,000 AI Projects Implementation Library.
"""

import sys
import os
import argparse
import subprocess
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataset import (
    load_feature_dataset,
    add_synthetic_noise_features
)
from feature_selection_engine import (
    run_univariate_filter,
    run_variance_threshold,
    run_multicollinearity_filter,
    run_rfe,
    run_rfecv,
    run_l1_regularization,
    run_tree_importance,
    run_permutation_importance,
    compute_consensus_ranking,
    evaluate_ground_truth_selection,
    compute_stability_score,
    benchmark_downstream_model,
    save_feature_selection_pipeline
)


def run_pipeline(
    dataset_name: str = "synthetic_classification",
    k: int = 4,
    method: str = "consensus",
    inject_noise: int = 0,
    run_benchmark: bool = False,
    run_stability: bool = False,
    save_pipeline_path: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Execute end-to-end feature selection pipeline.
    """
    print("=" * 78)
    print("🎯  PROJECT 014: FEATURE SELECTION TECHNIQUES ENGINE & BENCHMARK")
    print("=" * 78)

    # 1. Load Data
    print(f"\n[1/5] Loading Dataset: '{dataset_name}'...")
    data = load_feature_dataset(dataset_name, seed=seed)
    
    if inject_noise > 0:
        data = add_synthetic_noise_features(data, n_noise_features=inject_noise, seed=seed)
        print(f"      Injected {inject_noise} synthetic noise features.")
        
    print(f"      Samples: {data['n_samples']} | Features: {data['n_features']} | Task: {data['task_type'].upper()}")
    print(f"      Description: {data['description']}")
    
    k = min(k, data["n_features"])
    print(f"      Target Feature Budget: K = {k}")

    # 2. Run Feature Selection Techniques
    print("\n[2/5] Executing Feature Selection Paradigms...")
    
    filt_res = run_univariate_filter(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], method="f_test", k=k)
    print(f"      [Filter - ANOVA F-test]: Selected {filt_res['selected_features']}")
    
    rfe_res = run_rfe(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], n_features_to_select=k)
    print(f"      [Wrapper - RFE]:         Selected {rfe_res['selected_features']}")
    
    l1_res = run_l1_regularization(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], k=k)
    print(f"      [Embedded - L1]:         Selected {l1_res['selected_features']}")
    
    tree_res = run_tree_importance(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], k=k, seed=seed)
    print(f"      [Embedded - Tree MDI]:   Selected {tree_res['selected_features']}")

    # 3. Consensus Ensemble
    print("\n[3/5] Computing Multi-Method Consensus (Borda Count)...")
    ensemble_pool = {
        "ANOVA F-Test": filt_res,
        "RFE": rfe_res,
        "L1 Sparsity": l1_res,
        "Tree Importance": tree_res
    }
    consensus_res = compute_consensus_ranking(ensemble_pool, data["feature_names"], k=k)
    print(f"      [Consensus Ensemble]:    Selected {consensus_res['selected_features']}")

    # Ground-truth evaluation if applicable
    if data["has_ground_truth"]:
        gt = evaluate_ground_truth_selection(
            consensus_res["selected_indices"],
            data["ground_truth_informative"],
            data["ground_truth_noise"],
            data["n_features"]
        )
        print(f"\n      Ground Truth Verification:")
        print(f"      - Precision: {gt['precision']*100:.1f}%")
        print(f"      - Recall:    {gt['recall']*100:.1f}%")
        print(f"      - F1-Score:  {gt['f1']:.3f}")
        print(f"      - Noise Discovered: {gt['noise_features_selected']} features (FDR: {gt['false_discovery_rate']*100:.1f}%)")

    # 4. Stability Evaluation
    if run_stability:
        print("\n[4/5] Evaluating Selection Stability (Nogueira-Brown Index)...")
        stab = compute_stability_score(data["X"], data["y"], task_type=data["task_type"], method_type="f_test", k=k, n_bootstraps=5, seed=seed)
        print(f"      Stability Score (ANOVA): {stab['stability_score']:.4f} (across 5 bootstrap resamples)")

    # 5. Downstream Machine Learning Benchmark
    if run_benchmark:
        print("\n[5/5] Running Downstream ML Generalization Benchmark...")
        rng = np.random.RandomState(seed)
        subsets = {
            f"All Features ({data['n_features']})": list(range(data["n_features"])),
            f"Filter (Top-{k})": filt_res["selected_indices"],
            f"Wrapper RFE (Top-{k})": rfe_res["selected_indices"],
            f"Embedded Tree (Top-{k})": tree_res["selected_indices"],
            f"Consensus (Top-{k})": consensus_res["selected_indices"],
            f"Random Null (Top-{k})": rng.choice(data["n_features"], size=k, replace=False).tolist()
        }
        bench_df = benchmark_downstream_model(data["X"], data["y"], data["feature_names"], subsets, task_type=data["task_type"], seed=seed)
        print("\n" + bench_df.to_string(index=False))

    # Pipeline Saving
    if save_pipeline_path:
        pipeline_dict = {
            "selected_features": consensus_res["selected_features"],
            "selected_indices": consensus_res["selected_indices"],
            "k": k,
            "dataset_name": dataset_name,
            "task_type": data["task_type"]
        }
        save_feature_selection_pipeline(pipeline_dict, save_pipeline_path)
        print(f"\n💾 Saved feature selection pipeline artifact to: '{save_pipeline_path}'")

    print("\n" + "=" * 78)
    print("✨  Feature selection execution completed successfully.")
    print("=" * 78)

    return {
        "dataset": data,
        "filter": filt_res,
        "rfe": rfe_res,
        "l1": l1_res,
        "tree": tree_res,
        "consensus": consensus_res
    }


def launch_ui():
    """Launch the Streamlit interactive dashboard."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    print(f"🚀 Launching Feature Selection Studio at: {app_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project 014: Feature Selection Techniques CLI & Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="synthetic_classification",
                        choices=["synthetic_classification", "synthetic_regression", "breast_cancer", "wine", "diabetes", "california_housing"],
                        help="Dataset name to process")
    parser.add_argument("--k", type=int, default=4, help="Number of features to select")
    parser.add_argument("--method", type=str, default="consensus", choices=["consensus", "filter", "rfe", "l1", "tree"], help="Feature selection method")
    parser.add_argument("--inject-noise", type=int, default=0, help="Number of pure noise features to inject")
    parser.add_argument("--benchmark", action="store_true", help="Run downstream ML benchmark table")
    parser.add_argument("--stability", action="store_true", help="Compute bootstrap feature stability score")
    parser.add_argument("--save-pipeline", type=str, default=None, help="Save fitted pipeline to file path")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit Web UI")

    args = parser.parse_args()

    if args.ui:
        launch_ui()
    else:
        run_pipeline(
            dataset_name=args.dataset,
            k=args.k,
            method=args.method,
            inject_noise=args.inject_noise,
            run_benchmark=args.benchmark,
            run_stability=args.stability,
            save_pipeline_path=args.save_pipeline
        )

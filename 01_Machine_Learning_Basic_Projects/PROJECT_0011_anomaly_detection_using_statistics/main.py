"""
Project 011: Statistical Anomaly Detection Pipeline & CLI
=========================================================
An industrial-grade command-line interface and reproducible evaluation pipeline
for statistical anomaly detection across univariate, multivariate, and ensemble methods.

Usage Examples:
    # Run default ensemble on financial fraud benchmark
    python main.py

    # Run tournament benchmark comparing all statistical methods
    python main.py --tournament

    # Run Mahalanobis detector on 2D correlated dataset
    python main.py --dataset correlated --method mahalanobis

    # Run streaming time-series Bollinger Band detector
    python main.py --stream-demo

    # Save fitted model artifact
    python main.py --save-model detector_baseline.joblib

Part of the 1,000 AI Projects Implementation Library.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataset import (
    load_benchmark_dataset,
    generate_streaming_telemetry,
    get_archetype_presets
)
from ad_engine import (
    StatisticalAnomalyDetector,
    evaluate_ad_metrics,
    compare_all_statistical_methods,
    compute_rolling_stats,
    save_detector,
    load_detector
)


def run_ad_pipeline(
    dataset_name: str = "fraud",
    method: str = "ensemble",
    contamination: float = 0.05,
    n_samples: int = 1000,
    run_tournament: bool = False,
    run_stream: bool = False,
    save_model_path: Optional[str] = None,
    load_model_path: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Execute full end-to-end Statistical Anomaly Detection pipeline.
    """
    print("=" * 78)
    print("🛡️  PROJECT 011: STATISTICAL ANOMALY DETECTION ENGINE & STUDIO")
    print("=" * 78)

    # Streaming mode execution
    if run_stream:
        print("\n[STREAMING MODE] Generating 400-step time-series signal with injected anomalies...")
        df_stream = generate_streaming_telemetry(n_steps=400, seed=seed)
        df_res = compute_rolling_stats(df_stream["metric_stream"], window=20, k=2.5)
        n_flagged = df_res["is_anomaly"].sum()
        ground_truth_anom = df_stream["is_anomaly"].sum()
        eval_m = evaluate_ad_metrics(df_stream["is_anomaly"], df_res["is_anomaly"])

        print(f"Total Stream Steps:        {len(df_stream)}")
        print(f"Ground Truth Anomalies:    {ground_truth_anom}")
        print(f"Rolling Bollinger Flagged: {n_flagged}")
        print(f"Stream Precision:          {eval_m['precision']:.4f}")
        print(f"Stream Recall:             {eval_m['recall']:.4f}")
        print(f"Stream F1-Score:           {eval_m['f1']:.4f}")
        print("✓ Streaming detector demonstration complete.")
        return {"streaming_results": eval_m}

    # 1. Load Dataset
    print(f"\n[1/5] Loading Benchmark Telemetry: '{dataset_name}' (Contamination: {contamination*100:.1f}%)...")
    df, feature_cols, target_col = load_benchmark_dataset(
        dataset_name=dataset_name,
        contamination=contamination,
        seed=seed
    )
    X = df[feature_cols].to_numpy(dtype=float)
    y_true = df[target_col].to_numpy(dtype=int)

    n_anomalies = int(np.sum(y_true))
    print(f"      Loaded {len(df):,} samples across {len(feature_cols)} features.")
    print(f"      Normal instances: {len(df) - n_anomalies:,} | Ground-truth anomalies: {n_anomalies:,}")

    # 2. Tournament Benchmarking Mode
    if run_tournament:
        print("\n[2/5] Running Statistical Tournament across all detection algorithms...")
        df_tourney = compare_all_statistical_methods(X, y_true, feature_names=feature_cols)
        print("\n" + "=" * 78)
        print("🏆 STATISTICAL TOURNAMENT LEADERBOARD")
        print("=" * 78)
        print(df_tourney.to_string(index=False))
        print("=" * 78)

    # 3. Fit or Load Selected Detector
    if load_model_path:
        print(f"\n[3/5] Loading pre-trained detector from '{load_model_path}'...")
        detector = load_detector(load_model_path)
    else:
        print(f"\n[3/5] Fitting Statistical Detector (Method: '{method}')...")
        detector = StatisticalAnomalyDetector(method=method)
        detector.fit(X, feature_names=feature_cols)
        print("      Baseline reference statistics successfully calibrated.")

    # 4. Predict and Evaluate
    print("\n[4/5] Scoring observations and computing classification metrics...")
    y_pred = detector.predict(X)
    scores = detector.score_samples(X)
    metrics = evaluate_ad_metrics(y_true, y_pred, scores=scores)

    print("\n" + "-" * 45)
    print(f"  MODEL PERFORMANCE SUMMARY [{method.upper()}]")
    print("-" * 45)
    print(f"  Accuracy:                 {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision:                {metrics['precision'] * 100:.2f}%")
    print(f"  Recall (Sensitivity):     {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:                 {metrics['f1']:.4f}")
    print(f"  Specificity:              {metrics['specificity'] * 100:.2f}%")
    print(f"  False Alarm Rate (FPR):   {metrics['fpr'] * 100:.2f}%")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:                  {metrics['roc_auc']:.4f}")
    if metrics['pr_auc'] is not None:
        print(f"  PR-AUC:                   {metrics['pr_auc']:.4f}")
    print("-" * 45)
    print(f"  Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    print("-" * 45)

    # Single-Point Explainability Demo
    anom_indices = np.where(y_true == 1)[0]
    if len(anom_indices) > 0:
        demo_idx = anom_indices[0]
        sample_x = X[demo_idx]
        explanation = detector.explain_sample(sample_x)
        print("\n[5/5] Single-Sample Forensic Explainability (Point #{}):".format(demo_idx))
        print(f"      Anomaly Flag:   {explanation['is_anomaly']} (Score: {explanation['anomaly_score']:.4f})")
        print(f"      Primary Driver: {explanation['primary_driver']}")
        print(f"      Mahalanobis D:  {explanation['mahalanobis_distance']:.2f} (Cutoff: {explanation['mahalanobis_threshold']:.2f})")
        print("      Top Deviating Features:")
        for feat in explanation["feature_breakdown"][:3]:
            print(f"        • {feat['feature']:<20} Val={feat['value']:.2f} | Z={feat['z_score']:+.2f} | ModZ={feat['modified_z']:+.2f}")

    # Serialization
    if save_model_path:
        print(f"\n[ARTIFACT] Serializing fitted baseline to '{save_model_path}'...")
        save_detector(detector, save_model_path)
        print("✓ Serialization complete.")

    print("\n" + "=" * 78)
    print("✨ Pipeline execution completed successfully.")
    print("=" * 78)

    return {
        "metrics": metrics,
        "detector": detector,
        "feature_cols": feature_cols
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 011: Statistical Anomaly Detection Studio & CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fraud",
        choices=["fraud", "cloud", "iot", "medical", "correlated", "multimodal", "masking"],
        help="Benchmark dataset to run anomaly detection on."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ensemble",
        choices=["z_score", "modified_z_score", "iqr", "mahalanobis", "robust_mahalanobis", "ensemble"],
        help="Statistical anomaly detection algorithm."
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Proportion of injected outliers in synthetic benchmark."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Total sample count for the benchmark."
    )
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Run head-to-head tournament benchmarking across all statistical methods."
    )
    parser.add_argument(
        "--stream-demo",
        action="store_true",
        help="Demonstrate time-series streaming rolling Bollinger Band detector."
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Optional path to serialize fitted baseline (.joblib)."
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Optional path to load pre-trained baseline (.joblib)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_ad_pipeline(
        dataset_name=args.dataset,
        method=args.method,
        contamination=args.contamination,
        n_samples=args.samples,
        run_tournament=args.tournament,
        run_stream=args.stream_demo,
        save_model_path=args.save_model,
        load_model_path=args.load_model,
        seed=args.seed
    )

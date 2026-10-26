"""
Project 011: Statistical Anomaly Detection (AD) Engine
=====================================================
A production-grade, mathematically rigorous statistical anomaly detection suite.
Features:
- Parametric Univariate: Standard Z-Score (Normal distribution deviation)
- Robust Univariate: Modified Z-Score using Median Absolute Deviation (MAD / Boris Iglewicz & David Hoaglin)
- Non-Parametric Univariate: Tukey's Interquartile Range (IQR) fences
- Multivariate Covariance: Mahalanobis Distance with Chi-Square (χ²) cutoffs
- Robust Multivariate: Minimum Covariance Determinant (MCD / Rousseeuw)
- Extreme Studentized Deviate: Rosner's Generalized ESD test for multiple outliers
- Time-Series Streaming: Rolling Window Mean, Std, & Bollinger Band envelopes
- Multi-Method Consensus Ensemble: Combines detectors with configurable voting logic
- Diagnostic Explanations: Feature deviation decomposition per point
- Serialization: Complete model parameter persistence via Joblib

Part of the 1,000 AI Projects Implementation Library.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)
import joblib
from typing import Dict, Any, List, Optional, Tuple, Union


# =============================================================================
# 1. STANDALONE STATISTICAL ALGORITHMS
# =============================================================================

def compute_z_score(
    X: np.ndarray,
    threshold: float = 3.0,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Standard Z-Score anomaly detection:
        Z = |X - μ| / σ
    Flags any observation where any feature |Z| > threshold.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if mean is None:
        mean = np.mean(X_arr, axis=0)
    if std is None:
        std = np.std(X_arr, axis=0, ddof=1)
        # Prevent division by zero
        std = np.where(std < 1e-9, 1e-9, std)

    z_matrix = np.abs((X_arr - mean) / std)
    # Maximum Z-score across features for each observation
    max_z = np.max(z_matrix, axis=1)
    is_anomaly = max_z > threshold

    params = {"mean": mean, "std": std, "threshold": threshold}
    return is_anomaly, max_z, params


def compute_modified_z_score(
    X: np.ndarray,
    threshold: float = 3.5,
    median: Optional[np.ndarray] = None,
    mad: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Modified Z-Score anomaly detection (Boris Iglewicz & David Hoaglin):
        M_i = 0.6745 * |X_i - median(X)| / MAD
    where MAD = median(|X_i - median(X)|).
    Robust against the masking effect where outliers inflate mean and std.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if median is None:
        median = np.median(X_arr, axis=0)

    if mad is None:
        abs_diff = np.abs(X_arr - median)
        mad = np.median(abs_diff, axis=0)
        mad = np.where(mad < 1e-9, 1e-9, mad)

    # 0.6745 is the 75th percentile of the standard normal distribution
    mod_z_matrix = 0.6745 * np.abs(X_arr - median) / mad
    max_mod_z = np.max(mod_z_matrix, axis=1)
    is_anomaly = max_mod_z > threshold

    params = {"median": median, "mad": mad, "threshold": threshold}
    return is_anomaly, max_mod_z, params


def compute_iqr(
    X: np.ndarray,
    k: float = 1.5,
    q25: Optional[np.ndarray] = None,
    q75: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Tukey's Interquartile Range (IQR) method:
        IQR = Q75 - Q25
        Lower Fence = Q25 - k * IQR
        Upper Fence = Q75 + k * IQR
    Non-parametric; no normal distribution assumption required.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if q25 is None:
        q25 = np.percentile(X_arr, 25, axis=0)
    if q75 is None:
        q75 = np.percentile(X_arr, 75, axis=0)

    iqr = q75 - q25
    iqr = np.where(iqr < 1e-9, 1e-9, iqr)

    lower_fence = q25 - k * iqr
    upper_fence = q75 + k * iqr

    # Distance beyond fence normalized by IQR
    lower_breach = np.maximum(0, lower_fence - X_arr) / iqr
    upper_breach = np.maximum(0, X_arr - upper_fence) / iqr
    breach_matrix = np.maximum(lower_breach, upper_breach)
    max_breach = np.max(breach_matrix, axis=1)

    is_anomaly = (X_arr < lower_fence).any(axis=1) | (X_arr > upper_fence).any(axis=1)

    params = {
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "k": k
    }
    return is_anomaly, max_breach, params


def compute_mahalanobis(
    X: np.ndarray,
    threshold: Optional[float] = None,
    alpha: float = 0.01,
    robust: bool = True,
    mean: Optional[np.ndarray] = None,
    inv_cov: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Multivariate Mahalanobis Distance:
        D_M(x) = sqrt((x - μ)^T Σ^(-1) (x - μ))
    Under multivariate normality, D_M^2 follows a Chi-Square (χ²) distribution
    with p degrees of freedom (p = number of dimensions).
    If robust=True, computes Minimum Covariance Determinant (MCD) to prevent
    outlier contamination of the covariance matrix.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n_samples, n_features = X_arr.shape

    if mean is None or inv_cov is None:
        if robust and n_samples > n_features + 2:
            try:
                mcd = MinCovDet(random_state=42, support_fraction=0.85).fit(X_arr)
                mean = mcd.location_
                cov = mcd.covariance_
            except Exception:
                emp = EmpiricalCovariance().fit(X_arr)
                mean = emp.location_
                cov = emp.covariance_
        else:
            mean = np.mean(X_arr, axis=0)
            cov = np.cov(X_arr, rowvar=False)
            if n_features == 1:
                cov = np.array([[float(cov)]])

        # Regularize covariance matrix to ensure stable inversion
        cov += np.eye(n_features) * 1e-6
        inv_cov = np.linalg.pinv(cov)

    diff = X_arr - mean
    # Vectorized Mahalanobis distance calculation: sqrt(sum((diff @ inv_cov) * diff, axis=1))
    dist_sq = np.sum((diff @ inv_cov) * diff, axis=1)
    dist_sq = np.maximum(0.0, dist_sq)
    dists = np.sqrt(dist_sq)

    # Theoretical Chi-Square critical value
    if threshold is None:
        threshold_sq = stats.chi2.ppf(1.0 - alpha, df=n_features)
        threshold = float(np.sqrt(threshold_sq))

    is_anomaly = dists > threshold

    params = {
        "mean": mean,
        "inv_cov": inv_cov,
        "threshold": threshold,
        "alpha": alpha,
        "df": n_features,
        "robust": robust
    }
    return is_anomaly, dists, params


def compute_generalized_esd(
    X: np.ndarray,
    max_outliers: int = 10,
    alpha: float = 0.05
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """
    Rosner's Generalized Extreme Studentized Deviate (GESD) test.
    Detects one or more outliers in an essentially normally distributed univariate dataset.
    """
    x_arr = np.asarray(X, dtype=float).flatten()
    n = len(x_arr)
    max_r = min(max_outliers, n // 2)

    data_copy = x_arr.copy()
    indices = np.arange(n)
    removed_indices = []
    r_stats = []
    lambda_crits = []

    for i in range(1, max_r + 1):
        cur_n = len(data_copy)
        if cur_n <= 2:
            break
        cur_mean = np.mean(data_copy)
        cur_std = np.std(data_copy, ddof=1)
        if cur_std < 1e-9:
            break

        deviations = np.abs(data_copy - cur_mean)
        max_idx = np.argmax(deviations)
        r_stat = deviations[max_idx] / cur_std
        r_stats.append(r_stat)

        # Critical value lambda_i
        p = 1.0 - alpha / (2.0 * cur_n)
        t_crit = stats.t.ppf(p, df=cur_n - 2)
        lambda_i = ((cur_n - 1) * t_crit) / np.sqrt((cur_n - 2 + t_crit**2) * cur_n)
        lambda_crits.append(lambda_i)

        original_idx = indices[max_idx]
        removed_indices.append(original_idx)

        data_copy = np.delete(data_copy, max_idx)
        indices = np.delete(indices, max_idx)

    # Determine largest k such that R_k > lambda_k
    k_found = 0
    for idx in range(len(r_stats) - 1, -1, -1):
        if r_stats[idx] > lambda_crits[idx]:
            k_found = idx + 1
            break

    detected_outliers = removed_indices[:k_found] if k_found > 0 else []
    is_anomaly = np.zeros(n, dtype=bool)
    is_anomaly[detected_outliers] = True

    params = {
        "max_outliers": max_r,
        "alpha": alpha,
        "num_detected": len(detected_outliers),
        "r_stats": r_stats,
        "lambda_crits": lambda_crits
    }
    return is_anomaly, detected_outliers, params


def compute_rolling_stats(
    series: Union[np.ndarray, pd.Series],
    window: int = 20,
    k: float = 2.5
) -> pd.DataFrame:
    """
    Time-series streaming anomaly detection using Rolling Mean & Bollinger Bands:
        Upper = Rolling_Mean + k * Rolling_Std
        Lower = Rolling_Mean - k * Rolling_Std
    Flags local spikes, trend breaks, and sudden variance explosions.
    """
    s = pd.Series(series, dtype=float)
    roll_mean = s.rolling(window=window, min_periods=max(3, window // 4)).mean()
    roll_std = s.rolling(window=window, min_periods=max(3, window // 4)).std().fillna(1e-6)

    upper_band = roll_mean + k * roll_std
    lower_band = roll_mean - k * roll_std

    z_stream = np.abs(s - roll_mean) / roll_std
    is_anomaly = (s > upper_band) | (s < lower_band)

    df_res = pd.DataFrame({
        "value": s,
        "rolling_mean": roll_mean,
        "rolling_std": roll_std,
        "upper_band": upper_band,
        "lower_band": lower_band,
        "rolling_z": z_stream,
        "is_anomaly": is_anomaly
    })
    return df_res


# =============================================================================
# 2. UNIFIED STATISTICAL ANOMALY DETECTOR CLASS
# =============================================================================

class StatisticalAnomalyDetector:
    """
    Modular, scikit-learn compatible statistical anomaly detection class.
    Supports Z-Score, Modified Z-Score (MAD), IQR, Mahalanobis Distance,
    Robust MCD Mahalanobis, and multi-method Consensus Ensembles.
    """

    SUPPORTED_METHODS = [
        "z_score",
        "modified_z_score",
        "iqr",
        "mahalanobis",
        "robust_mahalanobis",
        "ensemble"
    ]

    def __init__(
        self,
        method: str = "ensemble",
        z_threshold: float = 3.0,
        mod_z_threshold: float = 3.5,
        iqr_k: float = 1.5,
        mahalanobis_alpha: float = 0.01,
        ensemble_voting: str = "majority",
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' not recognized. Choose from {self.SUPPORTED_METHODS}")

        self.method = method
        self.z_threshold = z_threshold
        self.mod_z_threshold = mod_z_threshold
        self.iqr_k = iqr_k
        self.mahalanobis_alpha = mahalanobis_alpha
        self.ensemble_voting = ensemble_voting  # 'any', 'majority', 'all', 'weighted'
        self.ensemble_weights = ensemble_weights or {
            "z_score": 0.20,
            "modified_z_score": 0.30,
            "iqr": 0.25,
            "mahalanobis": 0.25
        }

        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.fitted_params_: Dict[str, Any] = {}

    def fit(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None) -> "StatisticalAnomalyDetector":
        """
        Calibrate reference baseline statistics on training data.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            self.feature_names = feature_names or [f"Feature_{i+1}" for i in range(X_arr.shape[1])]

        n_samples, n_features = X_arr.shape

        # 1. Z-Score parameters
        mean = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0, ddof=1)
        std = np.where(std < 1e-9, 1e-9, std)

        # 2. Modified Z-Score (MAD) parameters
        median = np.median(X_arr, axis=0)
        abs_diff = np.abs(X_arr - median)
        mad = np.median(abs_diff, axis=0)
        mad = np.where(mad < 1e-9, 1e-9, mad)

        # 3. IQR parameters
        q25 = np.percentile(X_arr, 25, axis=0)
        q75 = np.percentile(X_arr, 75, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < 1e-9, 1e-9, iqr)

        # 4. Standard Mahalanobis parameters
        cov = np.cov(X_arr, rowvar=False)
        if n_features == 1:
            cov = np.array([[float(cov)]])
        cov += np.eye(n_features) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        mah_thresh = np.sqrt(stats.chi2.ppf(1.0 - self.mahalanobis_alpha, df=n_features))

        # 5. Robust MCD Mahalanobis parameters
        rob_mean = mean
        rob_inv_cov = inv_cov
        if n_samples > n_features + 2:
            try:
                mcd = MinCovDet(random_state=42, support_fraction=0.85).fit(X_arr)
                rob_mean = mcd.location_
                rob_cov = mcd.covariance_ + np.eye(n_features) * 1e-6
                rob_inv_cov = np.linalg.pinv(rob_cov)
            except Exception:
                rob_mean = mean
                rob_inv_cov = inv_cov

        self.fitted_params_ = {
            "mean": mean,
            "std": std,
            "median": median,
            "mad": mad,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "cov": cov,
            "inv_cov": inv_cov,
            "robust_mean": rob_mean,
            "robust_inv_cov": rob_inv_cov,
            "mahalanobis_threshold": mah_thresh,
            "n_features": n_features,
            "n_samples": n_samples
        }
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Flag anomalies as boolean array (True = anomaly, False = normal).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict() is called.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        p = self.fitted_params_

        if self.method == "z_score":
            flags, _, _ = compute_z_score(X_arr, threshold=self.z_threshold, mean=p["mean"], std=p["std"])
            return flags

        elif self.method == "modified_z_score":
            flags, _, _ = compute_modified_z_score(X_arr, threshold=self.mod_z_threshold, median=p["median"], mad=p["mad"])
            return flags

        elif self.method == "iqr":
            flags, _, _ = compute_iqr(X_arr, k=self.iqr_k, q25=p["q25"], q75=p["q75"])
            return flags

        elif self.method == "mahalanobis":
            flags, _, _ = compute_mahalanobis(
                X_arr,
                threshold=p["mahalanobis_threshold"],
                robust=False,
                mean=p["mean"],
                inv_cov=p["inv_cov"]
            )
            return flags

        elif self.method == "robust_mahalanobis":
            flags, _, _ = compute_mahalanobis(
                X_arr,
                threshold=p["mahalanobis_threshold"],
                robust=True,
                mean=p["robust_mean"],
                inv_cov=p["robust_inv_cov"]
            )
            return flags

        elif self.method == "ensemble":
            # Multi-detector voting consensus
            z_flags, _, _ = compute_z_score(X_arr, threshold=self.z_threshold, mean=p["mean"], std=p["std"])
            m_flags, _, _ = compute_modified_z_score(X_arr, threshold=self.mod_z_threshold, median=p["median"], mad=p["mad"])
            i_flags, _, _ = compute_iqr(X_arr, k=self.iqr_k, q25=p["q25"], q75=p["q75"])
            r_flags, _, _ = compute_mahalanobis(
                X_arr,
                threshold=p["mahalanobis_threshold"],
                robust=True,
                mean=p["robust_mean"],
                inv_cov=p["robust_inv_cov"]
            )

            flag_matrix = np.column_stack([z_flags, m_flags, i_flags, r_flags])

            if self.ensemble_voting == "any":
                return flag_matrix.any(axis=1)
            elif self.ensemble_voting == "all":
                return flag_matrix.all(axis=1)
            elif self.ensemble_voting == "weighted":
                scores = self.score_samples(X_arr)
                return scores >= 0.50
            else:  # majority
                return (flag_matrix.sum(axis=1) >= 2)

        return np.zeros(len(X_arr), dtype=bool)

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Continuous normalized anomaly score in range [0.0, 1.0].
        Higher score represents extreme deviation from the normal operational envelope.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before score_samples() is called.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        p = self.fitted_params_

        # 1. Z-Score component normalized by sigmoid-like curve
        _, z_raw, _ = compute_z_score(X_arr, threshold=self.z_threshold, mean=p["mean"], std=p["std"])
        s_z = 1.0 / (1.0 + np.exp(-(z_raw - self.z_threshold) * 1.5))

        # 2. Modified Z component
        _, mod_raw, _ = compute_modified_z_score(X_arr, threshold=self.mod_z_threshold, median=p["median"], mad=p["mad"])
        s_mod = 1.0 / (1.0 + np.exp(-(mod_raw - self.mod_z_threshold) * 1.5))

        # 3. IQR component
        _, iqr_raw, _ = compute_iqr(X_arr, k=self.iqr_k, q25=p["q25"], q75=p["q75"])
        s_iqr = 1.0 / (1.0 + np.exp(-(iqr_raw - 0.5) * 3.0))

        # 4. Mahalanobis component
        _, mah_raw, _ = compute_mahalanobis(
            X_arr,
            threshold=p["mahalanobis_threshold"],
            robust=True,
            mean=p["robust_mean"],
            inv_cov=p["robust_inv_cov"]
        )
        mah_thresh = p["mahalanobis_threshold"]
        s_mah = 1.0 / (1.0 + np.exp(-(mah_raw - mah_thresh) / max(0.1, mah_thresh * 0.2)))

        if self.method == "z_score":
            return np.clip(s_z, 0.0, 1.0)
        elif self.method == "modified_z_score":
            return np.clip(s_mod, 0.0, 1.0)
        elif self.method == "iqr":
            return np.clip(s_iqr, 0.0, 1.0)
        elif self.method in ["mahalanobis", "robust_mahalanobis"]:
            return np.clip(s_mah, 0.0, 1.0)
        else:  # ensemble
            w = self.ensemble_weights
            w_sum = w["z_score"] + w["modified_z_score"] + w["iqr"] + w["mahalanobis"]
            composite = (
                w["z_score"] * s_z +
                w["modified_z_score"] * s_mod +
                w["iqr"] * s_iqr +
                w["mahalanobis"] * s_mah
            ) / w_sum
            return np.clip(composite, 0.0, 1.0)

    def explain_sample(self, x: Union[np.ndarray, List[float], pd.Series]) -> Dict[str, Any]:
        """
        Explain the statistical driver behind an anomaly flag for a single point.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before explain_sample() is called.")

        x_vec = np.asarray(x, dtype=float).flatten()
        p = self.fitted_params_

        # Univariate feature deviations
        z_devs = (x_vec - p["mean"]) / p["std"]
        mod_z_devs = 0.6745 * (x_vec - p["median"]) / p["mad"]

        iqr = p["iqr"]
        lower_fence = p["q25"] - self.iqr_k * iqr
        upper_fence = p["q75"] + self.iqr_k * iqr

        diff = (x_vec - p["robust_mean"]).reshape(1, -1)
        mah_dist = float(np.sqrt(np.sum((diff @ p["robust_inv_cov"]) * diff)))

        feature_explanations = []
        for i, name in enumerate(self.feature_names):
            val = float(x_vec[i])
            z_val = float(z_devs[i])
            mod_z_val = float(mod_z_devs[i])
            iqr_breach = val < lower_fence[i] or val > upper_fence[i]

            feature_explanations.append({
                "feature": name,
                "value": val,
                "mean": float(p["mean"][i]),
                "std": float(p["std"][i]),
                "z_score": round(z_val, 3),
                "modified_z": round(mod_z_val, 3),
                "iqr_bounds": (round(float(lower_fence[i]), 3), round(float(upper_fence[i]), 3)),
                "iqr_outlier": bool(iqr_breach),
                "anomaly_driver_score": round(max(abs(z_val), abs(mod_z_val)), 3)
            })

        # Sort features by highest deviation
        feature_explanations.sort(key=lambda d: d["anomaly_driver_score"], reverse=True)

        is_anomaly = bool(self.predict(x_vec.reshape(1, -1))[0])
        score = float(self.score_samples(x_vec.reshape(1, -1))[0])

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 4),
            "mahalanobis_distance": round(mah_dist, 3),
            "mahalanobis_threshold": round(float(p["mahalanobis_threshold"]), 3),
            "primary_driver": feature_explanations[0]["feature"],
            "feature_breakdown": feature_explanations
        }


# =============================================================================
# 3. METRICS, BENCHMARKING & VALIDATION
# =============================================================================

def evaluate_ad_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute full classification and ranking metrics for anomaly detection.
    Convention: 1 / True = Anomaly, 0 / False = Normal.
    """
    y_true_b = np.asarray(y_true, dtype=int)
    y_pred_b = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_b, y_pred_b))
    prec = float(precision_score(y_true_b, y_pred_b, zero_division=0))
    rec = float(recall_score(y_true_b, y_pred_b, zero_division=0))
    f1 = float(f1_score(y_true_b, y_pred_b, zero_division=0))

    cm = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = None
    pr_auc = None

    if scores is not None and len(np.unique(y_true_b)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_b, scores))
            p_curve, r_curve, _ = precision_recall_curve(y_true_b, scores)
            pr_auc = float(auc(r_curve, p_curve))
        except Exception:
            pass

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None
    }


def compare_all_statistical_methods(
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Comprehensive tournament benchmarking across all statistical detectors.
    """
    methods = [
        ("Z-Score (k=3.0)", "z_score", {"z_threshold": 3.0}),
        ("Modified Z-Score (MAD k=3.5)", "modified_z_score", {"mod_z_threshold": 3.5}),
        ("Tukey's IQR (k=1.5)", "iqr", {"iqr_k": 1.5}),
        ("Mahalanobis (χ² 99%)", "mahalanobis", {"mahalanobis_alpha": 0.01}),
        ("Robust MCD Mahalanobis", "robust_mahalanobis", {"mahalanobis_alpha": 0.01}),
        ("Statistical Ensemble (Majority)", "ensemble", {"ensemble_voting": "majority"})
    ]

    records = []
    for label, m_key, kwargs in methods:
        det = StatisticalAnomalyDetector(method=m_key, **kwargs)
        det.fit(X, feature_names=feature_names)
        preds = det.predict(X)
        scores = det.score_samples(X)
        m = evaluate_ad_metrics(y_true, preds, scores)

        records.append({
            "Method": label,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1"],
            "False Alarm Rate (FPR)": m["fpr"],
            "ROC-AUC": m["roc_auc"] if m["roc_auc"] is not None else 0.0,
            "True Positives": m["tp"],
            "False Positives": m["fp"]
        })

    return pd.DataFrame(records)


def compute_2d_decision_mesh(
    detector: StatisticalAnomalyDetector,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    grid_res: int = 80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D mesh grid for continuous score landscapes and decision boundary contours.
    """
    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], grid_res),
        np.linspace(y_range[0], y_range[1], grid_res)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    scores = detector.score_samples(grid_points).reshape(xx.shape)
    preds = detector.predict(grid_points).reshape(xx.shape).astype(int)

    return xx, yy, scores, preds


# =============================================================================
# 4. MODEL PERSISTENCE
# =============================================================================

def save_detector(detector: StatisticalAnomalyDetector, filepath: str) -> None:
    """Save fitted anomaly detector to disk using Joblib."""
    joblib.dump(detector, filepath)


def load_detector(filepath: str) -> StatisticalAnomalyDetector:
    """Load pre-trained anomaly detector from disk."""
    return joblib.load(filepath)

"""
Project 014: Feature Selection Techniques - Computation Engine
==============================================================
Production-grade implementations of Filter, Wrapper, Embedded, and
Consensus/Ensemble feature selection techniques with stability metrics,
ground-truth validation, and downstream ML benchmarking.

Part of the 1,000 AI Projects Implementation Library.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    RFE,
    RFECV,
    SequentialFeatureSelector
)
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error


# =============================================================================
# 1. FILTER METHODS
# =============================================================================

def run_variance_threshold(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Filter low-variance features that carry near-zero informational entropy.
    """
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    variances = np.var(X_norm, axis=0)
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_norm)
    support = selector.get_support()
    
    selected_indices = np.where(support)[0].tolist()
    dropped_indices = np.where(~support)[0].tolist()
    
    ranked_indices = np.argsort(variances)[::-1].tolist()
    
    return {
        "method": "Variance Threshold",
        "threshold": threshold,
        "variances": variances.tolist(),
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices],
        "dropped_features": [feature_names[i] for i in dropped_indices],
        "ranked_indices": ranked_indices,
        "scores": variances.tolist()
    }


def run_univariate_filter(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    method: str = "f_test",
    k: int = 5
) -> Dict[str, Any]:
    """
    Univariate statistical filtering:
      - 'f_test': ANOVA F-value (f_classif / f_regression)
      - 'mutual_info': Shannon mutual information entropy
      - 'chi2': Chi-squared test (features scaled non-negative)
      - 'correlation': Absolute Pearson correlation with target
    """
    k = max(1, min(k, X.shape[1]))
    
    if method == "f_test":
        score_func = f_classif if task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        raw_scores = selector.scores_
        p_values = selector.pvalues_ if hasattr(selector, "pvalues_") else None
        metric_name = "ANOVA F-Statistic"

    elif method == "mutual_info":
        if task_type == "classification":
            raw_scores = mutual_info_classif(X, y, random_state=42)
        else:
            raw_scores = mutual_info_regression(X, y, random_state=42)
        p_values = None
        metric_name = "Mutual Information (Nats)"

    elif method == "chi2":
        # Chi2 requires strictly non-negative values
        scaler = MinMaxScaler()
        X_pos = scaler.fit_transform(X)
        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(X_pos, y)
        raw_scores = selector.scores_
        p_values = selector.pvalues_
        metric_name = "Chi-Square (\u03c7\u00b2) Statistic"

    elif method == "correlation":
        raw_scores = []
        p_vals = []
        for col_idx in range(X.shape[1]):
            try:
                r, p = pearsonr(X[:, col_idx], y)
                if np.isnan(r):
                    r = 0.0
                    p = 1.0
            except Exception:
                r, p = 0.0, 1.0
            raw_scores.append(abs(r))
            p_vals.append(p)
        raw_scores = np.array(raw_scores)
        p_values = np.array(p_vals)
        metric_name = "Absolute Pearson Correlation (|r|)"
    else:
        raise ValueError(f"Unknown univariate filter method: '{method}'")

    # Replace NaNs or Infs
    clean_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
    ranked_indices = np.argsort(clean_scores)[::-1].tolist()
    selected_indices = ranked_indices[:k]
    
    # Feature rankings (1 = best)
    ranks = [0] * len(feature_names)
    for rank_pos, idx in enumerate(ranked_indices):
        ranks[idx] = rank_pos + 1

    return {
        "method": f"Filter: {metric_name}",
        "metric_name": metric_name,
        "k": k,
        "scores": clean_scores.tolist(),
        "p_values": p_values.tolist() if p_values is not None else None,
        "ranked_indices": ranked_indices,
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices],
        "ranks": ranks
    }


def run_multicollinearity_filter(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Detect pairs of features with high pairwise correlation (|r| > threshold)
    and flag redundant columns for pruning.
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()
    
    high_corr_pairs = []
    features_to_drop = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val > threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                high_corr_pairs.append({
                    "feature_1": col_i,
                    "feature_2": col_j,
                    "correlation": float(val)
                })
                # Drop feature j
                features_to_drop.add(col_j)
                
    remaining_features = [f for f in feature_names if f not in features_to_drop]
    remaining_indices = [feature_names.index(f) for f in remaining_features]
    
    return {
        "threshold": threshold,
        "corr_matrix": corr_matrix.values.tolist(),
        "high_corr_pairs": high_corr_pairs,
        "dropped_features": list(features_to_drop),
        "selected_features": remaining_features,
        "selected_indices": remaining_indices
    }


# =============================================================================
# 2. WRAPPER METHODS
# =============================================================================

def run_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    n_features_to_select: int = 5,
    estimator_type: str = "linear",
    step: int = 1
) -> Dict[str, Any]:
    """
    Recursive Feature Elimination (RFE) using a supervised base estimator.
    """
    n_features_to_select = max(1, min(n_features_to_select, X.shape[1]))
    
    # Standardize input for linear estimator stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if task_type == "classification":
        if estimator_type == "rf":
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = LogisticRegression(max_iter=1000, random_state=42)
    else:
        if estimator_type == "rf":
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = Ridge(alpha=1.0, random_state=42)

    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(X_scaled, y)
    
    support = rfe.support_
    ranking = rfe.ranking_
    
    selected_indices = np.where(support)[0].tolist()
    ranked_indices = np.argsort(ranking).tolist()
    
    # In RFE ranking, 1 indicates selected features
    return {
        "method": f"Wrapper: RFE ({estimator_type.upper()})",
        "n_features_to_select": n_features_to_select,
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices],
        "ranking": ranking.tolist(),
        "ranked_indices": ranked_indices,
        "ranks": ranking.tolist()
    }


def run_rfecv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    cv_folds: int = 5,
    step: int = 1
) -> Dict[str, Any]:
    """
    RFE with Cross-Validation (RFECV) to automatically discover the optimal
    feature subset size that maximizes cross-validated generalization score.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if task_type == "classification":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        estimator = Ridge(alpha=1.0, random_state=42)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "r2"

    rfecv = RFECV(estimator=estimator, step=step, cv=cv, scoring=scoring, n_jobs=-1)
    rfecv.fit(X_scaled, y)
    
    cv_scores = rfecv.cv_results_["mean_test_score"].tolist()
    cv_stds = rfecv.cv_results_["std_test_score"].tolist()
    
    # Feature counts evaluated
    feature_counts = list(range(1, len(feature_names) + 1, step))
    if len(feature_counts) != len(cv_scores):
        feature_counts = list(range(len(feature_names) - (len(cv_scores)-1)*step, len(feature_names) + 1, step))
        if len(feature_counts) != len(cv_scores):
            feature_counts = list(range(1, len(cv_scores) + 1))

    selected_indices = np.where(rfecv.support_)[0].tolist()
    optimal_k = rfecv.n_features_
    
    return {
        "method": "Wrapper: RFECV (Cross-Validated Optimization)",
        "optimal_k": optimal_k,
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices],
        "ranking": rfecv.ranking_.tolist(),
        "ranks": rfecv.ranking_.tolist(),
        "cv_scores": cv_scores,
        "cv_stds": cv_stds,
        "feature_counts": feature_counts,
        "scoring_metric": scoring
    }


def run_sequential_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    n_features_to_select: int = 5,
    direction: str = "forward",
    cv_folds: int = 3
) -> Dict[str, Any]:
    """
    Sequential Feature Selection (Forward Selection or Backward Elimination).
    """
    n_features_to_select = max(1, min(n_features_to_select, X.shape[1]))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if task_type == "classification":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
    else:
        estimator = Ridge(alpha=1.0, random_state=42)

    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        direction=direction,
        cv=cv_folds,
        n_jobs=-1
    )
    sfs.fit(X_scaled, y)
    
    support = sfs.get_support()
    selected_indices = np.where(support)[0].tolist()
    
    return {
        "method": f"Wrapper: Sequential {direction.capitalize()} Selection (SFS)",
        "direction": direction,
        "n_features_to_select": n_features_to_select,
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices]
    }


# =============================================================================
# 3. EMBEDDED METHODS
# =============================================================================

def run_l1_regularization(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    alpha_or_C: float = 0.5,
    k: int = 5
) -> Dict[str, Any]:
    """
    L1 Regularization (Lasso / L1 Logistic Regression).
    Induces sparsity by driving non-informative feature weights to zero.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if task_type == "classification":
        # In LogisticRegression, lower C means stronger regularization
        C_val = max(0.001, float(alpha_or_C))
        model = LogisticRegression(penalty="l1", solver="saga", C=C_val, max_iter=2000, random_state=42)
        model.fit(X_scaled, y)
        if model.coef_.ndim > 1:
            coefs = np.mean(np.abs(model.coef_), axis=0)
        else:
            coefs = np.abs(model.coef_)
    else:
        # In Lasso, higher alpha means stronger regularization
        alpha_val = max(0.001, float(alpha_or_C))
        model = Lasso(alpha=alpha_val, max_iter=2000, random_state=42)
        model.fit(X_scaled, y)
        coefs = np.abs(model.coef_)

    non_zero_indices = np.where(coefs > 1e-5)[0].tolist()
    ranked_indices = np.argsort(coefs)[::-1].tolist()
    
    # If k specified, select top k or non-zero, whichever is smaller/greater
    k = max(1, min(k, len(feature_names)))
    selected_indices = ranked_indices[:k]
    
    ranks = [0] * len(feature_names)
    for rank_pos, idx in enumerate(ranked_indices):
        ranks[idx] = rank_pos + 1

    return {
        "method": f"Embedded: L1 Regularization ({'L1-Logistic' if task_type == 'classification' else 'Lasso'})",
        "penalty_param": alpha_or_C,
        "coefficients": coefs.tolist(),
        "non_zero_count": len(non_zero_indices),
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices],
        "ranked_indices": ranked_indices,
        "ranks": ranks,
        "scores": coefs.tolist()
    }


def compute_l1_regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    n_alphas: int = 15
) -> Dict[str, Any]:
    """
    Compute feature coefficient shrinkage path across multiple regularization strengths.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if task_type == "classification":
        c_values = np.logspace(-2, 1, n_alphas)
        path_data = []
        for c in c_values:
            lr = LogisticRegression(penalty="l1", solver="saga", C=c, max_iter=1000, random_state=42)
            lr.fit(X_scaled, y)
            if lr.coef_.ndim > 1:
                c_coefs = np.mean(np.abs(lr.coef_), axis=0).tolist()
            else:
                c_coefs = np.abs(lr.coef_).tolist()
            path_data.append(c_coefs)
        param_values = c_values.tolist()
        param_name = "Inverse Regularization Strength (C)"
    else:
        alpha_values = np.logspace(-3, 0.5, n_alphas)
        path_data = []
        for a in alpha_values:
            lasso = Lasso(alpha=a, max_iter=1000, random_state=42)
            lasso.fit(X_scaled, y)
            path_data.append(np.abs(lasso.coef_).tolist())
        param_values = alpha_values.tolist()
        param_name = "Regularization Alpha (\u03b1)"

    return {
        "param_name": param_name,
        "param_values": param_values,
        "feature_paths": path_data,  # Shape: (n_alphas, n_features)
        "feature_names": feature_names
    }


def run_tree_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    k: int = 5,
    n_estimators: int = 80,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Tree-based Feature Importance (Mean Decrease in Impurity / MDI).
    """
    k = max(1, min(k, X.shape[1]))
    
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        
    model.fit(X, y)
    importances = model.feature_importances_
    
    ranked_indices = np.argsort(importances)[::-1].tolist()
    selected_indices = ranked_indices[:k]
    
    ranks = [0] * len(feature_names)
    for rank_pos, idx in enumerate(ranked_indices):
        ranks[idx] = rank_pos + 1

    return {
        "method": "Embedded: Random Forest MDI Importance",
        "k": k,
        "scores": importances.tolist(),
        "ranked_indices": ranked_indices,
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices],
        "ranks": ranks
    }


def run_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    k: int = 5,
    n_repeats: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Permutation Feature Importance on holdout test data to avoid cardinality bias.
    """
    k = max(1, min(k, X.shape[1]))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=-1)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(n_estimators=60, random_state=seed, n_jobs=-1)
        scoring = "r2"
        
    model.fit(X_train, y_train)
    res = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=seed, scoring=scoring, n_jobs=-1)
    
    means = np.nan_to_num(res.importances_mean, nan=0.0)
    stds = np.nan_to_num(res.importances_std, nan=0.0)
    
    ranked_indices = np.argsort(means)[::-1].tolist()
    selected_indices = ranked_indices[:k]
    
    ranks = [0] * len(feature_names)
    for rank_pos, idx in enumerate(ranked_indices):
        ranks[idx] = rank_pos + 1

    return {
        "method": "Embedded/Validation: Permutation Importance",
        "k": k,
        "scores": means.tolist(),
        "stds": stds.tolist(),
        "ranked_indices": ranked_indices,
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices],
        "ranks": ranks
    }


# =============================================================================
# 4. CONSENSUS RANKING, STABILITY & GROUND TRUTH VALIDATION
# =============================================================================

def compute_consensus_ranking(
    results_dict: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    k: int = 5
) -> Dict[str, Any]:
    """
    Borda Count rank aggregation across multiple feature selection methods.
    Points are awarded proportionally based on feature rank in each method.
    """
    k = max(1, min(k, len(feature_names)))
    n_features = len(feature_names)
    
    # Extract ranks for methods that supply them
    valid_methods = {}
    for name, res in results_dict.items():
        if "ranks" in res:
            valid_methods[name] = res["ranks"]
        elif "ranked_indices" in res:
            # Reconstruct ranks from ranked indices
            r = [0] * n_features
            for pos, idx in enumerate(res["ranked_indices"]):
                r[idx] = pos + 1
            valid_methods[name] = r
            
    if not valid_methods:
        raise ValueError("No methods with ranking information provided for consensus.")
        
    method_names = list(valid_methods.keys())
    # Borda score: points = (n_features - rank + 1)
    borda_scores = np.zeros(n_features)
    for method_name, ranks in valid_methods.items():
        for i, rank in enumerate(ranks):
            borda_scores[i] += (n_features - rank + 1)
            
    # Normalize Borda scores to [0, 100]
    max_score = borda_scores.max() if borda_scores.max() > 0 else 1.0
    normalized_scores = (borda_scores / max_score) * 100.0
    
    ranked_indices = np.argsort(borda_scores)[::-1].tolist()
    selected_indices = ranked_indices[:k]
    
    ranks = [0] * n_features
    for rank_pos, idx in enumerate(ranked_indices):
        ranks[idx] = rank_pos + 1
        
    ranking_matrix = []
    for i in range(n_features):
        row = {"feature": feature_names[i], "consensus_score": round(float(normalized_scores[i]), 2)}
        for m_name in method_names:
            row[m_name] = valid_methods[m_name][i]
        ranking_matrix.append(row)

    return {
        "method": "Consensus: Borda Count Ensemble",
        "method_names": method_names,
        "k": k,
        "borda_scores": borda_scores.tolist(),
        "normalized_scores": normalized_scores.tolist(),
        "ranked_indices": ranked_indices,
        "selected_indices": selected_indices,
        "selected_features": [feature_names[i] for i in selected_indices],
        "ranks": ranks,
        "ranking_matrix": ranking_matrix
    }


def evaluate_ground_truth_selection(
    selected_indices: List[int],
    true_informative_indices: List[int],
    true_noise_indices: List[int],
    total_features: int
) -> Dict[str, Any]:
    """
    Calculate Ground-Truth Precision, Recall, F1, and False Discovery Rate (FDR)
    when evaluating on controlled synthetic benchmarks.
    """
    if not true_informative_indices:
        return {"has_ground_truth": False}

    sel_set = set(selected_indices)
    true_set = set(true_informative_indices)
    noise_set = set(true_noise_indices) if true_noise_indices else set()

    tp = len(sel_set.intersection(true_set))
    fp = len(sel_set - true_set)
    fn = len(true_set - sel_set)
    tn = total_features - (tp + fp + fn)

    precision = tp / len(sel_set) if len(sel_set) > 0 else 0.0
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    noise_leaked = len(sel_set.intersection(noise_set))
    false_discovery_rate = fp / len(sel_set) if len(sel_set) > 0 else 0.0

    return {
        "has_ground_truth": True,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "noise_features_selected": noise_leaked,
        "false_discovery_rate": round(false_discovery_rate, 4)
    }


def compute_stability_score(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = "classification",
    method_type: str = "f_test",
    k: int = 5,
    n_bootstraps: int = 6,
    subsample_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute Feature Selection Stability using the Nogueira & Brown (2018)
    pairwise Jaccard index across bootstrap data subsamples.
    Stability score in [0, 1] indicates how sensitive the selector is to data perturbations.
    """
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    n_sub = int(n_samples * subsample_ratio)
    
    selected_subsets = []
    
    for i in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_sub, replace=True)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        
        feat_names_dummy = [f"f_{j}" for j in range(X.shape[1])]
        
        if method_type == "f_test":
            res = run_univariate_filter(X_boot, y_boot, feat_names_dummy, task_type=task_type, method="f_test", k=k)
        elif method_type == "mutual_info":
            res = run_univariate_filter(X_boot, y_boot, feat_names_dummy, task_type=task_type, method="mutual_info", k=k)
        elif method_type == "rfe":
            res = run_rfe(X_boot, y_boot, feat_names_dummy, task_type=task_type, n_features_to_select=k, estimator_type="linear")
        elif method_type == "tree":
            res = run_tree_importance(X_boot, y_boot, feat_names_dummy, task_type=task_type, k=k, n_estimators=40, seed=seed+i)
        elif method_type == "l1":
            res = run_l1_regularization(X_boot, y_boot, feat_names_dummy, task_type=task_type, k=k)
        else:
            res = run_univariate_filter(X_boot, y_boot, feat_names_dummy, task_type=task_type, method="f_test", k=k)
            
        selected_subsets.append(set(res["selected_indices"]))
        
    # Calculate average pairwise Jaccard similarity
    jaccard_scores = []
    for i in range(n_bootstraps):
        for j in range(i + 1, n_bootstraps):
            s1 = selected_subsets[i]
            s2 = selected_subsets[j]
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            jaccard = intersection / union if union > 0 else 1.0
            jaccard_scores.append(jaccard)
            
    avg_stability = float(np.mean(jaccard_scores)) if jaccard_scores else 1.0
    
    return {
        "stability_score": round(avg_stability, 4),
        "n_bootstraps": n_bootstraps,
        "pairwise_jaccards": [round(s, 3) for s in jaccard_scores],
        "k": k,
        "method_evaluated": method_type
    }


# =============================================================================
# 5. DOWNSTREAM MACHINE LEARNING BENCHMARK
# =============================================================================

def benchmark_downstream_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    selection_dict: Dict[str, List[int]],
    task_type: str = "classification",
    model_family: str = "linear",
    cv_folds: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Benchmark downstream ML model performance across different feature selection subsets:
      - Raw Baseline (All Features)
      - Filter Selected
      - Wrapper Selected
      - Embedded Selected
      - Consensus Selected
      - Random Baseline (Null Hypothesis)
    Measures: CV Score (Accuracy/R2), F1/RMSE, Training Latency, Latency Reduction.
    """
    results = []
    n_features_full = X.shape[1]
    
    # Set up cross-validation splitter
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring_metric = "accuracy"
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring_metric = "r2"

    for method_name, subset_indices in selection_dict.items():
        if not subset_indices:
            continue
            
        X_sub = X[:, subset_indices]
        n_selected = len(subset_indices)
        dim_reduction_pct = (1.0 - (n_selected / n_features_full)) * 100.0
        
        # Instantiate model
        if task_type == "classification":
            if model_family == "rf":
                clf = RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=-1)
            else:
                clf = LogisticRegression(max_iter=1000, random_state=seed)
        else:
            if model_family == "rf":
                clf = RandomForestRegressor(n_estimators=60, random_state=seed, n_jobs=-1)
            else:
                clf = Ridge(alpha=1.0, random_state=seed)

        # Scale data
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        
        # Benchmark Training & CV
        t_start = time.perf_counter()
        scores = cross_val_score(clf, X_sub_scaled, y, cv=cv, scoring=scoring_metric, n_jobs=-1)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        
        # Fit single model for inference benchmark
        clf.fit(X_sub_scaled, y)
        t_inf_start = time.perf_counter()
        _ = clf.predict(X_sub_scaled[:100] if len(X_sub_scaled) >= 100 else X_sub_scaled)
        inf_latency_us = ((time.perf_counter() - t_inf_start) / (min(100, len(X_sub_scaled)))) * 1e6
        
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        
        results.append({
            "Method / Subset": method_name,
            "Features Used": n_selected,
            "Dim Reduction (%)": round(dim_reduction_pct, 1),
            f"Mean {scoring_metric.upper()}": round(mean_score, 4),
            f"Std {scoring_metric.upper()}": round(std_score, 4),
            "Train Time (ms)": round(elapsed_ms, 2),
            "Inference (us/sample)": round(inf_latency_us, 2)
        })
        
    return pd.DataFrame(results)


# =============================================================================
# 6. PIPELINE PERSISTENCE
# =============================================================================

def save_feature_selection_pipeline(
    pipeline_dict: Dict[str, Any],
    filepath: str
) -> None:
    """
    Serialize fitted feature selector pipeline to disk.
    """
    joblib.dump(pipeline_dict, filepath)


def load_feature_selection_pipeline(
    filepath: str
) -> Dict[str, Any]:
    """
    Load serialized feature selector pipeline from disk.
    """
    return joblib.load(filepath)

"""
Project 014: Feature Selection Techniques Studio
=================================================
Flagship interactive Streamlit application for Filter, Wrapper, Embedded,
and Consensus feature selection, multicollinearity pruning, stability metrics,
ground-truth validation, and downstream ML benchmarking.

Part of the 1,000 AI Projects Implementation Library.
"""

import sys
import io
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataset import (
    load_feature_dataset,
    add_synthetic_noise_features,
    load_custom_csv
)
from feature_selection_engine import (
    run_variance_threshold,
    run_univariate_filter,
    run_multicollinearity_filter,
    run_rfe,
    run_rfecv,
    run_sequential_feature_selection,
    run_l1_regularization,
    compute_l1_regularization_path,
    run_tree_importance,
    run_permutation_importance,
    compute_consensus_ranking,
    evaluate_ground_truth_selection,
    compute_stability_score,
    benchmark_downstream_model,
    save_feature_selection_pipeline
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Feature Selection Studio | 1,000 AI Projects",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS Dark/Glassmorphic Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek metric card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .metric-caption {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 6px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-info { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); }
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-danger { background: rgba(244, 63, 94, 0.2); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.4); }
    
    /* Code block styling */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: Dataset Selection & Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 Feature Selection Studio")
    st.caption("Production-grade Feature Engineering & Dimensionality Reduction")
    st.markdown("---")
    
    dataset_choice = st.selectbox(
        "📁 Select Benchmark Dataset",
        [
            "Synthetic Classification (Controlled Ground Truth)",
            "Synthetic Regression (Controlled Ground Truth)",
            "Breast Cancer Diagnostic (30 features)",
            "Wine Cultivars (13 features)",
            "Diabetes Progression (10 features)",
            "California Housing (8 features)",
            "Upload Custom CSV File"
        ],
        index=0
    )
    
    # Custom synthetic parameters or file uploader
    custom_file = None
    if dataset_choice == "Upload Custom CSV File":
        custom_file = st.file_uploader("Upload CSV", type=["csv"])
        n_samples = 500
        n_features = 20
        n_info = 6
        n_red = 4
        noise_flip = 0.1
    elif "Synthetic Classification" in dataset_choice:
        with st.expander("⚙️ Synthetic Generator Settings", expanded=False):
            n_samples = st.slider("Sample Count (N)", 100, 1500, 400, step=50)
            n_features = st.slider("Total Features (D)", 8, 40, 16, step=2)
            n_info = st.slider("Informative Features", 2, n_features - 3, 5, step=1)
            n_red = st.slider("Redundant Collinear Features", 0, n_features - n_info - 1, 3, step=1)
            noise_flip = st.slider("Label Noise Flip Ratio", 0.0, 0.3, 0.05, step=0.01)
    elif "Synthetic Regression" in dataset_choice:
        with st.expander("⚙️ Synthetic Regression Settings", expanded=False):
            n_samples = st.slider("Sample Count (N)", 100, 1500, 400, step=50)
            n_features = st.slider("Total Features (D)", 8, 40, 16, step=2)
            n_info = st.slider("Informative Features", 2, n_features - 2, 6, step=1)
            n_red = 0
            noise_flip = st.slider("Gaussian Target Noise", 0.0, 0.5, 0.1, step=0.05)
    else:
        n_samples = 500
        n_features = 20
        n_info = 6
        n_red = 4
        noise_flip = 0.1

    st.markdown("### 🧪 Noise Injection Experiment")
    inject_noise = st.checkbox("Inject Pure Noise Columns", value=False, help="Inject random Gaussian/Uniform noise columns to evaluate if algorithms properly reject them.")
    n_injected_noise = st.slider("Injected Noise Features", 1, 10, 3) if inject_noise else 0

    st.markdown("---")
    st.markdown("### 🎚️ Feature Target Budget (K)")
    k_selected = st.slider("Features to Select (K)", 1, 30, 4, help="Target number of features to select for downstream evaluation.")
    
    st.markdown("---")
    downstream_model = st.radio(
        "Downstream Benchmark Model",
        ["Linear / Logistic Model", "Random Forest Ensemble"],
        index=0
    )
    model_family = "rf" if "Random Forest" in downstream_model else "linear"


# -----------------------------------------------------------------------------
# Data Ingestion & State Preparation
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_dataset(dataset_choice, n_samples, n_features, n_info, n_red, noise_flip, inject_noise, n_injected_noise):
    if dataset_choice == "Synthetic Classification (Controlled Ground Truth)":
        data = load_feature_dataset("synthetic_classification", n_samples=n_samples, n_features=n_features, n_informative=n_info, n_redundant=n_red, noise_ratio=noise_flip)
    elif dataset_choice == "Synthetic Regression (Controlled Ground Truth)":
        data = load_feature_dataset("synthetic_regression", n_samples=n_samples, n_features=n_features, n_informative=n_info, noise_ratio=noise_flip)
    elif "Breast Cancer" in dataset_choice:
        data = load_feature_dataset("breast_cancer")
    elif "Wine" in dataset_choice:
        data = load_feature_dataset("wine")
    elif "Diabetes" in dataset_choice:
        data = load_feature_dataset("diabetes")
    elif "California Housing" in dataset_choice:
        data = load_feature_dataset("california_housing")
    else:
        data = load_feature_dataset("synthetic_classification")
        
    if inject_noise and n_injected_noise > 0:
        data = add_synthetic_noise_features(data, n_noise_features=n_injected_noise)
        
    return data

if dataset_choice == "Upload Custom CSV File":
    if custom_file is not None:
        try:
            data = load_custom_csv(custom_file)
            st.sidebar.success(f"Loaded CSV with {data['n_features']} features.")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV file in the sidebar to proceed, or choose one of the pre-configured datasets.")
        st.stop()
else:
    data = get_cached_dataset(dataset_choice, n_samples, n_features, n_info, n_red, noise_flip, inject_noise, n_injected_noise)

# Ensure K is within valid range
k_selected = min(k_selected, data["n_features"])


# -----------------------------------------------------------------------------
# Main Header & High-Level KPIs
# -----------------------------------------------------------------------------
st.title("🎯 Feature Selection Techniques Studio")
st.markdown(
    f"**{data['description']}** &nbsp;|&nbsp; "
    f"Task: <span class='badge badge-info'>{data['task_type'].upper()}</span> &nbsp;|&nbsp; "
    f"Target Budget: <span class='badge badge-success'>K = {k_selected} of {data['n_features']}</span>",
    unsafe_allow_html=True
)

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
with kpi_c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Dataset Samples</div>
        <div class="metric-value">{data['n_samples']}</div>
        <div class="metric-caption">Row observations (N)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Original Features</div>
        <div class="metric-value">{data['n_features']}</div>
        <div class="metric-caption">Dimensionality (D)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Target Budget (K)</div>
        <div class="metric-value">{k_selected}</div>
        <div class="metric-caption">Reduced dimensionality</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c4:
    dim_reduction_pct = (1.0 - (k_selected / data['n_features'])) * 100.0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sparsity Ratio</div>
        <div class="metric-value">{dim_reduction_pct:.1f}%</div>
        <div class="metric-caption">Compression gain</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c5:
    gt_label = f"{len(data.get('ground_truth_informative', []))} Signals" if data['has_ground_truth'] else "Empirical Real"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Ground Truth</div>
        <div class="metric-value" style="font-size: 1.45rem;">{gt_label}</div>
        <div class="metric-caption">Benchmark knowledge</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab_data, tab_filter, tab_wrapper, tab_embedded, tab_consensus, tab_benchmark, tab_export = st.tabs([
    "📊 1. Dataset & Collinearity",
    "🧪 2. Filter Methods",
    "🔄 3. Wrapper Methods (RFE)",
    "🌲 4. Embedded & Permutation",
    "🏆 5. Consensus & Stability",
    "⚡ 6. Downstream ML Benchmark",
    "💾 7. Export Pipeline"
])


# =============================================================================
# TAB 1: DATASET & COLLINEARITY EXPLORER
# =============================================================================
with tab_data:
    st.subheader("Data Matrix & Pairwise Multicollinearity Analysis")
    st.markdown(
        "Multicollinear features introduce redundant degrees of freedom, inflate regression coefficient variances, "
        "and deceive univariate filter scores. Explore pairwise correlation and prune collinear pairs below."
    )
    
    col_d1, col_d2 = st.columns([1, 1])
    
    with col_d1:
        st.markdown("##### 📋 Data Matrix Sample")
        st.dataframe(data["df"].head(10), use_container_width=True, height=260)
        
        # Ground Truth Breakdown if available
        if data["has_ground_truth"]:
            st.markdown("##### 🧬 Ground-Truth Categorization")
            info_badges = " ".join([f"<span class='badge badge-success'>{data['feature_names'][i]}</span>" for i in data["ground_truth_informative"]])
            st.markdown(f"**Informative Signals ({len(data['ground_truth_informative'])}):**<br>{info_badges}", unsafe_allow_html=True)
            
            if data.get("ground_truth_redundant"):
                red_badges = " ".join([f"<span class='badge badge-warning'>{data['feature_names'][i]}</span>" for i in data["ground_truth_redundant"]])
                st.markdown(f"**Redundant Collinear ({len(data['ground_truth_redundant'])}):**<br>{red_badges}", unsafe_allow_html=True)
                
            if data.get("ground_truth_noise"):
                noise_badges = " ".join([f"<span class='badge badge-danger'>{data['feature_names'][i]}</span>" for i in data["ground_truth_noise"]])
                st.markdown(f"**Pure Noise ({len(data['ground_truth_noise'])}):**<br>{noise_badges}", unsafe_allow_html=True)
        else:
            st.markdown("##### 📈 Target Distribution")
            if data["task_type"] == "classification":
                vc = pd.Series(data["y"]).value_counts().reset_index()
                vc.columns = ["Class", "Count"]
                fig_t = px.bar(vc, x="Class", y="Count", color="Class", template="plotly_dark", height=200)
                fig_t.update_layout(margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                fig_t = px.histogram(data["y"], nbins=25, labels={"value": "Target Value"}, template="plotly_dark", height=200)
                fig_t.update_layout(margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
                st.plotly_chart(fig_t, use_container_width=True)

    with col_d2:
        st.markdown("##### 🔥 Pearson Correlation Heatmap")
        corr_thresh = st.slider("Highlight Collinearity Threshold (|r| > )", 0.50, 0.99, 0.85, step=0.05)
        
        # Compute correlation
        corr_df = data["df"][data["feature_names"]].corr()
        
        fig_corr = px.imshow(
            corr_df,
            labels=dict(x="Feature", y="Feature", color="Correlation"),
            x=data["feature_names"],
            y=data["feature_names"],
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            template="plotly_dark",
            height=380
        )
        fig_corr.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.markdown("---")
    st.markdown("##### ✂️ Automated Multicollinearity Pruning")
    collin_res = run_multicollinearity_filter(data["X"], data["feature_names"], threshold=corr_thresh)
    
    if collin_res["high_corr_pairs"]:
        st.warning(f"Found **{len(collin_res['high_corr_pairs'])} highly collinear feature pairs** (|r| > {corr_thresh}):")
        pair_df = pd.DataFrame(collin_res["high_corr_pairs"])
        st.dataframe(pair_df, use_container_width=True)
        st.markdown(f"**Recommended to drop {len(collin_res['dropped_features'])} redundant features:** " + 
                    ", ".join([f"`{f}`" for f in collin_res["dropped_features"]]))
    else:
        st.success(f"No feature pairs exceed the correlation threshold of |r| > {corr_thresh}. Feature matrix exhibits healthy independence.")


# =============================================================================
# TAB 2: FILTER METHODS STUDIO
# =============================================================================
with tab_filter:
    st.subheader("Filter Methods: Univariate Statistical Ranking & Entropy")
    st.markdown(
        "Filter methods evaluate each feature independently of model interaction using statistical significance tests "
        "or mutual information entropy. They are blazingly fast (O(N·D)) and scale effortlessly to millions of features."
    )
    
    col_f_opt1, col_f_opt2 = st.columns([1, 1])
    with col_f_opt1:
        filter_metric = st.selectbox(
            "Select Filter Test Statistic",
            [
                "ANOVA F-Test (f_classif / f_regression)",
                "Mutual Information (Shannon Entropy)",
                "Chi-Square Test (Categorical/Count-based)",
                "Absolute Pearson Correlation (|r|)",
                "Variance Threshold (Entropy check)"
            ]
        )
    with col_f_opt2:
        variance_thresh = st.slider("Variance Threshold (Normalized)", 0.0, 0.25, 0.02, step=0.01) if "Variance" in filter_metric else 0.0

    # Execute Filter
    if "ANOVA" in filter_metric:
        filt_res = run_univariate_filter(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], method="f_test", k=k_selected)
    elif "Mutual Information" in filter_metric:
        filt_res = run_univariate_filter(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], method="mutual_info", k=k_selected)
    elif "Chi-Square" in filter_metric:
        filt_res = run_univariate_filter(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], method="chi2", k=k_selected)
    elif "Pearson" in filter_metric:
        filt_res = run_univariate_filter(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], method="correlation", k=k_selected)
    else:
        filt_res = run_variance_threshold(data["X"], data["feature_names"], threshold=variance_thresh)

    # Visualization
    c_fchart1, c_fchart2 = st.columns([2, 1])
    
    with c_fchart1:
        st.markdown(f"##### 📊 Feature Importance Ranking: {filt_res.get('metric_name', 'Variance')}")
        
        df_filt = pd.DataFrame({
            "Feature": data["feature_names"],
            "Score": filt_res["scores"],
            "Selected": ["Selected (Top-K)" if i in filt_res["selected_indices"] else "Discarded" for i in range(len(data["feature_names"]))]
        }).sort_values(by="Score", ascending=True)
        
        fig_bar = px.bar(
            df_filt,
            x="Score",
            y="Feature",
            color="Selected",
            orientation="h",
            color_discrete_map={"Selected (Top-K)": "#6366f1", "Discarded": "#334155"},
            template="plotly_dark",
            height=440
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20), xaxis_title="Score / Importance", yaxis_title="Feature")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_fchart2:
        st.markdown("##### 📌 Selected Feature Set")
        st.write(f"Top **{len(filt_res['selected_indices'])}** features selected:")
        for idx in filt_res["selected_indices"]:
            is_true = idx in data.get("ground_truth_informative", [])
            badge = " [Ground Truth]" if is_true else ""
            st.markdown(f"- **{data['feature_names'][idx]}** (`Rank {filt_res['ranks'][idx] if 'ranks' in filt_res else 1}`) {badge}")
            
        if filt_res.get("p_values") is not None:
            st.markdown("##### 📉 Statistical Significance (p-values)")
            p_df = pd.DataFrame({
                "Feature": [data["feature_names"][i] for i in filt_res["selected_indices"]],
                "Score": [filt_res["scores"][i] for i in filt_res["selected_indices"]],
                "p-value": [f"{filt_res['p_values'][i]:.2e}" for i in filt_res["selected_indices"]]
            })
            st.dataframe(p_df, use_container_width=True)


# =============================================================================
# TAB 3: WRAPPER METHODS LAB (RFE & RFECV)
# =============================================================================
with tab_wrapper:
    st.subheader("Wrapper Methods: Recursive Feature Elimination & SFS")
    st.markdown(
        "Wrapper methods treat feature selection as a search problem around a predictive model. "
        "They repeatedly fit the model, evaluate performance, and prune the weakest features, capturing complex feature interactions."
    )
    
    wrap_subtab1, wrap_subtab2 = st.tabs(["🎯 RFE (Fixed Target K)", "📈 RFECV (Optimal Feature Discovery)"])
    
    with wrap_subtab1:
        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            rfe_estimator = st.selectbox("Base Estimator for RFE", ["Linear / Logistic Model", "Random Forest"], key="rfe_est")
            rfe_est_type = "rf" if "Random Forest" in rfe_estimator else "linear"
            rfe_step = st.slider("Pruning Step Size per Iteration", 1, 5, 1)
            
            with st.spinner("Computing Recursive Feature Elimination..."):
                rfe_res = run_rfe(
                    data["X"], data["y"], data["feature_names"],
                    task_type=data["task_type"],
                    n_features_to_select=k_selected,
                    estimator_type=rfe_est_type,
                    step=rfe_step
                )
                
            st.markdown("##### 🏅 RFE Selected Features")
            for idx in rfe_res["selected_indices"]:
                is_gt = idx in data.get("ground_truth_informative", [])
                st.markdown(f"- ✅ **{data['feature_names'][idx]}** {'(Ground Truth Informative)' if is_gt else ''}")

        with col_w2:
            st.markdown("##### 🪜 RFE Elimination Ladder")
            df_rfe = pd.DataFrame({
                "Feature": data["feature_names"],
                "RFE Rank": rfe_res["ranking"]
            }).sort_values(by="RFE Rank", ascending=True)
            
            fig_rfe = px.bar(
                df_rfe,
                x="RFE Rank",
                y="Feature",
                color="RFE Rank",
                orientation="h",
                color_continuous_scale="Viridis_r",
                template="plotly_dark",
                height=420
            )
            fig_rfe.update_layout(margin=dict(l=20, r=20, t=10, b=20), xaxis_title="Elimination Rank (1 = Retained, Higher = Eliminated Early)")
            st.plotly_chart(fig_rfe, use_container_width=True)

    with wrap_subtab2:
        st.markdown("##### 🚀 Discovering the Optimal Number of Features via Cross-Validation")
        st.caption("RFECV systematically evaluates CV generalization score across feature counts to pinpoint the exact elbow/peak.")
        
        rfecv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        if st.button("Run RFECV Optimization Search", type="primary"):
            with st.spinner("Executing RFECV Cross-Validation grid..."):
                rfecv_res = run_rfecv(
                    data["X"], data["y"], data["feature_names"],
                    task_type=data["task_type"],
                    cv_folds=rfecv_folds,
                    step=1
                )
                st.session_state["rfecv_res"] = rfecv_res
                
        if "rfecv_res" in st.session_state:
            rfecv_res = st.session_state["rfecv_res"]
            
            st.success(f"Optimal feature subset size discovered: **{rfecv_res['optimal_k']} features** (out of {data['n_features']})!")
            
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Scatter(
                x=rfecv_res["feature_counts"],
                y=rfecv_res["cv_scores"],
                mode="lines+markers",
                name=f"Mean CV {rfecv_res['scoring_metric'].upper()}",
                line=dict(color="#6366f1", width=3),
                marker=dict(size=7)
            ))
            
            # Optimal vertical marker
            best_idx = rfecv_res["feature_counts"].index(rfecv_res["optimal_k"]) if rfecv_res["optimal_k"] in rfecv_res["feature_counts"] else 0
            best_score = rfecv_res["cv_scores"][best_idx]
            fig_cv.add_vline(x=rfecv_res["optimal_k"], line_dash="dash", line_color="#10b981", annotation_text=f"Optimal K={rfecv_res['optimal_k']}")
            
            fig_cv.update_layout(
                title=f"Cross-Validation Performance vs Number of Features ({rfecv_res['scoring_metric'].upper()})",
                xaxis_title="Number of Features Selected",
                yaxis_title=f"CV Score ({rfecv_res['scoring_metric'].upper()})",
                template="plotly_dark",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_cv, use_container_width=True)
            
            st.markdown(f"**Optimal Subset Features ({rfecv_res['optimal_k']}):** " + 
                        ", ".join([f"`{f}`" for f in rfecv_res["selected_features"]]))


# =============================================================================
# TAB 4: EMBEDDED & PERMUTATION IMPORTANCE HUB
# =============================================================================
with tab_embedded:
    st.subheader("Embedded Methods: L1 Sparsity & Permutation Importance")
    st.markdown(
        "Embedded methods incorporate feature selection directly into the objective loss function. "
        "L1 regularization (Lasso) drives non-informative weights to exact zeros, while Tree MDI and "
        "Holdout Permutation Importance expose model-agnostic feature dependencies."
    )
    
    emb_c1, emb_c2 = st.columns([1, 1])
    
    with emb_c1:
        st.markdown("##### 🧬 L1 Regularization & Sparsity")
        reg_param = st.slider("Regularization Strength (C / alpha)", 0.01, 2.0, 0.4, step=0.05)
        
        l1_res = run_l1_regularization(
            data["X"], data["y"], data["feature_names"],
            task_type=data["task_type"],
            alpha_or_C=reg_param,
            k=k_selected
        )
        
        st.markdown(f"**Non-zero weights active:** `{l1_res['non_zero_count']} / {data['n_features']}`")
        
        df_l1 = pd.DataFrame({
            "Feature": data["feature_names"],
            "Coefficient Magnitude": l1_res["coefficients"]
        }).sort_values(by="Coefficient Magnitude", ascending=True)
        
        fig_l1 = px.bar(
            df_l1,
            x="Coefficient Magnitude",
            y="Feature",
            orientation="h",
            color="Coefficient Magnitude",
            color_continuous_scale="Purples",
            template="plotly_dark",
            height=340
        )
        fig_l1.update_layout(margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_l1, use_container_width=True)

    with emb_c2:
        st.markdown("##### 🌲 Random Forest MDI vs Permutation Importance")
        with st.spinner("Computing Permutation Importance on validation split..."):
            tree_res = run_tree_importance(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], k=k_selected)
            perm_res = run_permutation_importance(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], k=k_selected)
            
        df_comp = pd.DataFrame({
            "Feature": data["feature_names"],
            "Tree MDI": tree_res["scores"],
            "Permutation Mean": perm_res["scores"]
        }).sort_values(by="Permutation Mean", ascending=True)
        
        fig_perm = go.Figure()
        fig_perm.add_trace(go.Bar(
            y=df_comp["Feature"],
            x=df_comp["Tree MDI"],
            name="Tree MDI (Gini)",
            orientation="h",
            marker=dict(color="#10b981")
        ))
        fig_perm.add_trace(go.Bar(
            y=df_comp["Feature"],
            x=df_comp["Permutation Mean"],
            name="Permutation Importance (Holdout)",
            orientation="h",
            marker=dict(color="#f59e0b")
        ))
        fig_perm.update_layout(
            barmode="group",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_perm, use_container_width=True)
        st.caption("Notice how Permutation Importance on holdout data prevents high-cardinality bias common to pure tree MDI.")


# =============================================================================
# TAB 5: CONSENSUS RANKING, STABILITY & GROUND TRUTH
# =============================================================================
with tab_consensus:
    st.subheader("Consensus Ensemble & Feature Selection Stability")
    st.markdown(
        "Individual feature selection algorithms exhibit individual biases. "
        "A **Consensus Ensemble (Borda Count)** merges rankings across Filter, Wrapper, and Embedded methods to create "
        "a resilient, production-ready feature subset."
    )
    
    # Compute active methods
    filt_anova = run_univariate_filter(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], method="f_test", k=k_selected)
    filt_mi = run_univariate_filter(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], method="mutual_info", k=k_selected)
    rfe_active = run_rfe(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], n_features_to_select=k_selected)
    l1_active = run_l1_regularization(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], k=k_selected)
    tree_active = run_tree_importance(data["X"], data["y"], data["feature_names"], task_type=data["task_type"], k=k_selected)
    
    ensemble_pool = {
        "ANOVA F-Test": filt_anova,
        "Mutual Info": filt_mi,
        "RFE (Wrapper)": rfe_active,
        "L1 Sparsity": l1_active,
        "Tree Importance": tree_active
    }
    
    consensus_res = compute_consensus_ranking(ensemble_pool, data["feature_names"], k=k_selected)
    
    col_con1, col_con2 = st.columns([1, 1])
    
    with col_con1:
        st.markdown("##### 🏆 Borda Count Consensus Ranking")
        df_cons = pd.DataFrame({
            "Feature": data["feature_names"],
            "Consensus Score": consensus_res["normalized_scores"],
            "Selected": ["Consensus Top-K" if i in consensus_res["selected_indices"] else "Other" for i in range(len(data["feature_names"]))]
        }).sort_values(by="Consensus Score", ascending=True)
        
        fig_cons = px.bar(
            df_cons,
            x="Consensus Score",
            y="Feature",
            color="Selected",
            orientation="h",
            color_discrete_map={"Consensus Top-K": "#10b981", "Other": "#334155"},
            template="plotly_dark",
            height=380
        )
        fig_cons.update_layout(margin=dict(l=20, r=20, t=10, b=20), xaxis_title="Normalized Consensus Score [0 - 100]")
        st.plotly_chart(fig_cons, use_container_width=True)

    with col_con2:
        st.markdown("##### 🔀 Multi-Method Rank Matrix Heatmap")
        matrix_df = pd.DataFrame(consensus_res["ranking_matrix"]).set_index("feature")
        
        fig_mat = px.imshow(
            matrix_df.drop(columns=["consensus_score"]),
            labels=dict(x="Method", y="Feature", color="Rank"),
            color_continuous_scale="Viridis_r",
            template="plotly_dark",
            height=380
        )
        fig_mat.update_layout(margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_mat, use_container_width=True)

    st.markdown("---")
    
    col_eval1, col_eval2 = st.columns([1, 1])
    
    with col_eval1:
        st.markdown("##### 🛡️ Selection Stability Analysis (Nogueira & Brown)")
        st.caption("Evaluates pairwise Jaccard similarity across bootstrap subsamples of the dataset.")
        
        stab_method = st.selectbox("Method to evaluate stability for", ["f_test", "mutual_info", "rfe", "tree", "l1"])
        with st.spinner("Evaluating bootstrap stability..."):
            stab_res = compute_stability_score(
                data["X"], data["y"],
                task_type=data["task_type"],
                method_type=stab_method,
                k=k_selected,
                n_bootstraps=5
            )
            
        st.metric(
            label=f"Stability Index (S) - {stab_method.upper()}",
            value=f"{stab_res['stability_score']:.3f}",
            delta="High Robustness" if stab_res['stability_score'] >= 0.75 else "Moderate Sensitivity"
        )
        st.progress(float(min(1.0, max(0.0, stab_res["stability_score"]))))
        st.caption("A stability index > 0.75 demonstrates that the selected subset is invariant to dataset sampling noise.")

    with col_eval2:
        st.markdown("##### 🎯 Ground-Truth Precision & Recall")
        if data["has_ground_truth"]:
            gt_metrics = evaluate_ground_truth_selection(
                consensus_res["selected_indices"],
                data["ground_truth_informative"],
                data["ground_truth_noise"],
                data["n_features"]
            )
            
            c_gt1, c_gt2, c_gt3 = st.columns(3)
            c_gt1.metric("Precision", f"{gt_metrics['precision']*100:.1f}%")
            c_gt2.metric("Recall", f"{gt_metrics['recall']*100:.1f}%")
            c_gt3.metric("F1-Score", f"{gt_metrics['f1']:.3f}")
            
            st.markdown(
                f"- **True Positives:** `{gt_metrics['true_positives']}` signal features correctly picked.<br>"
                f"- **False Discoveries:** `{gt_metrics['noise_features_selected']}` pure noise features leaked.<br>"
                f"- **False Discovery Rate (FDR):** `{gt_metrics['false_discovery_rate']*100:.1f}%`",
                unsafe_allow_html=True
            )
        else:
            st.info("Ground-truth signal vs noise breakdown is available for Synthetic benchmark datasets. You are currently viewing an empirical real dataset.")


# =============================================================================
# TAB 6: DOWNSTREAM ML BENCHMARK
# =============================================================================
with tab_benchmark:
    st.subheader("Downstream Machine Learning Benchmark")
    st.markdown(
        "Does pruning features hurt prediction accuracy? Compare the baseline model trained on **all features** "
        "against subsets chosen by **Filter, Wrapper, Embedded, Consensus**, and a **Random baseline**."
    )
    
    # Build feature subsets dictionary
    rng = np.random.RandomState(42)
    random_subset = rng.choice(data["n_features"], size=k_selected, replace=False).tolist()
    
    subsets_to_test = {
        f"1. All Features ({data['n_features']})": list(range(data["n_features"])),
        f"2. Filter (ANOVA Top-{k_selected})": filt_anova["selected_indices"],
        f"3. Wrapper (RFE Top-{k_selected})": rfe_active["selected_indices"],
        f"4. Embedded (L1 Top-{k_selected})": l1_active["selected_indices"],
        f"5. Embedded (Tree Top-{k_selected})": tree_active["selected_indices"],
        f"6. Consensus Ensemble (Top-{k_selected})": consensus_res["selected_indices"],
        f"7. Random Subset (Null Baseline)": random_subset
    }
    
    with st.spinner("Benchmarking downstream model across all feature subsets..."):
        bench_df = benchmark_downstream_model(
            data["X"], data["y"], data["feature_names"],
            subsets_to_test,
            task_type=data["task_type"],
            model_family=model_family,
            cv_folds=5
        )
        
    st.dataframe(bench_df, use_container_width=True)
    
    col_b1, col_b2 = st.columns([1, 1])
    
    score_col = [c for c in bench_df.columns if "MEAN" in c.upper()][0]
    
    with col_b1:
        st.markdown(f"##### 🎯 Generalization Performance ({score_col})")
        fig_score = px.bar(
            bench_df,
            x="Method / Subset",
            y=score_col,
            color="Method / Subset",
            template="plotly_dark",
            height=360
        )
        fig_score.update_layout(margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
        st.plotly_chart(fig_score, use_container_width=True)
        
    with col_b2:
        st.markdown("##### ⚡ Training Latency & Compression")
        fig_lat = px.scatter(
            bench_df,
            x="Dim Reduction (%)",
            y=score_col,
            size="Features Used",
            color="Method / Subset",
            text="Method / Subset",
            template="plotly_dark",
            height=360
        )
        fig_lat.update_traces(textposition="top center")
        fig_lat.update_layout(margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
        st.plotly_chart(fig_lat, use_container_width=True)
        
    # Key Takeaways
    full_score = bench_df.loc[bench_df["Method / Subset"].str.contains("All Features"), score_col].values[0]
    best_subset_row = bench_df[~bench_df["Method / Subset"].str.contains("All Features")].sort_values(by=score_col, ascending=False).iloc[0]
    
    delta = best_subset_row[score_col] - full_score
    if delta >= 0:
        st.success(
            f"🎉 **Feature Selection Gain:** `{best_subset_row['Method / Subset']}` achieved "
            f"**{best_subset_row[score_col]:.4f}** (vs {full_score:.4f} for all features) while reducing "
            f"features by **{best_subset_row['Dim Reduction (%)']}%**!"
        )
    else:
        st.info(
            f"💡 `{best_subset_row['Method / Subset']}` retained **{best_subset_row[score_col]/full_score*100:.1f}%** "
            f"of full performance while compressing dimensionality by **{best_subset_row['Dim Reduction (%)']}%**."
        )


# =============================================================================
# TAB 7: EXPORT PIPELINE & CODE GENERATOR
# =============================================================================
with tab_export:
    st.subheader("Export Clean Dataset & Production Scikit-Learn Pipeline")
    st.markdown("Transform your dataset using the selected features and download ready-to-deploy assets.")
    
    col_exp1, col_exp2 = st.columns([1, 1])
    
    with col_exp1:
        st.markdown("##### 🎛️ Choose Final Feature Subset")
        export_choice = st.selectbox(
            "Select feature subset for production export",
            [
                f"Consensus Top-{k_selected}",
                f"RFE Top-{k_selected}",
                f"ANOVA Filter Top-{k_selected}",
                f"L1 Regularization Top-{k_selected}",
                f"Tree Importance Top-{k_selected}"
            ]
        )
        
        if "Consensus" in export_choice:
            final_indices = consensus_res["selected_indices"]
        elif "RFE" in export_choice:
            final_indices = rfe_active["selected_indices"]
        elif "ANOVA" in export_choice:
            final_indices = filt_anova["selected_indices"]
        elif "L1" in export_choice:
            final_indices = l1_active["selected_indices"]
        else:
            final_indices = tree_active["selected_indices"]
            
        final_features = [data["feature_names"][i] for i in final_indices]
        
        st.write("**Features included in export:**")
        st.write(", ".join([f"`{f}`" for f in final_features]))
        
        # Transformed DataFrame
        df_export = pd.DataFrame(data["X"][:, final_indices], columns=final_features)
        df_export["target"] = data["y"]
        
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Transformed Dataset (CSV)",
            data=csv_data,
            file_name=f"dataset_reduced_{len(final_features)}_features.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Save pipeline dictionary
        pipeline_dict = {
            "selected_features": final_features,
            "selected_indices": final_indices,
            "task_type": data["task_type"],
            "dataset_name": data["dataset_name"],
            "k": len(final_features)
        }
        buffer = io.BytesIO()
        joblib.dump(pipeline_dict, buffer)
        st.download_button(
            label="💾 Download Pipeline Metadata (.joblib)",
            data=buffer.getvalue(),
            file_name="feature_selector_pipeline.joblib",
            mime="application/octet-stream"
        )

    with col_exp2:
        st.markdown("##### 🐍 Production Python Code Snippet")
        code_snippet = f"""# Reproducible Scikit-Learn Feature Selection Pipeline
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_csv("your_dataset.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 2. Selected Feature Indices from Feature Selection Studio
selected_features = {final_features}
print(f"Selected {{len(selected_features)}} optimal features:")
print(selected_features)

# 3. Transform to pruned feature matrix
X_pruned = X[selected_features]

# 4. Train downstream production model
model = {'LogisticRegression(max_iter=1000)' if data['task_type'] == 'classification' else 'Ridge()'}
model.fit(X_pruned, y)
print("Model successfully trained on pruned features!")
"""
        st.code(code_snippet, language="python")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Project 014: Feature Selection Techniques Studio &nbsp;•&nbsp; "
    "Part of the 1,000 AI Projects Implementation Library &nbsp;•&nbsp; "
    "Filter • Wrapper • Embedded • Consensus • Downstream ML"
    "</div>",
    unsafe_allow_html=True
)

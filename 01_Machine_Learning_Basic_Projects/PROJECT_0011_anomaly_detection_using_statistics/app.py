"""
Project 011: Statistical Anomaly Detection (AD) Studio
======================================================
A high-performance interactive Streamlit application showcasing statistical anomaly detection.
Features:
- Parametric Univariate: Standard Z-Score (Normal Gaussian deviation)
- Robust Univariate: Modified Z-Score using Median Absolute Deviation (MAD / Iglewicz & Hoaglin)
- Non-Parametric: Tukey's Interquartile Range (IQR) fences
- Multivariate: Mahalanobis Distance with Chi-Square (χ²) distribution cutoffs
- Robust Multivariate: Minimum Covariance Determinant (MCD / Rousseeuw)
- Multi-Method Consensus Ensemble with configurable voting
- 2D Bivariate Decision Boundaries & Confidence Ellipses
- Time-Series Streaming Anomaly Detection with Rolling Bollinger Bands
- Real-Time Live Inference & Feature Contribution Explainer
- Custom CSV File Scanner with Instant Outlier Export
- Mathematical Theory & Formulation Deep-Dive

Part of the 1,000 AI Projects Implementation Library.
"""

import time
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple

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
    compute_2d_decision_mesh
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Statistical Anomaly Detection Studio | 1,000 AI Projects",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Aesthetic Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Banner */
    .hero-container {
        padding: 1.6rem 2rem;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(59, 130, 246, 0.15) 50%, rgba(16, 185, 129, 0.12) 100%);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F87171, #60A5FA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.02rem;
        margin-top: 0.2rem;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(239, 68, 68, 0.4);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 0.25rem;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #34D399;
        margin-top: 0.2rem;
    }

    /* Badges */
    .tag-badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-right: 0.45rem;
    }
    .badge-ad {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-stat {
        background: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-robust {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Verdict Callout */
    .verdict-normal {
        padding: 1rem 1.4rem;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        border-radius: 10px;
        color: #D1FAE5;
    }
    .verdict-anomaly {
        padding: 1rem 1.4rem;
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid #EF4444;
        border-radius: 10px;
        color: #FEE2E2;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching Data & Model Fitting
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_dataset(dataset_key: str, contamination: float, seed: int):
    return load_benchmark_dataset(dataset_key, contamination=contamination, seed=seed)


# -----------------------------------------------------------------------------
# Sidebar: Controls, Presets & Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Statistical AD Studio")
    st.caption("Project 011 • 1,000 AI Projects Implementation Library")
    st.markdown("---")

    # Archetype Presets
    presets = get_archetype_presets()
    selected_preset_name = st.selectbox(
        "⚡ Quick Archetype Presets",
        ["Custom Configuration"] + list(presets.keys()),
        index=0
    )

    preset_config = presets.get(selected_preset_name, {})

    st.markdown("#### 📂 1. Telemetry Source")
    dataset_category = st.radio(
        "Dataset Category",
        ["Real-World Telemetry", "2D Geometric Manifolds", "Masking & Swamping Demo"],
        horizontal=False
    )

    if dataset_category == "Real-World Telemetry":
        ds_options = [
            "Financial Transactions & Fraud",
            "Cloud Server Infrastructure Telemetry",
            "Industrial IoT Sensor Telemetry",
            "Medical Vital Signs Telemetry"
        ]
        default_ds_idx = 0
        if preset_config and "dataset" in preset_config:
            for i, opt in enumerate(ds_options):
                if opt.lower() in preset_config["dataset"].lower():
                    default_ds_idx = i
                    break
        dataset_selection = st.selectbox("Select Domain Benchmark", ds_options, index=default_ds_idx)

    elif dataset_category == "2D Geometric Manifolds":
        ds_options = [
            "2D Correlated Gaussian (Covariance Breach)",
            "2D Multimodal Clusters (Day vs Night Mode)"
        ]
        dataset_selection = st.selectbox("Select 2D Manifold", ds_options, index=0)
    else:
        dataset_selection = "Masking & Swamping Demonstration"

    st.markdown("---")
    st.markdown("#### ⚙️ 2. Statistical Algorithm")
    method_choices = {
        "Statistical Ensemble (Consensus Voting)": "ensemble",
        "Standard Z-Score (k=3.0)": "z_score",
        "Modified Z-Score (Median Absolute Deviation)": "modified_z_score",
        "Tukey's IQR Fences (Non-Parametric)": "iqr",
        "Mahalanobis Distance (Covariance-Aware)": "mahalanobis",
        "Robust MCD Mahalanobis (Rousseeuw)": "robust_mahalanobis"
    }

    default_method_idx = 0
    if preset_config and "method" in preset_config:
        for idx, (_, val) in enumerate(method_choices.items()):
            if val == preset_config["method"]:
                default_method_idx = idx
                break

    selected_method_label = st.selectbox("Detection Model", list(method_choices.keys()), index=default_method_idx)
    selected_method = method_choices[selected_method_label]

    st.markdown("---")
    st.markdown("#### 🎛️ 3. Hyperparameters & Tuning")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        contamination_rate = st.slider("Contamination Rate", min_value=0.01, max_value=0.15, value=0.05, step=0.01)
    with col_h2:
        seed_val = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

    with st.expander("🔬 Fine-Tune Statistical Cutoffs", expanded=False):
        z_thresh = st.slider("Z-Score Threshold (σ)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        mod_z_thresh = st.slider("Modified Z (MAD) Threshold", min_value=2.0, max_value=6.0, value=3.5, step=0.1)
        iqr_k = st.slider("IQR Multiplier (k)", min_value=1.0, max_value=3.5, value=1.5, step=0.25)
        mah_alpha = st.slider("Mahalanobis χ² Significance (α)", min_value=0.001, max_value=0.10, value=0.01, step=0.005)

        ensemble_rule = st.selectbox(
            "Ensemble Voting Logic",
            ["majority", "any", "all", "weighted"],
            index=0,
            help="'majority': >=2 votes; 'any': any detector flags; 'all': unanimous; 'weighted': weighted anomaly score >= 0.50"
        )


# -----------------------------------------------------------------------------
# Data Loading & Model Execution
# -----------------------------------------------------------------------------
t_start = time.time()
df, feature_cols, target_col = get_cached_dataset(
    dataset_key=dataset_selection,
    contamination=contamination_rate,
    seed=seed_val
)

X_data = df[feature_cols].to_numpy(dtype=float)
y_true = df[target_col].to_numpy(dtype=int)

# Instantiate and fit detector
detector = StatisticalAnomalyDetector(
    method=selected_method,
    z_threshold=z_thresh,
    mod_z_threshold=mod_z_thresh,
    iqr_k=iqr_k,
    mahalanobis_alpha=mah_alpha,
    ensemble_voting=ensemble_rule
)
detector.fit(X_data, feature_names=feature_cols)

# Predict & Score
y_pred = detector.predict(X_data)
scores = detector.score_samples(X_data)
metrics = evaluate_ad_metrics(y_true, y_pred, scores=scores)
fit_duration_ms = round((time.time() - t_start) * 1000, 2)


# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <span class="tag-badge badge-ad">PROJECT 011</span>
            <span class="tag-badge badge-stat">STATISTICAL ANOMALY DETECTION</span>
            <span class="tag-badge badge-robust">{selected_method_label.upper()}</span>
            <h1 class="hero-title">Statistical Anomaly Detection Studio</h1>
            <p class="hero-subtitle">
                An industrial-grade laboratory exploring parametric Z-Score deviations, robust Median Absolute Deviation (MAD),
                Tukey's IQR fences, multivariate Mahalanobis distance, and multi-detector consensus ensembles.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Top KPI Metric Cards
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Detection F1-Score</div>
        <div class="metric-value">{metrics['f1']:.4f}</div>
        <div class="metric-sub">Precision: {metrics['precision']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Recall (Sensitivity)</div>
        <div class="metric-value">{metrics['recall']*100:.1f}%</div>
        <div class="metric-sub">{metrics['tp']} of {metrics['tp']+metrics['fn']} anomalies caught</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">False Alarm Rate</div>
        <div class="metric-value">{metrics['fpr']*100:.2f}%</div>
        <div class="metric-sub">{metrics['fp']} false positives</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    roc_display = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ROC-AUC Score</div>
        <div class="metric-value">{roc_display}</div>
        <div class="metric-sub">Accuracy: {metrics['accuracy']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Inference Latency</div>
        <div class="metric-value">{fit_duration_ms} ms</div>
        <div class="metric-sub">{len(df)} samples scored</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Detection & Visualizer",
    "⚔️ Method Tournament Arena",
    "🌐 Multivariate & Covariance",
    "🎯 Live Real-Time Simulator",
    "⏱️ Time-Series & Streaming",
    "📂 Custom CSV Scanner",
    "📚 Statistical Theory"
])


# =============================================================================
# TAB 1: Detection & Visualizer
# =============================================================================
with tab1:
    st.subheader("🔍 Statistical Anomaly Detection & Landscape Visualizer")
    st.caption("Inspect how the statistical model separates normal operating telemetry from anomalous outliers.")

    col_v1, col_v2 = st.columns([3, 1])

    with col_v1:
        if len(feature_cols) >= 2:
            f_x = st.selectbox("X-Axis Feature", feature_cols, index=0)
            f_y = st.selectbox("Y-Axis Feature", feature_cols, index=1 if len(feature_cols) > 1 else 0)

            # Scatter Plot with Anomaly Coloring
            df_plot = df.copy()
            df_plot["Predicted_Status"] = np.where(y_pred, "Flagged Anomaly ⚠️", "Normal Baseline ✅")
            df_plot["Anomaly_Score"] = scores
            df_plot["Ground_Truth"] = np.where(y_true, "Real Anomaly", "Normal")

            color_map = {
                "Normal Baseline ✅": "#3B82F6",
                "Flagged Anomaly ⚠️": "#EF4444"
            }

            fig_scatter = px.scatter(
                df_plot,
                x=f_x,
                y=f_y,
                color="Predicted_Status",
                color_discrete_map=color_map,
                hover_data={"Anomaly_Score": ":.4f", "Ground_Truth": True},
                title=f"Statistical Anomaly Distribution: {f_x} vs {f_y}",
                template="plotly_dark",
                height=520
            )

            fig_scatter.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="#FFFFFF")))
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font=dict(family="Plus Jakarta Sans"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # If 2D manifold, draw contour decision landscape
            if len(feature_cols) == 2 and selected_method in ["mahalanobis", "robust_mahalanobis", "ensemble"]:
                x_min, x_max = df[f_x].min() - 0.5, df[f_x].max() + 0.5
                y_min, y_max = df[f_y].min() - 0.5, df[f_y].max() + 0.5
                xx, yy, mesh_scores, _ = compute_2d_decision_mesh(detector, (x_min, x_max), (y_min, y_max), grid_res=50)

                fig_scatter.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, 50),
                    y=np.linspace(y_min, y_max, 50),
                    z=mesh_scores,
                    colorscale="Reds",
                    showscale=False,
                    opacity=0.25,
                    contours=dict(start=0.5, end=0.9, size=0.2),
                    hoverinfo="skip"
                ))

            st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            # 1D Distribution Histogram
            feat = feature_cols[0]
            df_plot = df.copy()
            df_plot["Predicted_Status"] = np.where(y_pred, "Flagged Anomaly", "Normal")

            fig_hist = px.histogram(
                df_plot,
                x=feat,
                color="Predicted_Status",
                color_discrete_map={"Normal": "#3B82F6", "Flagged Anomaly": "#EF4444"},
                marginal="box",
                barmode="overlay",
                nbins=40,
                title=f"1D Distribution Histogram: {feat}",
                template="plotly_dark",
                height=500
            )
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_v2:
        st.markdown("#### 📊 Anomaly Profile")
        total_obs = len(df)
        flagged_count = int(np.sum(y_pred))
        normal_count = total_obs - flagged_count

        st.metric("Total Records", f"{total_obs:,}")
        st.metric("Flagged Outliers", f"{flagged_count:,}", f"{(flagged_count/total_obs)*100:.1f}%")
        st.metric("Normal Population", f"{normal_count:,}")

        st.markdown("---")
        st.markdown("#### 🎯 Confusion Breakdown")
        st.write(f"• **True Positives (TP):** {metrics['tp']}")
        st.write(f"• **False Positives (FP):** {metrics['fp']}")
        st.write(f"• **True Negatives (TN):** {metrics['tn']}")
        st.write(f"• **False Negatives (FN):** {metrics['fn']}")


# =============================================================================
# TAB 2: Method Tournament Arena
# =============================================================================
with tab2:
    st.subheader("⚔️ Statistical Tournament Arena")
    st.caption("Benchmark all statistical methods head-to-head on the current dataset to identify the optimal detector.")

    with st.spinner("Running statistical tournament across all methods..."):
        df_tournament = compare_all_statistical_methods(X_data, y_true, feature_names=feature_cols)

    # Leaderboard Table
    st.dataframe(
        df_tournament.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-Score": "{:.4f}",
            "False Alarm Rate (FPR)": "{:.2%}",
            "ROC-AUC": "{:.4f}"
        }).highlight_max(subset=["F1-Score", "ROC-AUC", "Precision", "Recall"], color="#065F46")
          .highlight_min(subset=["False Alarm Rate (FPR)"], color="#065F46"),
        use_container_width=True
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        # F1 vs False Alarm Rate Scatter
        fig_tradeoff = px.scatter(
            df_tournament,
            x="False Alarm Rate (FPR)",
            y="F1-Score",
            text="Method",
            size="Recall",
            color="Precision",
            color_continuous_scale="Viridis",
            title="Algorithm Tradeoff: F1-Score vs False Alarm Rate",
            template="plotly_dark",
            height=420
        )
        fig_tradeoff.update_traces(textposition="top center", marker=dict(sizeref=0.08, sizemin=10))
        fig_tradeoff.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    with col_t2:
        # Radar Performance Comparison
        fig_radar = go.Figure()
        categories = ["Precision", "Recall", "Accuracy", "ROC-AUC"]

        for _, row in df_tournament.iterrows():
            r_vals = [row[c] for c in categories]
            r_vals.append(r_vals[0])  # close loop
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=categories + [categories[0]],
                name=row["Method"]
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Multi-Metric Radar Comparison",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=420
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================================
# TAB 3: Multivariate & Covariance Inspector
# =============================================================================
with tab3:
    st.subheader("🌐 Multivariate Covariance & Mahalanobis Analysis")
    st.caption("Understand why univariate tests fail when anomalies hide in the correlation structure between features.")

    if len(feature_cols) >= 2:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            # Correlation Matrix Heatmap
            corr_matrix = df[feature_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                title="Telemetry Feature Correlation Matrix",
                template="plotly_dark",
                height=420
            )
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_m2:
            # Mahalanobis Distance vs Chi-Square Cutoff Histogram
            p = detector.fitted_params_
            diff = X_data - p["robust_mean"]
            mah_distances = np.sqrt(np.sum((diff @ p["robust_inv_cov"]) * diff, axis=1))

            fig_mah = go.Figure()
            fig_mah.add_trace(go.Histogram(
                x=mah_distances[y_true == 0],
                name="Normal Population",
                marker_color="#3B82F6",
                opacity=0.75,
                nbinsx=35
            ))
            fig_mah.add_trace(go.Histogram(
                x=mah_distances[y_true == 1],
                name="True Outliers",
                marker_color="#EF4444",
                opacity=0.75,
                nbinsx=35
            ))

            cutoff_val = p["mahalanobis_threshold"]
            fig_mah.add_vline(
                x=cutoff_val,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"χ² Cutoff ({cutoff_val:.2f})",
                annotation_position="top right"
            )

            fig_mah.update_layout(
                barmode="overlay",
                title=f"Mahalanobis Distance Distribution (df={len(feature_cols)})",
                xaxis_title="Mahalanobis Distance D_M",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=420
            )
            st.plotly_chart(fig_mah, use_container_width=True)

        st.markdown("""
        > [!NOTE]
        > **The Off-Diagonal Covariance Phenomenon:** In multidimensional systems, an operational state can have individual feature values that are completely unremarkable when viewed in isolation (e.g. CPU=60% and Latency=120ms). However, if high CPU usually coincides with high latency, a point with **low CPU and high latency** constitutes a catastrophic failure (e.g., deadlock or network block). Only **Mahalanobis Distance** measures distance in standard deviations along the principal eigenvectors of covariance!
        """)

    else:
        st.info("Multivariate covariance inspection requires a dataset with at least 2 features.")


# =============================================================================
# TAB 4: Live Real-Time Simulator
# =============================================================================
with tab4:
    st.subheader("🎯 Live Real-Time Inference & What-If Simulator")
    st.caption("Simulate arbitrary sensor or transaction inputs to test detector alerts and observe which features drive the anomaly score.")

    col_s1, col_s2 = st.columns([1, 2])

    input_sample = []
    with col_s1:
        st.markdown("#### 🎛️ Input Telemetry Values")
        for feat in feature_cols:
            f_min = float(df[feat].min())
            f_max = float(df[feat].max())
            f_mean = float(df[feat].mean())
            f_std = float(df[feat].std())

            val = st.slider(
                f"{feat}",
                min_value=round(f_min - 2 * f_std, 2),
                max_value=round(f_max + 2 * f_std, 2),
                value=round(f_mean, 2),
                step=round(max(0.01, f_std / 10), 2)
            )
            input_sample.append(val)

    with col_s2:
        explanation = detector.explain_sample(input_sample)

        st.markdown("#### 🚨 Detector Verdict & Diagnostic Score")
        if explanation["is_anomaly"]:
            st.markdown(f"""
            <div class="verdict-anomaly">
                <h3>⚠️ ANOMALY ALERT TRIGGERED</h3>
                <p><strong>Composite Anomaly Score:</strong> {explanation['anomaly_score']:.4f} / 1.0000</p>
                <p><strong>Primary Root Cause:</strong> <code>{explanation['primary_driver']}</code></p>
                <p><strong>Mahalanobis Distance:</strong> {explanation['mahalanobis_distance']:.2f} (Threshold: {explanation['mahalanobis_threshold']:.2f})</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-normal">
                <h3>✅ NORMAL OPERATIONAL ENVELOPE</h3>
                <p><strong>Composite Anomaly Score:</strong> {explanation['anomaly_score']:.4f} / 1.0000</p>
                <p>Telemetry values are within acceptable statistical boundaries.</p>
                <p><strong>Mahalanobis Distance:</strong> {explanation['mahalanobis_distance']:.2f} (Threshold: {explanation['mahalanobis_threshold']:.2f})</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Feature Deviation Breakdown")

        df_breakdown = pd.DataFrame(explanation["feature_breakdown"])
        st.dataframe(
            df_breakdown[["feature", "value", "mean", "std", "z_score", "modified_z", "iqr_outlier"]].style.format({
                "value": "{:.2f}",
                "mean": "{:.2f}",
                "std": "{:.2f}",
                "z_score": "{:+.2f}",
                "modified_z": "{:+.2f}"
            }),
            use_container_width=True
        )

        # Feature Deviation Bar Chart
        fig_bar = px.bar(
            df_breakdown,
            x="feature",
            y="anomaly_driver_score",
            color="anomaly_driver_score",
            color_continuous_scale="Reds",
            title="Feature Deviation Magnitude (|Z| / |ModZ|)",
            template="plotly_dark",
            height=280
        )
        fig_bar.add_hline(y=3.0, line_dash="dash", line_color="#EF4444", annotation_text="3σ Anomaly Cutoff")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
        st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# TAB 5: Time-Series & Streaming Anomaly Detection
# =============================================================================
with tab5:
    st.subheader("⏱️ Time-Series Streaming & Bollinger Band Envelopes")
    st.caption("Monitor continuous telemetry streams using adaptive rolling window statistics and dynamic volatility bands.")

    col_ts1, col_ts2 = st.columns([1, 3])

    with col_ts1:
        st.markdown("#### ⚙️ Stream Configuration")
        stream_steps = st.slider("Stream Sequence Length", 100, 600, 350, step=50)
        roll_window = st.slider("Rolling Window Size (W)", 5, 50, 20, step=5)
        band_multiplier = st.slider("Bollinger Band Multiplier (k)", 1.5, 4.0, 2.5, step=0.25)
        noise_amp = st.slider("Noise Amplitude", 0.2, 3.0, 1.0, step=0.2)

    with col_ts2:
        df_stream = generate_streaming_telemetry(n_steps=stream_steps, noise_level=noise_amp, seed=seed_val)
        df_rolling = compute_rolling_stats(df_stream["metric_stream"], window=roll_window, k=band_multiplier)

        fig_stream = go.Figure()

        # Bollinger Upper & Lower Fill
        fig_stream.add_trace(go.Scatter(
            x=df_stream["timestamp_step"],
            y=df_rolling["upper_band"],
            mode="lines",
            line=dict(color="rgba(59, 130, 246, 0.3)", width=1),
            name="Upper Bollinger Band (+k·σ)",
            showlegend=False
        ))
        fig_stream.add_trace(go.Scatter(
            x=df_stream["timestamp_step"],
            y=df_rolling["lower_band"],
            mode="lines",
            line=dict(color="rgba(59, 130, 246, 0.3)", width=1),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.08)",
            name="Adaptive Confidence Envelope",
        ))

        # Rolling Mean
        fig_stream.add_trace(go.Scatter(
            x=df_stream["timestamp_step"],
            y=df_rolling["rolling_mean"],
            mode="lines",
            line=dict(color="#60A5FA", width=1.8, dash="dot"),
            name="Rolling Mean (μ_t)"
        ))

        # Stream Signal
        fig_stream.add_trace(go.Scatter(
            x=df_stream["timestamp_step"],
            y=df_stream["metric_stream"],
            mode="lines",
            line=dict(color="#E2E8F0", width=1.5),
            name="Incoming Telemetry Stream"
        ))

        # Flagged Anomalies
        anom_mask = df_rolling["is_anomaly"]
        fig_stream.add_trace(go.Scatter(
            x=df_stream["timestamp_step"][anom_mask],
            y=df_stream["metric_stream"][anom_mask],
            mode="markers",
            marker=dict(color="#EF4444", size=9, symbol="x"),
            name="Streaming Anomaly Alert"
        ))

        fig_stream.update_layout(
            title=f"Adaptive Streaming Anomaly Detector (Window={roll_window}, Multiplier={band_multiplier}σ)",
            xaxis_title="Time Step (t)",
            yaxis_title="Metric Amplitude",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_stream, use_container_width=True)

        m_stream = evaluate_ad_metrics(df_stream["is_anomaly"], df_rolling["is_anomaly"])
        st.markdown(f"**Stream Performance:** F1-Score: `{m_stream['f1']:.4f}` | Precision: `{m_stream['precision']*100:.1f}%` | Recall: `{m_stream['recall']*100:.1f}%` | Detected `{m_stream['tp']}` true events with `{m_stream['fp']}` false alarms.")


# =============================================================================
# TAB 6: Custom CSV Scanner
# =============================================================================
with tab6:
    st.subheader("📂 Custom Dataset Scanner & Batch Anomaly Profiler")
    st.caption("Upload your own CSV data to run automated statistical anomaly scans and export labeled results.")

    uploaded_file = st.file_uploader("Upload CSV File for Anomaly Audit", type=["csv"])

    if uploaded_file is not None:
        try:
            df_custom = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df_custom):,} rows and {df_custom.shape[1]} columns!")

            num_cols = list(df_custom.select_dtypes(include=[np.number]).columns)
            if not num_cols:
                st.error("Uploaded CSV has no numeric columns for statistical analysis.")
            else:
                selected_cols = st.multiselect("Select Numeric Features to Audit", num_cols, default=num_cols[:min(6, len(num_cols))])

                if selected_cols:
                    custom_X = df_custom[selected_cols].dropna().to_numpy()
                    custom_detector = StatisticalAnomalyDetector(method=selected_method)
                    custom_detector.fit(custom_X, feature_names=selected_cols)

                    custom_flags = custom_detector.predict(custom_X)
                    custom_scores = custom_detector.score_samples(custom_X)

                    df_scored = df_custom.loc[df_custom[selected_cols].dropna().index].copy()
                    df_scored["anomaly_flag"] = custom_flags.astype(int)
                    df_scored["anomaly_score"] = np.round(custom_scores, 4)

                    n_flagged = custom_flags.sum()
                    st.metric("Detected Anomalies", f"{n_flagged:,}", f"{(n_flagged/len(df_scored))*100:.2f}% of rows")

                    st.markdown("#### Scored Dataset Preview")
                    st.dataframe(df_scored.head(50), use_container_width=True)

                    # Export Scored CSV
                    csv_buffer = io.StringIO()
                    df_scored.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Scored CSV with Anomaly Labels",
                        data=csv_buffer.getvalue(),
                        file_name="anomaly_detection_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")
    else:
        st.info("Upload any CSV file containing numerical metrics (e.g., server logs, IoT telemetry, transaction records).")


# =============================================================================
# TAB 7: Statistical Theory
# =============================================================================
with tab7:
    st.subheader("📚 Statistical Foundations & Mathematical Formulations")
    st.caption("A rigorous mathematical breakdown of the statistical tests powering this studio.")

    st.markdown(r"""
    ### 1. Standard Z-Score (Normal Gaussian Deviate)
    The classical standard score measures the signed number of standard deviations an observation $x$ lies from the sample mean $\mu$:
    $$Z = \frac{|x - \mu|}{\sigma}$$
    An anomaly is flagged when $Z > k$ (typically $k=3.0$, corresponding to $p \approx 0.0027$ under Gaussian normality).

    > [!WARNING]
    > **Masking Effect:** Extreme outliers heavily inflate both sample mean $\mu$ and sample standard deviation $\sigma$. Consequently, a cluster of severe outliers can increase $\sigma$ so much that they mask each other, keeping $Z \le 3.0$!

    ---

    ### 2. Modified Z-Score (Boris Iglewicz & David Hoaglin)
    To overcome masking, the Modified Z-score replaces mean and standard deviation with the **median** and **Median Absolute Deviation (MAD)**:
    $$\text{MAD} = \text{median}\left(|x_i - \text{median}(x)|\right)$$
    $$M_i = \frac{0.6745 \cdot |x_i - \text{median}(x)|}{\text{MAD}}$$
    where $0.6745$ is the 75th percentile of the standard normal distribution ($E[\text{MAD}] = \sigma / 0.6745$ for normal data).
    Outliers are typically declared when $M_i > 3.5$.

    ---

    ### 3. Tukey's Interquartile Range (IQR) Fences
    Tukey's method is completely non-parametric and makes zero distributional assumptions:
    $$\text{IQR} = Q_3 - Q_1$$
    $$\text{Lower Fence} = Q_1 - k \cdot \text{IQR}, \quad \text{Upper Fence} = Q_3 + k \cdot \text{IQR}$$
    - $k = 1.5$: Outlier boundary
    - $k = 3.0$: "Far out" extreme anomaly boundary

    ---

    ### 4. Multivariate Mahalanobis Distance ($D_M$)
    When features share covariance, univariate tests cannot detect anomalies that violate correlation. Mahalanobis distance measures the multidimensional distance of point $\mathbf{x} \in \mathbb{R}^p$ from the centroid $\boldsymbol{\mu}$, weighted by the inverse covariance matrix $\boldsymbol{\Sigma}^{-1}$:
    $$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

    Under the null hypothesis of multivariate normality:
    $$D_M^2(\mathbf{x}) \sim \chi^2_p$$
    where $p$ is the number of feature dimensions. An observation is deemed an anomaly if:
    $$D_M^2(\mathbf{x}) > \chi^2_{p, 1-\alpha}$$

    ---

    ### 5. Robust Minimum Covariance Determinant (MCD / Rousseeuw)
    Just as extreme outliers distort $\mu$ and $\sigma$, they also distort the empirical covariance matrix $\boldsymbol{\Sigma}$.
    The **Minimum Covariance Determinant (MCD)** estimator finds a subset of $h$ observations (typically $85\%$ of the data) whose empirical covariance matrix has the **lowest possible determinant**, providing a breakdown point of up to $50\%$ against contamination.
    """)

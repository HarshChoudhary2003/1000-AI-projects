"""
Project 013: Principal Component Analysis (PCA) Studio
======================================================
Flagship interactive Streamlit application for dimensionality reduction,
spectral analysis, scree plots, biplots, image compression reconstruction,
first-principles SVD math verification, and downstream ML benchmarking.

Part of the 1,000 AI Projects Implementation Library.
"""

import io
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    compare_scratch_vs_sklearn
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Principal Component Analysis Studio | 1,000 AI Projects",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Theme Styling
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
        padding: 1.8rem 2.2rem;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(99, 102, 241, 0.15) 50%, rgba(244, 63, 94, 0.12) 100%);
        border: 1px solid rgba(6, 182, 212, 0.28);
        border-radius: 16px;
        margin-bottom: 1.6rem;
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.02rem;
        margin-top: 0.35rem;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.90) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(6, 182, 212, 0.4);
    }
    .metric-title {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.74rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Badges */
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid rgba(6, 182, 212, 0.35);
        border-radius: 9999px;
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        border-color: rgba(168, 85, 247, 0.35);
        color: #C084FC;
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.35);
        color: #34D399;
    }

    /* Info card callout */
    .info-callout {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38BDF8;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin: 12px 0;
        color: #CBD5E1;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🌌 PCA Control Deck")

presets = get_archetype_presets()
preset_names = ["Custom Configuration"] + list(presets.keys())
selected_preset = st.sidebar.selectbox("🎯 Quick Archetype Preset", preset_names, index=1)

# Default values
default_dataset = "digits"
default_k = 2
default_scaling = "standard"

if selected_preset != "Custom Configuration":
    p_info = presets[selected_preset]
    default_dataset = p_info["dataset"]
    default_k = p_info["n_components"]
    default_scaling = "standard" if p_info["standardize"] else "none"
    st.sidebar.caption(f"💡 {p_info['notes']}")

# Dataset selection
dataset_options = {
    "digits": "Handwritten Digits (64D - 8x8 Grayscale Images)",
    "breast_cancer": "Breast Cancer Biopsy (30D Cell Morphometry)",
    "wine": "Wine Cultivars (13D Chemical Constituents)",
    "iris": "Botanical Iris (4D Flower Dimensions)",
    "swiss_roll": "Synthetic Swiss Roll (6D Nonlinear Manifold)",
    "correlated_gaussian": "Correlated Sensor Array (12D Gaussian Ellipsoid)"
}
dataset_keys = list(dataset_options.keys())
dataset_idx = dataset_keys.index(default_dataset) if default_dataset in dataset_keys else 0

selected_dataset = st.sidebar.selectbox(
    "📁 Benchmark Dataset",
    options=dataset_keys,
    format_func=lambda k: dataset_options[k],
    index=dataset_idx
)

# Scaling option
scaling_option = st.sidebar.selectbox(
    "⚖️ Feature Preprocessing",
    options=["standard", "robust", "none"],
    index=0 if default_scaling == "standard" else 2,
    format_func=lambda s: {
        "standard": "StandardScaler (Mean=0, Unit Variance)",
        "robust": "RobustScaler (Median / IQR for Outliers)",
        "none": "No Scaling (Raw Coordinate Scale)"
    }[s]
)

# PCA Algorithm Variant
pca_variant = st.sidebar.selectbox(
    "🔬 PCA Variant",
    options=["standard", "incremental", "kernel", "truncated_svd"],
    format_func=lambda v: {
        "standard": "Standard PCA (Exact SVD / LAPACK)",
        "incremental": "Incremental PCA (Mini-batch / Out-of-Core)",
        "kernel": "Kernel PCA (Nonlinear Manifold Embedding)",
        "truncated_svd": "Truncated SVD (Sparse Matrix / Uncentered)"
    }[v]
)

kernel_type = "rbf"
gamma_val = None
if pca_variant == "kernel":
    col_k1, col_k2 = st.sidebar.columns(2)
    with col_k1:
        kernel_type = st.selectbox("Kernel", ["rbf", "poly", "cosine", "sigmoid"])
    with col_k2:
        gamma_val = st.number_input("Gamma", value=0.05, step=0.01, format="%.3f")

# Number of components slider
max_slider_k = 10 if selected_dataset != "digits" else 30
target_k = st.sidebar.slider("✂️ Principal Components (k)", min_value=1, max_value=max_slider_k, value=default_k)

# Subsampling
subsample_toggle = st.sidebar.checkbox("Subsample Data (Faster Render)", value=False)
n_subsamples = None
if subsample_toggle:
    n_subsamples = st.sidebar.slider("Sample Count", min_value=200, max_value=2000, value=600, step=100)

st.sidebar.markdown("---")
st.sidebar.caption("🚀 1,000 AI Projects • Implementation 013")

# -----------------------------------------------------------------------------
# Data Loading & Pipeline Execution
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data_cached(name: str, n_samples: Optional[int]):
    return load_benchmark_dataset(name, n_samples=n_samples)

data_bundle = load_data_cached(selected_dataset, n_subsamples)
X = data_bundle["X"]
y = data_bundle["y"]
df_full = data_bundle["df"]
feature_names = data_bundle["feature_names"]
target_names = data_bundle["target_names"]
is_image = data_bundle["is_image_data"]
image_shape = data_bundle["image_shape"]

# Fit primary PCA pipeline
fit_res = fit_pca_pipeline(
    X,
    n_components=target_k,
    scaling=scaling_option,
    pca_type=pca_variant,
    kernel=kernel_type,
    gamma=gamma_val
)
X_proj = fit_res["X_proj"]
model = fit_res["model"]
scaler = fit_res["scaler"]
exp_var_ratio = fit_res["explained_variance_ratio_"]
cum_var_ratio = fit_res["cumulative_variance_ratio_"]

# Full scree analysis
scree_res = compute_scree_analysis(fit_res["X_scaled"], max_components=min(X.shape[1], 35))

# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">🌌 Principal Component Analysis Studio</h1>
            <p class="hero-subtitle">Interactive Dimensionality Reduction, Spectral Decomposition & Orthogonal Manifold Lab</p>
        </div>
        <div style="margin-top: 8px;">
            <span class="badge-tag">{selected_dataset.replace('_', ' ').title()}</span>
            <span class="badge-tag badge-purple">{scaling_option.title()} Scaling</span>
            <span class="badge-tag badge-emerald">{pca_variant.upper()}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top-Level KPI Metric Deck
# -----------------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

tot_var_retained = float(cum_var_ratio[-1] * 100) if len(cum_var_ratio) > 0 else 0.0
dim_reduction = float((1.0 - (target_k / X.shape[1])) * 100)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Input Dimensions</div>
        <div class="metric-value">{X.shape[1]} <span style="font-size:0.9rem; color:#94A3B8;">feats</span></div>
        <div class="metric-sub">{X.shape[0]} Observations</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Target Latent Space</div>
        <div class="metric-value">{target_k} <span style="font-size:0.9rem; color:#06B6D4;">PCs</span></div>
        <div class="metric-sub">Compression: {dim_reduction:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Retained Variance</div>
        <div class="metric-value" style="color: #34D399;">{tot_var_retained:.1f}%</div>
        <div class="metric-sub">Across top {target_k} components</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    kaiser_val = scree_res["kaiser_k"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Kaiser Criterion</div>
        <div class="metric-value">{kaiser_val} <span style="font-size:0.9rem; color:#C084FC;">PCs</span></div>
        <div class="metric-sub">Eigenvalues &ge; 1.0</div>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    k95_val = scree_res["k_95"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">95% Variance Knee</div>
        <div class="metric-value">{k95_val} <span style="font-size:0.9rem; color:#F59E0B;">PCs</span></div>
        <div class="metric-sub">Preserves 95% total variance</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Studio Tabs
# -----------------------------------------------------------------------------
tab_proj, tab_scree, tab_loadings, tab_reconstruct, tab_math, tab_benchmark, tab_upload = st.tabs([
    "🔮 2D & 3D Projections",
    "📊 Scree & Variance Spectrum",
    "🧭 Feature Loadings & Biplot",
    "🖼️ Image Compression Lab",
    "🧮 Math & SVD Verification",
    "⚡ Downstream ML Impact",
    "📁 Custom Dataset Scanner"
])

# =============================================================================
# TAB 1: 2D & 3D Interactive Projections
# =============================================================================
with tab_proj:
    st.markdown("### 🔮 Latent Principal Component Manifolds")
    st.caption("Visualizing high-dimensional observations projected onto the dominant orthogonal eigenvectors.")

    col_view1, col_view2 = st.columns([1, 1])

    # Build Projection DataFrame
    proj_cols = [f"PC{i+1}" for i in range(X_proj.shape[1])]
    proj_df = pd.DataFrame(X_proj, columns=proj_cols)
    proj_df["Class"] = df_full["target_label"]
    proj_df["Target_ID"] = y

    with col_view1:
        st.markdown("#### 🗺️ 2D Principal Subspace (PC1 vs PC2)")
        if X_proj.shape[1] >= 2:
            pc1_var = exp_var_ratio[0] * 100 if len(exp_var_ratio) > 0 else 0
            pc2_var = exp_var_ratio[1] * 100 if len(exp_var_ratio) > 1 else 0

            fig_2d = px.scatter(
                proj_df,
                x="PC1",
                y="PC2",
                color="Class",
                hover_data=["Target_ID"],
                title=f"PC1 ({pc1_var:.1f}% var) vs PC2 ({pc2_var:.1f}% var)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Prism,
                opacity=0.82
            )
            fig_2d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_2d, use_container_width=True)
        else:
            st.info("Select at least 2 components in the sidebar to inspect 2D projections.")

    with col_view2:
        st.markdown("#### 🌐 3D Principal Subspace (PC1 vs PC2 vs PC3)")
        if X_proj.shape[1] >= 3:
            pc3_var = exp_var_ratio[2] * 100 if len(exp_var_ratio) > 2 else 0

            fig_3d = px.scatter_3d(
                proj_df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Class",
                hover_data=["Target_ID"],
                title=f"PC1 ({pc1_var:.1f}%) | PC2 ({pc2_var:.1f}%) | PC3 ({pc3_var:.1f}%)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Prism,
                opacity=0.78
            )
            fig_3d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(15,23,42,0.8)", gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(backgroundcolor="rgba(15,23,42,0.8)", gridcolor="rgba(255,255,255,0.1)"),
                    zaxis=dict(backgroundcolor="rgba(15,23,42,0.8)", gridcolor="rgba(255,255,255,0.1)"),
                ),
                legend=dict(orientation="h", yanchor="bottom", y=0.98)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("💡 Set Principal Components (k) &ge; 3 in the sidebar to activate the interactive 3D orbit projector.")

    # Component Statistics Table
    with st.expander("📋 Inspect Projected Data Sample Coordinates", expanded=False):
        st.dataframe(proj_df.head(25), use_container_width=True)


# =============================================================================
# TAB 2: Scree Plot & Variance Spectrum
# =============================================================================
with tab_scree:
    st.markdown("### 📊 Scree Plot & Spectral Energy Distribution")
    st.caption("Determine the intrinsic dimensionality of the dataset using eigenvalues and cumulative explained variance.")

    scree_df = scree_res["scree_df"]

    col_scree_chart, col_scree_stat = st.columns([2, 1])

    with col_scree_chart:
        fig_scree = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar chart for individual variance
        fig_scree.add_trace(
            go.Bar(
                x=scree_df["Component"],
                y=scree_df["Explained_Variance_Pct"],
                name="Individual Variance (%)",
                marker_color="rgba(6, 182, 212, 0.75)",
                marker_line=dict(color="#06B6D4", width=1.5)
            ),
            secondary_y=False
        )

        # Line chart for cumulative variance
        fig_scree.add_trace(
            go.Scatter(
                x=scree_df["Component"],
                y=scree_df["Cumulative_Variance_Pct"],
                name="Cumulative Variance (%)",
                mode="lines+markers",
                line=dict(color="#F43F5E", width=3),
                marker=dict(size=7, color="#FDA4AF", symbol="circle")
            ),
            secondary_y=True
        )

        # 95% Cutoff horizontal threshold
        fig_scree.add_hline(
            y=95.0,
            line_dash="dash",
            line_color="#10B981",
            secondary_y=True,
            annotation_text="95% Cutoff Threshold",
            annotation_position="bottom right"
        )

        # 80% Cutoff horizontal threshold
        fig_scree.add_hline(
            y=80.0,
            line_dash="dot",
            line_color="#F59E0B",
            secondary_y=True,
            annotation_text="80% Cutoff Threshold",
            annotation_position="bottom right"
        )

        fig_scree.update_layout(
            title="Scree Plot: Individual & Cumulative Explained Variance",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
        )
        fig_scree.update_xaxes(title_text="Principal Component", tickangle=-45)
        fig_scree.update_yaxes(title_text="Individual Variance (%)", secondary_y=False)
        fig_scree.update_yaxes(title_text="Cumulative Variance (%)", range=[0, 105], secondary_y=True)

        st.plotly_chart(fig_scree, use_container_width=True)

    with col_scree_stat:
        st.markdown("#### 🎯 Dimensionality Heuristics")

        st.markdown(f"""
        <div class="info-callout">
            <b>1. Kaiser-Guttman Rule:</b> <code>{scree_res['kaiser_k']} Components</code><br>
            Retains all components having eigenvalues &ge; 1.0 (explains more variance than an average single standardized variable).
        </div>
        <div class="info-callout" style="border-left-color: #F59E0B;">
            <b>2. Scree Elbow Point:</b> <code>{scree_res['elbow_k']} Components</code><br>
            Point of maximum geometric curvature where marginal variance drops sharply into noise.
        </div>
        <div class="info-callout" style="border-left-color: #10B981;">
            <b>3. 90% Variance Target:</b> <code>{scree_res['k_90']} Components</code><br>
            Minimum components required to preserve &ge;90% of total variance.
        </div>
        <div class="info-callout" style="border-left-color: #EC4899;">
            <b>4. 95% Variance Target:</b> <code>{scree_res['k_95']} Components</code><br>
            Gold-standard cutoff for lossless data reconstruction & noise filtering.
        </div>
        """, unsafe_allow_html=True)

    # Scree Spectrum Table
    with st.expander("📑 Detailed Component Spectrum Table", expanded=False):
        st.dataframe(scree_df.style.format({
            "Eigenvalue": "{:.4f}",
            "Explained_Variance_Ratio": "{:.4f}",
            "Explained_Variance_Pct": "{:.2f}%",
            "Cumulative_Variance_Ratio": "{:.4f}",
            "Cumulative_Variance_Pct": "{:.2f}%"
        }), use_container_width=True)


# =============================================================================
# TAB 3: Feature Loadings & Biplot
# =============================================================================
with tab_loadings:
    st.markdown("### 🧭 Feature Loadings & 2D Biplot Analysis")
    st.caption("Inspect how original input features project onto principal axes. Loading vectors reveal correlation and variable significance.")

    loadings_res = compute_loadings_and_biplot(model, feature_names)

    if "error" in loadings_res:
        st.warning(loadings_res["error"])
    else:
        loadings_df = loadings_res["loadings_df"]
        contrib_df = loadings_res["contrib_df"]
        biplot_vectors = loadings_res["biplot_vectors"]

        col_bip, col_rank = st.columns([1.6, 1])

        with col_bip:
            st.markdown("#### 🎯 Interactive 2D Biplot (Observations + Feature Vectors)")
            if X_proj.shape[1] >= 2:
                # Base scatter of observations
                fig_biplot = go.Figure()

                # Add sample points
                for c_name in np.unique(df_full["target_label"]):
                    mask = (df_full["target_label"] == c_name)
                    fig_biplot.add_trace(go.Scatter(
                        x=X_proj[mask, 0],
                        y=X_proj[mask, 1],
                        mode="markers",
                        name=str(c_name),
                        marker=dict(size=6, opacity=0.65)
                    ))

                # Arrow scaling factor so vectors match observation bounds
                x_range = np.max(np.abs(X_proj[:, 0])) * 0.85
                y_range = np.max(np.abs(X_proj[:, 1])) * 0.85

                # Plot top-12 longest feature vectors to keep biplot legible
                top_vectors = biplot_vectors[:12]
                for v in top_vectors:
                    vx_scaled = v["vx"] * x_range
                    vy_scaled = v["vy"] * y_range

                    # Arrow trace
                    fig_biplot.add_trace(go.Scatter(
                        x=[0, vx_scaled],
                        y=[0, vy_scaled],
                        mode="lines+text",
                        text=["", f"<b>{v['feature']}</b>"],
                        textposition="top right",
                        line=dict(color="#F43F5E", width=2),
                        hoverinfo="text",
                        hovertext=f"Feature: {v['feature']}<br>PC1 Loading: {v['loading_pc1']:.3f}<br>PC2 Loading: {v['loading_pc2']:.3f}",
                        showlegend=False
                    ))

                fig_biplot.update_layout(
                    title="2D Biplot: Observation Projection & Top Feature Axes",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    xaxis=dict(title=f"PC1 ({exp_var_ratio[0]*100:.1f}%)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                    yaxis=dict(title=f"PC2 ({exp_var_ratio[1]*100:.1f}%)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_biplot, use_container_width=True)
            else:
                st.info("Requires at least 2 components for Biplot visualization.")

        with col_rank:
            st.markdown("#### 🏆 Top Contributing Features")
            top_tab1, top_tab2 = st.tabs(["Top for PC1", "Top for PC2"])
            with top_tab1:
                top_pc1 = loadings_res["top_pc1"]
                fig_top1 = px.bar(
                    x=top_pc1.values,
                    y=top_pc1.index,
                    orientation="h",
                    title="PC1 Absolute Loadings",
                    template="plotly_dark",
                    color=top_pc1.values,
                    color_continuous_scale="Viridis"
                )
                fig_top1.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_top1, use_container_width=True)

            with top_tab2:
                top_pc2 = loadings_res["top_pc2"]
                fig_top2 = px.bar(
                    x=top_pc2.values,
                    y=top_pc2.index,
                    orientation="h",
                    title="PC2 Absolute Loadings",
                    template="plotly_dark",
                    color=top_pc2.values,
                    color_continuous_scale="Plasma"
                )
                fig_top2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_top2, use_container_width=True)

        # Full Loadings Matrix Heatmap
        with st.expander("🔥 Full Component Loadings Heatmap", expanded=False):
            # Display first 8 PCs if available
            n_disp_pcs = min(8, loadings_df.shape[1])
            disp_loadings = loadings_df.iloc[:, :n_disp_pcs]
            fig_hm = px.imshow(
                disp_loadings,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
                title=f"Feature Loadings Matrix (Top {n_disp_pcs} Components)"
            )
            fig_hm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_hm, use_container_width=True)


# =============================================================================
# TAB 4: Image Compression & Reconstruction Lab
# =============================================================================
with tab_reconstruct:
    st.markdown("### 🖼️ Lossy Compression & Signal Reconstruction Lab")
    st.caption("Inspect the inverse transform projection: X_reconstructed = Z * V^T + mu. Observe how increasing k restores high-frequency visual details.")

    rec_col1, rec_col2 = st.columns([1, 2])

    with rec_col1:
        st.markdown("#### 🎛️ Reconstruction Controls")
        sample_idx = st.slider("Select Observation Index", min_value=0, max_value=min(100, len(X)-1), value=7)
        max_rec_k = min(X.shape[1], 40)
        rec_k = st.slider("Reconstruction Components (k)", min_value=1, max_value=max_rec_k, value=min(target_k, max_rec_k))

        rec_res = compute_reconstruction_analysis(X, scaler, k_components=rec_k, sample_index=sample_idx)

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div class="metric-title">Observation Label</div>
            <div class="metric-value" style="font-size: 1.35rem; color:#38BDF8;">{df_full['target_label'].iloc[sample_idx]}</div>
            <div class="metric-sub">Sample #{sample_idx}</div>
        </div>
        <div class="metric-card" style="margin-bottom: 12px;">
            <div class="metric-title">Sample Reconstruction MSE</div>
            <div class="metric-value" style="color: #F43F5E;">{rec_res['sample_mse']:.4f}</div>
            <div class="metric-sub">Average dataset MSE: {rec_res['mse_total']:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Energy Preserved</div>
            <div class="metric-value" style="color: #34D399;">{rec_res['explained_variance_ratio_sum']*100:.2f}%</div>
            <div class="metric-sub">Using {rec_k} of {X.shape[1]} features</div>
        </div>
        """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown("#### 🔬 Visual Comparison")
        if is_image and image_shape is not None:
            # Grayscale Image Reshaping (e.g. 8x8 digits)
            img_raw = rec_res["sample_raw"].reshape(image_shape)
            img_rec = rec_res["sample_rec"].reshape(image_shape)
            img_diff = np.abs(rec_res["sample_diff"]).reshape(image_shape)

            c_img1, c_img2, c_img3 = st.columns(3)
            with c_img1:
                st.markdown("<p style='text-align:center; font-weight:700;'>Original (64 Pixels)</p>", unsafe_allow_html=True)
                fig_o = px.imshow(img_raw, color_continuous_scale="gray", template="plotly_dark")
                fig_o.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_o, use_container_width=True)

            with c_img2:
                st.markdown(f"<p style='text-align:center; font-weight:700;'>Reconstructed (k={rec_k})</p>", unsafe_allow_html=True)
                fig_r = px.imshow(img_rec, color_continuous_scale="gray", template="plotly_dark")
                fig_r.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_r, use_container_width=True)

            with c_img3:
                st.markdown("<p style='text-align:center; font-weight:700;'>Absolute Error Residual</p>", unsafe_allow_html=True)
                fig_d = px.imshow(img_diff, color_continuous_scale="Hot", template="plotly_dark")
                fig_d.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_d, use_container_width=True)
        else:
            # Tabular feature profile comparison
            n_show_feats = min(20, X.shape[1])
            comp_df = pd.DataFrame({
                "Feature": feature_names[:n_show_feats],
                "Original": rec_res["sample_raw"][:n_show_feats],
                "Reconstructed": rec_res["sample_rec"][:n_show_feats]
            })
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=comp_df["Feature"], y=comp_df["Original"], name="Original", marker_color="#38BDF8"))
            fig_bar.add_trace(go.Bar(x=comp_df["Feature"], y=comp_df["Reconstructed"], name=f"Reconstructed (k={rec_k})", marker_color="#F43F5E"))
            fig_bar.update_layout(
                title=f"Feature Values: Original vs Reconstructed (Sample #{sample_idx})",
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Reconstruction Distortion Curve
    st.markdown("#### 📉 Rate-Distortion Curve: Reconstruction Error (MSE) vs Number of Components")
    curve_df = compute_reconstruction_curve(X, scaler, max_k=min(X.shape[1], 40), step=1)

    fig_curve = make_subplots(specs=[[{"secondary_y": True}]])
    fig_curve.add_trace(
        go.Scatter(
            x=curve_df["k_components"],
            y=curve_df["reconstruction_mse"],
            mode="lines+markers",
            name="Reconstruction MSE",
            line=dict(color="#F43F5E", width=3)
        ),
        secondary_y=False
    )
    fig_curve.add_trace(
        go.Scatter(
            x=curve_df["k_components"],
            y=curve_df["compression_ratio_pct"],
            mode="lines+markers",
            name="Dimensionality Compression Savings (%)",
            line=dict(color="#10B981", width=2, dash="dash")
        ),
        secondary_y=True
    )
    fig_curve.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_curve.update_xaxes(title_text="Number of Components (k)")
    fig_curve.update_yaxes(title_text="Mean Squared Error (MSE)", secondary_y=False)
    fig_curve.update_yaxes(title_text="Compression Ratio (%)", range=[0, 105], secondary_y=True)

    st.plotly_chart(fig_curve, use_container_width=True)


# =============================================================================
# TAB 5: Mathematical Engine & SVD Verification
# =============================================================================
with tab_math:
    st.markdown("### 🧮 First-Principles Mathematical Verification")
    st.caption("Proof that direct Covariance Matrix Eigendecomposition and Thin Singular Value Decomposition (SVD) produce numerically identical solutions.")

    math_col1, math_col2 = st.columns([1, 1])

    with math_col1:
        st.markdown("#### 📐 Mathematical Formulation")
        st.latex(r"X_c = X - \mu \quad \text{where } \mu = \frac{1}{n} \sum_{i=1}^n x_i")
        st.latex(r"\Sigma = \frac{1}{n - 1} X_c^T X_c \quad \text{(Sample Covariance Matrix)}")
        st.latex(r"\Sigma v_i = \lambda_i v_i \quad \text{(Eigendecomposition)}")
        st.latex(r"X_c = U S V^T \implies \lambda_i = \frac{S_i^2}{n - 1} \quad \text{(SVD Equivalence)}")
        st.latex(r"Z = X_c V_k \quad \text{(Orthogonal Subspace Projection)}")

    with math_col2:
        st.markdown("#### 🔬 Numerical Equivalence Validation")
        val_res = compare_scratch_vs_sklearn(X, n_components=min(target_k, 5))

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 10px;">
            <div class="metric-title">Max Projection Delta</div>
            <div class="metric-value" style="color: #34D399; font-size: 1.5rem;">{val_res['diff_projection_max']:.2e}</div>
            <div class="metric-sub">|X_scratch - X_sklearn| max absolute difference</div>
        </div>
        <div class="metric-card" style="margin-bottom: 10px;">
            <div class="metric-title">Max Variance Ratio Delta</div>
            <div class="metric-value" style="color: #38BDF8; font-size: 1.5rem;">{val_res['diff_variance_ratio_max']:.2e}</div>
            <div class="metric-sub">Eigenvalues vs SVD Singular values squared</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Numerical Equality Status</div>
            <div class="metric-value" style="color: #10B981; font-size: 1.4rem;">{'✅ EXACT MATCH' if val_res['is_numerically_identical'] else '⚠️ NUMERICAL DRIFT'}</div>
            <div class="metric-sub">Machine floating-point precision verified</div>
        </div>
        """, unsafe_allow_html=True)

    # Scratch Pipeline Trace
    with st.expander("🔍 Step-by-Step Scratch Deconstruction Trace", expanded=False):
        scratch_engine = CustomPCAScratch(n_components=target_k)
        scaler_temp = StandardScaler()
        X_sc_scaled = scaler_temp.fit_transform(X)
        scratch_engine.fit(X_sc_scaled)

        st.markdown(f"**1. Input Matrix Dimensions:** `{X.shape[0]} rows` &times; `{X.shape[1]} columns`")
        st.markdown(f"**2. Covariance Matrix Shape:** `{scratch_engine.covariance_matrix_.shape[0]} &times; {scratch_engine.covariance_matrix_.shape[1]}`")
        st.markdown(f"**3. First 5 Covariance Eigenvalues:** `{np.round(scratch_engine.eigenvalues_[:5], 4).tolist()}`")
        st.markdown(f"**4. First 5 SVD Singular Values:** `{np.round(scratch_engine.singular_values_[:5], 4).tolist()}`")
        st.markdown(f"**5. Projection Matrix $V_k^T$ Shape:** `{scratch_engine.components_.shape}`")


# =============================================================================
# TAB 6: Downstream ML Benchmark
# =============================================================================
with tab_benchmark:
    st.markdown("### ⚡ Downstream ML Impact: Combatting the Curse of Dimensionality")
    st.caption("Benchmark how feeding reduced principal components into a downstream classifier affects accuracy, F1-score, and training latency.")

    bench_col1, bench_col2 = st.columns([1, 3])

    with bench_col1:
        st.markdown("#### ⚙️ Benchmark Setup")
        clf_choice = st.selectbox("Classifier", ["logistic_regression", "random_forest"], format_func=lambda c: "Logistic Regression (Linear)" if c == "logistic_regression" else "Random Forest (Nonlinear Ensemble)")
        test_split = st.slider("Test Set Ratio", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
        run_btn = st.button("🚀 Run Downstream Benchmark", type="primary", use_container_width=True)

    with bench_col2:
        if run_btn or "bench_results" in st.session_state:
            with st.spinner("Training models across latent feature subsets..."):
                if run_btn:
                    # Choose smart candidate k values based on dataset size
                    n_feats = X.shape[1]
                    raw_cands = [1, 2, 3, 5, 10, 15, 20, 30]
                    cands = sorted(list(set([c for c in raw_cands if c < n_feats] + [target_k])))
                    bench_df = run_downstream_benchmark(
                        X, y,
                        scaling=scaling_option,
                        classifier_name=clf_choice,
                        k_candidates=cands,
                        test_size=test_split
                    )
                    st.session_state["bench_results"] = bench_df
                else:
                    bench_df = st.session_state["bench_results"]

            st.markdown("#### 📈 Benchmark Results Table")
            st.dataframe(bench_df.style.format({
                "variance_retained_pct": "{:.1f}%",
                "test_accuracy": "{:.4f}",
                "test_f1": "{:.4f}",
                "train_latency_ms": "{:.2f} ms",
                "dimension_savings_pct": "{:.1f}%"
            }), use_container_width=True)

            # Comparative Chart
            fig_bench = make_subplots(specs=[[{"secondary_y": True}]])
            fig_bench.add_trace(
                go.Bar(
                    x=bench_df["Model_Type"],
                    y=bench_df["test_accuracy"] * 100,
                    name="Test Accuracy (%)",
                    marker_color="#06B6D4"
                ),
                secondary_y=False
            )
            fig_bench.add_trace(
                go.Scatter(
                    x=bench_df["Model_Type"],
                    y=bench_df["train_latency_ms"],
                    mode="lines+markers",
                    name="Training Latency (ms)",
                    line=dict(color="#F43F5E", width=3),
                    marker=dict(size=8, color="#FDA4AF")
                ),
                secondary_y=True
            )
            fig_bench.update_layout(
                title=f"Accuracy vs Training Latency Across Dimensionality ({clf_choice.replace('_', ' ').title()})",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_bench.update_yaxes(title_text="Accuracy (%)", secondary_y=False)
            fig_bench.update_yaxes(title_text="Training Latency (ms)", secondary_y=True)

            st.plotly_chart(fig_bench, use_container_width=True)
        else:
            st.info("Click '🚀 Run Downstream Benchmark' to evaluate classification performance and latency across PCA dimensions.")


# =============================================================================
# TAB 7: Custom Dataset Scanner & Exporter
# =============================================================================
with tab_upload:
    st.markdown("### 📁 Custom CSV Scanner & Component Exporter")
    st.caption("Upload your own tabular dataset to perform instant PCA, view scree curves, project into 2D/3D space, and export transformed coordinates.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV successfully: `{custom_df.shape[0]} rows` &times; `{custom_df.shape[1]} columns`")

            # Extract numeric columns
            numeric_cols = custom_df.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_cols = [c for c in custom_df.columns if c not in numeric_cols]

            if len(numeric_cols) < 2:
                st.error("Uploaded CSV must contain at least 2 numeric feature columns.")
            else:
                c_up1, c_up2 = st.columns([1, 1])
                with c_up1:
                    sel_features = st.multiselect("Select Feature Columns for PCA", options=numeric_cols, default=numeric_cols[:min(15, len(numeric_cols))])
                with c_up2:
                    label_col = st.selectbox("Select Optional Label / Group Column (for coloring)", options=["None"] + custom_df.columns.tolist())

                if len(sel_features) >= 2:
                    X_custom = custom_df[sel_features].dropna().values
                    custom_scaler = StandardScaler()
                    X_cust_scaled = custom_scaler.fit_transform(X_custom)

                    custom_pca = PCA(n_components=min(len(sel_features), 3))
                    X_cust_proj = custom_pca.fit_transform(X_cust_scaled)

                    cust_proj_df = pd.DataFrame(X_cust_proj, columns=[f"PC{i+1}" for i in range(X_cust_proj.shape[1])])
                    if label_col != "None":
                        cust_proj_df[label_col] = custom_df[label_col].iloc[:len(cust_proj_df)].values

                    # Quick 2D Plot
                    fig_cust = px.scatter(
                        cust_proj_df,
                        x="PC1",
                        y="PC2",
                        color=label_col if label_col != "None" else None,
                        title=f"Custom Dataset 2D PCA (PC1: {custom_pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {custom_pca.explained_variance_ratio_[1]*100:.1f}%)",
                        template="plotly_dark",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig_cust.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
                    st.plotly_chart(fig_cust, use_container_width=True)

                    # Export options
                    st.markdown("#### 💾 Export Transformed Outputs")
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        csv_proj = cust_proj_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Projected Coordinates (Z) CSV",
                            data=csv_proj,
                            file_name="pca_projected_coordinates.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col_exp2:
                        loadings_matrix = pd.DataFrame(
                            custom_pca.components_.T,
                            index=sel_features,
                            columns=[f"PC{i+1}" for i in range(custom_pca.n_components_)]
                        )
                        csv_loadings = loadings_matrix.to_csv().encode("utf-8")
                        st.download_button(
                            label="📥 Download Feature Loadings Matrix (V) CSV",
                            data=csv_loadings,
                            file_name="pca_feature_loadings.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"Error parsing CSV: {str(e)}")
    else:
        st.info("Upload any tabular CSV dataset above to test PCA with custom coordinates and export projected data.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem;">
    <b>Project 013: Principal Component Analysis Implementation Studio</b> • Part of the 1,000 AI Projects Portfolio<br>
    Built with Python, Scikit-Learn, Streamlit, and Plotly.
</div>
""", unsafe_allow_html=True)

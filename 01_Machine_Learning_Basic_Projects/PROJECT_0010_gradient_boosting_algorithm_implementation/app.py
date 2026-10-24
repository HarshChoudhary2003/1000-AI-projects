"""
Project 010: Gradient Boosting Machine (GBM) Studio
===================================================
A high-performance interactive Streamlit application showcasing Gradient Tree Boosting.
Features:
- Stage-by-stage deviance & residual error convergence curves
- Early stopping detection and overfitting threshold analysis
- 2D non-linear decision boundary & continuous probability landscapes
- Mean Decrease in Impurity (MDI) feature importances
- Individual boosting stage tree structural inspector
- Real-time live inference & what-if scenario simulator
- Batch CSV scoring with download capabilities
- Mathematical theory deep-dive (negative gradient pseudo-residuals & shrinkage)

Part of the 1,000 AI Projects Implementation Library.
"""

import time
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any, Tuple
import joblib

from dataset import (
    load_real_dataset,
    generate_2d_manifold,
    get_archetype_presets
)
from gbm_engine import (
    prepare_dataset_split,
    train_gradient_boosting,
    compute_staged_metrics,
    evaluate_model,
    extract_feature_importances,
    extract_forest_diagnostics,
    inspect_stage_tree,
    compute_2d_decision_mesh,
    run_cross_validation,
    predict_single_sample
)

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gradient Boosting Studio | 1,000 AI Projects",
    page_icon="⚡",
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
    
    /* Header Gradient & Badge */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        margin-bottom: 2rem;
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10B981, #3B82F6, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        line-height: 1.5;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.4);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 0.25rem;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #10B981;
        margin-top: 0.2rem;
    }
    
    /* Callout & Info Badges */
    .callout-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 1rem;
    }
    .tag-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    .badge-gbm {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-stage {
        background: rgba(99, 102, 241, 0.2);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar: Controls, Presets & Hyperparameters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Gradient Boosting Studio")
    st.caption("Project 010 • 1,000 AI Projects Implementation Library")
    st.markdown("---")

    # 1. Dataset Selection
    st.markdown("#### 📂 1. Benchmark Dataset")
    dataset_category = st.radio(
        "Dataset Source",
        ["Real-World Tabular", "2D Geometric Manifolds"],
        horizontal=True
    )

    if dataset_category == "Real-World Tabular":
        dataset_name = st.selectbox(
            "Select Benchmark",
            [
                "Credit Default & Risk Scoring (1,000 samples, 10 features)",
                "Breast Cancer Wisconsin (569 samples, 30 features)",
                "Wine Cultivars (178 samples, 13 features, 3 classes)"
            ],
            index=0
        )
        if "Credit" in dataset_name:
            ds_key = "credit"
        elif "Breast" in dataset_name:
            ds_key = "breast_cancer"
        else:
            ds_key = "wine"
        is_2d_manifold = False
    else:
        manifold_name = st.selectbox(
            "Select 2D Manifold",
            ["Two Moons", "Concentric Circles", "Interlocking Spirals", "Overlapping Blobs", "Linearly Separable"],
            index=0
        )
        noise_level = st.slider("Manifold Noise", 0.05, 0.45, 0.18, 0.02)
        n_manifold_samples = st.slider("Sample Count", 150, 1000, 400, 50)
        ds_key = manifold_name
        is_2d_manifold = True

    st.markdown("---")

    # 2. Operational Presets
    st.markdown("#### 🎛️ 2. Architectural Presets")
    presets = get_archetype_presets()
    preset_choice = st.selectbox(
        "Load Configuration",
        ["Custom Tuning"] + list(presets.keys())
    )

    preset_defaults = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 1.0,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }

    if preset_choice != "Custom Tuning":
        cfg = presets[preset_choice]
        preset_defaults.update(cfg)
        st.info(f"💡 {cfg.get('description', '')}")

    st.markdown("---")

    # 3. Hyperparameters
    st.markdown("#### ⚙️ 3. Hyperparameters")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        n_estimators = st.slider(
            "Boosting Stages (M)",
            min_value=10,
            max_value=300,
            value=int(preset_defaults["n_estimators"]),
            step=10,
            help="Total number of sequential decision tree iterations to boost."
        )
    with col_h2:
        learning_rate = st.slider(
            "Shrinkage (η)",
            min_value=0.01,
            max_value=1.0,
            value=float(preset_defaults["learning_rate"]),
            step=0.01,
            format="%.2f",
            help="Shrinkage factor scaling the contribution of each subsequent tree."
        )

    col_h3, col_h4 = st.columns(2)
    with col_h3:
        max_depth = st.slider(
            "Max Tree Depth",
            min_value=1,
            max_value=8,
            value=int(preset_defaults["max_depth"]),
            step=1,
            help="Maximum depth of each individual regression tree. 1 = Decision Stump."
        )
    with col_h4:
        subsample = st.slider(
            "Subsample Ratio",
            min_value=0.2,
            max_value=1.0,
            value=float(preset_defaults["subsample"]),
            step=0.05,
            help="Fraction of samples randomly drawn (stochastic gradient boosting) to train each tree."
        )

    with st.expander("Advanced Tree Regularization"):
        min_split = st.slider("Min Samples Split", 2, 20, int(preset_defaults.get("min_samples_split", 2)))
        min_leaf = st.slider("Min Samples Leaf", 1, 15, int(preset_defaults.get("min_samples_leaf", 1)))
        test_split_ratio = st.slider("Test Partition Ratio", 0.15, 0.40, 0.25, 0.05)
        seed_val = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")
    retrain_clicked = st.button("🚀 Retrain Model", use_container_width=True, type="primary")


# ---------------------------------------------------------
# Data Loading & Model Execution Pipeline
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_dataset(key: str, is_2d: bool, noise: float = 0.20, n_samples: int = 400, seed: int = 42):
    if is_2d:
        return generate_2d_manifold(manifold_type=key, n_samples=n_samples, noise=noise, random_state=seed)
    return load_real_dataset(key)

if is_2d_manifold:
    df, feature_cols, target_col, target_names = get_cached_dataset(ds_key, True, noise_level, n_manifold_samples, seed_val)
else:
    df, feature_cols, target_col, target_names = get_cached_dataset(ds_key, False, seed=seed_val)

# Prepare split
X_train, X_test, y_train, y_test = prepare_dataset_split(
    df=df,
    feature_cols=feature_cols,
    target_col=target_col,
    test_size=test_split_ratio,
    random_state=seed_val
)

# Train model
with st.spinner("Training Gradient Boosting ensemble..."):
    t0 = time.time()
    model = train_gradient_boosting(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        random_state=seed_val
    )
    training_time_ms = round((time.time() - t0) * 1000, 2)

# Compute metrics, diagnostics & staged curves
staged_metrics = compute_staged_metrics(model, X_train, y_train, X_test, y_test)
eval_results = evaluate_model(model, X_test, y_test, target_names=target_names)
forest_diagnostics = extract_forest_diagnostics(model)
feature_importances = extract_feature_importances(model, feature_cols)


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <span class="tag-badge badge-gbm">PROJECT 010</span>
            <span class="tag-badge badge-stage">GRADIENT BOOSTING MACHINE</span>
            <span class="tag-badge badge-gbm">SEQUENTIAL STAGE ENSEMBLE</span>
            <h1 class="hero-title">Gradient Boosting Algorithm Studio</h1>
            <p class="hero-subtitle">
                An industrial-grade laboratory for exploring sequential residual gradient descent,
                shrinkage convergence, deviance minimization, 2D decision landscapes, and real-time inference.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Primary Top KPIs
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Test Accuracy</div>
        <div class="metric-value">{eval_results['accuracy']*100:.1f}%</div>
        <div class="metric-sub">Top Stage {staged_metrics['best_stage_accuracy']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    auc_val = f"{eval_results['roc_auc']:.3f}" if eval_results['roc_auc'] > 0 else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ROC-AUC Score</div>
        <div class="metric-value">{auc_val}</div>
        <div class="metric-sub">{'Binary' if eval_results['is_binary'] else 'Multi-class OVR'}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Min Test Deviance</div>
        <div class="metric-value">{staged_metrics['min_test_deviance']:.3f}</div>
        <div class="metric-sub">Achieved @ Stage {staged_metrics['best_stage']}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Trees Built</div>
        <div class="metric-value">{forest_diagnostics['total_trees']}</div>
        <div class="metric-sub">Avg Depth: {forest_diagnostics['avg_depth']:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fit Duration</div>
        <div class="metric-value">{training_time_ms} ms</div>
        <div class="metric-sub">Subsample: {int(subsample*100)}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Tabs Organization
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📉 Stage-by-Stage Convergence",
    "🗺️ 2D Decision Landscape",
    "🌲 Tree Diagnostics & Importances",
    "🎯 Live Real-Time Inference",
    "📊 Dataset Profiler",
    "📈 Evaluation & Cross-Validation",
    "📚 Gradient Boosting Theory"
])


# =========================================================
# TAB 1: Stage-by-Stage Deviance & Residual Convergence
# =========================================================
with tab1:
    st.subheader("📉 Deviance (Loss) Minimization Across Boosting Stages")
    st.caption("Gradient Boosting builds trees sequentially to minimize the loss function deviance. Observe how train deviance continually decreases while test deviance reaches an optimal early stopping point.")

    col_c1, col_c2 = st.columns([3, 1])

    with col_c1:
        # Plotly Deviance Curve
        fig_loss = go.Figure()

        fig_loss.add_trace(go.Scatter(
            x=staged_metrics["iterations"],
            y=staged_metrics["train_deviance"],
            mode="lines",
            name="Training Deviance (Loss)",
            line=dict(color="#10B981", width=2.5),
            hovertemplate="Stage %{x}<br>Train Loss: %{y:.4f}<extra></extra>"
        ))

        fig_loss.add_trace(go.Scatter(
            x=staged_metrics["iterations"],
            y=staged_metrics["test_deviance"],
            mode="lines",
            name="Test Deviance (Validation)",
            line=dict(color="#6366F1", width=2.5),
            hovertemplate="Stage %{x}<br>Test Loss: %{y:.4f}<extra></extra>"
        ))

        # Highlight optimal stopping stage
        best_st = staged_metrics["best_stage"]
        best_loss = staged_metrics["min_test_deviance"]

        fig_loss.add_trace(go.Scatter(
            x=[best_st],
            y=[best_loss],
            mode="markers",
            name="Optimal Stage (Early Stop)",
            marker=dict(color="#EF4444", size=12, symbol="star"),
            hovertemplate=f"Optimal Stage: {best_st}<br>Min Loss: {best_loss:.4f}<extra></extra>"
        ))

        fig_loss.add_vline(
            x=best_st,
            line_dash="dash",
            line_color="rgba(239, 68, 68, 0.6)",
            annotation_text=f"Optimal: Tree #{best_st}",
            annotation_position="top right"
        )

        fig_loss.update_layout(
            title="Stage-by-Stage Deviance (Log-Loss) Trajectory",
            xaxis_title="Boosting Iteration (Tree Stage m)",
            yaxis_title="Deviance / Negative Log-Likelihood",
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_c2:
        st.markdown(f"""
        <div class="callout-box">
            <h4 style="margin:0 0 0.5rem 0; color:#10B981;">💡 Convergence Insights</h4>
            <p style="font-size: 0.88rem; color:#CBD5E1; margin-bottom: 0.5rem;">
                <b>Optimal Stage:</b> Tree #{best_st} of {n_estimators}
            </p>
            <p style="font-size: 0.88rem; color:#CBD5E1; margin-bottom: 0.5rem;">
                <b>Min Test Loss:</b> {best_loss:.4f}
            </p>
            <p style="font-size: 0.88rem; color:#CBD5E1; margin-bottom: 0.5rem;">
                <b>Final Stage Loss:</b> {staged_metrics['test_deviance'][-1]:.4f}
            </p>
            <p style="font-size: 0.82rem; color:#94A3B8; margin-top: 0.75rem;">
                {'⚠️ <b>Overfitting Warning:</b> Test loss begins climbing past stage ' + str(best_st) + '. Consider reducing n_estimators or increasing shrinkage (smaller learning rate).' if best_st < n_estimators * 0.7 else '✅ <b>Healthy Convergence:</b> Model maintains stable generalization throughout boosting stages.'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Optimal Early Stopping Stage",
            f"Tree #{best_st}",
            delta=f"{n_estimators - best_st} trees ahead" if best_st < n_estimators else "At final stage"
        )
        st.metric(
            "Deviance Reduction Ratio",
            f"{((staged_metrics['train_deviance'][0] - staged_metrics['train_deviance'][-1]) / staged_metrics['train_deviance'][0] * 100):.1f}%",
            delta="Loss eliminated"
        )

    # Staged Accuracy Chart
    st.markdown("#### Cumulative Test Accuracy Progression")
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=staged_metrics["iterations"],
        y=[v * 100 for v in staged_metrics["test_accuracy"]],
        mode="lines",
        name="Test Accuracy (%)",
        line=dict(color="#38BDF8", width=2.5)
    ))
    fig_acc.update_layout(
        title="Accuracy (%) as successive trees contribute to ensemble prediction",
        xaxis_title="Boosting Stage (Trees)",
        yaxis_title="Test Accuracy (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_acc, use_container_width=True)


# =========================================================
# TAB 2: 2D Decision Boundary & Probability Landscape
# =========================================================
with tab2:
    st.subheader("🗺️ 2D Continuous Decision Surface & Probability Mesh")
    st.caption("Inspect how the additive ensemble of decision trees combines orthogonal axis-aligned splits into smooth, non-linear probability surfaces.")

    if len(feature_cols) >= 2:
        col_m1, col_m2 = st.columns([1, 3])

        with col_m1:
            st.markdown("##### Projection Coordinates")
            if is_2d_manifold or len(feature_cols) == 2:
                feat_x = feature_cols[0]
                feat_y = feature_cols[1]
                st.info(f"Visualizing direct 2D features: **{feat_x}** and **{feat_y}**.")
            else:
                st.caption("Select two features to project onto the 2D decision plane:")
                feat_x = st.selectbox("X-Axis Feature", feature_cols, index=0)
                feat_y = st.selectbox("Y-Axis Feature", feature_cols, index=1 if len(feature_cols) > 1 else 0)

            mesh_res = st.slider("Mesh Resolution", 60, 180, 100, 20)
            show_test_points = st.checkbox("Overlay Testing Points", value=True)
            show_errors = st.checkbox("Highlight Misclassifications", value=True)

        with col_m2:
            # Prepare 2D subset model if dataset has >2 features
            if len(feature_cols) == 2:
                X_mesh_train = X_train
                X_mesh_test = X_test
                mesh_model = model
            else:
                idx_x = feature_cols.index(feat_x)
                idx_y = feature_cols.index(feat_y)
                X_mesh_train = X_train[:, [idx_x, idx_y]]
                X_mesh_test = X_test[:, [idx_x, idx_y]]
                mesh_model = train_gradient_boosting(
                    X_train=X_mesh_train,
                    y_train=y_train,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=seed_val
                )

            xx, yy, preds_mesh, prob_surface = compute_2d_decision_mesh(
                mesh_model,
                np.vstack([X_mesh_train, X_mesh_test]),
                resolution=mesh_res
            )

            fig_surface = go.Figure()

            # Continuous Probability Heatmap
            fig_surface.add_trace(go.Contour(
                x=np.linspace(xx.min(), xx.max(), mesh_res),
                y=np.linspace(yy.min(), yy.max(), mesh_res),
                z=prob_surface,
                colorscale="Viridis",
                opacity=0.75,
                contours=dict(coloring="heatmap", showlabels=True),
                colorbar=dict(title="Confidence / Prob"),
                name="Decision Surface"
            ))

            # Add Training Scatter Points
            for cls_idx in np.unique(y_train):
                cls_mask = y_train == cls_idx
                cls_name = target_names[cls_idx] if cls_idx < len(target_names) else f"Class {cls_idx}"
                fig_surface.add_trace(go.Scatter(
                    x=X_mesh_train[cls_mask, 0],
                    y=X_mesh_train[cls_mask, 1],
                    mode="markers",
                    name=f"Train: {cls_name}",
                    marker=dict(size=7, line=dict(width=1, color="black")),
                    hovertemplate=f"Train {cls_name}<br>{feat_x}: %{{x:.2f}}<br>{feat_y}: %{{y:.2f}}<extra></extra>"
                ))

            # Add Testing Points
            if show_test_points:
                test_preds = mesh_model.predict(X_mesh_test)
                for cls_idx in np.unique(y_test):
                    cls_mask = y_test == cls_idx
                    cls_name = target_names[cls_idx] if cls_idx < len(target_names) else f"Class {cls_idx}"
                    fig_surface.add_trace(go.Scatter(
                        x=X_mesh_test[cls_mask, 0],
                        y=X_mesh_test[cls_mask, 1],
                        mode="markers",
                        name=f"Test: {cls_name}",
                        marker=dict(size=9, symbol="diamond", line=dict(width=1.5, color="white")),
                        hovertemplate=f"Test {cls_name}<br>{feat_x}: %{{x:.2f}}<br>{feat_y}: %{{y:.2f}}<extra></extra>"
                    ))

                # Highlight Errors
                if show_errors:
                    err_mask = (test_preds != y_test)
                    if np.any(err_mask):
                        fig_surface.add_trace(go.Scatter(
                            x=X_mesh_test[err_mask, 0],
                            y=X_mesh_test[err_mask, 1],
                            mode="markers",
                            name="Misclassified (Test)",
                            marker=dict(size=14, symbol="x", color="#EF4444", line=dict(width=2)),
                            hovertemplate=f"Error<br>{feat_x}: %{{x:.2f}}<br>{feat_y}: %{{y:.2f}}<extra></extra>"
                        ))

            fig_surface.update_layout(
                title=f"Gradient Boosting Probability Boundary: {feat_x} vs {feat_y}",
                xaxis_title=feat_x,
                yaxis_title=feat_y,
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                height=560,
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_surface, use_container_width=True)
    else:
        st.warning("2D Decision Mesh requires at least 2 features.")


# =========================================================
# TAB 3: Tree Diagnostics & Feature Importances
# =========================================================
with tab3:
    st.subheader("🌲 Tree Architecture Diagnostics & Feature Importances")
    st.caption("Inspect the structural properties of the boosted ensemble and determine which features contribute most to reducing impurity across all stages.")

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.markdown("#### Mean Decrease in Impurity (MDI)")
        fig_imp = px.bar(
            feature_importances.head(15),
            x="Percentage",
            y="Feature",
            orientation="h",
            color="Percentage",
            color_continuous_scale="Viridis",
            labels={"Percentage": "Relative Importance (%)", "Feature": "Feature Name"},
            title="Feature Importance Ranking (Top 15)"
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=450
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_t2:
        st.markdown("#### Ensemble Architectural Summary")
        diag_data = {
            "Metric": [
                "Total Boosting Iterations (M)",
                "Configured Max Depth",
                "Observed Average Tree Depth",
                "Average Leaf Nodes per Tree",
                "Total Decision Nodes in Ensemble",
                "Configured Subsampling Ratio",
                "Learning Rate (Shrinkage η)"
            ],
            "Value": [
                f"{forest_diagnostics['n_estimators']}",
                f"{forest_diagnostics['max_depth_configured']}",
                f"{forest_diagnostics['avg_depth']:.2f}",
                f"{forest_diagnostics['avg_leaves']:.1f}",
                f"{forest_diagnostics['total_nodes']}",
                f"{forest_diagnostics['subsample'] * 100:.0f}%",
                f"{forest_diagnostics['learning_rate']:.3f}"
            ]
        }
        st.table(pd.DataFrame(diag_data))

    st.markdown("---")

    # Individual Stage Inspector
    st.markdown("#### 🔍 Individual Boosting Stage Tree Inspector")
    st.caption("Examine the specific regression tree trained to predict the negative gradient residuals at stage k.")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        selected_stage = st.slider(
            "Select Tree Stage to Inspect",
            min_value=1,
            max_value=n_estimators,
            value=min(1, n_estimators),
            step=1
        )
        stage_info = inspect_stage_tree(model, selected_stage - 1, feature_cols)

        st.markdown(f"""
        <div class="callout-box">
            <b>Tree #{stage_info['stage']} Profile</b><br>
            • Depth: {stage_info['max_depth']}<br>
            • Total Nodes: {stage_info['node_count']}<br>
            • Terminal Leaves: {stage_info['n_leaves']}<br>
            • Leaf Value Range: [{stage_info['leaf_values_min']:.3f}, {stage_info['leaf_values_max']:.3f}]
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        splits = stage_info["feature_split_counts"]
        if splits:
            df_splits = pd.DataFrame(list(splits.items()), columns=["Feature", "Splits Made"]).sort_values(by="Splits Made", ascending=False)
            fig_splits = px.bar(
                df_splits,
                x="Splits Made",
                y="Feature",
                orientation="h",
                color="Splits Made",
                color_continuous_scale="Teal",
                title=f"Features Split by Tree #{stage_info['stage']}"
            )
            fig_splits.update_layout(
                yaxis=dict(autorange="reversed"),
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                height=260
            )
            st.plotly_chart(fig_splits, use_container_width=True)
        else:
            st.info("No interior splits in this tree (Decision Stump).")


# =========================================================
# TAB 4: Live Real-Time Inference Sandbox
# =========================================================
with tab4:
    st.subheader("🎯 Live Inference & What-If Scenario Sandbox")
    st.caption("Adjust input parameters dynamically to trigger instantaneous model inference, probability scoring, and confidence diagnostics.")

    inf_col1, inf_col2 = st.columns([1, 1])

    with inf_col1:
        st.markdown("##### 🧪 Dynamic Feature Input Bench")
        input_vector = []
        for i, col in enumerate(feature_cols):
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            step_size = (col_max - col_min) / 100.0 if col_max > col_min else 0.01

            val = st.slider(
                f"{col}",
                min_value=round(col_min, 3),
                max_value=round(col_max, 3),
                value=round(col_mean, 3),
                step=round(step_size, 3),
                key=f"inf_{col}_{i}"
            )
            input_vector.append(val)

    with inf_col2:
        st.markdown("##### 🔮 Prediction Telemetry")
        prediction_result = predict_single_sample(
            model=model,
            feature_vector=input_vector,
            feature_names=feature_cols,
            target_names=target_names
        )

        pred_class = prediction_result["predicted_label"]
        confidence = prediction_result["confidence_percentage"]

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 1.5rem; border-color: rgba(16, 185, 129, 0.5);">
            <div class="metric-label">Predicted Decision</div>
            <div class="metric-value" style="color: #34D399;">{pred_class}</div>
            <div class="metric-sub">Confidence: <b>{confidence}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Plot Probability Distribution
        df_probs = pd.DataFrame(
            list(prediction_result["probability_distribution"].items()),
            columns=["Class", "Probability"]
        )

        fig_prob = px.bar(
            df_probs,
            x="Probability",
            y="Class",
            orientation="h",
            color="Probability",
            color_continuous_scale="Blues",
            range_x=[0, 1],
            title="Class Probability Distribution"
        )
        fig_prob.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=280
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    # Batch CSV Inference
    st.markdown("#### 📥 Batch Inference Scoring")
    st.caption("Upload a CSV with the matching feature columns to perform bulk batch inference and download tagged predictions.")

    uploaded_file = st.file_uploader("Upload CSV for scoring", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing = [c for c in feature_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns in CSV: {missing}")
            else:
                batch_X = batch_df[feature_cols].to_numpy()
                batch_preds = model.predict(batch_X)
                batch_probas = model.predict_proba(batch_X)

                batch_df["Predicted_Class_Idx"] = batch_preds
                batch_df["Predicted_Class"] = [
                    target_names[i] if i < len(target_names) else f"Class {i}" for i in batch_preds
                ]
                batch_df["Confidence"] = np.max(batch_probas, axis=1)

                st.success(f"Successfully processed {len(batch_df)} records.")
                st.dataframe(batch_df.head(10))

                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "💾 Download Annotated Predictions CSV",
                    csv_buffer.getvalue(),
                    "batch_predictions.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error(f"Failed to process CSV: {str(e)}")


# =========================================================
# TAB 5: Dataset Profiler
# =========================================================
with tab5:
    st.subheader("📊 Dataset Profiler & Statistical Overview")
    st.caption("Inspect raw distributions, feature summaries, correlation matrices, and class balance.")

    col_dp1, col_dp2 = st.columns([1, 2])

    with col_dp1:
        st.markdown("##### Dataset Metadata")
        st.markdown(f"""
        - **Total Records:** {len(df):,}
        - **Feature Dimensions:** {len(feature_cols)}
        - **Target Classes:** {len(target_names)} ({', '.join(target_names)})
        - **Training Partition:** {len(X_train)} samples ({int((1 - test_split_ratio)*100)}%)
        - **Testing Partition:** {len(X_test)} samples ({int(test_split_ratio*100)}%)
        """)

        # Class balance pie chart
        counts = df["target_label"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        fig_pie = px.pie(
            counts,
            values="Count",
            names="Class",
            title="Target Class Balance",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_dp2:
        st.markdown("##### Data Preview (First 10 Records)")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("##### Summary Statistics")
        st.dataframe(df[feature_cols].describe().round(3), use_container_width=True)

    # Correlation Matrix
    if len(feature_cols) > 2:
        st.markdown("##### Feature Correlation Heatmap")
        corr = df[feature_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Pearson Correlation Matrix"
        )
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            height=500
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# =========================================================
# TAB 6: Evaluation & Cross-Validation
# =========================================================
with tab6:
    st.subheader("📈 Model Evaluation & Cross-Validation")
    st.caption("Validate model reliability across test partitions and stratified cross-validation folds.")

    col_ev1, col_ev2 = st.columns(2)

    with col_ev1:
        st.markdown("#### Confusion Matrix")
        cm = np.array(eval_results["confusion_matrix"])
        labels = eval_results["class_labels"]

        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            labels=dict(x="Predicted Class", y="Actual True Class", color="Count"),
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=380
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_ev2:
        st.markdown("#### Classification Metrics Breakdown")
        report = eval_results["classification_report"]
        rows = []
        for key, vals in report.items():
            if isinstance(vals, dict):
                rows.append({
                    "Class": key,
                    "Precision": f"{vals.get('precision', 0):.3f}",
                    "Recall": f"{vals.get('recall', 0):.3f}",
                    "F1-Score": f"{vals.get('f1-score', 0):.3f}",
                    "Support": int(vals.get('support', 0))
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")

    # K-Fold Cross Validation Bench
    st.markdown("#### 🔁 Stratified K-Fold Cross-Validation")
    cv_folds = st.slider("Select Cross-Validation Folds", 3, 10, 5, 1)

    if st.button("Run K-Fold Cross-Validation Bench"):
        with st.spinner(f"Executing {cv_folds}-Fold Stratified Cross-Validation..."):
            cv_summary = run_cross_validation(
                X=df[feature_cols].to_numpy(),
                y=df[target_col].to_numpy(),
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                cv_folds=cv_folds,
                random_state=seed_val
            )

        col_cv1, col_cv2 = st.columns([1, 2])
        with col_cv1:
            st.metric("CV Mean Accuracy", f"{cv_summary['accuracy_mean']*100:.2f}%", f"± {cv_summary['accuracy_std']*100:.2f}%")
            st.metric("CV Mean F1-Score", f"{cv_summary['f1_mean']*100:.2f}%", f"± {cv_summary['f1_std']*100:.2f}%")
            if "roc_auc_mean" in cv_summary:
                st.metric("CV Mean ROC-AUC", f"{cv_summary['roc_auc_mean']:.3f}", f"± {cv_summary['roc_auc_std']:.3f}")

        with col_cv2:
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(cv_folds)],
                y=[acc * 100 for acc in cv_summary["fold_accuracies"]],
                marker_color="#10B981"
            ))
            fig_cv.add_hline(
                y=cv_summary["accuracy_mean"] * 100,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"Mean: {cv_summary['accuracy_mean']*100:.2f}%"
            )
            fig_cv.update_layout(
                title=f"Per-Fold Accuracy Across {cv_folds} Folds",
                xaxis_title="Validation Fold",
                yaxis_title="Accuracy (%)",
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                height=300
            )
            st.plotly_chart(fig_cv, use_container_width=True)


# =========================================================
# TAB 7: Gradient Boosting Theory & Mathematics
# =========================================================
with tab7:
    st.subheader("📚 Gradient Tree Boosting Mathematical Derivation")
    st.markdown("""
    Gradient Boosting is an additive ensemble technique originally pioneered by Jerome H. Friedman (1999).
    Unlike bagging (Random Forest), which builds independent trees in parallel to reduce variance,
    Gradient Boosting builds shallow trees sequentially to greedily minimize a differentiable loss function.
    """)

    st.markdown("### 1. The Additive Model Formulation")
    st.latex(r"""
    F_M(x) = F_0(x) + \sum_{m=1}^M \eta \cdot h_m(x)
    """)
    st.markdown("""
    Where:
    - $F_0(x)$ is the base initial prediction (typically the log-odds of the positive class for classification).
    - $h_m(x)$ is the weak learner (regression tree) trained at stage $m$.
    - $\eta \in (0, 1]$ is the **shrinkage parameter** (learning rate) scaling each tree's contribution to prevent overfitting.
    - $M$ is the total number of boosting stages (`n_estimators`).
    """)

    st.markdown("### 2. Pseudo-Residuals as Negative Gradients")
    st.markdown("""
    At each iteration $m$, the algorithm computes the pseudo-residuals $r_{im}$, which are the negative gradients of the loss function $L(y_i, F(x_i))$ evaluated at the current model $F_{m-1}$:
    """)
    st.latex(r"""
    r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
    """)

    st.markdown("For binary classification using Bernoulli log-loss:")
    st.latex(r"""
    L(y_i, F(x_i)) = \log(1 + e^{-2 y_i^* F(x_i)}) \quad \implies \quad r_{im} = y_i - p(x_i)
    """)
    st.markdown("""
    Where $p(x_i) = \frac{1}{1 + e^{-F_{m-1}(x_i)}}$ is the predicted probability.
    The pseudo-residual $r_{im}$ is simply the **prediction error** $(y_i - p_i)$. Each new tree $h_m(x)$ fits directly to these residual errors!
    """)

    st.markdown("### 3. Algorithm Comparison Matrix")
    comparison_table = {
        "Attribute": [
            "Ensemble Paradigm",
            "Learner Independence",
            "Target Objective",
            "Variance vs Bias Reduction",
            "Shrinkage / Learning Rate",
            "Out-of-Bag Evaluation",
            "Sensitivity to Outliers"
        ],
        "Random Forest (Bagging)": [
            "Parallel Bagging + Subsampling",
            "Independent deep decision trees",
            "Original target values $y$",
            "Primarily reduces Variance",
            "None (equal vote weighting)",
            "Native OOB estimation",
            "Robust against noisy labels"
        ],
        "Gradient Boosting (GBM)": [
            "Sequential Additive Modeling",
            "Dependent shallow decision trees",
            "Negative gradient pseudo-residuals",
            "Primarily reduces Bias (and Variance)",
            "Shrinkage parameter $\eta$",
            "Requires validation / staged loss",
            "Higher sensitivity to label noise"
        ]
    }
    st.table(pd.DataFrame(comparison_table))


# ---------------------------------------------------------
# Footer & Artifact Export
# ---------------------------------------------------------
st.markdown("---")
foot_col1, foot_col2 = st.columns([3, 1])

with foot_col1:
    st.caption("Part of the 1,000 AI Projects Implementation Library • Gradient Boosting Algorithm Implementation")

with foot_col2:
    # Save model artifact in-memory for download
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    st.download_button(
        label="💾 Export Trained Model (.joblib)",
        data=model_buffer.getvalue(),
        file_name="gbm_model.joblib",
        mime="application/octet-stream",
        use_container_width=True
    )

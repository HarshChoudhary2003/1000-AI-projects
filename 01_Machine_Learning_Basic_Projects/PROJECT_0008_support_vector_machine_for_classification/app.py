"""
Project 008: Support Vector Machine (SVM) Classification Studio
================================================================
Flagship interactive Streamlit application featuring real-world biomedical
diagnostics (Wisconsin Breast Cancer), multi-kernel benchmarking, continuous
2D margin & support vector geometry, grid search hyperparameter exploration,
and real-time diagnostic inference.

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve

from dataset import (
    load_real_dataset,
    generate_2d_manifold,
    get_clinical_patient_presets
)
from svm_engine import (
    prepare_dataset_split,
    train_svm_model,
    extract_support_vector_diagnostics,
    evaluate_svm_model,
    run_cross_validation,
    compare_kernel_suite,
    run_grid_search_tuning,
    compute_2d_decision_mesh,
    predict_single_instance
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SVM Classification Studio | 1,000 AI Projects",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Theme Styling
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.90) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Badge & Alert elements */
    .badge-primary {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #818cf8;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-success {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-warning {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fbbf24;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-danger {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #f87171;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Highlight Banner */
    .banner-container {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.15) 0%, rgba(147, 51, 234, 0.1) 100%);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding-bottom: 12px;">
    <div style="font-size: 2.2rem; margin-bottom: 4px;">⚡</div>
    <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700;">SVM Studio</h2>
    <span class="badge-primary" style="margin-top: 4px;">PROJECT 008 • CLASSIFICATION</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 1. Dataset Configuration")

dataset_choice = st.sidebar.selectbox(
    "Benchmark Dataset",
    [
        "Wisconsin Breast Cancer (Diagnostic)",
        "Wine Cultivars (Multi-Class)",
        "Iris Botanical Benchmark",
        "Two Moons (Synthetic 2D)",
        "Concentric Circles (Synthetic 2D)",
        "Linearly Separable (Synthetic 2D)",
        "Overlapping Blobs (Synthetic 2D)",
        "Upload Custom CSV"
    ],
    index=0
)

# Custom CSV Handler
custom_df = None
if dataset_choice == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded: {custom_df.shape[0]} rows, {custom_df.shape[1]} cols")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 2. Hyperparameter Controls")

kernel_type = st.sidebar.selectbox(
    "Kernel Function",
    ["rbf", "linear", "poly", "sigmoid"],
    format_func=lambda x: {
        "rbf": "RBF (Radial Basis / Gaussian)",
        "linear": "Linear (Hyperplane)",
        "poly": "Polynomial (Degree d)",
        "sigmoid": "Sigmoid (Hyperbolic Tangent)"
    }[x],
    index=0
)

# C Regularization Parameter
c_val = st.sidebar.slider(
    "Regularization (C)",
    min_value=0.01,
    max_value=50.0,
    value=1.0,
    step=0.1,
    help="Inverse regularization strength. Smaller C encourages larger margin with more margin violations (slack); larger C strictly penalizes misclassifications."
)

# Gamma (Kernel Coefficient)
gamma_mode = st.sidebar.radio("Gamma Setting", ["scale", "auto", "manual"], horizontal=True)
if gamma_mode == "manual":
    gamma_val = st.sidebar.slider(
        "Gamma Value",
        min_value=0.001,
        max_value=2.0,
        value=0.1,
        step=0.005,
        help="Kernel coefficient for RBF, Poly, and Sigmoid. Defines individual training sample influence radius."
    )
else:
    gamma_val = gamma_mode

# Poly Degree
poly_degree = 3
coef0_val = 0.0
if kernel_type == "poly":
    poly_degree = st.sidebar.slider("Polynomial Degree (d)", min_value=2, max_value=6, value=3)
    coef0_val = st.sidebar.slider("Independent Term (coef0)", min_value=0.0, max_value=5.0, value=1.0)
elif kernel_type == "sigmoid":
    coef0_val = st.sidebar.slider("Independent Term (coef0)", min_value=-2.0, max_value=2.0, value=0.0)

# Class Weighting & Scaling
class_weight_opt = st.sidebar.selectbox("Class Weighting", ["None", "balanced"], index=0)
class_weight = "balanced" if class_weight_opt == "balanced" else None

scale_method = st.sidebar.selectbox(
    "Feature Scaling",
    ["standard", "minmax", "none"],
    format_func=lambda x: {
        "standard": "StandardScaler (Zero mean, unit variance)",
        "minmax": "MinMaxScaler (Bounded [0, 1])",
        "none": "No Scaling (Raw values)"
    }[x],
    index=0
)

test_ratio = st.sidebar.slider("Holdout Test Fraction", min_value=0.10, max_value=0.40, value=0.25, step=0.05)


# ---------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------
@st.cache_data
def get_dataset(choice: str, uploaded_content: Optional[bytes] = None):
    if choice == "Wisconsin Breast Cancer (Diagnostic)":
        return load_real_dataset("breast_cancer")
    elif choice == "Wine Cultivars (Multi-Class)":
        return load_real_dataset("wine")
    elif choice == "Iris Botanical Benchmark":
        return load_real_dataset("iris")
    elif choice == "Two Moons (Synthetic 2D)":
        return generate_2d_manifold("Two Moons")
    elif choice == "Concentric Circles (Synthetic 2D)":
        return generate_2d_manifold("Concentric Circles")
    elif choice == "Linearly Separable (Synthetic 2D)":
        return generate_2d_manifold("Linearly Separable")
    elif choice == "Overlapping Blobs (Synthetic 2D)":
        return generate_2d_manifold("Overlapping Blobs")
    elif choice == "Upload Custom CSV" and uploaded_content is not None:
        df = pd.read_csv(io.BytesIO(uploaded_content))
        # Autodetect numeric columns and target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return None
        target_col = numeric_cols[-1]
        feature_cols = [c for c in numeric_cols if c != target_col]
        # Target classes
        unique_targets = df[target_col].unique()
        target_names = [f"Class {t}" for t in unique_targets]
        df["target_label"] = df[target_col].astype(str)
        return df, feature_cols, target_col, target_names
    return load_real_dataset("breast_cancer")

uploaded_bytes = uploaded_file.getvalue() if custom_df is not None else None
dataset_pack = get_dataset(dataset_choice, uploaded_bytes)

if dataset_pack is None:
    st.error("Please upload a valid CSV containing at least 1 feature column and 1 numeric target column.")
    st.stop()

df_active, feature_names, target_col, target_names = dataset_pack

# Prepare Train/Test split with scaling
data_bundle = prepare_dataset_split(
    df=df_active,
    feature_cols=feature_names,
    target_col=target_col,
    test_size=test_ratio,
    scale_method=scale_method,
    random_state=42
)
X_train, X_test = data_bundle["X_train"], data_bundle["X_test"]
y_train, y_test = data_bundle["y_train"], data_bundle["y_test"]
scaler = data_bundle["scaler"]

# Train Active Model
model_active, train_latency_ms = train_svm_model(
    X_train=X_train,
    y_train=y_train,
    kernel=kernel_type,
    C=c_val,
    gamma=gamma_val,
    degree=poly_degree,
    coef0=coef0_val,
    class_weight=class_weight,
    probability=True,
    random_state=42
)

# Diagnostics & Evaluation
diag_active = extract_support_vector_diagnostics(model_active, X_train, y_train)
eval_active = evaluate_svm_model(model_active, X_test, y_test, target_names=target_names)


# ---------------------------------------------------------
# Main Page Header & Banner
# ---------------------------------------------------------
st.markdown("""
<div class="banner-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;">
                Support Vector Machine Studio
            </h1>
            <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 0.95rem;">
                Max-Margin Hyperplane Geometry, Multi-Kernel Optimization & Diagnostic Inference
            </p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="badge-primary">Kernel: {}</span>
            <span class="badge-success">C = {}</span>
            <span class="badge-warning">Scaling: {}</span>
        </div>
    </div>
</div>
""".format(kernel_type.upper(), c_val, scale_method.capitalize()), unsafe_allow_html=True)


# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab_benchmark, tab_decision, tab_tournament, tab_clinical, tab_theory = st.tabs([
    "📊 Executive Studio & Benchmark",
    "🔮 2D Decision Space & Margin Explorer",
    "⚔️ Kernel Battle & Grid Search",
    "🩺 Clinical Diagnostic Inference",
    "📚 SVM Theory & Architecture"
])


# =========================================================
# TAB 1: Executive Studio & Benchmark
# =========================================================
with tab_benchmark:
    st.subheader("Model Performance Telemetry & Champion Metrics")

    # High-level metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">{eval_active['accuracy']:.1%}</div>
            <div class="metric-sub">Holdout test set</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">F1-Score (Wt)</div>
            <div class="metric-value">{eval_active['f1']:.3f}</div>
            <div class="metric-sub">Harmonic precision-recall</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        roc_disp = f"{eval_active['roc_auc']:.3f}" if "roc_auc" in eval_active else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ROC-AUC</div>
            <div class="metric-value">{roc_disp}</div>
            <div class="metric-sub">Discriminative ranking</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Support Vectors</div>
            <div class="metric-value">{diag_active['n_support_vectors']}</div>
            <div class="metric-sub">{diag_active['support_vector_ratio_pct']:.1f}% of train points</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        viol_pct = diag_active.get('margin_violations_pct', 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Margin Violations</div>
            <div class="metric-value">{viol_pct:.1f}%</div>
            <div class="metric-sub">Slack xi &gt; 0 instances</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Train Latency</div>
            <div class="metric-value">{train_latency_ms:.1f}<span style="font-size: 1rem;">ms</span></div>
            <div class="metric-sub">Fit execution speed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column layout for visual telemetry
    col_chart_left, col_chart_right = st.columns([1, 1])

    with col_chart_left:
        st.markdown("#### 🎯 Confusion Matrix & Margin Counts")
        cm = eval_active["confusion_matrix"]
        
        # Interactive Plotly Heatmap for Confusion Matrix
        cm_labels_y = [f"Actual {n}" for n in target_names]
        cm_labels_x = [f"Pred {n}" for n in target_names]
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Ground Truth Class", color="Sample Count"),
            x=cm_labels_x,
            y=cm_labels_y,
            color_continuous_scale="Purples",
            text_auto=True,
            aspect="auto"
        )
        fig_cm.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        if eval_active.get("is_binary", False) and "specificity" in eval_active:
            sub_c1, sub_c2, sub_c3 = st.columns(3)
            sub_c1.metric("Specificity (TNR)", f"{eval_active['specificity']:.1%}")
            sub_c2.metric("Sensitivity / Recall", f"{eval_active['recall']:.1%}")
            sub_c3.metric("Matthews Corr (MCC)", f"{eval_active['mcc']:.3f}")

    with col_chart_right:
        if eval_active.get("is_binary", False) and "y_proba" in eval_active:
            st.markdown("#### 📈 Discrimination Curves (ROC & PR)")
            y_proba = eval_active["y_proba"][:, 1]
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
            prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)

            fig_roc = go.Figure()
            # ROC Curve
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {eval_active['roc_auc']:.3f})",
                line=dict(color="#6366f1", width=3)
            ))
            # Chance line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random Guess",
                line=dict(color="#64748b", dash="dash")
            ))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC)",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Recall)",
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
                legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.markdown("#### 📊 Per-Class Precision & Recall Breakdown")
            rep_df = pd.DataFrame(eval_active["classification_report"]).transpose()
            st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

    # Cross-validation exploration
    st.markdown("---")
    st.markdown("#### 🔄 Stratified K-Fold Generalization Stability")
    cv_k = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5)
    if st.button("Run Cross-Validation Benchmark", key="run_cv_btn"):
        with st.spinner(f"Evaluating {cv_k}-fold cross validation..."):
            cv_bundle = run_cross_validation(model_active, X_train, y_train, cv_folds=cv_k)
            
            col_cv_1, col_cv_2 = st.columns([1, 2])
            with col_cv_1:
                st.metric(
                    "Mean CV Accuracy",
                    f"{cv_bundle['mean_accuracy']:.2%}",
                    f"± {cv_bundle['std_accuracy']:.2%}"
                )
                st.write(f"**Min Fold Score:** {cv_bundle['min_accuracy']:.2%}")
                st.write(f"**Max Fold Score:** {cv_bundle['max_accuracy']:.2%}")

            with col_cv_2:
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Bar(
                    x=[f"Fold {i+1}" for i in range(cv_k)],
                    y=cv_bundle["fold_scores"],
                    marker=dict(
                        color=cv_bundle["fold_scores"],
                        colorscale="Viridis",
                        showscale=False
                    ),
                    text=[f"{s:.1%}" for s in cv_bundle["fold_scores"]],
                    textposition="auto"
                ))
                fig_cv.add_hline(
                    y=cv_bundle["mean_accuracy"],
                    line_dash="dash",
                    line_color="#ec4899",
                    annotation_text=f"Mean ({cv_bundle['mean_accuracy']:.1%})"
                )
                fig_cv.update_layout(
                    title=f"{cv_k}-Fold Stratified Accuracy Distribution",
                    yaxis=dict(range=[0.75, 1.02]),
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="rgba(15, 23, 42, 0.5)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig_cv, use_container_width=True)

    # Raw Dataset Explorer
    with st.expander("🔍 Explore Raw Dataset & Correlation Insights"):
        st.dataframe(df_active.head(10), use_container_width=True)
        st.write(f"**Total Samples:** {df_active.shape[0]} | **Feature Dimension:** {len(feature_names)}")


# =========================================================
# TAB 2: 2D Decision Space & Margin Explorer
# =========================================================
with tab_decision:
    st.markdown("### 🔮 Interactive 2D Decision Space & Continuous Margin Contour")
    st.markdown("""
    Explore the geometric beauty of Support Vector Machines. 
    Notice the **solid separating boundary** ($f(x) = 0$), the **positive and negative margin gutters** ($f(x) = \pm 1$), 
    and the **highlighted support vectors** (instances that define the maximum-margin hyperplane).
    """)

    col_ctrl_1, col_ctrl_2, col_ctrl_3 = st.columns(3)
    
    with col_ctrl_1:
        if len(feature_names) >= 2:
            feat_x = st.selectbox("Horizontal Feature (X1)", feature_names, index=0)
            feat_y = st.selectbox("Vertical Feature (X2)", feature_names, index=1 if len(feature_names) > 1 else 0)
        else:
            feat_x, feat_y = feature_names[0], feature_names[0]

    with col_ctrl_2:
        vis_c = st.slider("Dynamic C (Boundary Stiffness)", 0.01, 30.0, float(c_val), 0.1, key="vis_c")
        vis_kernel = st.selectbox("Dynamic Kernel", ["rbf", "linear", "poly", "sigmoid"], index=["rbf", "linear", "poly", "sigmoid"].index(kernel_type), key="vis_kernel")

    with col_ctrl_3:
        vis_gamma = gamma_val
        if vis_kernel in ["rbf", "poly", "sigmoid"]:
            vis_gamma_mode = st.radio("Dynamic Gamma", ["scale", "0.05", "0.2", "1.0", "5.0"], horizontal=True, key="vis_gamma_mode")
            vis_gamma = vis_gamma_mode if vis_gamma_mode == "scale" else float(vis_gamma_mode)

    # Train a 2D SVM dedicated to the chosen 2 features
    X_2d_raw = df_active[[feat_x, feat_y]].values
    y_2d = df_active[target_col].values

    # Scale 2D coordinates for numerical stability
    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d_raw)

    model_2d, _ = train_svm_model(
        X_train=X_2d_scaled,
        y_train=y_2d,
        kernel=vis_kernel,
        C=vis_c,
        gamma=vis_gamma,
        degree=poly_degree,
        coef0=coef0_val,
        probability=True,
        random_state=42
    )
    diag_2d = extract_support_vector_diagnostics(model_2d, X_2d_scaled, y_2d)

    # Meshgrid computation
    mesh_res = compute_2d_decision_mesh(model_2d, X_2d_scaled, resolution=180, margin_padding=0.8)
    xx, yy = mesh_res["xx"], mesh_res["yy"]
    Z_dec = mesh_res["Z_dec"]
    Z_label = mesh_res["Z_label"]

    fig_contour = go.Figure()

    # Continuous decision background
    if Z_dec is not None:
        fig_contour.add_trace(go.Contour(
            x=np.linspace(mesh_res["x_range"][0], mesh_res["x_range"][1], 180),
            y=np.linspace(mesh_res["y_range"][0], mesh_res["y_range"][1], 180),
            z=Z_dec,
            colorscale="RdBu",
            opacity=0.45,
            showscale=True,
            colorbar=dict(title="f(x) Distance"),
            contours=dict(
                coloring="heatmap",
                showlines=True
            ),
            name="Decision Surface"
        ))

        # Explicit Decision Boundary line f(x) = 0
        fig_contour.add_trace(go.Contour(
            x=np.linspace(mesh_res["x_range"][0], mesh_res["x_range"][1], 180),
            y=np.linspace(mesh_res["y_range"][0], mesh_res["y_range"][1], 180),
            z=Z_dec,
            showscale=False,
            contours=dict(
                start=0,
                end=0,
                coloring="none"
            ),
            line=dict(color="#ffffff", width=3),
            name="Separating Hyperplane f(x)=0"
        ))

        # Positive Margin Line f(x) = +1
        fig_contour.add_trace(go.Contour(
            x=np.linspace(mesh_res["x_range"][0], mesh_res["x_range"][1], 180),
            y=np.linspace(mesh_res["y_range"][0], mesh_res["y_range"][1], 180),
            z=Z_dec,
            showscale=False,
            contours=dict(
                start=1.0,
                end=1.0,
                coloring="none"
            ),
            line=dict(color="#10b981", width=2, dash="dash"),
            name="Positive Margin f(x)=+1"
        ))

        # Negative Margin Line f(x) = -1
        fig_contour.add_trace(go.Contour(
            x=np.linspace(mesh_res["x_range"][0], mesh_res["y_range"][1], 180),
            y=np.linspace(mesh_res["y_range"][0], mesh_res["y_range"][1], 180),
            z=Z_dec,
            showscale=False,
            contours=dict(
                start=-1.0,
                end=-1.0,
                coloring="none"
            ),
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="Negative Margin f(x)=-1"
        ))

    # Scatter data samples
    unique_y = np.unique(y_2d)
    colors = ["#3b82f6", "#f43f5e", "#10b981", "#fbbf24"]
    for i, cls_val in enumerate(unique_y):
        mask = (y_2d == cls_val)
        cls_name = target_names[int(cls_val)] if int(cls_val) < len(target_names) else f"Class {cls_val}"
        fig_contour.add_trace(go.Scatter(
            x=X_2d_scaled[mask, 0],
            y=X_2d_scaled[mask, 1],
            mode="markers",
            name=cls_name,
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                line=dict(width=1, color="#ffffff")
            )
        ))

    # Highlight Support Vectors
    sv_indices = diag_2d["support_indices"]
    fig_contour.add_trace(go.Scatter(
        x=X_2d_scaled[sv_indices, 0],
        y=X_2d_scaled[sv_indices, 1],
        mode="markers",
        name=f"Support Vectors ({len(sv_indices)})",
        marker=dict(
            size=15,
            symbol="circle-open",
            color="#fbbf24",
            line=dict(width=2.5, color="#fbbf24")
        )
    ))

    fig_contour.update_layout(
        title=f"2D Feature Projection Space: {feat_x} vs {feat_y} | Kernel: {vis_kernel.upper()} (C={vis_c})",
        xaxis_title=f"{feat_x} (Standardized)",
        yaxis_title=f"{feat_y} (Standardized)",
        height=620,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="rgba(148, 163, 184, 0.2)"
        )
    )

    st.plotly_chart(fig_contour, use_container_width=True)

    # 2D Geometry Stats Box
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("2D Support Vectors", f"{len(sv_indices)} / {len(X_2d_scaled)}")
    col_stat2.metric("Support Vector Ratio", f"{(len(sv_indices)/len(X_2d_scaled))*100:.1f}%")
    if "margin_width" in diag_2d:
        col_stat3.metric("Linear Margin Width (2/||w||)", f"{diag_2d['margin_width']:.4f}")
        col_stat4.metric("Hyperplane Norm ||w||", f"{diag_2d['w_norm']:.4f}")
    else:
        col_stat3.metric("Kernel Space", f"{vis_kernel.upper()} Transform")
        col_stat4.metric("Dual Alpha Multipliers", f"{len(sv_indices)} active")


# =========================================================
# TAB 3: Kernel Battle & Grid Search
# =========================================================
with tab_tournament:
    st.markdown("### ⚔️ Kernel Tournament & Hyperparameter Grid Search")
    
    st.markdown("#### 1. Multi-Kernel Tournament")
    st.caption("Benchmark Linear, RBF, Polynomial, and Sigmoid kernels under identical train/test splits.")
    
    tournament_df = compare_kernel_suite(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        C=c_val,
        random_state=42
    )

    # Display styled tournament table
    st.dataframe(
        tournament_df.style.highlight_max(subset=["Accuracy", "F1-Score", "ROC-AUC"], color="#1e3a5f")
        .highlight_min(subset=["Support Vectors", "Train Latency (ms)"], color="#14532d")
        .format({
            "Accuracy": "{:.2%}",
            "F1-Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "SV Ratio (%)": "{:.1f}%",
            "Train Latency (ms)": "{:.2f}"
        }),
        use_container_width=True
    )

    # Bar chart comparison
    fig_battle = px.bar(
        tournament_df,
        x="Kernel",
        y="Accuracy",
        color="Kernel",
        text_auto=".2%",
        title=f"Holdout Accuracy by Kernel (C = {c_val})",
        color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#ec4899"]
    )
    fig_battle.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
        showlegend=False
    )
    st.plotly_chart(fig_battle, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 2. RBF Hyperparameter Grid Search (C vs Gamma)")
    st.caption("Scan cross-validated accuracy across regularization C and kernel coefficient gamma.")

    if st.button("Execute Hyperparameter Grid Search Matrix", key="btn_grid_search"):
        with st.spinner("Computing 5-fold cross-validation across 25 parameter configurations..."):
            grid_res = run_grid_search_tuning(
                X_train=X_train,
                y_train=y_train,
                kernel="rbf",
                c_values=[0.01, 0.1, 1.0, 10.0, 100.0],
                gamma_values=[0.001, 0.01, 0.1, 1.0, 10.0],
                cv_folds=5
            )

            col_grid_l, col_grid_r = st.columns([2, 1])
            with col_grid_l:
                c_strs = [f"C={c}" for c in grid_res["c_values"]]
                g_strs = [f"γ={g}" for g in grid_res["gamma_values"]]

                fig_grid = px.imshow(
                    grid_res["grid_matrix"],
                    x=g_strs,
                    y=c_strs,
                    labels=dict(x="Gamma (Influence Radius)", y="C (Soft Margin Penalty)", color="CV Accuracy"),
                    color_continuous_scale="Viridis",
                    text_auto=".1%",
                    title="Stratified 5-Fold Cross-Validation Accuracy Surface"
                )
                fig_grid.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
                )
                st.plotly_chart(fig_grid, use_container_width=True)

            with col_grid_r:
                st.markdown("""
                <div class="metric-card" style="margin-top: 40px;">
                    <div class="metric-title">Optimal Grid Configuration</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: #10b981;">
                        C = {}, γ = {}
                    </div>
                    <div class="metric-sub">Mean CV Score: {:.2%}</div>
                </div>
                """.format(
                    grid_res['best_params']['C'],
                    grid_res['best_params']['gamma'],
                    grid_res['best_score']
                ), unsafe_allow_html=True)

                st.markdown("""
                **💡 Engineering Takeaways:**
                - **Low C, Low Gamma**: High bias / underfitting; margin is excessively wide and smooth.
                - **High C, High Gamma**: High variance / overfitting; islands form around isolated data points.
                - **Optimal Zone**: Maximum generalization occurs where the boundary captures global manifold curvature without memorizing noise.
                """)


# =========================================================
# TAB 4: Clinical Diagnostic Inference
# =========================================================
with tab_clinical:
    st.markdown("### 🩺 Real-Time Diagnostic Clinical Inference Engine")
    st.markdown("""
    Test simulated patient profiles or manipulate biopsy cytologic measurements 
    to obtain real-time SVM diagnostic predictions, Platt-calibrated malignancy probabilities, 
    and geometric boundary distance metrics.
    """)

    presets = get_clinical_patient_presets()
    preset_choice = st.selectbox(
        "Load Clinical Patient Archetype",
        ["Custom Manual Entry"] + list(presets.keys())
    )

    # Initialize feature values dictionary
    default_vals = {}
    if preset_choice in presets:
        default_vals = presets[preset_choice]

    st.markdown("#### Patient Cytological Features")
    
    # Render input controls in 3 responsive columns
    cin_cols = st.columns(3)
    user_inputs = {}

    for i, f_col in enumerate(feature_names[:12]):
        col_idx = i % 3
        # Reference statistics from dataset
        col_min = float(df_active[f_col].min())
        col_max = float(df_active[f_col].max())
        col_mean = float(df_active[f_col].mean())
        def_v = default_vals.get(f_col, col_mean)

        with cin_cols[col_idx]:
            user_inputs[f_col] = st.slider(
                f"{f_col.title()}",
                min_value=round(col_min * 0.7, 3),
                max_value=round(col_max * 1.3, 3),
                value=float(def_v),
                step=round((col_max - col_min) / 100.0, 4) if col_max > col_min else 0.01,
                key=f"clin_{f_col}"
            )

    # For any remaining features not shown in the first 12, use their mean or preset value
    for f_col in feature_names[12:]:
        col_mean = float(df_active[f_col].mean())
        user_inputs[f_col] = default_vals.get(f_col, col_mean)

    # Run Real-Time Prediction
    st.markdown("---")
    infer_res = predict_single_instance(
        model=model_active,
        scaler=scaler,
        raw_feature_dict=user_inputs,
        feature_order=feature_names,
        target_names=target_names
    )

    col_res_l, col_res_r = st.columns([1, 1])

    with col_res_l:
        pred_label = infer_res["prediction_label"]
        is_malignant = "malignant" in pred_label.lower()
        badge_style = "badge-danger" if is_malignant else "badge-success"
        card_border = "#ef4444" if is_malignant else "#10b981"

        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {card_border}; padding: 24px;">
            <div class="metric-title">Diagnostic Prediction Result</div>
            <div class="metric-value" style="color: {'#f87171' if is_malignant else '#34d399'};">
                {pred_label}
            </div>
            <span class="{badge_style}" style="margin-top: 8px;">
                CONFIDENCE LEVEL: {max(infer_res['probabilities'].values()):.1%}
            </span>
            <div class="metric-sub" style="margin-top: 12px; font-size: 0.85rem; line-height: 1.4;">
                Separating Hyperplane Distance f(x): <strong>{infer_res['decision_value']:.4f}</strong>
                <br>
                {"⚠️ Critical: Patient sample falls on the malignant side of the SVM margin." if is_malignant else "✅ Favorable: Sample features reside safely within the benign classification envelope."}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res_r:
        st.markdown("#### 📊 Platt-Calibrated Class Probabilities")
        proba_df = pd.DataFrame([
            {"Class": k, "Probability": v}
            for k, v in infer_res["probabilities"].items()
        ])
        fig_prob = px.bar(
            proba_df,
            x="Probability",
            y="Class",
            orientation="h",
            text=proba_df["Probability"].map(lambda p: f"{p:.1%}"),
            color="Class",
            color_discrete_sequence=["#ef4444", "#10b981"] if len(proba_df) == 2 else px.colors.qualitative.Plotly
        )
        fig_prob.update_layout(
            xaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=10, b=10),
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)


# =========================================================
# TAB 5: Mathematical Foundations & Architecture
# =========================================================
with tab_theory:
    st.markdown("### 📚 Support Vector Machine Foundations & Mathematical Rigor")
    
    st.markdown("""
    #### 1. Maximum Margin Primal Optimization Problem
    Given a labeled training set $\\{(x_1, y_1), \\dots, (x_N, y_N)\\}$ where $x_i \\in \\mathbb{R}^D$ and $y_i \\in \\{-1, +1\\}$, 
    the goal of a Support Vector Machine is to discover the separating hyperplane:
    
    $$w \\cdot x + b = 0$$
    
    that maximizes the geometric margin $\\gamma = \\frac{2}{\\|w\\|_2}$ while permitting constrained margin violations via slack variables $\\xi_i \\ge 0$:
    
    $$\\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^N \\xi_i$$
    $$\\text{subject to } y_i (w \\cdot x_i + b) \\ge 1 - \\xi_i, \\quad \\xi_i \\ge 0, \\quad \\forall i$$
    
    - **Parameter $C$**: Governs the trade-off between maximizing margin width (generalization) and penalizing slack errors $\\xi_i$.
    """)

    st.markdown("""
    #### 2. The Dual Formulation & Karush-Kuhn-Tucker (KKT) Conditions
    Applying Lagrange multipliers $\\alpha_i \\ge 0$, the primal transforms into the convex Wolfe Dual:
    
    $$\\max_{\\alpha} \\sum_{i=1}^N \\alpha_i - \\frac{1}{2} \\sum_{i=1}^N \\sum_{j=1}^N \\alpha_i \\alpha_j y_i y_j \\langle x_i, x_j \\rangle$$
    $$\\text{subject to } 0 \\le \\alpha_i \\le C \\quad \\text{and} \\quad \\sum_{i=1}^N \\alpha_i y_i = 0$$
    
    **The Support Vector Principle:**
    - If $\\alpha_i = 0$, sample $x_i$ lies safely outside the margin and has zero influence on the decision boundary.
    - If $0 < \\alpha_i < C$, sample $x_i$ is a **free support vector** lying exactly on the margin boundary ($y_i(w \\cdot x_i + b) = 1$).
    - If $\\alpha_i = C$, sample $x_i$ is a **bounded support vector** violating the margin ($0 < y_i(w \\cdot x_i + b) < 1$) or misclassified.
    """)

    st.markdown("""
    #### 3. The Kernel Trick (Mercer's Theorem)
    By replacing the Euclidean inner product $\\langle x_i, x_j \\rangle$ with a positive semi-definite kernel function $K(x_i, x_j) = \\langle \\phi(x_i), \\phi(x_j) \\rangle$, 
    the SVM implicitly maps input data into an arbitrary (or infinite-dimensional) Hilbert feature space without explicit transformation coordinate computation:
    
    | Kernel Name | Mathematical Formulation | Characteristics & Best Use-Case |
    | :--- | :--- | :--- |
    | **Linear** | $K(x, x') = x^T x'$ | Fast, sparse, highly effective for high-dimensional data ($D > N$) |
    | **RBF (Gaussian)** | $K(x, x') = \\exp(-\\gamma \\|x - x'\\|^2)$ | Universally continuous, maps to infinite dimensions, default gold standard |
    | **Polynomial** | $K(x, x') = (\\gamma x^T x' + r)^d$ | Models multi-feature interactions up to degree $d$ |
    | **Sigmoid** | $K(x, x') = \\tanh(\\gamma x^T x' + r)$ | Bridges SVM with multilayer perceptron neural networks |
    """)

    st.markdown("""
    #### 4. Production Hardening & Deployment Architecture
    ```
    ┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
    │  Raw Biopsy Telemetry   │ ──> │ Standard Feature Scaler│ ──> │ Calibrated SVC Engine  │
    │  (30 Continuous Inputs) │     │ (Zero Mean, Unit Var)  │     │ (RBF Kernel, C=1.0)    │
    └─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                               │
                                                                               ▼
    ┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
    │ Out-of-Distribution     │ <── │ Signed Margin Distance │ <── │  Platt-Calibrated      │
    │ Alert & Drift Telemetry │     │ f(x) Geometric Proximity│     │  Malignancy Probability│
    └─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
    Support Vector Machine Studio • Part of the <strong>1,000 AI Projects Implementation Library</strong>
</div>
""", unsafe_allow_html=True)

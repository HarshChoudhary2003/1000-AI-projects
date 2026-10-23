"""
Project 009: Random Forest Classifier & Ensemble Studio
=======================================================
Flagship interactive Streamlit application featuring real-world multi-class
chemical profiling (Wine Cultivars), Out-of-Bag (OOB) error convergence,
Mean Decrease in Impurity (Gini) feature importances, 2D ensemble voting surfaces,
individual tree voting telemetry, and real-time decision inference.

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
    get_archetype_presets
)
from rf_engine import (
    prepare_dataset_split,
    train_random_forest,
    evaluate_rf_model,
    extract_feature_importances,
    compute_oob_convergence,
    extract_forest_diagnostics,
    inspect_tree_votes,
    compute_2d_ensemble_mesh,
    run_cross_validation
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Random Forest Studio | 1,000 AI Projects",
    page_icon="🌲",
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
        border-color: rgba(16, 185, 129, 0.4);
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

    /* Badges */
    .badge-forest {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-amber {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fbbf24;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-blue {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Banner Container */
    .banner-container {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.25) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
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
    <div style="font-size: 2.2rem; margin-bottom: 4px;">🌲</div>
    <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700;">Random Forest</h2>
    <span class="badge-forest" style="margin-top: 4px;">PROJECT 009 • ENSEMBLE</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 1. Dataset Configuration")

dataset_choice = st.sidebar.selectbox(
    "Benchmark Dataset",
    [
        "Wine Cultivars (Chemical Profiling)",
        "Wisconsin Breast Cancer (Diagnostic)",
        "Iris Botanical Benchmark",
        "Two Moons (Synthetic 2D)",
        "Concentric Circles (Synthetic 2D)",
        "Overlapping Blobs (Synthetic 2D)",
        "Complex Multimodal (Synthetic 2D)",
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
st.sidebar.subheader("⚙️ 2. Forest Hyperparameters")

n_estimators = st.sidebar.slider(
    "Number of Trees (n_estimators)",
    min_value=10,
    max_value=300,
    value=100,
    step=10,
    help="Total decision trees in the ensemble. More trees reduce variance without risk of overfitting."
)

max_depth_mode = st.sidebar.radio("Max Depth Constraint", ["Unlimited (None)", "Constrained"], horizontal=True)
if max_depth_mode == "Constrained":
    max_depth_val = st.sidebar.slider("Maximum Depth", min_value=1, max_value=25, value=5)
else:
    max_depth_val = None

criterion = st.sidebar.selectbox("Splitting Criterion", ["gini", "entropy", "log_loss"], index=0)

max_features_mode = st.sidebar.selectbox(
    "Max Features per Split",
    ["sqrt", "log2", "all"],
    format_func=lambda x: {
        "sqrt": "Square Root (sqrt)",
        "log2": "Log2 (log2)",
        "all": "All Features (None)"
    }[x],
    index=0
)
max_features_val = None if max_features_mode == "all" else max_features_mode

min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=10, value=2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=10, value=1)
bootstrap = st.sidebar.checkbox("Bootstrap Sampling", value=True, help="Sample with replacement for tree training. Enables OOB evaluation.")
class_weight_opt = st.sidebar.selectbox("Class Weighting", ["None", "balanced"], index=0)
class_weight = "balanced" if class_weight_opt == "balanced" else None

test_ratio = st.sidebar.slider("Holdout Test Fraction", min_value=0.10, max_value=0.40, value=0.25, step=0.05)


# ---------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------
@st.cache_data
def get_dataset(choice: str, uploaded_content: Optional[bytes] = None):
    if choice == "Wine Cultivars (Chemical Profiling)":
        return load_real_dataset("wine")
    elif choice == "Wisconsin Breast Cancer (Diagnostic)":
        return load_real_dataset("cancer")
    elif choice == "Iris Botanical Benchmark":
        return load_real_dataset("iris")
    elif choice == "Two Moons (Synthetic 2D)":
        return generate_2d_manifold("Two Moons")
    elif choice == "Concentric Circles (Synthetic 2D)":
        return generate_2d_manifold("Concentric Circles")
    elif choice == "Overlapping Blobs (Synthetic 2D)":
        return generate_2d_manifold("Overlapping Blobs")
    elif choice == "Complex Multimodal (Synthetic 2D)":
        return generate_2d_manifold("Complex Multimodal")
    elif choice == "Upload Custom CSV" and uploaded_content is not None:
        df = pd.read_csv(io.BytesIO(uploaded_content))
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return None
        target_col = numeric_cols[-1]
        feature_cols = [c for c in numeric_cols if c != target_col]
        unique_targets = df[target_col].unique()
        target_names = [f"Class {t}" for t in unique_targets]
        df["target_label"] = df[target_col].astype(str)
        return df, feature_cols, target_col, target_names
    return load_real_dataset("wine")

uploaded_bytes = uploaded_file.getvalue() if custom_df is not None else None
dataset_pack = get_dataset(dataset_choice, uploaded_bytes)

if dataset_pack is None:
    st.error("Please upload a valid CSV containing at least 1 feature column and 1 numeric target column.")
    st.stop()

df_active, feature_names, target_col, target_names = dataset_pack

# Data split
data_bundle = prepare_dataset_split(
    df=df_active,
    feature_cols=feature_names,
    target_col=target_col,
    test_size=test_ratio,
    scale_method="none",
    random_state=42
)
X_train, X_test = data_bundle["X_train"], data_bundle["X_test"]
y_train, y_test = data_bundle["y_train"], data_bundle["y_test"]

# Train Active Random Forest Model
model_active, train_latency_ms = train_random_forest(
    X_train=X_train,
    y_train=y_train,
    n_estimators=n_estimators,
    max_depth=max_depth_val,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features_val,
    criterion=criterion,
    bootstrap=bootstrap,
    oob_score=bootstrap,
    class_weight=class_weight,
    random_state=42
)

# Evaluate and diagnose
eval_active = evaluate_rf_model(model_active, X_test, y_test, target_names=target_names)
forest_diag = extract_forest_diagnostics(model_active)
feature_imp_df = extract_feature_importances(model_active, feature_names)


# ---------------------------------------------------------
# Main Page Header & Banner
# ---------------------------------------------------------
st.markdown("""
<div class="banner-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;">
                Random Forest Classifier Studio
            </h1>
            <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 0.95rem;">
                Ensemble Bagging Architecture, Out-of-Bag (OOB) Telemetry & Feature Importance Profiling
            </p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="badge-forest">Forest Size: {} Trees</span>
            <span class="badge-amber">Criterion: {}</span>
            <span class="badge-blue">OOB Validation: {}</span>
        </div>
    </div>
</div>
""".format(n_estimators, criterion.upper(), "Enabled" if bootstrap else "Disabled"), unsafe_allow_html=True)


# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab_benchmark, tab_forest, tab_decision, tab_inference, tab_theory = st.tabs([
    "📊 Executive Studio & Benchmark",
    "🌲 Forest Architecture & OOB Lab",
    "🔮 2D Ensemble Space & Consensus",
    "🎯 Real-Time Decision & Vote Inspector",
    "📚 Random Forest Theory & Bagging"
])


# =========================================================
# TAB 1: Executive Studio & Benchmark
# =========================================================
with tab_benchmark:
    st.subheader("Model Performance Telemetry & Champion Metrics")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test Accuracy</div>
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
        oob_disp = f"{eval_active['oob_score']:.1%}" if "oob_score" in eval_active else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">OOB Accuracy</div>
            <div class="metric-value">{oob_disp}</div>
            <div class="metric-sub">Internal cross-validation</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        roc_disp = f"{eval_active['roc_auc']:.3f}" if "roc_auc" in eval_active else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ROC-AUC</div>
            <div class="metric-value">{roc_disp}</div>
            <div class="metric-sub">Discriminative ranking</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Depth</div>
            <div class="metric-value">{forest_diag['avg_depth']:.1f}</div>
            <div class="metric-sub">Range: [{forest_diag['min_depth']} - {forest_diag['max_depth']}]</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Train Latency</div>
            <div class="metric-value">{train_latency_ms:.1f}<span style="font-size: 1rem;">ms</span></div>
            <div class="metric-sub">{n_estimators} parallel estimators</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart_l, col_chart_r = st.columns([1, 1])

    with col_chart_l:
        st.markdown("#### 🎯 Confusion Matrix")
        cm = eval_active["confusion_matrix"]
        cm_labels_y = [f"Actual {n}" for n in target_names]
        cm_labels_x = [f"Pred {n}" for n in target_names]

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Ground Truth Class", color="Sample Count"),
            x=cm_labels_x,
            y=cm_labels_y,
            color_continuous_scale="Greens",
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
        st.plotly_chart(fig_cm, width="stretch")

        if eval_active.get("is_binary", False) and "specificity" in eval_active:
            sub_c1, sub_c2, sub_c3 = st.columns(3)
            sub_c1.metric("Specificity (TNR)", f"{eval_active['specificity']:.1%}")
            sub_c2.metric("Sensitivity (TPR)", f"{eval_active['recall']:.1%}")
            sub_c3.metric("Matthews Corr (MCC)", f"{eval_active['mcc']:.3f}")

    with col_chart_r:
        if eval_active.get("is_binary", False) and "y_proba" in eval_active:
            st.markdown("#### 📈 Discrimination Curves (ROC & PR)")
            y_proba = eval_active["y_proba"][:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {eval_active['roc_auc']:.3f})",
                line=dict(color="#10b981", width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random Chance",
                line=dict(color="#64748b", dash="dash")
            ))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
                legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
            )
            st.plotly_chart(fig_roc, width="stretch")
        else:
            st.markdown("#### 📊 Per-Class Precision & Recall Breakdown")
            rep_df = pd.DataFrame(eval_active["classification_report"]).transpose()
            st.dataframe(rep_df.style.format(precision=3), width="stretch")

    # Cross-validation exploration
    st.markdown("---")
    st.markdown("#### 🔄 Stratified K-Fold Generalization Stability")
    cv_k = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5)
    if st.button("Run Cross-Validation Benchmark", key="run_cv_btn"):
        with st.spinner(f"Evaluating {cv_k}-fold cross validation across ensemble..."):
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
                        colorscale="Tealgrn",
                        showscale=False
                    ),
                    text=[f"{s:.1%}" for s in cv_bundle["fold_scores"]],
                    textposition="auto"
                ))
                fig_cv.add_hline(
                    y=cv_bundle["mean_accuracy"],
                    line_dash="dash",
                    line_color="#fbbf24",
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
                st.plotly_chart(fig_cv, width="stretch")

    with st.expander("🔍 Explore Raw Dataset & Correlation Insights"):
        st.dataframe(df_active.head(10), width="stretch")
        st.write(f"**Total Samples:** {df_active.shape[0]} | **Feature Dimension:** {len(feature_names)}")


# =========================================================
# TAB 2: Forest Architecture & OOB Lab
# =========================================================
with tab_forest:
    st.markdown("### 🌲 Forest Architecture, OOB Convergence & Feature Importance")
    st.markdown("""
    Random Forest leverages **Bagging (Bootstrap Aggregation)** and **Random Feature Subspace Selection**. 
    Inspect the Mean Decrease in Impurity (Gini importance) across features and observe the 
    **Out-of-Bag (OOB) error convergence curve** as the number of estimators increases.
    """)

    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.markdown("#### 🌟 Gini Feature Importance Ranking (MDI)")
        fig_imp = px.bar(
            feature_imp_df.head(12),
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            error_x="Std_Dev",
            color="Importance_Pct",
            color_continuous_scale="Viridis",
            labels=dict(Importance_Pct="Importance (%)", Feature="Predictor"),
            title="Top 12 Most Informative Features (with Inter-Tree Variance)"
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, width="stretch")

    with col_f2:
        st.markdown("#### 📈 Cumulative Importance Curve")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=list(range(1, len(feature_imp_df) + 1)),
            y=feature_imp_df["Cumulative"] * 100.0,
            mode="lines+markers",
            line=dict(color="#f59e0b", width=3),
            name="Cumulative Importance"
        ))
        fig_cum.add_hline(y=90.0, line_dash="dash", line_color="#ef4444", annotation_text="90% Variance Threshold")
        fig_cum.update_layout(
            title="Cumulative Feature Explanatory Power",
            xaxis_title="Number of Top Features Included",
            yaxis_title="Cumulative Importance (%)",
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
        )
        st.plotly_chart(fig_cum, width="stretch")

    st.markdown("---")
    st.markdown("#### 🎯 Out-of-Bag (OOB) Convergence Profiler")
    st.caption("Demonstrate the asymptotic law of large numbers: adding trees reduces variance without risk of overfitting.")

    if st.button("Generate OOB Convergence Curve", key="btn_oob_curve"):
        with st.spinner("Computing incremental OOB error across tree depths..."):
            oob_df = compute_oob_convergence(X_train, y_train, max_trees=min(200, n_estimators + 50), step=10)

            fig_oob = go.Figure()
            fig_oob.add_trace(go.Scatter(
                x=oob_df["n_estimators"],
                y=oob_df["oob_error"],
                mode="lines+markers",
                name="OOB Error Rate (1 - OOB Accuracy)",
                line=dict(color="#ef4444", width=3)
            ))
            fig_oob.add_trace(go.Scatter(
                x=oob_df["n_estimators"],
                y=oob_df["oob_score"],
                mode="lines+markers",
                name="OOB Accuracy",
                line=dict(color="#10b981", width=3)
            ))
            fig_oob.update_layout(
                title="Out-of-Bag (OOB) Performance vs Number of Trees",
                xaxis_title="Number of Trees (n_estimators)",
                yaxis_title="Metric Rate",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", color="#cbd5e1")
            )
            st.plotly_chart(fig_oob, width="stretch")

    # Forest Structural Details
    st.markdown("---")
    st.markdown("#### 🏛️ Forest Structural Distribution")
    col_str1, col_str2, col_str3, col_str4 = st.columns(4)
    col_str1.metric("Total Trees in Forest", forest_diag["n_trees"])
    col_str2.metric("Average Tree Depth", f"{forest_diag['avg_depth']:.1f}")
    col_str3.metric("Average Leaves per Tree", f"{forest_diag['avg_leaves']:.1f}")
    col_str4.metric("Total Forest Decision Nodes", f"{forest_diag['total_nodes_forest']:,}")


# =========================================================
# TAB 3: 2D Ensemble Space & Consensus
# =========================================================
with tab_decision:
    st.markdown("### 🔮 2D Ensemble Decision Space & Continuous Voting Consensus")
    st.markdown("""
    Explore how an ensemble of orthogonal decision trees creates smooth, non-linear classification surfaces. 
    The background heatmap illustrates the **Ensemble Voting Consensus Probability** (confidence in the prediction).
    """)

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        if len(feature_names) >= 2:
            feat_x = st.selectbox("Horizontal Feature (X1)", feature_names, index=0, key="rf_feat_x")
            feat_y = st.selectbox("Vertical Feature (X2)", feature_names, index=1 if len(feature_names) > 1 else 0, key="rf_feat_y")
        else:
            feat_x, feat_y = feature_names[0], feature_names[0]

    with col_c2:
        vis_trees = st.slider("Dynamic Trees (n_estimators)", 5, 200, 50, 5, key="vis_trees")

    with col_c3:
        vis_depth = st.slider("Dynamic Max Depth", 1, 15, 5, 1, key="vis_depth")

    # Dedicated 2D Random Forest
    X_2d_raw = df_active[[feat_x, feat_y]].values
    y_2d = df_active[target_col].values

    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d_raw)

    model_2d, _ = train_random_forest(
        X_train=X_2d_scaled,
        y_train=y_2d,
        n_estimators=vis_trees,
        max_depth=vis_depth,
        bootstrap=True,
        oob_score=False,
        random_state=42
    )

    mesh_res = compute_2d_ensemble_mesh(model_2d, X_2d_scaled, resolution=180, margin_padding=0.8)
    xx, yy = mesh_res["xx"], mesh_res["yy"]
    Z_label = mesh_res["Z_label"]
    Z_conf = mesh_res["Z_conf"]

    fig_contour = go.Figure()

    # Voting Consensus Heatmap
    fig_contour.add_trace(go.Contour(
        x=np.linspace(mesh_res["x_range"][0], mesh_res["x_range"][1], 180),
        y=np.linspace(mesh_res["y_range"][0], mesh_res["y_range"][1], 180),
        z=Z_conf,
        colorscale="Viridis",
        opacity=0.45,
        showscale=True,
        colorbar=dict(title="Consensus %"),
        name="Ensemble Voting Confidence"
    ))

    # Discrete Decision Boundary Contours
    fig_contour.add_trace(go.Contour(
        x=np.linspace(mesh_res["x_range"][0], mesh_res["x_range"][1], 180),
        y=np.linspace(mesh_res["y_range"][0], mesh_res["y_range"][1], 180),
        z=Z_label,
        showscale=False,
        contours=dict(coloring="none", showlines=True),
        line=dict(color="#ffffff", width=2.5),
        name="Decision Boundaries"
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

    fig_contour.update_layout(
        title=f"2D Feature Projection: {feat_x} vs {feat_y} | Forest of {vis_trees} Trees (Max Depth {vis_depth})",
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
    st.plotly_chart(fig_contour, width="stretch")


# =========================================================
# TAB 4: Real-Time Decision & Vote Inspector
# =========================================================
with tab_inference:
    st.markdown("### 🎯 Real-Time Decision Inference & Tree Vote Inspector")
    st.markdown("""
    Select a pre-loaded sample archetype or manipulate chemical/biopsy measurements. 
    Observe the **exact voting distribution across every individual tree** in the ensemble!
    """)

    presets = get_archetype_presets()
    preset_choice = st.selectbox(
        "Load Sample Archetype Profile",
        ["Custom Manual Entry"] + list(presets.keys())
    )

    default_vals = {}
    if preset_choice in presets:
        default_vals = presets[preset_choice]

    st.markdown("#### Input Sample Features")
    cin_cols = st.columns(3)
    user_inputs = {}

    for i, f_col in enumerate(feature_names[:12]):
        col_idx = i % 3
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
                key=f"rf_in_{f_col}"
            )

    for f_col in feature_names[12:]:
        col_mean = float(df_active[f_col].mean())
        user_inputs[f_col] = default_vals.get(f_col, col_mean)

    # Perform Tree Vote Inspection
    st.markdown("---")
    vote_res = inspect_tree_votes(
        model=model_active,
        scaler=None,
        raw_feature_dict=user_inputs,
        feature_order=feature_names,
        target_names=target_names
    )

    col_res_l, col_res_r = st.columns([1, 1])

    with col_res_l:
        pred_label = vote_res["prediction_label"]
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid #10b981; padding: 24px;">
            <div class="metric-title">Ensemble Consensus Classification</div>
            <div class="metric-value" style="color: #34d399;">
                {pred_label}
            </div>
            <span class="badge-forest" style="margin-top: 8px;">
                VOTING CONSENSUS: {vote_res['consensus_confidence']:.1f}% OF TREES
            </span>
            <div class="metric-sub" style="margin-top: 12px; font-size: 0.85rem; line-height: 1.4;">
                Winning Class Received <strong>{vote_res['vote_counts'].get(pred_label, 0)}</strong> out of <strong>{vote_res['total_trees']}</strong> total tree votes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res_r:
        st.markdown("#### 🗳️ Individual Tree Voting Breakdown")
        vote_df = pd.DataFrame([
            {"Class": k, "Tree Votes": v, "Percentage": f"{(v/vote_res['total_trees'])*100:.1f}%"}
            for k, v in vote_res["vote_counts"].items()
        ])
        fig_votes = px.bar(
            vote_df,
            x="Tree Votes",
            y="Class",
            orientation="h",
            text="Percentage",
            color="Class",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_votes.update_layout(
            xaxis=dict(range=[0, vote_res["total_trees"] * 1.05]),
            margin=dict(l=20, r=20, t=10, b=10),
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#cbd5e1"),
            showlegend=False
        )
        st.plotly_chart(fig_votes, width="stretch")


# =========================================================
# TAB 5: Random Forest Theory & Bagging
# =========================================================
with tab_theory:
    st.markdown("### 📚 Random Forest Foundations & Ensemble Bagging")

    st.markdown("""
    #### 1. Bootstrap Aggregation (Bagging)
    A Random Forest is an ensemble meta-estimator that fits multiple decision tree classifiers on various sub-samples of the dataset and uses averaging to improve predictive accuracy and control over-fitting:

    Given a dataset $D$ of size $N$, for each tree $b = 1, \dots, B$:
    1. Draw a bootstrap sample $D_b$ of size $N$ with replacement from $D$.
    2. Train an unpruned decision tree $T_b$ on $D_b$.
    3. At each candidate split node, select $m$ features at random from the total $P$ features (typically $m \approx \sqrt{P}$).
    4. Find the best split among the $m$ features according to Gini impurity or Entropy.

    The ensemble prediction is computed by majority voting:
    
    $$\\hat{y} = \\text{mode} \\{ T_1(x), T_2(x), \\dots, T_B(x) \\}$$
    """)

    st.markdown("""
    #### 2. The Bias-Variance Reduction Theorem
    For an ensemble of $B$ identically distributed trees, each with variance $\\sigma^2$ and positive pairwise correlation $\\rho$:

    $$\\text{Var}(\\bar{T}) = \\rho \\sigma^2 + \\frac{1 - \\rho}{B} \\sigma^2$$

    - As $B \\to \\infty$, the second term $\\frac{1 - \\rho}{B} \\sigma^2 \\to 0$.
    - The remaining variance is bounded by $\\rho \\sigma^2$.
    - **Crucial Insight:** By restricting each split to a random subset of $m \\approx \\sqrt{P}$ features, Random Forests substantially decrease the inter-tree correlation $\\rho$, driving total variance down dramatically!
    """)

    st.markdown("""
    #### 3. Out-of-Bag (OOB) Mathematical Validation
    Because each bootstrap sample is drawn with replacement, the probability that a specific training instance is **not** selected in a bootstrap sample of size $N$ is:

    $$P(\\text{Not Chosen}) = \\left( 1 - \\frac{1}{N} \\right)^N \\approx \\frac{1}{e} \\approx 36.8\\%$$

    - Approximately $36.8\\%$ of training observations are left out of each tree's training fold (called **Out-of-Bag samples**).
    - Each observation is predicted using only the subset of trees where it was Out-of-Bag, providing an unbiased cross-validation score without separate validation sets!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
    Random Forest Classifier Studio • Part of the <strong>1,000 AI Projects Implementation Library</strong>
</div>
""", unsafe_allow_html=True)

"""
Project 007: K-Nearest Neighbors (KNN) Implementation & Spatial Geometry Studio
================================================================================
An industrial-grade, interactive Streamlit workbench for K-Nearest Neighbors
classification, Voronoi decision boundaries, hyperparameter optimization,
feature scaling impact analysis, and the curse of dimensionality.

Part of the 1,000 AI Projects Implementation Library.
"""

import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

from dataset import (
    load_real_dataset,
    generate_2d_manifold,
    generate_curse_of_dimensionality_experiment
)


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="KNN Spatial Geometry Studio | AI Algorithms Workbench",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System (Titanium Obsidian & Champagne Amber)
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
    
    /* Concept Callout Box */
    .studio-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        background: #0E121A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .studio-card-accent {
        border-left: 5px solid #F59E0B;
    }
    .studio-card-cyan {
        border-left: 5px solid #38BDF8;
    }
    .studio-card-danger {
        border-left: 5px solid #EF4444;
    }
    .studio-card-success {
        border-left: 5px solid #10B981;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        font-weight: 700;
        font-size: 0.90rem;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper & Training Functions
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_dataset_cached(dataset_choice, manifold_noise=0.20, n_samples_2d=400):
    """Load or synthesize the selected benchmark dataset."""
    if dataset_choice == "Breast Cancer Diagnostic (Real, 30 Features)":
        return load_real_dataset("breast_cancer")
    elif dataset_choice == "Wine Chemical Cultivars (Real, 13 Features)":
        return load_real_dataset("wine")
    elif dataset_choice == "Two Moons (2D Non-Linear)":
        return generate_2d_manifold("Two Moons", n_samples=n_samples_2d, noise=manifold_noise)
    elif dataset_choice == "Concentric Circles (2D Non-Linear)":
        return generate_2d_manifold("Concentric Circles", n_samples=n_samples_2d, noise=manifold_noise)
    elif dataset_choice == "Intertwined Spirals (2D Complex)":
        return generate_2d_manifold("Intertwined Spirals", n_samples=n_samples_2d, noise=manifold_noise)
    elif dataset_choice == "Gaussian Blobs (2D Multi-Class)":
        return generate_2d_manifold("Gaussian Blobs (Multi-Class)", n_samples=n_samples_2d, noise=manifold_noise)
    else:
        return load_real_dataset("breast_cancer")


def get_scaler(scaler_type: str):
    """Instantiate selected feature scaler."""
    if scaler_type == "StandardScaler (Z-Score)":
        return StandardScaler()
    elif scaler_type == "MinMaxScaler [0, 1]":
        return MinMaxScaler()
    elif scaler_type == "RobustScaler (IQR)":
        return RobustScaler()
    else:
        return None  # Raw unscaled data


@st.cache_data(show_spinner=False)
def compute_k_optimization_curve(X_train_scaled, y_train, k_max=35, cv=5):
    """Calculates cross-validation accuracy across a range of K values."""
    k_range = list(range(1, min(k_max + 1, len(X_train_scaled) - 5), 2 if len(X_train_scaled) > 50 else 1))
    cv_scores_mean = []
    cv_scores_std = []
    
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_scores_mean.append(scores.mean())
        cv_scores_std.append(scores.std())
        
    return k_range, cv_scores_mean, cv_scores_std


# ==============================================================================
# Sidebar: Interactive Configuration Studio
# ==============================================================================
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 12px;">
    <h2 style="margin: 0; font-size: 1.4rem; color: #F59E0B; font-family: 'Outfit', sans-serif;">🧭 KNN Geometry Studio</h2>
    <span style="font-size: 0.75rem; color: #94A3B8;">Project #0007 • Distance Metric & Spatial Lab</span>
</div>
""", unsafe_allow_html=True)

# 1. Dataset Selection
st.sidebar.markdown("### 🗂️ 1. Dataset Benchmark")
dataset_choice = st.sidebar.selectbox(
    "Benchmark Dataset",
    [
        "Breast Cancer Diagnostic (Real, 30 Features)",
        "Wine Chemical Cultivars (Real, 13 Features)",
        "Two Moons (2D Non-Linear)",
        "Concentric Circles (2D Non-Linear)",
        "Intertwined Spirals (2D Complex)",
        "Gaussian Blobs (2D Multi-Class)",
        "Upload Custom Tabular CSV"
    ],
    index=0
)

custom_df = None
if dataset_choice == "Upload Custom Tabular CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Select feature and target columns)", type=["csv"])
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(custom_df)} records!")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")

manifold_noise = 0.20
if "2D" in dataset_choice:
    manifold_noise = st.sidebar.slider("Synthetic Noise Level", min_value=0.05, max_value=0.50, value=0.20, step=0.05)

# Load Selected Data
if custom_df is not None:
    numeric_cols = custom_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        target_candidate = custom_df.columns[-1]
        target_col = st.sidebar.selectbox("Target Column", custom_df.columns, index=len(custom_df.columns)-1)
        feature_names = [c for c in numeric_cols if c != target_col]
        df = custom_df.copy()
        df[target_col] = df[target_col].astype('category').cat.codes
        target_names = [str(c) for c in custom_df[target_col].unique()]
        df["target_label"] = custom_df[target_col].astype(str)
    else:
        st.sidebar.error("CSV must contain at least 2 numeric feature columns.")
        df, feature_names, target_col, target_names = load_real_dataset("breast_cancer")
else:
    df, feature_names, target_col, target_names = load_dataset_cached(dataset_choice, manifold_noise)

# 2. KNN Hyperparameters
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 2. Hyperparameter Controls")

max_possible_k = min(51, len(df) - 10)
k_neighbors = st.sidebar.slider(
    "Number of Neighbors (K)",
    min_value=1,
    max_value=max_possible_k,
    value=5,
    step=1,
    help="Odd K values are recommended for binary classification to break ties."
)

weights_mode = st.sidebar.radio(
    "Neighbor Weighting Function",
    ["uniform (equal vote)", "distance (inverse distance: 1/d)"],
    index=0,
    help="Uniform gives all K neighbors 1 vote. Distance weights closer neighbors more heavily."
)
weights_clean = 'uniform' if 'uniform' in weights_mode else 'distance'

metric_choice = st.sidebar.selectbox(
    "Distance Metric",
    [
        "Euclidean (L2 norm)",
        "Manhattan (L1 norm / City Block)",
        "Chebyshev (L∞ norm / Maximum)",
        "Minkowski (Lp norm)",
        "Cosine Distance"
    ],
    index=0
)

metric_map = {
    "Euclidean (L2 norm)": "euclidean",
    "Manhattan (L1 norm / City Block)": "manhattan",
    "Chebyshev (L∞ norm / Maximum)": "chebyshev",
    "Minkowski (Lp norm)": "minkowski",
    "Cosine Distance": "cosine"
}
metric_name = metric_map[metric_choice]

minkowski_p = 2
if metric_choice == "Minkowski (Lp norm)":
    minkowski_p = st.sidebar.slider("Minkowski Power Parameter (p)", min_value=1.0, max_value=6.0, value=3.0, step=0.5)

# 3. Feature Scaling & Preprocessing
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ 3. Preprocessing & Scaling")
scaler_choice = st.sidebar.selectbox(
    "Feature Scaling Strategy",
    [
        "StandardScaler (Z-Score)",
        "MinMaxScaler [0, 1]",
        "RobustScaler (IQR)",
        "None (Raw Features - Observe Distortion!)"
    ],
    index=0,
    help="KNN calculates geometric distance. Features with large scales dwarf smaller features if unscaled."
)

algorithm_choice = st.sidebar.selectbox(
    "Search Algorithm Architecture",
    ["auto", "kd_tree", "ball_tree", "brute"],
    index=0,
    help="KD-Tree is fast for low dimensions; Ball-Tree handles higher dimensions; Brute computes all pairwise distances."
)


# ==============================================================================
# Model Training & Evaluation Execution
# ==============================================================================
X = df[feature_names].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

scaler = get_scaler(scaler_choice)
if scaler is not None:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)
else:
    X_train_scaled = X_train
    X_test_scaled = X_test
    X_all_scaled = X

# Construct KNN Classifier
knn_kwargs = {
    'n_neighbors': k_neighbors,
    'weights': weights_clean,
    'algorithm': algorithm_choice
}

if metric_name == "minkowski":
    knn_kwargs['metric'] = 'minkowski'
    knn_kwargs['p'] = minkowski_p
else:
    knn_kwargs['metric'] = metric_name

t_start = time.perf_counter()
model = KNeighborsClassifier(**knn_kwargs)
model.fit(X_train_scaled, y_train)
t_fit_ms = (time.perf_counter() - t_start) * 1000

# Predictions & Scores
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

is_binary = len(np.unique(y)) == 2
roc_auc_val = 0.0
if is_binary and y_pred_proba.shape[1] == 2:
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc_val = auc(fpr, tpr)
    except Exception:
        roc_auc_val = 0.0


# ==============================================================================
# Header & Executive Metric Strip
# ==============================================================================
st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="margin: 0; font-size: 2.3rem; font-family: 'Outfit', sans-serif;">
        K-Nearest Neighbors (KNN) <span style="color: #F59E0B;">Spatial Geometry Studio</span>
    </h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-top: 6px;">
        An industrial-grade workbench for distance-metric learning, Voronoi decision boundaries, neighbor vector tracing, and hyperparameter calibration.
    </p>
</div>
""", unsafe_allow_html=True)

# Top 6 KPI Metric Cards
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Validation Accuracy", f"{acc * 100:.1f}%", help="Correct test predictions")
kpi2.metric("Weighted Precision", f"{prec * 100:.1f}%", help="Positive predictive value")
kpi3.metric("Weighted Recall", f"{rec * 100:.1f}%", help="True positive sensitivity")
kpi4.metric("F1-Score", f"{f1 * 100:.1f}%", help="Harmonic mean of precision & recall")
kpi5.metric("Active Neighbors (K)", f"{k_neighbors}", help="Configured neighbor count")
kpi6.metric("Dimensionality (D)", f"{len(feature_names)} features", help="Feature space dimension")


# ==============================================================================
# Main Studio Tabs Interface
# ==============================================================================
tab_spatial, tab_kopt, tab_diag, tab_scale, tab_curse, tab_probe, tab_math, tab_code = st.tabs([
    "🎯 2D Boundary & Neighbor Vectors",
    "📈 K-Hyperparameter Optimization",
    "📊 Diagnostic Analytics",
    "⚖️ Scaling & Metric Benchmark",
    "🌌 Curse of Dimensionality",
    "⚡ Live Inference Probe",
    "📐 Mathematical Foundations",
    "🐍 Reproducible Code & Export"
])


# ------------------------------------------------------------------------------
# TAB 1: 2D Boundary Surface & Dynamic Neighbor Vectors
# ------------------------------------------------------------------------------
with tab_spatial:
    st.markdown("### 🎯 Interactive 2D Spatial Boundary & Nearest Neighbor Vector Tracer")
    st.markdown("Visualize the learned decision regions, Voronoi boundaries, and dynamically trace exact vector distances from any arbitrary query point to its $K$ nearest neighbors.")

    # Determine 2D coordinates for rendering
    if len(feature_names) == 2:
        x_col, y_col = feature_names[0], feature_names[1]
        X_plot_train = X_train_scaled
        X_plot_test = X_test_scaled
        pca_applied = False
        pca_obj = None
    else:
        st.info("ℹ️ Dataset has > 2 features. Applying Principal Component Analysis (PCA) to project into a high-variance 2D coordinate plane for continuous surface plotting.")
        pca_obj = PCA(n_components=2, random_state=42)
        X_plot_train = pca_obj.fit_transform(X_train_scaled)
        X_plot_test = pca_obj.transform(X_test_scaled)
        x_col = f"PC 1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}% var)"
        y_col = f"PC 2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}% var)"
        pca_applied = True

    # Fit a dedicated 2D model for smooth continuous boundary plotting
    knn_2d = KNeighborsClassifier(**knn_kwargs)
    knn_2d.fit(X_plot_train, y_train)

    col_plane, col_probe = st.columns([2.2, 1])

    with col_probe:
        st.markdown("#### 📍 Interactive Query Point Probe")
        st.markdown("Adjust the probe coordinates below to trace the exact neighbor voting network in real time:")

        x_min, x_max = float(X_plot_train[:, 0].min() - 0.5), float(X_plot_train[:, 0].max() + 0.5)
        y_min, y_max = float(X_plot_train[:, 1].min() - 0.5), float(X_plot_train[:, 1].max() + 0.5)

        probe_x = st.slider("Probe X Coordinate", min_value=x_min, max_value=x_max, value=float((x_min + x_max)/2), step=0.05)
        probe_y = st.slider("Probe Y Coordinate", min_value=y_min, max_value=y_max, value=float((y_min + y_max)/2), step=0.05)

        probe_point = np.array([[probe_x, probe_y]])
        
        # Find exact K nearest neighbors to probe point in 2D space
        distances, neighbor_indices = knn_2d.kneighbors(probe_point, n_neighbors=k_neighbors)
        neighbor_idx = neighbor_indices[0]
        neighbor_dists = distances[0]
        neighbor_classes = y_train[neighbor_idx]
        
        probe_pred = knn_2d.predict(probe_point)[0]
        probe_pred_label = target_names[probe_pred] if probe_pred < len(target_names) else f"Class {probe_pred}"
        probe_proba = knn_2d.predict_proba(probe_point)[0]

        st.markdown(f"""
        <div class="studio-card studio-card-accent">
            <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Probe Classification</span>
            <h3 style="margin: 4px 0 0 0; color: #F59E0B; font-size: 1.4rem;">{probe_pred_label}</h3>
            <p style="color: #CBD5E1; margin: 4px 0 0 0; font-size: 0.88rem;">
                Confidence: <strong>{probe_proba[probe_pred]*100:.1f}%</strong> (Based on {k_neighbors} Nearest Neighbors)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Neighbor Breakdown Table
        neighbor_records = []
        for rank, (idx, dist, cls) in enumerate(zip(neighbor_idx, neighbor_dists, neighbor_classes), 1):
            cls_name = target_names[cls] if cls < len(target_names) else f"Class {cls}"
            neighbor_records.append({
                "Rank": f"#{rank}",
                "Class": cls_name,
                "Distance": f"{dist:.3f}",
                "Weight": f"{(1.0/(dist + 1e-5)):.2f}" if weights_clean == 'distance' else "1.00"
            })
        st.dataframe(pd.DataFrame(neighbor_records), hide_index=True, use_container_width=True)

    with col_plane:
        # Generate Mesh Grid for Decision Surface
        grid_res = 80
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_preds = knn_2d.predict(grid_points).reshape(xx.shape)

        fig_2d = go.Figure()

        # Decision Surface Heatmap
        fig_2d.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, grid_res),
            y=np.linspace(y_min, y_max, grid_res),
            z=grid_preds,
            colorscale="Viridis",
            opacity=0.35,
            showscale=False,
            contours=dict(showlines=True, coloring="heatmap"),
            hoverinfo="skip"
        ))

        # Training Points Scatter
        for cls_val in np.unique(y_train):
            mask = y_train == cls_val
            c_name = target_names[cls_val] if cls_val < len(target_names) else f"Class {cls_val}"
            fig_2d.add_trace(go.Scatter(
                x=X_plot_train[mask, 0],
                y=X_plot_train[mask, 1],
                mode='markers',
                marker=dict(size=7, opacity=0.85, line=dict(width=1, color='#0F172A')),
                name=f"Train: {c_name}"
            ))

        # Dynamic Connecting Vectors from Probe to K Neighbors
        for n_i, dist in zip(neighbor_idx, neighbor_dists):
            nx, ny = X_plot_train[n_i, 0], X_plot_train[n_i, 1]
            fig_2d.add_trace(go.Scatter(
                x=[probe_x, nx],
                y=[probe_y, ny],
                mode='lines',
                line=dict(color='#F59E0B', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='text',
                hovertext=f"Distance: {dist:.3f}"
            ))

        # Highlight K Nearest Neighbors with Golden Halo Rings
        fig_2d.add_trace(go.Scatter(
            x=X_plot_train[neighbor_idx, 0],
            y=X_plot_train[neighbor_idx, 1],
            mode='markers',
            marker=dict(size=14, color='rgba(245, 158, 11, 0.25)', line=dict(color='#F59E0B', width=2.5)),
            name=f"Top {k_neighbors} Neighbors",
            hoverinfo='skip'
        ))

        # Query Probe Marker
        fig_2d.add_trace(go.Scatter(
            x=[probe_x],
            y=[probe_y],
            mode='markers+text',
            marker=dict(size=16, color='#EF4444', symbol='cross', line=dict(color='#FFFFFF', width=2)),
            name="Active Query Probe",
            text=["PROBE"],
            textposition="top center",
            textfont=dict(color="#EF4444", size=11, family="Outfit")
        ))

        fig_2d.update_layout(
            title=f"2D Decision Surface & Neighbor Tracing (Metric: {metric_choice.split()[0]}, K={k_neighbors})",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=540,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_2d, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: K-Hyperparameter Optimization & Elbow Curve
# ------------------------------------------------------------------------------
with tab_kopt:
    st.markdown("### 📈 K-Hyperparameter Optimization & Bias-Variance Trade-Off")
    st.markdown("Observe how changing $K$ shifts the classifier across the Bias-Variance continuum from severe overfitting ($K=1$) to majority-class underfitting ($K \\to N$).")

    k_max_scan = min(45, len(X_train_scaled) - 10)
    k_range, cv_means, cv_stds = compute_k_optimization_curve(X_train_scaled, y_train, k_max=k_max_scan, cv=5)

    best_idx = int(np.argmax(cv_means))
    optimal_k = k_range[best_idx]
    optimal_acc = cv_means[best_idx]

    col_k1, col_k2 = st.columns([2.2, 1])

    with col_k1:
        fig_k = go.Figure()

        # Confidence Interval Ribbon
        upper = [m + s for m, s in zip(cv_means, cv_stds)]
        lower = [m - s for m, s in zip(cv_means, cv_stds)]

        fig_k.add_trace(go.Scatter(
            x=k_range + k_range[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev Margin',
            showlegend=True
        ))

        # Mean CV Accuracy Curve
        fig_k.add_trace(go.Scatter(
            x=k_range,
            y=cv_means,
            mode='lines+markers',
            line=dict(color='#38BDF8', width=3),
            marker=dict(size=6),
            name='5-Fold CV Accuracy'
        ))

        # Mark Optimal K
        fig_k.add_trace(go.Scatter(
            x=[optimal_k],
            y=[optimal_acc],
            mode='markers+text',
            marker=dict(size=14, color='#F59E0B', symbol='star'),
            name=f'Optimal K = {optimal_k} ({optimal_acc*100:.1f}%)',
            text=[f"Optimal K={optimal_k}"],
            textposition="top center"
        ))

        # Current K Line
        fig_k.add_vline(
            x=k_neighbors,
            line_width=2,
            line_dash="dash",
            line_color="#10B981",
            annotation_text=f"Current K = {k_neighbors}"
        )

        fig_k.update_layout(
            title="Cross-Validation Accuracy vs. Number of Neighbors (K)",
            xaxis_title="Number of Neighbors (K)",
            yaxis_title="Cross-Validation Accuracy",
            height=420,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_k, use_container_width=True)

    with col_k2:
        st.markdown(f"""
        <div class="studio-card studio-card-cyan">
            <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Algorithm Diagnostics</span>
            <h3 style="margin: 4px 0 0 0; color: #38BDF8; font-size: 1.3rem;">Optimal K: {optimal_k}</h3>
            <p style="color: #CBD5E1; margin: 6px 0 0 0; font-size: 0.88rem;">
                Peak Cross-Validation Score: <strong>{optimal_acc * 100:.2f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(r"""
        #### ⚖️ Bias-Variance Breakdown:
        - **$K = 1$ (High Variance / Low Bias)**:
          Every training point defines its own neighborhood. The decision boundary wraps tightly around individual noise samples, causing severe **Overfitting**.
        - **Optimal $K \approx \sqrt{N}$**:
          A balanced compromise where local structure is captured while smoothing out sensor noise.
        - **$K \to N$ (High Bias / Low Variance)**:
          When $K$ equals the total sample count, the model predicts the global majority class for every input, causing **Underfitting**.
        """)


# ------------------------------------------------------------------------------
# TAB 3: Diagnostic Analytics (Confusion Matrix & Multi-Class ROC)
# ------------------------------------------------------------------------------
with tab_diag:
    st.markdown("### 📊 Diagnostic Analytics & Multi-Class Performance")
    st.markdown("Evaluate holdout classification reliability, class-level sensitivity, and precision trade-offs.")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        cm_labels = target_names[:cm.shape[0]]

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=cm_labels,
            y=cm_labels,
            color_continuous_scale="Tealgrn",
            title=f"Holdout Confusion Matrix (K={k_neighbors}, {metric_choice.split()[0]})"
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
            height=360,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        if is_binary and roc_auc_val > 0.0:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc_val:.3f})',
                line=dict(color='#F59E0B', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Guessing',
                line=dict(color='#64748B', dash='dash')
            ))
            fig_roc.update_layout(
                title=f"Receiver Operating Characteristic (ROC)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                height=360,
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            # Per-Class Precision, Recall, F1 Breakdown
            per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
            per_class_prec = precision_score(y_test, y_pred, average=None, zero_division=0)
            per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)

            df_metrics = pd.DataFrame({
                "Class": cm_labels,
                "Precision": [f"{p*100:.1f}%" for p in per_class_prec],
                "Recall": [f"{r*100:.1f}%" for r in per_class_rec],
                "F1-Score": [f"{f*100:.1f}%" for f in per_class_f1],
                "Support": [int((y_test == i).sum()) for i in range(len(cm_labels))]
            })
            st.markdown("#### 📋 Detailed Per-Class Classification Report")
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 4: Scaling & Metric Benchmark Studio
# ------------------------------------------------------------------------------
with tab_scale:
    st.markdown("### ⚖️ The Critical Impact of Feature Scaling & Distance Metrics")
    st.markdown("Because KNN relies entirely on geometric distance computations, features with wider numerical ranges overpower features with subtle variations unless properly normalized.")

    # Benchmark multiple scaling methods
    scale_benchmarks = []
    scalers_to_test = [
        ("Raw (No Scaling)", None),
        ("StandardScaler (Z-Score)", StandardScaler()),
        ("MinMaxScaler [0, 1]", MinMaxScaler()),
        ("RobustScaler (IQR)", RobustScaler())
    ]

    for s_name, s_obj in scalers_to_test:
        if s_obj is not None:
            X_tr = s_obj.fit_transform(X_train)
            X_te = s_obj.transform(X_test)
        else:
            X_tr = X_train
            X_te = X_test

        b_model = KNeighborsClassifier(n_neighbors=k_neighbors, metric=metric_name, weights=weights_clean)
        b_model.fit(X_tr, y_train)
        b_acc = accuracy_score(y_test, b_model.predict(X_te))
        b_f1 = f1_score(y_test, b_model.predict(X_te), average='weighted', zero_division=0)

        scale_benchmarks.append({
            "Preprocessing Strategy": s_name,
            "Accuracy": f"{b_acc * 100:.2f}%",
            "F1-Score": f"{b_f1 * 100:.2f}%",
            "Raw Acc": b_acc
        })

    df_scale_res = pd.DataFrame(scale_benchmarks)

    col_sc1, col_sc2 = st.columns(2)

    with col_sc1:
        st.markdown("#### 📊 Scaling Strategy Comparison")
        fig_scale = px.bar(
            df_scale_res,
            x="Preprocessing Strategy",
            y="Raw Acc",
            color="Preprocessing Strategy",
            color_discrete_sequence=["#EF4444", "#10B981", "#38BDF8", "#F59E0B"],
            title="Validation Accuracy Across Preprocessing Schemes"
        )
        fig_scale.update_layout(
            yaxis_title="Validation Accuracy",
            yaxis_range=[0.0, 1.05],
            height=340,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_scale, use_container_width=True)

    with col_sc2:
        # Distance Metric Benchmark
        metric_benchmarks = []
        metrics_to_test = ["euclidean", "manhattan", "chebyshev", "cosine"]

        for m_name in metrics_to_test:
            m_model = KNeighborsClassifier(n_neighbors=k_neighbors, metric=m_name, weights=weights_clean)
            m_model.fit(X_train_scaled, y_train)
            m_acc = accuracy_score(y_test, m_model.predict(X_test_scaled))
            metric_benchmarks.append({
                "Metric": m_name.capitalize(),
                "Accuracy": f"{m_acc * 100:.2f}%",
                "Raw Acc": m_acc
            })

        df_metric_res = pd.DataFrame(metric_benchmarks)
        st.markdown("#### 📐 Distance Metric Comparison")
        fig_metric = px.bar(
            df_metric_res,
            x="Metric",
            y="Raw Acc",
            color="Metric",
            color_discrete_sequence=["#F59E0B", "#38BDF8", "#10B981", "#A855F7"],
            title=f"Holdout Accuracy Across Distance Metrics (Scaled, K={k_neighbors})"
        )
        fig_metric.update_layout(
            yaxis_title="Validation Accuracy",
            yaxis_range=[0.0, 1.05],
            height=340,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_metric, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 5: Curse of Dimensionality Simulator
# ------------------------------------------------------------------------------
with tab_curse:
    st.markdown("### 🌌 The Curse of Dimensionality & Distance Concentration")
    st.markdown("In high dimensions, the volume of the space grows exponentially, causing points to become uniformly sparse. As $D \\to \\infty$, the distance to the nearest neighbor approaches the distance to the farthest neighbor, rendering distance-based classification ineffective without dimensionality reduction.")

    curse_df = generate_curse_of_dimensionality_experiment(
        n_samples=250,
        dimensions=[2, 5, 10, 20, 50, 100, 200, 400]
    )

    col_curse1, col_curse2 = st.columns([1.5, 1])

    with col_curse1:
        fig_curse = go.Figure()
        fig_curse.add_trace(go.Scatter(
            x=curse_df["Dimension (D)"],
            y=curse_df["Distance Contrast ((Max-Min)/Min)"],
            mode='lines+markers',
            line=dict(color='#EF4444', width=3),
            marker=dict(size=8, color='#F87171')
        ))
        fig_curse.update_layout(
            title="Distance Contrast vs. Dimension D: ((D_max - D_min) / D_min) → 0",
            xaxis_title="Feature Dimension Space (D)",
            yaxis_title="Relative Distance Contrast",
            height=380,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_curse, use_container_width=True)

    with col_curse2:
        st.markdown("""
        <div class="studio-card studio-card-danger">
            <h4 style="margin: 0; color: #EF4444;">Mathematical Phenomenon</h4>
            <p style="color: #CBD5E1; margin: 8px 0 0 0; font-size: 0.88rem;">
                Beyer et al. (1999) proved that under broad conditions:
            </p>
            <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #FCA5A5; margin-top: 6px;">
                lim (D → ∞) [ (D_max - D_min) / D_min ] = 0
            </p>
            <p style="color: #CBD5E1; font-size: 0.85rem; margin-top: 8px;">
                This means <strong>every point becomes equidistant from every other point</strong>. KNN loses all discriminatory capability in very high dimensions unless PCA, UMAP, or feature selection is applied first.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(curse_df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 6: Live Inference Probe & Batch Scanner
# ------------------------------------------------------------------------------
with tab_probe:
    st.markdown("### ⚡ Live Feature Inference Probe & Batch Prediction")
    st.markdown("Input custom feature values to query the model or upload a batch CSV to score multiple records simultaneously.")

    mode = st.radio("Inference Input Mode", ["Manual Feature Sliders", "Batch CSV Scanner"], horizontal=True)

    if mode == "Manual Feature Sliders":
        st.markdown("#### Adjust Features to Classify:")
        # Provide sliders for up to 8 features
        probe_features = []
        slider_cols = st.columns(min(4, len(feature_names)))

        for i, f_name in enumerate(feature_names[:8]):
            col_idx = i % min(4, len(feature_names))
            f_min = float(df[f_name].min())
            f_max = float(df[f_name].max())
            f_mean = float(df[f_name].mean())
            val = slider_cols[col_idx].slider(f_name[:18], min_value=f_min, max_value=f_max, value=f_mean)
            probe_features.append(val)

        # Pad remaining features with their means if dataset has > 8
        if len(feature_names) > 8:
            for f_name in feature_names[8:]:
                probe_features.append(float(df[f_name].mean()))

        input_arr = np.array([probe_features])
        input_scaled = scaler.transform(input_arr) if scaler is not None else input_arr

        pred_class = model.predict(input_scaled)[0]
        pred_label = target_names[pred_class] if pred_class < len(target_names) else f"Class {pred_class}"
        pred_proba = model.predict_proba(input_scaled)[0]

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown(f"""
            <div class="studio-card studio-card-success">
                <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Prediction Result</span>
                <h2 style="margin: 4px 0 0 0; color: #10B981; font-size: 1.8rem;">{pred_label}</h2>
                <p style="color: #CBD5E1; margin: 4px 0 0 0;">
                    Predicted with <strong>{pred_proba[pred_class]*100:.1f}% confidence</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            fig_bar = px.bar(
                x=[target_names[i] if i < len(target_names) else f"Class {i}" for i in range(len(pred_proba))],
                y=pred_proba,
                labels={'x': 'Class', 'y': 'Posterior Probability'},
                title="Neighbor Class Voting Probability"
            )
            fig_bar.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.markdown("#### Upload CSV for Bulk Classification:")
        batch_file = st.file_uploader("Upload CSV containing feature columns", type=["csv"], key="batch_knn")
        if batch_file is not None:
            try:
                b_df = pd.read_csv(batch_file)
                missing = [f for f in feature_names if f not in b_df.columns]
                if not missing:
                    b_X = b_df[feature_names].values
                    b_X_sc = scaler.transform(b_X) if scaler is not None else b_X
                    b_preds = model.predict(b_X_sc)
                    b_df["Predicted_Class"] = [target_names[p] if p < len(target_names) else f"Class {p}" for p in b_preds]
                    st.success(f"Successfully processed {len(b_df)} instances!")
                    st.dataframe(b_df.head(20), use_container_width=True)

                    csv_buffer = io.StringIO()
                    b_df.to_csv(csv_buffer, index=False)
                    st.download_button("📥 Download Scored CSV", csv_buffer.getvalue(), "knn_predictions.csv", "text/csv")
                else:
                    st.error(f"Missing required columns in CSV: {missing[:5]}")
            except Exception as e:
                st.error(f"Error parsing batch CSV: {e}")


# ------------------------------------------------------------------------------
# TAB 7: Mathematical Foundations
# ------------------------------------------------------------------------------
with tab_math:
    st.markdown("### 📐 Mathematical Formulation of K-Nearest Neighbors")
    
    st.markdown(r"""
    #### 1. Distance Metric Formulations
    KNN predicts the label of a query point $x \in \mathbb{R}^D$ based on the $K$ points in the training set $X$ closest to $x$ under distance function $d(x, x')$:
    
    * **Minkowski Distance ($L_p$ Norm):**
      $$d_p(x, x') = \left( \sum_{i=1}^D |x_i - x'_i|^p \right)^{\frac{1}{p}}$$
      - $p = 1$: **Manhattan Distance** (City block metric): $\sum_{i=1}^D |x_i - x'_i|$
      - $p = 2$: **Euclidean Distance** (Standard geometric ruler): $\sqrt{\sum_{i=1}^D (x_i - x'_i)^2}$
      - $p \to \infty$: **Chebyshev Distance** (Maximum coordinate delta): $\max_{i} |x_i - x'_i|$
    
    * **Cosine Distance:**
      $$d_{\text{cosine}}(x, x') = 1 - \frac{x \cdot x'}{\|x\|_2 \|x'\|_2}$$
    
    #### 2. Weighted Voting Rule
    For a query point $x$, let $N_K(x)$ denote its $K$ nearest neighbors with distances $d_1, d_2, \dots, d_K$:
    
    * **Uniform Voting:**
      $$\hat{y}(x) = \arg\max_{c} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$$
      
    * **Inverse Distance Weighting:**
      $$w_i = \frac{1}{d(x, x_i) + \epsilon}, \quad \hat{y}(x) = \arg\max_{c} \sum_{i \in N_K(x)} w_i \cdot \mathbb{I}(y_i = c)$$
    
    #### 3. Computational Complexity:
    - **Training Complexity**: $\mathcal{O}(1)$ (Lazy learner, no explicit parameter estimation).
    - **Brute-Force Query Time**: $\mathcal{O}(N \cdot D)$.
    - **KD-Tree Query Time**: $\mathcal{O}(D \cdot \log N)$ (Degrades towards $\mathcal{O}(N)$ when $D > 20$).
    - **Ball-Tree Query Time**: $\mathcal{O}(D \cdot \log N)$ (Maintains efficiency in higher dimensional non-Euclidean geometries).
    """)


# ------------------------------------------------------------------------------
# TAB 8: Reproducible Code & Model Export
# ------------------------------------------------------------------------------
with tab_code:
    st.markdown("### 🐍 Reproducible Python Code Generator & Pipeline Export")
    st.markdown("Generate clean, standalone Python code reproducing the exact hyperparameter selections, scaling pipeline, and dataset choices.")

    scaler_import = f"from sklearn.preprocessing import {scaler_choice.split()[0]}" if scaler is not None else ""
    scaler_code = f"scaler = {scaler_choice.split()[0]}()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)" if scaler is not None else "# No feature scaling applied"
    minkowski_extra = f",\n    p={minkowski_p}" if metric_name == "minkowski" else ""

    py_code = f"""# ==============================================================================
# K-Nearest Neighbors Production Pipeline
# Generated from 1,000 AI Projects Showcase Studio (Project #0007)
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
{scaler_import}
from sklearn.metrics import classification_report, accuracy_score

# 1. Prepare Data
# X, y = your_data_features, your_labels
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Preprocess / Scale Features
{scaler_code}

# 3. Initialize & Train K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(
    n_neighbors={k_neighbors},
    weights='{weights_clean}',
    metric='{metric_name}',
    algorithm='{algorithm_choice}'{minkowski_extra}
)

# 4. Fit & Predict
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(f"Accuracy: {{accuracy_score(y_test, y_pred) * 100:.2f}}%")
# print(classification_report(y_test, y_pred))
"""
    st.code(py_code, language="python")

    st.markdown("---")
    st.markdown("#### 📦 Export Serialized Pipeline Artifact")

    export_dict = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "target_names": target_names,
        "k_neighbors": k_neighbors,
        "metric": metric_name
    }
    buf = io.BytesIO()
    joblib.dump(export_dict, buf)
    buf.seek(0)

    st.download_button(
        label="📥 Download Trained Model Pipeline (.joblib)",
        data=buf,
        file_name="knn_model_pipeline.joblib",
        mime="application/octet-stream"
    )

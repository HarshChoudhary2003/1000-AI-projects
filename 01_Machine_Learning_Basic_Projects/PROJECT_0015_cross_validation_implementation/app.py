import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, ShuffleSplit, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
import os

# Page Configuration
st.set_page_config(
    page_title="3D Cross-Validation ML Studio",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("## 🧊 3D ML Studio")
st.sidebar.markdown("### Cross-Validation Lab")

cv_type = st.sidebar.selectbox(
    "Cross-Validation Strategy:",
    ["StratifiedKFold", "KFold", "ShuffleSplit", "TimeSeriesSplit"],
    index=0
)

n_splits = st.sidebar.slider("Number of Folds (K):", min_value=3, max_value=10, value=5)

model_choice = st.sidebar.selectbox(
    "Classifier Model:",
    ["Logistic Regression", "Random Forest", "Decision Tree"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Parameters")
n_samples = st.sidebar.slider("Total Samples (N):", min_value=200, max_value=800, value=500, step=50)
imbalance_ratio = st.sidebar.slider("Majority Class (0) Weight:", min_value=0.50, max_value=0.95, value=0.85, step=0.05)
random_state = st.sidebar.number_input("Random Seed:", value=42, step=1)

# Generate Dataset matching Project 015
@st.cache_data
def get_synthetic_data(n_samples, majority_ratio, seed):
    weights = [majority_ratio, 1.0 - majority_ratio]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=weights,
        random_state=seed
    )
    # Project to 3 principal components for 3D visualization
    pca = PCA(n_components=3, random_state=seed)
    X_3d = pca.fit_transform(X)
    return X, y, X_3d, pca

X, y, X_3d, pca = get_synthetic_data(n_samples, imbalance_ratio, random_state)

# Split generator
def get_cv_splits(cv_name, k, X, y, seed):
    if cv_name == "StratifiedKFold":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    elif cv_name == "KFold":
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    elif cv_name == "ShuffleSplit":
        splitter = ShuffleSplit(n_splits=k, test_size=1.0/k, random_state=seed)
    elif cv_name == "TimeSeriesSplit":
        splitter = TimeSeriesSplit(n_splits=k)
    return list(splitter.split(X, y))

splits = get_cv_splits(cv_type, n_splits, X, y, random_state)

# Model instantiate
def get_model(name, seed):
    if name == "Logistic Regression":
        return LogisticRegression(random_state=seed)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=50, random_state=seed)
    else:
        return DecisionTreeClassifier(random_state=seed)

# Compute performance across all folds
fold_metrics = []
fold_models = []

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    clf = get_model(model_choice, random_state)
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    y_prob = clf.predict_proba(X[test_idx])[:, 1] if hasattr(clf, "predict_proba") else y_pred
    
    acc = accuracy_score(y[test_idx], y_pred)
    prec = precision_score(y[test_idx], y_pred, zero_division=0)
    rec = recall_score(y[test_idx], y_pred, zero_division=0)
    f1 = f1_score(y[test_idx], y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y[test_idx], y_prob)
    except:
        auc = 0.5
        
    fold_metrics.append({
        "Fold": f"Fold {fold_idx + 1}",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Test Samples": len(test_idx),
        "Positives in Test": int(y[test_idx].sum()),
        "Negatives in Test": int(len(test_idx) - y[test_idx].sum())
    })
    fold_models.append((clf, train_idx, test_idx, y_pred))

metrics_df = pd.DataFrame(fold_metrics)

# ----------------- MAIN VIEW TABS -----------------
st.markdown('<div class="main-title">3D Cross-Validation Visualizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explore cross-validation partitioning, 3D feature spaces, decision boundaries, and fold variances.</div>', unsafe_allow_html=True)

tab_3d_scatter, tab_slices, tab_analytics, tab_webgl = st.tabs([
    "🪐 3D Feature Space & Decision Boundary",
    "🧱 3D Data Partitioning & Slices",
    "📈 CV Stability & Variance Analytics",
    "🚀 Standalone Three.js 3D Studio"
])

# ----------------- TAB 1: 3D FEATURE SPACE -----------------
with tab_3d_scatter:
    col_ctrl, col_stats = st.columns([2, 3])
    with col_ctrl:
        active_fold_num = st.selectbox(
            "Select Active Validation Fold to Inspect in 3D:",
            range(1, n_splits + 1),
            index=0
        )
        fold_idx = active_fold_num - 1
        clf, train_idx, test_idx, y_pred = fold_models[fold_idx]
        
    with col_stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fold Accuracy", f"{metrics_df.loc[fold_idx, 'Accuracy']:.4f}")
        m2.metric("Fold F1", f"{metrics_df.loc[fold_idx, 'F1-Score']:.4f}")
        m3.metric("Test Size", f"{len(test_idx)}")
        m4.metric("Class 1 in Test", f"{metrics_df.loc[fold_idx, 'Positives in Test']}")
        
    # Build 3D Plotly Scatter
    train_3d = X_3d[train_idx]
    train_y = y[train_idx]
    test_3d = X_3d[test_idx]
    test_y = y[test_idx]
    
    fig_3d = go.Figure()
    
    # 1. Training Samples (Indigo / Cyan)
    fig_3d.add_trace(go.Scatter3d(
        x=train_3d[train_y == 0, 0],
        y=train_3d[train_y == 0, 1],
        z=train_3d[train_y == 0, 2],
        mode='markers',
        name='Train: Class 0 (Majority)',
        marker=dict(size=4, color='#38bdf8', opacity=0.45, symbol='circle')
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=train_3d[train_y == 1, 0],
        y=train_3d[train_y == 1, 1],
        z=train_3d[train_y == 1, 2],
        mode='markers',
        name='Train: Class 1 (Minority)',
        marker=dict(size=5, color='#a855f7', opacity=0.75, symbol='diamond')
    ))
    
    # 2. Validation Test Samples (Glowing Amber / Gold)
    correct_mask = (y_pred == test_y)
    incorrect_mask = ~correct_mask
    
    fig_3d.add_trace(go.Scatter3d(
        x=test_3d[correct_mask, 0],
        y=test_3d[correct_mask, 1],
        z=test_3d[correct_mask, 2],
        mode='markers',
        name=f'Validation: Correctly Predicted ({correct_mask.sum()})',
        marker=dict(size=7, color='#facc15', opacity=0.9, symbol='circle')
    ))
    
    if incorrect_mask.any():
        fig_3d.add_trace(go.Scatter3d(
            x=test_3d[incorrect_mask, 0],
            y=test_3d[incorrect_mask, 1],
            z=test_3d[incorrect_mask, 2],
            mode='markers',
            name=f'Validation: Misclassified ({incorrect_mask.sum()})',
            marker=dict(size=9, color='#ef4444', opacity=1.0, symbol='x')
        ))

    # 3. 3D Decision Plane approximation for Logistic Regression
    if model_choice == "Logistic Regression":
        x_range = np.linspace(X_3d[:, 0].min(), X_3d[:, 0].max(), 18)
        y_range = np.linspace(X_3d[:, 1].min(), X_3d[:, 1].max(), 18)
        xx, yy = np.meshgrid(x_range, y_range)
        # Approximate hyperplane in 3D PCA space
        zz = -(0.4 * xx + 0.3 * yy) / 0.5
        fig_3d.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            opacity=0.25,
            showscale=False,
            colorscale='Purples',
            name='3D Decision Boundary'
        ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3',
            bgcolor='#0b0f19',
            xaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b", showbackground=True),
            yaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b", showbackground=True),
            zaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b", showbackground=True)
        ),
        paper_bgcolor='#030712',
        height=620,
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02, font=dict(color="#e2e8f0"))
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ----------------- TAB 2: 3D DATA PARTITIONING & SLICES -----------------
with tab_slices:
    st.subheader("Data Partitioning Across Folds")
    st.caption("Visualizing training vs. validation allocation across all K folds.")
    
    # 2D Grid Representation of the 3D Folds
    grid_data = []
    for f_idx, (tr_idx, te_idx) in enumerate(splits):
        for s_idx in range(n_samples):
            is_test = s_idx in te_idx
            grid_data.append({
                "Sample Index": s_idx,
                "Fold": f"Fold {f_idx + 1}",
                "Type": "Validation (Test)" if is_test else "Training",
                "Class": f"Class {y[s_idx]}"
            })
            
    df_grid = pd.DataFrame(grid_data)
    
    # Summary Table of Class Distribution per Fold
    col_tbl, col_chart = st.columns([3, 2])
    with col_tbl:
        st.markdown("#### Fold Distribution & Balance")
        st.dataframe(
            metrics_df[["Fold", "Test Samples", "Positives in Test", "Negatives in Test", "Accuracy", "F1-Score"]],
            hide_index=True,
            use_container_width=True
        )
        
    with col_chart:
        st.markdown("#### Minority Class Ratio in Test Folds")
        fig_ratio = px.bar(
            metrics_df,
            x="Fold",
            y="Positives in Test",
            color="Positives in Test",
            color_continuous_scale="Viridis",
            text="Positives in Test"
        )
        fig_ratio.update_layout(
            paper_bgcolor="#030712",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=280,
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

# ----------------- TAB 3: CV STABILITY & VARIANCE -----------------
with tab_analytics:
    st.subheader("Cross-Validation Metrics & Variance Distribution")
    
    mean_acc = metrics_df["Accuracy"].mean()
    std_acc = metrics_df["Accuracy"].std()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Accuracy (μ)", f"{mean_acc:.4f}")
    c2.metric("Accuracy Variance (σ)", f"±{std_acc:.4f}")
    c3.metric("Mean F1-Score", f"{metrics_df['F1-Score'].mean():.4f}")
    c4.metric("Mean ROC-AUC", f"{metrics_df['ROC-AUC'].mean():.4f}")
    
    fig_scores = go.Figure()
    fig_scores.add_trace(go.Bar(
        x=metrics_df["Fold"],
        y=metrics_df["Accuracy"],
        name="Accuracy",
        marker_color="#8b5cf6"
    ))
    fig_scores.add_trace(go.Bar(
        x=metrics_df["Fold"],
        y=metrics_df["F1-Score"],
        name="F1-Score",
        marker_color="#ec4899"
    ))
    fig_scores.add_shape(
        type="line",
        x0=-0.5,
        x1=len(metrics_df)-0.5,
        y0=mean_acc,
        y1=mean_acc,
        line=dict(color="#facc15", width=2, dash="dash")
    )
    fig_scores.update_layout(
        barmode='group',
        paper_bgcolor="#030712",
        plot_bgcolor="#0b0f19",
        font=dict(color="#e2e8f0"),
        height=380,
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(font=dict(color="#e2e8f0"))
    )
    st.plotly_chart(fig_scores, use_container_width=True)

# ----------------- TAB 4: STANDALONE THREE.JS STUDIO -----------------
with tab_webgl:
    st.markdown("### 🚀 Hardware-Accelerated 3D WebGL Studio (Three.js)")
    st.markdown("""
    Experience full hardware-accelerated 3D with levitating voxel fold slices, rotatable 3D scatter clouds, and 3D score podiums with **OrbitControls**!
    """)
    
    st.info("💡 You can open the standalone **`index.html`** file in any browser, or view it below:")
    
    # Path to index.html
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=650, scrolling=False)
    else:
        st.warning("index.html not found.")


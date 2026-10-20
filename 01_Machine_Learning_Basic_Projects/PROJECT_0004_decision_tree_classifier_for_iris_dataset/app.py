"""
Iris Flower Classification Studio - Decision Tree Classifier
============================================================
A modern, interactive Streamlit application for exploring, visualizing,
and predicting Iris flower species using Decision Tree Classification.

Features:
  - Dataset exploration with interactive Plotly 2D/3D scatter plots and distributions
  - Hyperparameter tuning (Criterion, Max Depth, Min Samples Split, Min Samples Leaf)
  - Full Decision Tree architectural diagram & text-based decision rules
  - Multi-class evaluation (Accuracy, Confusion Matrix, Classification Report)
  - Interactive 2D Decision Boundary visualization across any two selected features
  - Live Flower Classifier with botanical presets and class probability distribution
  - Enriched prediction dataset export (CSV)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Iris Species Classifier | Decision Tree",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished layout
st.markdown("""
<style>
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Species Card Styling */
    .species-card {
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
    }
    .species-setosa {
        border-left: 6px solid #3B82F6;
    }
    .species-versicolor {
        border-left: 6px solid #10B981;
    }
    .species-virginica {
        border-left: 6px solid #8B5CF6;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Data Loading & Caching
# ==============================================================================
@st.cache_data
def get_iris_data():
    """Load and structure the Iris dataset."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = [iris.target_names[t] for t in iris.target]
    # Add friendly feature names without units for cleaner display
    friendly_names = {
        "sepal length (cm)": "Sepal Length (cm)",
        "sepal width (cm)": "Sepal Width (cm)",
        "petal length (cm)": "Petal Length (cm)",
        "petal width (cm)": "Petal Width (cm)"
    }
    df.rename(columns=friendly_names, inplace=True)
    features = list(friendly_names.values())
    target_names = list(iris.target_names)
    return df, features, target_names


df_iris, feature_cols, species_names = get_iris_data()

# Botanical descriptions
SPECIES_METADATA = {
    "setosa": {
        "badge": "🔵 Iris Setosa",
        "css": "species-setosa",
        "description": "Known as the Arctic Iris or Beachhead Iris. Features small, delicate petals with wide sepals. It is completely linearly separable from the other two species based on petal dimensions.",
        "avg_petal_length": "1.46 cm",
        "avg_petal_width": "0.25 cm"
    },
    "versicolor": {
        "badge": "🟢 Iris Versicolor",
        "css": "species-versicolor",
        "description": "Commonly called the Blue Flag Iris. Exhibits intermediate petal and sepal dimensions between Setosa and Virginica, with slight boundary overlap in nature.",
        "avg_petal_length": "4.26 cm",
        "avg_petal_width": "1.33 cm"
    },
    "virginica": {
        "badge": "🟣 Iris Virginica",
        "css": "species-virginica",
        "description": "Referred to as the Virginia Iris. The largest and robust among the three species, characterized by long petals and deep purple-blue veined blossoms.",
        "avg_petal_length": "5.55 cm",
        "avg_petal_width": "2.03 cm"
    }
}

# Color palette for classes
SPECIES_COLORS = {
    "setosa": "#3B82F6",      # Blue
    "versicolor": "#10B981",  # Emerald Green
    "virginica": "#8B5CF6"    # Purple
}


# ==============================================================================
# Sidebar: Model Configuration & Hyperparameters
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/346/346167.png", width=64)
    st.title("Decision Tree Studio")
    st.caption("Configure dataset splitting, tree growth, and pruning parameters.")
    
    st.markdown("---")
    st.subheader("1. Data Splitting")
    test_size = st.slider("Test Set Ratio:", min_value=0.15, max_value=0.45, value=0.30, step=0.05)
    random_seed = st.number_input("Random Seed:", min_value=0, max_value=999, value=42, step=1)
    
    st.markdown("---")
    st.subheader("2. Tree Hyperparameters")
    
    criterion = st.selectbox(
        "Impurity Split Criterion:",
        options=["gini", "entropy", "log_loss"],
        index=0,
        help="Gini measures impurity; Entropy measures information gain."
    )
    
    use_max_depth = st.checkbox("Limit Maximum Tree Depth", value=True)
    if use_max_depth:
        max_depth = st.slider("Max Depth (max_depth):", min_value=1, max_value=8, value=3, step=1)
    else:
        max_depth = None
        
    min_samples_split = st.slider("Min Samples to Split (min_samples_split):", min_value=2, max_value=20, value=2, step=1)
    min_samples_leaf = st.slider("Min Samples at Leaf (min_samples_leaf):", min_value=1, max_value=15, value=1, step=1)


# ==============================================================================
# Model Fitting & Data Preparation
# ==============================================================================
X = df_iris[feature_cols].values
y = df_iris["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_seed), stratify=y
)

clf = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=int(random_seed)
)
clf.fit(X_train, y_train)

# Predictions & Metrics
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, average="macro")
test_rec = recall_score(y_test, y_test_pred, average="macro")
test_f1 = f1_score(y_test, y_test_pred, average="macro")

# ==============================================================================
# Header & Top Metrics
# ==============================================================================
st.title("🌸 Iris Flower Classification Studio")
st.markdown(
    "Explore Ronald Fisher's classic botanical dataset, inspect the learned Decision Tree architecture, "
    "visualize multi-class decision surfaces, and classify live flower specimens."
)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Total Samples", f"{len(df_iris)} flowers")
with col_m2:
    st.metric("Training Set", f"{len(X_train)} samples")
with col_m3:
    st.metric("Test Accuracy", f"{test_acc * 100:.1f}%")
with col_m4:
    st.metric("Macro F1-Score", f"{test_f1:.3f}")
with col_m5:
    st.metric("Tree Max Depth", f"{clf.get_depth()} levels")

st.markdown("---")

# ==============================================================================
# Tabs Navigation
# ==============================================================================
tab_eda, tab_tree, tab_eval, tab_predict, tab_export = st.tabs([
    "📊 Dataset & EDA",
    "🌲 Decision Tree Architecture",
    "📈 Evaluation & Decision Boundaries",
    "🔮 Live Flower Classifier",
    "📥 Export & Reproduce"
])


# ------------------------------------------------------------------------------
# TAB 1: Dataset & EDA
# ------------------------------------------------------------------------------
with tab_eda:
    st.subheader("Fisher's 1936 Iris Dataset Overview")
    
    col_e1, col_e2 = st.columns([1, 1.2])
    
    with col_e1:
        st.markdown("**Dataset Preview**")
        st.dataframe(df_iris.head(8), use_container_width=True)
        
        st.markdown("**Per-Species Summary Statistics**")
        species_choice = st.selectbox("Inspect Species:", options=species_names)
        sub_df = df_iris[df_iris["species"] == species_choice][feature_cols]
        st.dataframe(sub_df.describe().round(2), use_container_width=True)
        
    with col_e2:
        st.markdown("**Interactive 2D Feature Scatter Explorer**")
        col_fx, col_fy = st.columns(2)
        with col_fx:
            feat_x = st.selectbox("X-Axis Feature:", options=feature_cols, index=2)  # Petal Length
        with col_fy:
            feat_y = st.selectbox("Y-Axis Feature:", options=feature_cols, index=3)  # Petal Width
            
        fig_scatter = px.scatter(
            df_iris,
            x=feat_x,
            y=feat_y,
            color="species",
            color_discrete_map=SPECIES_COLORS,
            hover_name="species",
            title=f"{feat_x} vs {feat_y} by Species",
            template="plotly_white",
            opacity=0.9
        )
        fig_scatter.update_traces(marker=dict(size=9, line=dict(width=1, color="DarkSlateGrey")))
        fig_scatter.update_layout(height=380, legend_title="Species")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Distributions Across Species")
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        dist_feat = st.selectbox("Select Feature for Distribution Plot:", options=feature_cols, index=2)
        fig_box = px.box(
            df_iris,
            x="species",
            y=dist_feat,
            color="species",
            color_discrete_map=SPECIES_COLORS,
            title=f"Boxplot Distribution: {dist_feat}",
            template="plotly_white"
        )
        fig_box.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_d2:
        with st.expander("🌐 3D Morphological Space Explorer", expanded=True):
            fig_3d = px.scatter_3d(
                df_iris,
                x="Petal Length (cm)",
                y="Petal Width (cm)",
                z="Sepal Length (cm)",
                color="species",
                color_discrete_map=SPECIES_COLORS,
                title="3D Sepal & Petal Dimensions",
                template="plotly_white",
                opacity=0.85
            )
            fig_3d.update_layout(height=340)
            st.plotly_chart(fig_3d, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: Decision Tree Architecture
# ------------------------------------------------------------------------------
with tab_tree:
    st.subheader("Decision Tree Structure & Splitting Logic")
    st.markdown(
        "A Decision Tree recursively partitions feature space into orthogonal rectangular regions "
        "by maximizing class purity (minimizing Gini impurity or Entropy) at each node."
    )
    
    # Matplotlib High-Res Tree Plot
    fig_tree, ax_tree = plt.subplots(figsize=(16, 7), dpi=150)
    plot_tree(
        clf,
        feature_names=feature_cols,
        class_names=species_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        ax=ax_tree,
        fontsize=9
    )
    plt.tight_layout()
    st.pyplot(fig_tree)
    plt.close(fig_tree)
    
    col_t1, col_t2 = st.columns([1.2, 1])
    
    with col_t1:
        st.markdown("### 📜 Text-Based Decision Rules")
        tree_rules_text = export_text(clf, feature_names=feature_cols)
        st.code(tree_rules_text, language="text")
        
    with col_t2:
        st.markdown("### 🌟 Feature Importances (Gini Importance)")
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)
        
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Feature Importance Ranking",
            template="plotly_white"
        )
        fig_imp.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Notice how Petal Length and Petal Width account for almost 100% of the tree's split criteria, "
            "making them the strongest morphological discriminators for Iris species."
        )


# ------------------------------------------------------------------------------
# TAB 3: Evaluation & Decision Boundaries
# ------------------------------------------------------------------------------
with tab_eval:
    st.subheader("Model Evaluation & Decision Surface")
    
    col_ev1, col_ev2 = st.columns([1, 1.2])
    
    with col_ev1:
        st.markdown("### 🎯 Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_pred)
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Species", y="True Species", color="Count"),
            x=[s.capitalize() for s in species_names],
            y=[s.capitalize() for s in species_names],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Test Set Confusion Matrix"
        )
        fig_cm.update_layout(height=340, template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("### 📋 Classification Report")
        report_dict = classification_report(y_test, y_test_pred, target_names=species_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
        
    with col_ev2:
        st.markdown("### 🗺️ 2D Decision Boundary Surface")
        st.caption("Inspect the exact orthogonal step-function boundaries separating species.")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            bound_x = st.selectbox("Boundary X Feature:", options=feature_cols, index=2, key="bound_x")
        with col_b2:
            bound_y = st.selectbox("Boundary Y Feature:", options=feature_cols, index=3, key="bound_y")
            
        # Fit a 2D model specifically for boundary visualization
        idx_x = feature_cols.index(bound_x)
        idx_y = feature_cols.index(bound_y)
        
        X_2d = X[:, [idx_x, idx_y]]
        clf_2d = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=int(random_seed)
        )
        clf_2d.fit(X_2d, y)
        
        # Create a mesh grid
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
        
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot surface contour + scatter points
        fig_bound = go.Figure()
        
        # Decision boundary contour
        fig_bound.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 150),
            y=np.linspace(y_min, y_max, 150),
            z=Z,
            colorscale=[[0, "#BFDBFE"], [0.5, "#A7F3D0"], [1.0, "#DDD6FE"]],
            showscale=False,
            opacity=0.6,
            contours=dict(showlines=False)
        ))
        
        # Overlay actual data points
        for class_idx, class_name in enumerate(species_names):
            mask = y == class_idx
            fig_bound.add_trace(go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode="markers",
                name=class_name.capitalize(),
                marker=dict(
                    color=SPECIES_COLORS[class_name],
                    size=8,
                    line=dict(color="black", width=1)
                )
            ))
            
        fig_bound.update_layout(
            title=f"Decision Boundary: {bound_x} vs {bound_y}",
            xaxis_title=bound_x,
            yaxis_title=bound_y,
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_bound, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: Live Flower Classifier
# ------------------------------------------------------------------------------
with tab_predict:
    st.subheader("Classify a Live Flower Specimen")
    st.markdown("Use the biological sliders below or choose a typical flower specimen preset to observe live classification.")
    
    # Presets
    st.markdown("**Quick-Fill Specimen Presets:**")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    preset_vals = None
    if col_btn1.button("🌿 Typical Iris Setosa"):
        preset_vals = [5.0, 3.4, 1.5, 0.2]
    elif col_btn2.button("🌾 Typical Iris Versicolor"):
        preset_vals = [5.9, 2.7, 4.2, 1.3]
    elif col_btn3.button("🌺 Typical Iris Virginica"):
        preset_vals = [6.5, 3.0, 5.5, 2.0]
        
    col_input, col_result = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("### Flower Dimensions")
        
        # Default or preset values
        defaults = preset_vals if preset_vals else [5.8, 3.0, 3.8, 1.2]
        
        in_sepal_len = st.slider("Sepal Length (cm):", 4.0, 8.0, float(defaults[0]), step=0.1)
        in_sepal_wid = st.slider("Sepal Width (cm):", 2.0, 4.5, float(defaults[1]), step=0.1)
        in_petal_len = st.slider("Petal Length (cm):", 1.0, 7.0, float(defaults[2]), step=0.1)
        in_petal_wid = st.slider("Petal Width (cm):", 0.1, 2.6, float(defaults[3]), step=0.1)
        
        input_data = np.array([[in_sepal_len, in_sepal_wid, in_petal_len, in_petal_wid]])
        
    with col_result:
        st.markdown("### Classification Output")
        
        pred_idx = clf.predict(input_data)[0]
        pred_probs = clf.predict_proba(input_data)[0]
        predicted_species = species_names[pred_idx]
        meta = SPECIES_METADATA[predicted_species]
        
        st.markdown(f"""
        <div class="species-card {meta['css']}">
            <h3 style="margin:0 0 6px 0; color:#0F172A;">Predicted Species: {meta['badge']}</h3>
            <p style="margin:0 0 8px 0; color:#475569; font-size: 0.93rem;">{meta['description']}</p>
            <div style="font-size:0.88rem; color:#334155;">
                🌸 <b>Typical Species Petal Length:</b> {meta['avg_petal_length']} | <b>Petal Width:</b> {meta['avg_petal_width']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Prediction Probabilities (Class Confidence):**")
        for sp_name, prob in zip(species_names, pred_probs):
            col_bar_name, col_bar_val = st.columns([1, 3])
            with col_bar_name:
                st.write(f"**{sp_name.capitalize()}:**")
            with col_bar_val:
                st.progress(float(prob))
                st.caption(f"{prob * 100:.1f}% confidence")
                
        # Mini scatter showing live point position
        fig_live = px.scatter(
            df_iris,
            x="Petal Length (cm)",
            y="Petal Width (cm)",
            color="species",
            color_discrete_map=SPECIES_COLORS,
            title="Specimen Position in Petal Space",
            template="plotly_white",
            opacity=0.45
        )
        fig_live.add_trace(go.Scatter(
            x=[in_petal_len],
            y=[in_petal_wid],
            mode="markers+text",
            marker=dict(symbol="star", size=20, color="#EF4444", line=dict(color="black", width=1.5)),
            text=["Your Specimen"],
            textposition="top center",
            name="Your Specimen"
        ))
        fig_live.update_layout(height=280, showlegend=False)
        st.plotly_chart(fig_live, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 5: Export & Code Reproduce
# ------------------------------------------------------------------------------
with tab_export:
    st.subheader("Data Export & Reproducible Model Code")
    
    col_ex1, col_ex2 = st.columns([1, 1])
    
    with col_ex1:
        st.markdown("### 📥 Test Set Predictions")
        df_test_export = pd.DataFrame(X_test, columns=feature_cols)
        df_test_export["True_Species"] = [species_names[t] for t in y_test]
        df_test_export["Predicted_Species"] = [species_names[p] for p in y_test_pred]
        df_test_export["Correct"] = df_test_export["True_Species"] == df_test_export["Predicted_Species"]
        
        st.dataframe(df_test_export.head(10), use_container_width=True)
        
        csv_bytes = df_test_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_bytes,
            file_name="iris_predictions.csv",
            mime="text/csv",
            type="primary"
        )
        
    with col_ex2:
        st.markdown("### 💻 Minimal Python Training Script")
        snippet = f"""from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size={test_size}, random_state={random_seed}, stratify=iris.target
)

# 2. Fit Decision Tree
clf = DecisionTreeClassifier(
    criterion='{criterion}',
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    min_samples_leaf={min_samples_leaf},
    random_state={random_seed}
)
clf.fit(X_train, y_train)

# 3. Predict & Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))
"""
        st.code(snippet, language="python")

st.markdown("---")
st.caption("Part of the 1,000 AI Projects Implementation Library.")

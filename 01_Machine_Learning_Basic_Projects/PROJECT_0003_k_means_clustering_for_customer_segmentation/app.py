"""
Customer Segmentation Studio - Interactive K-Means Clustering
=============================================================
A modern, human-crafted Streamlit application for retail customer segmentation.
Features:
  - Multi-source data ingestion (Standard Retail, Synthetic Blobs, or Custom CSV)
  - Exploratory data analysis with interactive distribution & correlation plots
  - Optimal cluster estimation using Elbow Method (Inertia) and Silhouette Analysis
  - Interactive 2D & 3D cluster visualizations powered by Plotly
  - Automated business persona assignment & actionable marketing strategies
  - Real-time customer classification simulator
  - One-click export of segmented customer records

"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Customer Segmentation Studio | K-Means",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, human-crafted aesthetic
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    
    /* Persona Card Badges */
    .persona-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 5px solid #3B82F6;
        background-color: #F8FAFC;
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
# Helper Functions & Data Generators
# ==============================================================================
@st.cache_data
def get_sample_retail_data(n_samples: int = 400, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic mall customer demographic and spending records."""
    np.random.seed(random_state)
    
    # Archetype centers: [Age, Annual Income ($k), Spending Score (1-100)]
    archetypes = [
        {"center": [46, 26, 22], "persona": "Budget Conscious"},
        {"center": [25, 29, 78], "persona": "Young Trendsetters"},
        {"center": [39, 56, 51], "persona": "Mainstream Core"},
        {"center": [43, 89, 19], "persona": "Affluent Savers"},
        {"center": [33, 86, 84], "persona": "VIP High Rollers"}
    ]
    
    per_cluster = n_samples // len(archetypes)
    records = []
    
    for arc in archetypes:
        c = arc["center"]
        ages = np.random.normal(c[0], 5, per_cluster).clip(18, 70)
        incomes = np.random.normal(c[1], 7, per_cluster).clip(15, 140)
        spendings = np.random.normal(c[2], 7, per_cluster).clip(1, 99)
        genders = np.random.choice(["Female", "Male"], size=per_cluster, p=[0.54, 0.46])
        
        for a, inc, sp, g in zip(ages, incomes, spendings, genders):
            records.append({
                "Age": int(round(a)),
                "Annual_Income_k$": round(float(inc), 1),
                "Spending_Score": int(round(sp)),
                "Gender": g
            })
            
    df = pd.DataFrame(records)
    df.insert(0, "Customer_ID", [f"CUST-{1001 + i}" for i in range(len(df))])
    return df


@st.cache_data
def get_synthetic_blobs(n_samples: int = 500, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """Generate generic synthetic blobs dataset."""
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2, cluster_std=1.2, random_state=random_state)
    # Scale to resemble income and spending
    incomes = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), (18, 110))
    spendings = np.interp(X[:, 1], (X[:, 1].min(), X[:, 1].max()), (5, 95))
    
    df = pd.DataFrame({
        "Customer_ID": [f"CUST-{1001 + i}" for i in range(len(X))],
        "Annual_Income_k$": np.round(incomes, 1),
        "Spending_Score": np.round(spendings, 0).astype(int),
        "Age": np.random.randint(18, 65, size=len(X)),
        "Gender": np.random.choice(["Female", "Male"], size=len(X))
    })
    return df


def calculate_elbow_and_silhouette(X: np.ndarray, max_k: int = 8):
    """Calculate WCSS (Inertia) and Silhouette Scores across K values."""
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        
    return k_range, inertias, silhouettes


def determine_business_persona(avg_income: float, avg_spending: float, median_income: float, median_spending: float) -> dict:
    """Map cluster centroids to realistic marketing personas & strategic playbooks."""
    if avg_income >= median_income * 1.15 and avg_spending >= median_spending * 1.15:
        return {
            "title": "👑 VIP High Rollers",
            "desc": "High earning power and high propensity to spend. Receptive to luxury, exclusivity, and premium services.",
            "strategy": "Invite-only VIP loyalty tiers, concierge support, exclusive product launches, and personalized white-glove communications."
        }
    elif avg_income >= median_income * 1.15 and avg_spending < median_spending * 0.85:
        return {
            "title": "💼 Affluent Savers",
            "desc": "High income earners who are measured and selective with their retail purchases.",
            "strategy": "Value-first campaigns emphasizing durability, investment quality, warranties, and prestige value propositions."
        }
    elif avg_income < median_income * 0.85 and avg_spending >= median_spending * 1.15:
        return {
            "title": "🛍️ Young Trendsetters",
            "desc": "Lower or moderate income customers with disproportionately high retail spending habits.",
            "strategy": "Limited-time flash promotions, trend-driven viral marketing, Buy-Now-Pay-Later (BNPL) payment plans, and seasonal discounts."
        }
    elif avg_income < median_income * 0.85 and avg_spending < median_spending * 0.85:
        return {
            "title": "🪙 Budget Conscious",
            "desc": "Price-sensitive customers seeking reliable everyday essentials at minimum cost.",
            "strategy": "Clearance deals, buy-one-get-one (BOGO) bundles, discount codes, and essential everyday value messaging."
        }
    else:
        return {
            "title": "⚖️ Mainstream Core",
            "desc": "Balanced middle-ground consumers representing the steady backbone of regular customer flow.",
            "strategy": "Broad seasonal marketing, standard loyalty points, recurring customer email newsletters, and cross-sell bundles."
        }


# ==============================================================================
# Sidebar: Configuration & Controls
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=64)
    st.title("Segmentation Setup")
    st.caption("Configure dataset, feature scaling, and K-Means parameters.")
    
    st.markdown("---")
    st.subheader("1. Data Source")
    data_source = st.radio(
        "Choose Data Origin:",
        ["Mall Customers (Realistic Retail)", "Synthetic Benchmark (Blobs)", "Upload Custom CSV"],
        index=0
    )
    
    df_raw = None
    if data_source == "Mall Customers (Realistic Retail)":
        n_samples = st.slider("Sample Size", min_value=150, max_value=800, value=400, step=50)
        df_raw = get_sample_retail_data(n_samples=n_samples)
    elif data_source == "Synthetic Benchmark (Blobs)":
        n_samples = st.slider("Sample Size", min_value=100, max_value=800, value=500, step=50)
        df_raw = get_synthetic_blobs(n_samples=n_samples)
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.info("Please upload a CSV file or switch to sample data.")
            st.stop()
            
    st.markdown("---")
    st.subheader("2. Feature Selection")
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out ID columns if present
    feature_candidates = [col for col in numeric_cols if not any(x in col.lower() for x in ["id", "index", "cluster"])]
    if len(feature_candidates) < 2:
        feature_candidates = numeric_cols

    default_features = [col for col in ["Annual_Income_k$", "Spending_Score"] if col in feature_candidates]
    if len(default_features) < 2:
        default_features = feature_candidates[:2]
        
    selected_features = st.multiselect(
        "Select Features for Clustering:",
        options=feature_candidates,
        default=default_features
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 numerical features to perform clustering.")
        st.stop()
        
    st.markdown("---")
    st.subheader("3. Model Hyperparameters")
    
    scaling_choice = st.selectbox(
        "Feature Scaling:",
        ["StandardScaler (Zero-Mean, Unit-Var)", "MinMaxScaler (0 to 1)", "No Scaling"],
        index=0
    )
    
    k_clusters = st.slider("Number of Clusters (K):", min_value=2, max_value=8, value=5, step=1)
    
    with st.expander("Advanced Algorithm Settings"):
        init_method = st.selectbox("Centroid Initialization:", ["k-means++", "random"], index=0)
        max_iter = st.number_input("Max Iterations:", min_value=100, max_value=1000, value=300, step=50)
        random_seed = st.number_input("Random Seed:", min_value=0, max_value=999, value=42, step=1)


# ==============================================================================
# Data Preprocessing & Modeling Pipeline
# ==============================================================================
X_raw = df_raw[selected_features].values

# Apply scaling
if scaling_choice.startswith("StandardScaler"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
elif scaling_choice.startswith("MinMaxScaler"):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
else:
    scaler = None
    X_scaled = X_raw.copy()

# Fit K-Means
kmeans_model = KMeans(
    n_clusters=k_clusters,
    init=init_method,
    max_iter=int(max_iter),
    random_state=int(random_seed),
    n_init=10
)
cluster_labels = kmeans_model.fit_predict(X_scaled)
overall_silhouette = silhouette_score(X_scaled, cluster_labels)

# Build augmented working DataFrame
df_clustered = df_raw.copy()
df_clustered["Cluster"] = cluster_labels
df_clustered["Cluster_Label"] = [f"Cluster {c + 1}" for c in cluster_labels]


# ==============================================================================
# Main Dashboard Header
# ==============================================================================
st.title("🛍️ Customer Segmentation Studio")
st.markdown(
    "Uncover natural customer archetypes, evaluate clustering performance, and translate "
    "unsupervised machine learning into targeted retail marketing strategies."
)

# Top KPI Metric Banners
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Total Customers", f"{len(df_clustered):,}")
with col_m2:
    if "Annual_Income_k$" in df_raw.columns:
        st.metric("Avg Annual Income", f"${df_raw['Annual_Income_k$'].mean():.1f}k")
    else:
        st.metric("Avg " + selected_features[0], f"{df_raw[selected_features[0]].mean():.1f}")
with col_m3:
    if "Spending_Score" in df_raw.columns:
        st.metric("Avg Spending Score", f"{df_raw['Spending_Score'].mean():.1f} / 100")
    else:
        st.metric("Avg " + selected_features[1], f"{df_raw[selected_features[1]].mean():.1f}")
with col_m4:
    st.metric("Configured Clusters (K)", f"{k_clusters}")
with col_m5:
    st.metric("Silhouette Score", f"{overall_silhouette:.3f}")

st.markdown("---")

# ==============================================================================
# Dashboard Tabs
# ==============================================================================
tab_explore, tab_diagnostics, tab_clusters, tab_prediction, tab_export = st.tabs([
    "📊 Exploratory Analysis",
    "📈 Optimal K Diagnostics",
    "🎯 Interactive Clusters & Personas",
    "🔮 Live Customer Classifier",
    "📥 Data Export"
])


# ------------------------------------------------------------------------------
# TAB 1: Exploratory Analysis
# ------------------------------------------------------------------------------
with tab_explore:
    st.subheader("Data Overview & Feature Distributions")
    
    col_t1, col_t2 = st.columns([1, 1])
    
    with col_t1:
        st.markdown("**Dataset Preview (First 5 Rows)**")
        st.dataframe(df_raw.head(8), use_container_width=True)
        
        st.markdown("**Summary Statistics**")
        st.dataframe(df_raw[selected_features].describe().round(2), use_container_width=True)
        
    with col_t2:
        st.markdown("**Feature Distributions**")
        feature_to_plot = st.selectbox("Inspect Distribution For:", options=selected_features)
        
        fig_dist = px.histogram(
            df_raw,
            x=feature_to_plot,
            nbins=25,
            marginal="box",
            color_discrete_sequence=["#3B82F6"],
            title=f"Distribution & Spread of {feature_to_plot}"
        )
        fig_dist.update_layout(bargap=0.1, template="plotly_white", height=340)
        st.plotly_chart(fig_dist, use_container_width=True)

    if len(selected_features) >= 2:
        st.markdown("---")
        st.subheader("Feature Correlation & Scatter Exploration")
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            corr = df_raw[selected_features].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Pairwise Correlation Matrix"
            )
            fig_corr.update_layout(template="plotly_white", height=320)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_c2:
            fig_pair = px.scatter(
                df_raw,
                x=selected_features[0],
                y=selected_features[1],
                color="Gender" if "Gender" in df_raw.columns else None,
                hover_data=["Customer_ID"] if "Customer_ID" in df_raw.columns else None,
                title=f"{selected_features[0]} vs {selected_features[1]}",
                template="plotly_white"
            )
            fig_pair.update_layout(height=320)
            st.plotly_chart(fig_pair, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: Optimal K Diagnostics (Elbow Method & Silhouette)
# ------------------------------------------------------------------------------
with tab_diagnostics:
    st.subheader("Finding the Ideal Number of Clusters (K)")
    st.markdown(
        "Two classic unsupervised learning heuristics help identify the natural grouping density: "
        "**The Elbow Method** (Within-Cluster Sum of Squares) and **Silhouette Analysis**."
    )
    
    with st.spinner("Computing Inertia and Silhouette scores across K = 2 to 8..."):
        k_values, inertias, silhouettes = calculate_elbow_and_silhouette(X_scaled, max_k=8)
    
    best_k_idx = int(np.argmax(silhouettes))
    best_k = k_values[best_k_idx]
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # Elbow plot
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_values,
            y=inertias,
            mode='lines+markers',
            marker=dict(size=9, color="#2563EB"),
            line=dict(color="#2563EB", width=2.5),
            name="WCSS (Inertia)"
        ))
        fig_elbow.update_layout(
            title="The Elbow Method (WCSS vs K)",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Within-Cluster Sum of Squares (WCSS)",
            template="plotly_white",
            height=360
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.caption("Look for the 'elbow' bend where the reduction in inertia flattens out.")

    with col_d2:
        # Silhouette plot
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=k_values,
            y=silhouettes,
            mode='lines+markers',
            marker=dict(size=9, color="#10B981"),
            line=dict(color="#10B981", width=2.5),
            name="Silhouette Score"
        ))
        fig_sil.add_vline(
            x=best_k,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Peak K={best_k} ({silhouettes[best_k_idx]:.3f})",
            annotation_position="top right"
        )
        fig_sil.update_layout(
            title="Silhouette Coefficient vs K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Mean Silhouette Coefficient",
            template="plotly_white",
            height=360
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        st.caption("Higher silhouette coefficient indicates better separation and cohesion between clusters.")

    # Recommendation Callout
    if k_clusters == best_k:
        st.success(f"🎯 **Your current selection (K = {k_clusters}) matches the optimal Silhouette Score peak!**")
    else:
        st.info(f"💡 **Recommendation:** The mathematical optimal separation peaks at **K = {best_k}** (Silhouette = {silhouettes[best_k_idx]:.3f}). You are currently exploring **K = {k_clusters}**.")


# ------------------------------------------------------------------------------
# TAB 3: Interactive Clusters & Business Personas
# ------------------------------------------------------------------------------
with tab_clusters:
    st.subheader("Segment Visualization & Customer Personas")
    
    # Calculate Cluster Centroids in Original Space
    centroids_scaled = kmeans_model.cluster_centers_
    if scaler is not None:
        centroids_original = scaler.inverse_transform(centroids_scaled)
    else:
        centroids_original = centroids_scaled
        
    df_centroids = pd.DataFrame(centroids_original, columns=selected_features)
    df_centroids["Cluster"] = range(k_clusters)
    df_centroids["Cluster_Label"] = [f"Centroid {i + 1}" for i in range(k_clusters)]
    
    # 2D Scatter Plot
    feat_x = selected_features[0]
    feat_y = selected_features[1]
    
    color_palette = px.colors.qualitative.Bold
    
    fig_cluster = px.scatter(
        df_clustered,
        x=feat_x,
        y=feat_y,
        color="Cluster_Label",
        hover_data=["Customer_ID", "Age"] if "Customer_ID" in df_clustered.columns and "Age" in df_clustered.columns else None,
        color_discrete_sequence=color_palette,
        title=f"2D Customer Segments: {feat_x} vs {feat_y}",
        template="plotly_white",
        opacity=0.85
    )
    
    # Add Centroid Star Markers
    fig_cluster.add_trace(go.Scatter(
        x=df_centroids[feat_x],
        y=df_centroids[feat_y],
        mode="markers+text",
        marker=dict(symbol="star", size=18, color="gold", line=dict(color="black", width=1.5)),
        text=[f"C{i+1}" for i in range(k_clusters)],
        textposition="top center",
        name="Cluster Centroids"
    ))
    
    fig_cluster.update_layout(height=480, legend_title="Segments")
    st.plotly_chart(fig_cluster, use_container_width=True)

    # 3D Scatter if 3 or more features
    if len(selected_features) >= 3:
        with st.expander("🌐 View 3D Cluster Visualization"):
            feat_z = selected_features[2]
            fig_3d = px.scatter_3d(
                df_clustered,
                x=feat_x,
                y=feat_y,
                z=feat_z,
                color="Cluster_Label",
                color_discrete_sequence=color_palette,
                title=f"3D Representation: {feat_x} vs {feat_y} vs {feat_z}",
                template="plotly_white",
                opacity=0.8
            )
            fig_3d.update_layout(height=550)
            st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")
    st.subheader("Business Personas & Actionable Marketing Playbooks")
    
    # Aggregate stats per cluster
    med_inc = df_raw[feat_x].median() if feat_x in df_raw.columns else 50.0
    med_sp = df_raw[feat_y].median() if feat_y in df_raw.columns else 50.0
    
    persona_cols = st.columns(min(k_clusters, 4))
    
    for i in range(k_clusters):
        c_df = df_clustered[df_clustered["Cluster"] == i]
        c_count = len(c_df)
        c_pct = (c_count / len(df_clustered)) * 100
        
        c_avg_x = df_centroids.iloc[i][feat_x]
        c_avg_y = df_centroids.iloc[i][feat_y]
        
        persona_info = determine_business_persona(c_avg_x, c_avg_y, med_inc, med_sp)
        
        # Display in cycling column grid
        col_target = persona_cols[i % len(persona_cols)]
        with col_target:
            st.markdown(f"""
            <div class="persona-card">
                <h4 style="margin:0 0 4px 0; color:#1E3A8A;">Cluster {i+1}: {persona_info['title']}</h4>
                <p style="margin:0; font-size: 0.88rem; color:#475569;"><b>Size:</b> {c_count} customers ({c_pct:.1f}%)</p>
                <p style="margin:0; font-size: 0.88rem; color:#475569;"><b>Avg {feat_x}:</b> {c_avg_x:.1f}</p>
                <p style="margin:0 0 8px 0; font-size: 0.88rem; color:#475569;"><b>Avg {feat_y}:</b> {c_avg_y:.1f}</p>
                <p style="font-size:0.83rem; margin:0 0 6px 0; color:#334155;">{persona_info['desc']}</p>
                <div style="background-color:#E2E8F0; border-radius:6px; padding:6px 8px; font-size:0.8rem; color:#0F172A;">
                    🎯 <b>Action:</b> {persona_info['strategy']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 4: Live Customer Classifier Simulator
# ------------------------------------------------------------------------------
with tab_prediction:
    st.subheader("Simulate & Classify a New Customer")
    st.markdown("Enter demographic & behavioral metrics for a prospective customer to immediately evaluate which cluster and persona they match.")
    
    col_p1, col_p2 = st.columns([1, 1.2])
    
    with col_p1:
        st.markdown("**Customer Attributes Input**")
        input_values = []
        for feature in selected_features:
            f_min = float(df_raw[feature].min())
            f_max = float(df_raw[feature].max())
            f_mean = float(df_raw[feature].mean())
            val = st.slider(
                f"{feature}:",
                min_value=round(f_min * 0.8, 1),
                max_value=round(f_max * 1.2, 1),
                value=round(f_mean, 1),
                step=1.0 if f_max > 10 else 0.1
            )
            input_values.append(val)
            
        predict_btn = st.button("Classify Customer", type="primary", use_container_width=True)

    with col_p2:
        if predict_btn or True:  # Live reactive prediction
            input_arr = np.array([input_values])
            if scaler is not None:
                input_scaled = scaler.transform(input_arr)
            else:
                input_scaled = input_arr
                
            pred_cluster = int(kmeans_model.predict(input_scaled)[0])
            
            c_avg_x = df_centroids.iloc[pred_cluster][feat_x]
            c_avg_y = df_centroids.iloc[pred_cluster][feat_y]
            persona_res = determine_business_persona(c_avg_x, c_avg_y, med_inc, med_sp)
            
            st.success(f"### Assigned Segment: Cluster {pred_cluster + 1}")
            st.markdown(f"#### Persona Archetype: {persona_res['title']}")
            st.write(persona_res['desc'])
            
            st.info(f"**Recommended Retention & Marketing Strategy:**\n\n{persona_res['strategy']}")
            
            # Interactive mini scatter showing new customer live
            fig_sim = px.scatter(
                df_clustered,
                x=feat_x,
                y=feat_y,
                color="Cluster_Label",
                color_discrete_sequence=color_palette,
                template="plotly_white",
                opacity=0.4,
                title="Customer Position on Feature Map"
            )
            fig_sim.add_trace(go.Scatter(
                x=[input_values[0]],
                y=[input_values[1]],
                mode="markers+text",
                marker=dict(symbol="cross", size=22, color="#EF4444", line=dict(color="black", width=2)),
                text=["New Customer"],
                textposition="top center",
                name="New Customer"
            ))
            fig_sim.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_sim, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 5: Data Export
# ------------------------------------------------------------------------------
with tab_export:
    st.subheader("Export Segmented Customer Records")
    st.markdown("Download the complete customer dataset enriched with assigned cluster labels and persona profiles.")
    
    # Attach human persona tags to the export table
    cluster_persona_map = {}
    for c_id in range(k_clusters):
        cx = df_centroids.iloc[c_id][feat_x]
        cy = df_centroids.iloc[c_id][feat_y]
        cluster_persona_map[c_id] = determine_business_persona(cx, cy, med_inc, med_sp)["title"]
        
    df_export = df_clustered.copy()
    df_export["Marketing_Persona"] = df_export["Cluster"].map(cluster_persona_map)
    
    st.dataframe(df_export.head(10), use_container_width=True)
    
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Enriched Customer Segments (CSV)",
        data=csv_bytes,
        file_name="segmented_customers.csv",
        mime="text/csv",
        type="primary"
    )

st.markdown("---")
st.caption("Built with Streamlit, Scikit-Learn, and Plotly | Part of the 1,000 AI Projects Portfolio")

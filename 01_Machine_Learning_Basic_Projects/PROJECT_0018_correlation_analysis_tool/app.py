import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import io

# Page Configuration
st.set_page_config(
    page_title="Correlation Analysis Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0ea5e9, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: monospace;
    }
    .stat-lbl {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94a3b8;
        margin-top: 4px;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR DATA & SETTINGS -----------------
st.sidebar.markdown("## 📊 Correlation Studio")
st.sidebar.markdown("### Dataset Selection")

data_source = st.sidebar.selectbox(
    "Choose Dataset Source:",
    [
        "Project 018 Baseline (Synthetic 4-Metric)",
        "Non-Linear Simulation (Quadratic & Sine)",
        "Iris Botanical Dataset",
        "California Housing Benchmark",
        "Upload Custom CSV File"
    ]
)

# Load / Generate Data
@st.cache_data
def get_dataset(source, csv_file=None):
    if source.startswith("Project 018"):
        np.random.seed(42)
        x1 = np.linspace(0, 50, 100)
        x2 = 2.5 * x1 + np.random.normal(0, 5, 100)
        x3 = -1.8 * x1 + np.random.normal(0, 8, 100)
        x4 = np.random.uniform(0, 100, 100)
        return pd.DataFrame({"Metric_A": x1, "Metric_B": x2, "Metric_C": x3, "Noise": x4})
    
    elif source.startswith("Non-Linear"):
        np.random.seed(42)
        x = np.linspace(-10, 10, 150)
        y_quad = x**2 + np.random.normal(0, 6, 150)
        y_sin = 20 * np.sin(x) + np.random.normal(0, 3, 150)
        y_exp = np.exp(x * 0.25) + np.random.normal(0, 2, 150)
        y_noise = np.random.normal(0, 15, 150)
        return pd.DataFrame({"X_Input": x, "Quadratic_Y": y_quad, "Sine_Wave_Y": y_sin, "Exponential_Y": y_exp, "Random_Noise": y_noise})
        
    elif source.startswith("Iris"):
        iris = load_iris(as_frame=True)
        return iris.frame.drop(columns=['target'])
        
    elif source.startswith("California"):
        cal = fetch_california_housing(as_frame=True)
        return cal.frame.iloc[:300, :6] # subset for fast interactive exploration
        
    else: # Custom CSV
        if csv_file is not None:
            try:
                df_user = pd.read_csv(csv_file)
                # Keep numeric columns only
                num_df = df_user.select_dtypes(include=[np.number])
                if num_df.empty:
                    st.sidebar.error("Uploaded CSV contains no numeric columns!")
                    return None
                return num_df
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {e}")
                return None
        return None

uploaded_file = None
if data_source == "Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV:", type=["csv"])

df = get_dataset(data_source, uploaded_file)
if df is None:
    st.info("Please upload a numeric CSV file to proceed.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Correlation Parameters")

method_choice = st.sidebar.radio(
    "Correlation Coefficient:",
    ["pearson", "spearman", "kendall"],
    format_func=lambda x: {
        "pearson": "Pearson (Linear)",
        "spearman": "Spearman (Monotonic / Rank)",
        "kendall": "Kendall's Tau (Concordant Pairs)"
    }[x]
)

mask_triangle = st.sidebar.selectbox("Matrix Masking:", ["Full Matrix", "Lower Triangle Only", "Upper Triangle Only"])
threshold = st.sidebar.slider("Highlight Filter (|r| ≥ Threshold):", min_value=0.0, max_value=0.95, value=0.0, step=0.05)

# ----------------- COMPUTATIONS -----------------
# 1. Correlation Matrix
corr_matrix = df.corr(method=method_choice)

# 2. P-Value Matrix Computation
def compute_pvalues(dataframe, method):
    cols = dataframe.columns
    p_df = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                p_df.loc[c1, c2] = 0.0
            else:
                x = dataframe[c1].dropna()
                y = dataframe[c2].dropna()
                idx = x.index.intersection(y.index)
                if method == 'pearson':
                    _, p = stats.pearsonr(x.loc[idx], y.loc[idx])
                elif method == 'spearman':
                    _, p = stats.spearmanr(x.loc[idx], y.loc[idx])
                else:
                    _, p = stats.kendalltau(x.loc[idx], y.loc[idx])
                p_df.loc[c1, c2] = p
    return p_df

p_matrix = compute_pvalues(df, method_choice)

# 3. Multicollinearity / VIF Computation
def calculate_vif(dataframe):
    vif_dict = {}
    clean_df = dataframe.dropna()
    cols = clean_df.columns
    if len(cols) < 2:
        return pd.Series()
    for col in cols:
        other_cols = [c for c in cols if c != col]
        X = clean_df[other_cols]
        y = clean_df[col]
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        vif = 1.0 / (1.0 - r2) if r2 < 0.999999 else 999.0
        vif_dict[col] = vif
    return pd.Series(vif_dict).sort_values(ascending=False)

vif_series = calculate_vif(df)

# Header
st.markdown('<div class="main-title">Correlation Analysis Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced statistical exploration of linear and monotonic dependencies, p-value matrices, and multicollinearity diagnostics.</div>', unsafe_allow_html=True)

# Top KPI Summary Cards
n_features = df.shape[1]
n_samples = df.shape[0]

# Compute strongest non-identity pair
corr_unstacked = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).unstack().dropna()
strongest_pair = corr_unstacked.abs().idxmax()
strongest_val = corr_matrix.loc[strongest_pair[0], strongest_pair[1]]

severe_vif_count = (vif_series > 10).sum()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#38bdf8;">{n_features}</div><div class="stat-lbl">Numeric Features</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#a855f7;">{n_samples}</div><div class="stat-lbl">Dataset Samples</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#10b981;">{strongest_val:+.3f}</div><div class="stat-lbl">Max Correlation</div></div>""", unsafe_allow_html=True)
with c4:
    pair_label = f"{strongest_pair[0][:8]} & {strongest_pair[1][:8]}"
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#facc15; font-size:1.1rem; line-height:2rem;">{pair_label}</div><div class="stat-lbl">Strongest Pair</div></div>""", unsafe_allow_html=True)
with c5:
    vif_color = "#ef4444" if severe_vif_count > 0 else "#10b981"
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:{vif_color};">{severe_vif_count}</div><div class="stat-lbl">High VIF (Multicollinearity)</div></div>""", unsafe_allow_html=True)

st.markdown("")

# ----------------- TABS INTERFACE -----------------
tab_heatmap, tab_scatter, tab_3d, tab_vif, tab_matrix_data = st.tabs([
    "🔥 Correlation Heatmap & Significance",
    "📈 Bivariate Scatter & Regression",
    "🪐 3D Feature Space Manifold",
    "⚠️ Multicollinearity & VIF Diagnostics",
    "📋 Raw Data & Matrix Export"
])

# ----------------- TAB 1: HEATMAP & SIGNIFICANCE -----------------
with tab_heatmap:
    col_h1, col_h2 = st.columns([3, 2])
    
    with col_h1:
        st.subheader("Interactive Correlation Matrix")
        
        display_corr = corr_matrix.copy()
        if threshold > 0:
            display_corr = display_corr.where(display_corr.abs() >= threshold, np.nan)
            
        if mask_triangle == "Lower Triangle Only":
            mask = np.triu(np.ones_like(display_corr, dtype=bool), k=1)
            display_corr = display_corr.mask(mask)
        elif mask_triangle == "Upper Triangle Only":
            mask = np.tril(np.ones_like(display_corr, dtype=bool), k=-1)
            display_corr = display_corr.mask(mask)

        # Plotly Diverging Heatmap
        fig_heat = px.imshow(
            display_corr,
            text_auto=".3f",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            labels=dict(x="Variable", y="Variable", color="Coefficient")
        )
        fig_heat.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=460,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_h2:
        st.subheader("P-Value Significance Matrix")
        st.caption("Hypothesis test against null ($H_0: \\rho = 0$). Values $p < 0.05$ indicate statistically significant correlation.")
        
        fig_p = px.imshow(
            p_matrix,
            text_auto=".3e",
            color_continuous_scale="Blues_r",
            range_color=[0, 0.05],
            labels=dict(x="Variable", y="Variable", color="P-Value")
        )
        fig_p.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=460,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_p, use_container_width=True)

# ----------------- TAB 2: BIVARIATE SCATTER & REGRESSION -----------------
with tab_scatter:
    st.subheader("Pairwise Bivariate Scatter & Trendline Analysis")
    
    col_sel_x, col_sel_y = st.columns(2)
    with col_sel_x:
        x_var = st.selectbox("Select X-Axis Variable:", df.columns, index=0)
    with col_sel_y:
        y_var = st.selectbox("Select Y-Axis Variable:", [c for c in df.columns if c != x_var], index=0)

    # Compute regression parameters
    x_vals = df[x_var].values
    y_vals = df[y_var].values
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_vals, y_vals)
    r_squared = r_val**2
    spearman_rho, spearman_p = stats.spearmanr(x_vals, y_vals)

    col_sc1, col_sc2 = st.columns([3, 1])
    with col_sc1:
        fig_scatter = px.scatter(
            df,
            x=x_var,
            y=y_var,
            trendline="ols",
            marginal_x="histogram",
            marginal_y="histogram",
            color_discrete_sequence=["#38bdf8"],
            title=f"{x_var} vs. {y_var}"
        )
        fig_scatter.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=480,
            margin=dict(l=10, r=10, t=30, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_sc2:
        st.markdown("#### Relationship Metrics")
        st.metric("Pearson r", f"{r_val:+.4f}")
        st.metric("R² Variance Explained", f"{r_squared * 100:.2f}%")
        st.metric("Spearman Rank ρ", f"{spearman_rho:+.4f}")
        st.metric("Linear Slope (m)", f"{slope:+.4f}")
        st.metric("P-Value", f"{p_val:.3e}")
        
        strength = "Strong" if abs(r_val) >= 0.7 else ("Moderate" if abs(r_val) >= 0.4 else "Weak")
        direction = "Positive" if r_val > 0 else "Negative"
        st.info(f"**Interpretation**: {strength} {direction} correlation. Linearity captures {r_squared*100:.1f}% of variance.")

# ----------------- TAB 3: 3D FEATURE SPACE -----------------
with tab_3d:
    st.subheader("3D Multi-Variable Relationship Manifold")
    st.caption("Rotate and inspect 3 simultaneous feature dimensions to spot collinear planes and spatial clusters.")

    if len(df.columns) >= 3:
        c3_1, c3_2, c3_3 = st.columns(3)
        with c3_1:
            var_x = st.selectbox("X Dimension:", df.columns, index=0, key="3dx")
        with c3_2:
            var_y = st.selectbox("Y Dimension:", [c for c in df.columns if c != var_x], index=0, key="3dy")
        with c3_3:
            var_z = st.selectbox("Z Dimension:", [c for c in df.columns if c not in [var_x, var_y]], index=0, key="3dz")

        fig_3d = px.scatter_3d(
            df,
            x=var_x,
            y=var_y,
            z=var_z,
            color=var_z,
            color_continuous_scale="Viridis",
            opacity=0.85
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title=var_x,
                yaxis_title=var_y,
                zaxis_title=var_z,
                bgcolor="#0b0f19",
                xaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b"),
                yaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b"),
                zaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b")
            ),
            paper_bgcolor="#030712",
            font=dict(color="#e2e8f0"),
            height=520,
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("At least 3 numeric variables are required to generate the 3D manifold.")

# ----------------- TAB 4: VIF & MULTICOLLINEARITY -----------------
with tab_vif:
    st.subheader("Multicollinearity Diagnostics (Variance Inflation Factor)")
    st.markdown("""
    **VIF** quantifies the severity of multicollinearity in regression models:
    - **VIF < 5**: Low multicollinearity (safe for linear/logistic models).
    - **5 ≤ VIF < 10**: Moderate correlation; caution recommended.
    - **VIF ≥ 10**: Severe multicollinearity; estimates are unstable. Consider feature selection or PCA.
    """)

    col_v1, col_v2 = st.columns([3, 2])
    with col_v1:
        vif_df = vif_series.reset_index()
        vif_df.columns = ["Feature", "VIF Score"]
        vif_df["Risk Level"] = vif_df["VIF Score"].apply(
            lambda v: "Severe (VIF ≥ 10)" if v >= 10 else ("Moderate (5 ≤ VIF < 10)" if v >= 5 else "Low (VIF < 5)")
        )

        fig_vif = px.bar(
            vif_df,
            x="VIF Score",
            y="Feature",
            orientation="h",
            color="Risk Level",
            color_discrete_map={
                "Low (VIF < 5)": "#10b981",
                "Moderate (5 ≤ VIF < 10)": "#f59e0b",
                "Severe (VIF ≥ 10)": "#ef4444"
            },
            text=vif_df["VIF Score"].apply(lambda v: f"{v:.2f}")
        )
        fig_vif.add_vline(x=5, line_dash="dash", line_color="#f59e0b")
        fig_vif.add_vline(x=10, line_dash="dash", line_color="#ef4444")
        fig_vif.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=360,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_vif, use_container_width=True)

    with col_v2:
        st.markdown("#### Feature Recommendation")
        for f, v in vif_series.items():
            if v >= 10:
                st.error(f"⚠️ **{f}** (VIF: {v:.2f}): Redundant / collinear with other features. Dropping or combining is strongly advised.")
            elif v >= 5:
                st.warning(f"⚡ **{f}** (VIF: {v:.2f}): Moderate collinearity detected.")
            else:
                st.success(f"✅ **{f}** (VIF: {v:.2f}): Stable predictor.")

# ----------------- TAB 5: RAW DATA & EXPORT -----------------
with tab_matrix_data:
    st.subheader("Data Inspector & Matrix Export")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### Correlation Matrix Table")
        st.dataframe(corr_matrix.style.format("{:.4f}").background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)
        
        # Download Button
        csv_buffer = io.StringIO()
        corr_matrix.to_csv(csv_buffer)
        st.download_button(
            label="📥 Download Correlation Matrix CSV",
            data=csv_buffer.getvalue(),
            file_name=f"correlation_matrix_{method_choice}.csv",
            mime="text/csv"
        )

    with col_d2:
        st.markdown("#### Sample Raw Records")
        st.dataframe(df.head(10), use_container_width=True)

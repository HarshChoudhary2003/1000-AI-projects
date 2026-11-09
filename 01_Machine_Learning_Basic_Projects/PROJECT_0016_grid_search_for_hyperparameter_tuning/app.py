import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import time

# Page Configuration
st.set_page_config(
    page_title="Hyperparameter Tuning Studio | Grid Search",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06b6d4, #6366f1, #a855f7);
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

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("## ⚡ Grid Search Studio")
st.sidebar.markdown("### Model & Parameter Grid")

model_type = st.sidebar.selectbox(
    "Choose Classifier Algorithm:",
    ["Support Vector Machine (SVC)", "Random Forest", "Logistic Regression", "K-Nearest Neighbors"],
    index=0
)

# Feature Scaling
use_scaling = st.sidebar.checkbox("Apply Standard Scaler (Recommended)", value=True)

# Grid definitions per model
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Hyperparameter Grid")

if model_type == "Support Vector Machine (SVC)":
    c_options = st.sidebar.multiselect("C (Regularization):", [0.01, 0.1, 1, 10, 100], default=[0.1, 1, 10])
    kernels = st.sidebar.multiselect("Kernel Functions:", ['linear', 'rbf', 'poly', 'sigmoid'], default=['linear', 'rbf'])
    gammas = st.sidebar.multiselect("Gamma (Kernel Coefficient):", ['scale', 'auto', 0.01, 0.1, 1.0], default=['scale', 'auto'])
    
    if not c_options or not kernels or not gammas:
        st.sidebar.warning("Please select at least one value for each parameter!")
        c_options = [1]
        kernels = ['linear']
        gammas = ['scale']
        
    param_grid = {'C': c_options, 'kernel': kernels, 'gamma': gammas}
    base_estimator = SVC(probability=True, random_state=42)

elif model_type == "Random Forest":
    n_est = st.sidebar.multiselect("Number of Estimators:", [20, 50, 100, 200], default=[50, 100])
    max_depth = st.sidebar.multiselect("Max Depth:", [3, 5, 10, None], default=[3, 5, 10])
    min_samples_split = st.sidebar.multiselect("Min Samples Split:", [2, 5, 10], default=[2, 5])
    
    param_grid = {'n_estimators': n_est, 'max_depth': max_depth, 'min_samples_split': min_samples_split}
    base_estimator = RandomForestClassifier(random_state=42)

elif model_type == "Logistic Regression":
    c_opts = st.sidebar.multiselect("Inverse Regularization (C):", [0.01, 0.1, 1.0, 10.0, 100.0], default=[0.1, 1.0, 10.0])
    penalties = st.sidebar.multiselect("Penalty Norm:", ['l2'], default=['l2'])
    solvers = st.sidebar.multiselect("Solver:", ['lbfgs', 'liblinear'], default=['lbfgs'])
    
    param_grid = {'C': c_opts, 'penalty': penalties, 'solver': solvers}
    base_estimator = LogisticRegression(max_iter=1000, random_state=42)

else: # KNN
    n_neighbors = st.sidebar.multiselect("N Neighbors:", [3, 5, 7, 9, 15], default=[3, 5, 7])
    weights = st.sidebar.multiselect("Weights:", ['uniform', 'distance'], default=['uniform', 'distance'])
    metrics = st.sidebar.multiselect("Distance Metric:", ['euclidean', 'manhattan'], default=['euclidean', 'manhattan'])
    
    param_grid = {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metrics}
    base_estimator = KNeighborsClassifier()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Search Configuration")
cv_folds = st.sidebar.slider("Cross-Validation Folds (K):", min_value=2, max_value=10, value=3)
scoring_metric = st.sidebar.selectbox("Scoring Metric:", ["accuracy", "f1", "precision", "recall", "roc_auc"], index=0)
test_size = st.sidebar.slider("Holdout Test Split:", min_value=0.15, max_value=0.40, value=0.25, step=0.05)

# Calculate total combinations
total_combinations = 1
for v in param_grid.values():
    total_combinations *= len(v)
total_fits = total_combinations * cv_folds

st.sidebar.caption(f"Total Combinations: **{total_combinations}** | Total Fits: **{total_fits}**")

# ----------------- DATA PREPARATION -----------------
@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    return cancer.data, cancer.target, cancer.feature_names, cancer.target_names

X_raw, y_raw, feature_names, target_names = load_data()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=test_size, random_state=42, stratify=y_raw
)

if use_scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
else:
    X_train = X_train_raw
    X_test = X_test_raw
    scaler = None

# Header View
st.markdown('<div class="main-title">Grid Search Hyperparameter Tuning Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explore parameter spaces, 3D response landscapes, parallel sweeps, and holdout performance on Breast Cancer diagnostic data.</div>', unsafe_allow_html=True)

# ----------------- RUN GRID SEARCH -----------------
@st.cache_resource
def run_grid_search(base_est, grid, cv, scoring, X_tr, y_tr):
    start_time = time.time()
    search = GridSearchCV(
        estimator=base_est,
        param_grid=grid,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1
    )
    search.fit(X_tr, y_tr)
    duration = time.time() - start_time
    return search, duration

with st.spinner(f"Executing Grid Search across {total_combinations} parameter configurations ({total_fits} fits)..."):
    grid_search, elapsed = run_grid_search(base_estimator, param_grid, cv_folds, scoring_metric, X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_prec = precision_score(y_test, y_pred, zero_division=0)
test_rec = recall_score(y_test, y_pred, zero_division=0)

# Top Stat KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#38bdf8;">{grid_search.best_score_ * 100:.2f}%</div><div class="stat-lbl">Best CV Score</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#4ade80;">{test_acc * 100:.2f}%</div><div class="stat-lbl">Test Accuracy</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#f472b6;">{test_f1 * 100:.2f}%</div><div class="stat-lbl">Test F1-Score</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#fbbf24;">{total_combinations}</div><div class="stat-lbl">Params Evaluated</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#c084fc;">{elapsed:.2f}s</div><div class="stat-lbl">Search Duration</div></div>""", unsafe_allow_html=True)

st.markdown("")

# Best Params Pill Display
best_params_str = ", ".join([f"**{k}**: `{v}`" for k, v in grid_search.best_params_.items()])
st.success(f"🌟 **Best Hyperparameter Configuration Found:** {best_params_str}")

# Convert cv_results_ to clean DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['rank_test_score'] = results_df['rank_test_score'].astype(int)

# ----------------- TABS WORKSPACE -----------------
tab_landscape, tab_parallel, tab_table, tab_eval, tab_predict = st.tabs([
    "🪐 2D & 3D Hyperparameter Landscape",
    "🔀 Parallel Coordinates Sweep",
    "📋 Search Results & Overfitting Curves",
    "🎯 Model Evaluation & Confusion Matrix",
    "🧪 Live Patient Diagnosis Lab"
])

# ----------------- TAB 1: 2D & 3D LANDSCAPE -----------------
with tab_landscape:
    col_l1, col_l2 = st.columns([1, 1])
    
    with col_l1:
        st.subheader("2D Parameter Score Heatmap")
        
        # Pick 2 parameters to create a pivot heatmap
        param_keys = list(param_grid.keys())
        if len(param_keys) >= 2:
            p_x = st.selectbox("X-Axis Parameter:", param_keys, index=0, key="hx")
            p_y = st.selectbox("Y-Axis Parameter:", [k for k in param_keys if k != p_x], index=0, key="hy")
            
            # Pivot table for mean test score
            pivot_df = results_df.pivot_table(
                index=f'param_{p_y}',
                columns=f'param_{p_x}',
                values='mean_test_score',
                aggfunc='max'
            )
            
            fig_heat = px.imshow(
                pivot_df,
                labels=dict(x=p_x, y=p_y, color=f"Mean {scoring_metric.capitalize()}"),
                x=[str(c) for c in pivot_df.columns],
                y=[str(r) for r in pivot_df.index],
                color_continuous_scale="Viridis",
                text_auto=".3f"
            )
            fig_heat.update_layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#0b0f19",
                font=dict(color="#e2e8f0"),
                height=420,
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Select at least 2 parameters in the sidebar to render the 2D Heatmap.")

    with col_l2:
        st.subheader("3D Response Surface Landscape")
        
        if len(param_keys) >= 2:
            # 3D Scatter or Surface
            fig_3d = go.Figure()
            
            p1 = param_keys[0]
            p2 = param_keys[1]
            
            # Convert values to categorical strings or numeric for 3D coordinates
            x_vals = results_df[f'param_{p1}'].astype(str)
            y_vals = results_df[f'param_{p2}'].astype(str)
            z_vals = results_df['mean_test_score']
            
            # Find best index to highlight
            best_idx = results_df['rank_test_score'].idxmin()
            
            fig_3d.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=8,
                    color=z_vals,
                    colorscale='Plasma',
                    colorbar=dict(title=f"CV {scoring_metric}"),
                    opacity=0.85
                ),
                text=[f"Score: {s:.4f}" for s in z_vals],
                name='Grid Points'
            ))
            
            # Highlight Best Point with Star
            fig_3d.add_trace(go.Scatter3d(
                x=[str(results_df.loc[best_idx, f'param_{p1}'])],
                y=[str(results_df.loc[best_idx, f'param_{p2}'])],
                z=[results_df.loc[best_idx, 'mean_test_score']],
                mode='markers+text',
                marker=dict(size=14, color='#fbbf24', symbol='diamond'),
                text=['🌟 Global Best'],
                textposition='top center',
                name='Global Optimum'
            ))
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title=p1,
                    yaxis_title=p2,
                    zaxis_title=f"CV {scoring_metric.capitalize()}",
                    bgcolor="#0b0f19",
                    xaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b"),
                    yaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b"),
                    zaxis=dict(backgroundcolor="#0b0f19", gridcolor="#1e293b")
                ),
                paper_bgcolor="#030712",
                font=dict(color="#e2e8f0"),
                height=420,
                margin=dict(l=0, r=0, t=10, b=10)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

# ----------------- TAB 2: PARALLEL COORDINATES -----------------
with tab_parallel:
    st.subheader("Multi-Dimensional Hyperparameter Sweep")
    st.caption("Trace how parameter combinations traverse the search space to produce high-performing configurations (colored by CV score).")
    
    sweep_df = results_df.copy()
    dimensions = []
    
    for key in param_grid.keys():
        col_name = f'param_{key}'
        # If categorical, map to numeric codes
        if sweep_df[col_name].dtype == 'object' or sweep_df[col_name].apply(type).eq(str).any():
            unique_vals = list(sweep_df[col_name].unique())
            val_to_num = {v: i for i, v in enumerate(unique_vals)}
            sweep_df[f'{key}_code'] = sweep_df[col_name].map(val_to_num)
            dimensions.append(dict(
                range=[0, len(unique_vals) - 1],
                tickvals=list(range(len(unique_vals))),
                ticktext=[str(v) for v in unique_vals],
                label=key,
                values=sweep_df[f'{key}_code']
            ))
        else:
            dimensions.append(dict(
                range=[sweep_df[col_name].min(), sweep_df[col_name].max()],
                label=key,
                values=sweep_df[col_name]
            ))
            
    # Add Mean Test Score as final axis
    dimensions.append(dict(
        range=[sweep_df['mean_test_score'].min(), sweep_df['mean_test_score'].max()],
        label=f'CV {scoring_metric.capitalize()}',
        values=sweep_df['mean_test_score']
    ))
    
    fig_parallel = go.Figure(data=go.Parcoords(
        line=dict(
            color=sweep_df['mean_test_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=f"CV Score")
        ),
        dimensions=dimensions
    ))
    
    fig_parallel.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#e2e8f0"),
        height=450,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    st.plotly_chart(fig_parallel, use_container_width=True)

# ----------------- TAB 3: SEARCH RESULTS & OVERFITTING -----------------
with tab_table:
    st.subheader("Search Results Breakdown & Overfitting Analysis")
    
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("#### Ranked Configurations")
        display_cols = ['rank_test_score'] + [f'param_{k}' for k in param_grid.keys()] + ['mean_test_score', 'std_test_score', 'mean_train_score']
        ranked_table = results_df[display_cols].sort_values('rank_test_score')
        ranked_table.columns = ['Rank'] + list(param_grid.keys()) + ['Mean Test Score', 'Std Test Score', 'Mean Train Score']
        st.dataframe(
            ranked_table.style.highlight_min(subset=['Rank'], color='#065f46')
                              .format({'Mean Test Score': '{:.4f}', 'Std Test Score': '{:.4f}', 'Mean Train Score': '{:.4f}'}),
            use_container_width=True,
            hide_index=True
        )

    with col_t2:
        st.markdown("#### Train vs. Validation (Overfitting Check)")
        st.caption("Large gaps between Train and Validation indicate overfitting.")
        fig_overfit = go.Figure()
        sorted_res = results_df.sort_values('mean_test_score', ascending=False).reset_index()
        
        fig_overfit.add_trace(go.Scatter(
            x=sorted_res.index,
            y=sorted_res['mean_train_score'],
            mode='lines+markers',
            name='Train Score',
            line=dict(color='#38bdf8', dash='dot')
        ))
        fig_overfit.add_trace(go.Scatter(
            x=sorted_res.index,
            y=sorted_res['mean_test_score'],
            mode='lines+markers',
            name='CV Test Score',
            line=dict(color='#a855f7', width=2.5)
        ))
        fig_overfit.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=320,
            xaxis_title="Configurations (Ranked Descending)",
            yaxis_title="Score",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_overfit, use_container_width=True)

# ----------------- TAB 4: MODEL EVALUATION -----------------
with tab_eval:
    st.subheader("Holdout Test Set Performance")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {t}" for t in target_names],
            columns=[f"Pred {t}" for t in target_names]
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig_cm.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=360,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        st.markdown("#### ROC Curve")
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc_val = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc_val:.3f})', line=dict(color='#06b6d4', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance Baseline', line=dict(color='#64748b', dash='dash')))
            fig_roc.update_layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#0b0f19",
                font=dict(color="#e2e8f0"),
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=360,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception as e:
            st.warning("ROC Curve unavailable for current model output.")

# ----------------- TAB 5: PATIENT DIAGNOSIS SIMULATOR -----------------
with tab_predict:
    st.subheader("🧪 Live Patient Diagnostic Inference")
    st.markdown("Adjust key clinical measurement values to evaluate real-time classification using the tuned model:")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        f_radius = st.slider("Mean Radius:", float(X_raw[:, 0].min()), float(X_raw[:, 0].max()), float(X_raw[:, 0].mean()))
        f_texture = st.slider("Mean Texture:", float(X_raw[:, 1].min()), float(X_raw[:, 1].max()), float(X_raw[:, 1].mean()))
    with col_f2:
        f_perimeter = st.slider("Mean Perimeter:", float(X_raw[:, 2].min()), float(X_raw[:, 2].max()), float(X_raw[:, 2].mean()))
        f_area = st.slider("Mean Area:", float(X_raw[:, 3].min()), float(X_raw[:, 3].max()), float(X_raw[:, 3].mean()))
    with col_f3:
        f_smoothness = st.slider("Mean Smoothness:", float(X_raw[:, 4].min()), float(X_raw[:, 4].max()), float(X_raw[:, 4].mean()))
        f_concavity = st.slider("Mean Concavity:", float(X_raw[:, 6].min()), float(X_raw[:, 6].max()), float(X_raw[:, 6].mean()))
        
    if st.button("🔬 Classify Patient Sample", type="primary", use_container_width=True):
        # Create full sample vector populated with means for remaining features
        sample = X_raw.mean(axis=0)
        sample[0] = f_radius
        sample[1] = f_texture
        sample[2] = f_perimeter
        sample[3] = f_area
        sample[4] = f_smoothness
        sample[6] = f_concavity
        
        sample_transformed = scaler.transform([sample]) if scaler else [sample]
        pred_class = best_model.predict(sample_transformed)[0]
        pred_name = target_names[pred_class]
        
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(sample_transformed)[0]
            confidence = probs[pred_class] * 100
        else:
            confidence = 100.0
            
        color = "#10b981" if pred_class == 1 else "#ef4444"
        st.markdown(f"""
        <div style="background:#1e293b; border-left: 6px solid {color}; border-radius:12px; padding:20px; margin-top:16px;">
            <div style="font-size:1.4rem; font-weight:700; color:{color};">
                Predicted Diagnosis: {pred_name.upper()}
            </div>
            <div style="font-size:1rem; color:#e2e8f0; margin-top:6px;">
                Prediction Confidence: <strong>{confidence:.1f}%</strong> using optimal tuned {model_type}.
            </div>
        </div>
        """, unsafe_allow_html=True)

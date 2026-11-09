import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import time

# Page Configuration
st.set_page_config(
    page_title="Bayesian Hyperparameter Optimization Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ec4899, #8b5cf6, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: monospace;
    }
    .metric-lbl {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94a3b8;
        margin-top: 4px;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA PREPARATION -----------------
@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    return cancer.data, cancer.target, cancer.feature_names, cancer.target_names

X_raw, y_raw, feature_names, target_names = load_data()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("## 🔮 Bayesian Optimizer")
st.sidebar.markdown("### Optimization Setup")

model_choice = st.sidebar.selectbox(
    "Target Model to Optimize:",
    ["Random Forest", "Gradient Boosting", "Support Vector Machine (SVC)"]
)

acq_func = st.sidebar.selectbox(
    "Acquisition Function α(x):",
    ["Expected Improvement (EI)", "Upper Confidence Bound (UCB)", "Probability of Improvement (PI)"],
    index=0
)

col_exp1, col_exp2 = st.sidebar.columns(2)
with col_exp1:
    xi_val = st.number_input("Exploration Tradeoff (ξ):", value=0.01, step=0.005, format="%.3f")
with col_exp2:
    kappa_val = st.number_input("UCB Weight (κ):", value=2.576, step=0.5, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Search Budget")
n_init_samples = st.sidebar.slider("Initial Random Probes:", min_value=3, max_value=8, value=4)
n_bayes_iters = st.sidebar.slider("Bayesian Iterations:", min_value=5, max_value=25, value=12)
cv_folds = st.sidebar.slider("Cross-Validation Folds:", min_value=2, max_value=5, value=3)

# Hyperparameter search boundaries based on model choice
if model_choice == "Random Forest":
    param_bounds = {
        'n_estimators': (20, 250),
        'max_depth': (2, 16),
        'min_samples_split': (2, 10)
    }
elif model_choice == "Gradient Boosting":
    param_bounds = {
        'n_estimators': (20, 200),
        'max_depth': (2, 8),
        'min_samples_split': (2, 10)
    }
else:
    param_bounds = {
        'C': (0.01, 50.0),
        'gamma': (0.0001, 1.0),
        'min_samples_split': (2, 5) # dummy for consistent dimension
    }

# ----------------- BAYESIAN OPTIMIZATION CORE -----------------
def evaluate_objective(params_dict, X_tr, y_tr, model_type, cv):
    if model_type == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=int(params_dict['n_estimators']),
            max_depth=int(params_dict['max_depth']),
            min_samples_split=int(params_dict['min_samples_split']),
            random_state=42
        )
    elif model_type == "Gradient Boosting":
        clf = GradientBoostingClassifier(
            n_estimators=int(params_dict['n_estimators']),
            max_depth=int(params_dict['max_depth']),
            min_samples_split=int(params_dict['min_samples_split']),
            random_state=42
        )
    else:
        clf = SVC(
            C=float(params_dict['C']),
            gamma=float(params_dict['gamma']),
            random_state=42
        )
    scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring='accuracy')
    return scores.mean()

# Acquisition Functions Math
def compute_acquisition(mu, sigma, f_best, acq_type, xi=0.01, kappa=2.576):
    sigma = np.maximum(sigma, 1e-9)
    if acq_type.startswith("Expected Improvement"):
        diff = mu - f_best - xi
        Z = diff / sigma
        ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)
        return np.maximum(ei, 0.0)
    elif acq_type.startswith("Upper Confidence Bound"):
        return mu + kappa * sigma
    else: # Probability of Improvement
        diff = mu - f_best - xi
        Z = diff / sigma
        return norm.cdf(Z)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.25, random_state=42, stratify=y_raw
)

# Optimization Run with Session Caching
@st.cache_resource
def run_full_bayesian_optimization(model_type, acq_type, xi, kappa, n_init, n_iters, cv, seed=42):
    np.random.seed(seed)
    keys = list(param_bounds.keys())
    
    # 1. Generate initial random probes
    history = []
    for i in range(n_init):
        sample = {}
        for k in keys:
            low, high = param_bounds[k]
            if isinstance(low, int) and isinstance(high, int):
                sample[k] = int(np.random.randint(low, high + 1))
            else:
                sample[k] = float(np.random.uniform(low, high))
                
        score = evaluate_objective(sample, X_train, y_train, model_type, cv)
        history.append({
            'iteration': i + 1,
            'type': 'Initial Random Probe',
            'params': sample,
            'score': score
        })

    # Parallel Random Search Trace for comparison benchmark
    random_benchmark = [h['score'] for h in history]
    for _ in range(n_iters):
        r_sample = {}
        for k in keys:
            low, high = param_bounds[k]
            if isinstance(low, int) and isinstance(high, int):
                r_sample[k] = int(np.random.randint(low, high + 1))
            else:
                r_sample[k] = float(np.random.uniform(low, high))
        r_score = evaluate_objective(r_sample, X_train, y_train, model_type, cv)
        random_benchmark.append(r_score)

    # 2. Sequential Bayesian Optimization Loop
    gp_kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0]*len(keys), nu=2.5)
    gp = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-4, normalize_y=True, n_restarts_optimizer=5, random_state=seed)

    for it in range(n_iters):
        # Prepare training data for GP
        X_gp = np.array([[h['params'][k] for k in keys] for h in history])
        y_gp = np.array([h['score'] for h in history])
        f_best = np.max(y_gp)

        # Fit GP Surrogate
        gp.fit(X_gp, y_gp)

        # Generate candidate grid for acquisition optimization
        candidate_count = 500
        candidates = []
        for _ in range(candidate_count):
            cand = {}
            for k in keys:
                low, high = param_bounds[k]
                if isinstance(low, int) and isinstance(high, int):
                    cand[k] = int(np.random.randint(low, high + 1))
                else:
                    cand[k] = float(np.random.uniform(low, high))
            candidates.append(cand)
            
        cand_matrix = np.array([[c[k] for k in keys] for c in candidates])
        mu, sigma = gp.predict(cand_matrix, return_std=True)

        # Compute acquisition values
        acq_vals = compute_acquisition(mu, sigma, f_best, acq_type, xi, kappa)
        best_cand_idx = np.argmax(acq_vals)
        next_params = candidates[best_cand_idx]

        # Evaluate objective function
        score = evaluate_objective(next_params, X_train, y_train, model_type, cv)
        history.append({
            'iteration': n_init + it + 1,
            'type': 'Bayesian Guided Probe',
            'params': next_params,
            'score': score
        })

    return history, random_benchmark, gp

# Header
st.markdown('<div class="main-title">Bayesian Hyperparameter Optimization Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production-grade Gaussian Process surrogate modeling, acquisition function landscapes, and sample-efficient model tuning.</div>', unsafe_allow_html=True)

with st.spinner("Running Bayesian Optimization loop with Gaussian Process surrogate..."):
    history, random_benchmark, final_gp = run_full_bayesian_optimization(
        model_choice, acq_func, xi_val, kappa_val, n_init_samples, n_bayes_iters, cv_folds
    )

# Extract best configuration
scores = [h['score'] for h in history]
best_idx = int(np.argmax(scores))
best_record = history[best_idx]
best_params = best_record['params']
best_cv_score = best_record['score']

# Fit optimal model on full train set and evaluate test set
if model_choice == "Random Forest":
    final_model = RandomForestClassifier(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
        random_state=42
    )
elif model_choice == "Gradient Boosting":
    final_model = GradientBoostingClassifier(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
        random_state=42
    )
else:
    final_model = SVC(
        C=float(best_params['C']),
        gamma=float(best_params['gamma']),
        probability=True,
        random_state=42
    )

final_model.fit(X_train, y_train)
y_pred_test = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

# KPI Metrics Row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color:#ec4899;">{best_cv_score * 100:.2f}%</div><div class="metric-lbl">Best CV Score</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color:#10b981;">{test_accuracy * 100:.2f}%</div><div class="metric-lbl">Holdout Test Acc</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color:#a855f7;">{test_f1 * 100:.2f}%</div><div class="metric-lbl">Holdout Test F1</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color:#38bdf8;">{best_record['iteration']}</div><div class="metric-lbl">Optimal Iteration</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color:#facc15;">{len(history)}</div><div class="metric-lbl">Total Probes</div></div>""", unsafe_allow_html=True)

st.markdown("")
params_badge_str = ", ".join([f"**{k}**: `{v}`" for k, v in best_params.items()])
st.success(f"✨ **Optimal Hyperparameters Discovered:** {params_badge_str}")

# ----------------- TABS INTERFACE -----------------
tab_gp, tab_convergence, tab_3d, tab_eval, tab_infer = st.tabs([
    "📈 Gaussian Process & Acquisition Landscape",
    "⚡ Bayesian vs Random Convergence",
    "🪐 3D Posterior Response Surface",
    "🎯 Holdout Model Evaluation",
    "🧪 Real-Time Diagnostic Lab"
])

# ----------------- TAB 1: GP SURROGATE & ACQUISITION -----------------
with tab_gp:
    st.subheader("Gaussian Process Posterior & Acquisition Function")
    st.caption("Visualizing the surrogate model's predicted mean, epistemic uncertainty (±1.96σ confidence ribbon), and acquisition function peaks.")

    keys = list(param_bounds.keys())
    focus_param = st.selectbox("Inspect GP Along Hyperparameter:", keys, index=0)
    
    # Generate 1D slice across focus_param holding other params at their best values
    low, high = param_bounds[focus_param]
    x_axis = np.linspace(low, high, 100)
    
    slice_candidates = []
    for val in x_axis:
        cand = best_params.copy()
        cand[focus_param] = val
        slice_candidates.append([cand[k] for k in keys])
        
    slice_matrix = np.array(slice_candidates)
    mu_slice, sigma_slice = final_gp.predict(slice_matrix, return_std=True)
    f_best = np.max([h['score'] for h in history])
    acq_slice = compute_acquisition(mu_slice, sigma_slice, f_best, acq_func, xi_val, kappa_val)

    # 1. GP Posterior Plot
    fig_gp = go.Figure()
    
    # Upper & Lower 95% Confidence Band
    fig_gp.add_trace(go.Scatter(
        x=np.concatenate([x_axis, x_axis[::-1]]),
        y=np.concatenate([mu_slice + 1.96 * sigma_slice, (mu_slice - 1.96 * sigma_slice)[::-1]]),
        fill='toself',
        fillcolor='rgba(168, 85, 247, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty (95% CI ±1.96σ)',
        hoverinfo="skip"
    ))
    
    # Predicted Mean Curve
    fig_gp.add_trace(go.Scatter(
        x=x_axis,
        y=mu_slice,
        mode='lines',
        name='GP Posterior Mean μ(x)',
        line=dict(color='#a855f7', width=3)
    ))
    
    # Evaluated Historical Observations
    obs_x = [h['params'][focus_param] for h in history]
    obs_y = [h['score'] for h in history]
    fig_gp.add_trace(go.Scatter(
        x=obs_x,
        y=obs_y,
        mode='markers',
        name='Sampled Probes',
        marker=dict(size=8, color='#38bdf8', symbol='circle', line=dict(color='#ffffff', width=1))
    ))

    fig_gp.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#e2e8f0"),
        height=320,
        xaxis_title=focus_param,
        yaxis_title="Objective Score (Accuracy)",
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_gp, use_container_width=True)

    # 2. Acquisition Function Curve
    fig_acq = go.Figure()
    fig_acq.add_trace(go.Scatter(
        x=x_axis,
        y=acq_slice,
        mode='lines',
        name=f'{acq_func}',
        line=dict(color='#ec4899', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(236, 72, 153, 0.1)'
    ))
    
    # Mark maximum acquisition
    best_acq_x = x_axis[np.argmax(acq_slice)]
    fig_acq.add_trace(go.Scatter(
        x=[best_acq_x],
        y=[np.max(acq_slice)],
        mode='markers+text',
        name='Next Acquisition Target',
        marker=dict(size=10, color='#facc15', symbol='star'),
        text=['Next Probe Target'],
        textposition='top center'
    ))

    fig_acq.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#e2e8f0"),
        height=200,
        xaxis_title=focus_param,
        yaxis_title="α(x) Utility",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    st.plotly_chart(fig_acq, use_container_width=True)

# ----------------- TAB 2: CONVERGENCE BENCHMARK -----------------
with tab_convergence:
    st.subheader("Optimization Efficiency: Bayesian vs. Random Search")
    st.caption("Bayesian optimization leverages prior iterations to focus on high-yield regions, converging significantly faster than random exploration.")

    # Calculate cumulative maximums
    bayes_scores = [h['score'] for h in history]
    bayes_cum_max = np.maximum.accumulate(bayes_scores)
    random_cum_max = np.maximum.accumulate(random_benchmark)

    iterations = list(range(1, len(history) + 1))

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=iterations,
        y=bayes_cum_max,
        mode='lines+markers',
        name='Bayesian Optimization (GP + Acquisition)',
        line=dict(color='#ec4899', width=3),
        marker=dict(size=7)
    ))
    fig_conv.add_trace(go.Scatter(
        x=iterations,
        y=random_cum_max,
        mode='lines+markers',
        name='Random Search Baseline',
        line=dict(color='#64748b', width=2, dash='dash'),
        marker=dict(size=5)
    ))

    fig_conv.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#e2e8f0"),
        height=380,
        xaxis_title="Optimization Iteration (t)",
        yaxis_title="Best Objective Score Encountered",
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98)
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # Detailed Search Log Table
    with st.expander("📋 View Complete Exploration History Table"):
        log_rows = []
        for h in history:
            row = {'Iteration': h['iteration'], 'Probe Type': h['type'], 'CV Score': h['score']}
            row.update(h['params'])
            log_rows.append(row)
        log_df = pd.DataFrame(log_rows)
        st.dataframe(
            log_df.style.highlight_max(subset=['CV Score'], color='#065f46')
                        .format({'CV Score': '{:.4f}'}),
            use_container_width=True,
            hide_index=True
        )

# ----------------- TAB 3: 3D RESPONSE SURFACE -----------------
with tab_3d:
    st.subheader("3D Bayesian Posterior Response Surface")
    st.caption("Interactive 3D topographic manifold showing objective function peaks across two chosen hyperparameter dimensions.")

    keys = list(param_bounds.keys())
    p1 = keys[0]
    p2 = keys[1]
    
    # 2D Meshgrid
    low1, high1 = param_bounds[p1]
    low2, high2 = param_bounds[p2]
    
    grid1 = np.linspace(low1, high1, 24)
    grid2 = np.linspace(low2, high2, 24)
    X1, X2 = np.meshgrid(grid1, grid2)
    
    pts = []
    for g1, g2 in zip(X1.ravel(), X2.ravel()):
        c = best_params.copy()
        c[p1] = g1
        c[p2] = g2
        pts.append([c[k] for k in keys])
        
    Z_mean = final_gp.predict(np.array(pts)).reshape(X1.shape)

    fig_3d = go.Figure()
    
    # 3D GP Surface
    fig_3d.add_trace(go.Surface(
        x=X1, y=X2, z=Z_mean,
        colorscale='Viridis',
        opacity=0.85,
        name='GP Posterior Surface'
    ))
    
    # Historical Evaluation Points floating on surface
    hx1 = [h['params'][p1] for h in history]
    hx2 = [h['params'][p2] for h in history]
    hz = [h['score'] for h in history]
    
    fig_3d.add_trace(go.Scatter3d(
        x=hx1, y=hx2, z=hz,
        mode='markers',
        name='Evaluated Probes',
        marker=dict(size=5, color='#ec4899', symbol='circle')
    ))
    
    # Optimal point marker
    fig_3d.add_trace(go.Scatter3d(
        x=[best_params[p1]], y=[best_params[p2]], z=[best_cv_score],
        mode='markers+text',
        name='Global Optimum',
        marker=dict(size=12, color='#facc15', symbol='diamond'),
        text=['🌟 Optimal'],
        textposition='top center'
    ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title=p1,
            yaxis_title=p2,
            zaxis_title='Predicted Score',
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

# ----------------- TAB 4: HOLDOUT MODEL EVALUATION -----------------
with tab_eval:
    st.subheader("Holdout Test Set Validation")
    st.caption("Final evaluation of the tuned model on pristine unseen test instances.")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_test)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {t}" for t in target_names],
            columns=[f"Pred {t}" for t in target_names]
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Purples",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig_cm.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.markdown("#### ROC Curve")
        if hasattr(final_model, "predict_proba"):
            y_probs = final_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc_val = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Tuned Model (AUC = {roc_auc_val:.3f})', line=dict(color='#ec4899', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance Baseline', line=dict(color='#64748b', dash='dash')))
            fig_roc.update_layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#0b0f19",
                font=dict(color="#e2e8f0"),
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=340,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

# ----------------- TAB 5: REAL-TIME DIAGNOSTIC LAB -----------------
with tab_infer:
    st.subheader("🧪 Live Patient Diagnostic Inference")
    st.markdown("Test the Bayesian-optimized classifier on clinical tumor measurement inputs:")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        s_radius = st.slider("Mean Radius:", float(X_raw[:, 0].min()), float(X_raw[:, 0].max()), float(X_raw[:, 0].mean()))
        s_texture = st.slider("Mean Texture:", float(X_raw[:, 1].min()), float(X_raw[:, 1].max()), float(X_raw[:, 1].mean()))
    with col_s2:
        s_perimeter = st.slider("Mean Perimeter:", float(X_raw[:, 2].min()), float(X_raw[:, 2].max()), float(X_raw[:, 2].mean()))
        s_area = st.slider("Mean Area:", float(X_raw[:, 3].min()), float(X_raw[:, 3].max()), float(X_raw[:, 3].mean()))
    with col_s3:
        s_smoothness = st.slider("Mean Smoothness:", float(X_raw[:, 4].min()), float(X_raw[:, 4].max()), float(X_raw[:, 4].mean()))
        s_concavity = st.slider("Mean Concavity:", float(X_raw[:, 6].min()), float(X_raw[:, 6].max()), float(X_raw[:, 6].mean()))
        
    if st.button("🔬 Predict Clinical Diagnosis", type="primary", use_container_width=True):
        sample = X_raw.mean(axis=0)
        sample[0] = s_radius
        sample[1] = s_texture
        sample[2] = s_perimeter
        sample[3] = s_area
        sample[4] = s_smoothness
        sample[6] = s_concavity
        
        pred_class = final_model.predict([sample])[0]
        pred_label = target_names[pred_class]
        
        confidence = 100.0
        if hasattr(final_model, "predict_proba"):
            probs = final_model.predict_proba([sample])[0]
            confidence = probs[pred_class] * 100.0
            
        color = "#10b981" if pred_class == 1 else "#ef4444"
        st.markdown(f"""
        <div style="background:#1e293b; border-left: 6px solid {color}; border-radius:12px; padding:20px; margin-top:16px;">
            <div style="font-size:1.4rem; font-weight:700; color:{color};">
                Predicted Diagnosis: {pred_label.upper()}
            </div>
            <div style="font-size:0.95rem; color:#e2e8f0; margin-top:6px;">
                Prediction Confidence: <strong>{confidence:.1f}%</strong> computed via Bayesian-optimized {model_choice}.
            </div>
        </div>
        """, unsafe_allow_html=True)


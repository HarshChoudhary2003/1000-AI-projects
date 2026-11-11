import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score,
    matthews_corrcoef, roc_curve, precision_recall_curve, auc
)
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Confusion Matrix Diagnostic Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981, #06b6d4, #6366f1);
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
st.sidebar.markdown("## 🎯 Diagnostic Studio")
st.sidebar.markdown("### Dataset & Model Setup")

dataset_choice = st.sidebar.selectbox(
    "Choose Classification Task:",
    [
        "Iris Botanical (3 Classes - Multiclass)",
        "Breast Cancer Diagnostic (2 Classes - Binary)",
        "Wine Recognition (3 Classes - Multiclass)"
    ],
    index=0
)

model_choice = st.sidebar.selectbox(
    "Classifier Algorithm:",
    ["Logistic Regression", "Random Forest", "Support Vector Machine (SVC)", "K-Nearest Neighbors"],
    index=0
)

test_size = st.sidebar.slider("Test Set Split Ratio:", min_value=0.15, max_value=0.50, value=0.30, step=0.05)
random_seed = st.sidebar.number_input("Random Seed:", value=42, step=1)

# Normalization method for Confusion Matrix
norm_mode = st.sidebar.selectbox(
    "Matrix Normalization Mode:",
    ["None (Raw Counts)", "pred (Precision / By Column)", "true (Recall / By Row)", "all (Total %)"]
)

# ----------------- DATA LOADING & TRAINING -----------------
@st.cache_data
def get_data(choice):
    if choice.startswith("Iris"):
        data = load_iris()
        is_binary = False
    elif choice.startswith("Breast"):
        data = load_breast_cancer()
        is_binary = True
    else:
        data = load_wine()
        is_binary = False
    return data.data, data.target, data.feature_names, data.target_names, is_binary

X, y, feature_names, target_names, is_binary = get_data(dataset_choice)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed, stratify=y
)

# Fit model
def train_model(model_name, X_tr, y_tr, seed):
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=500, random_state=seed)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    elif model_name == "Support Vector Machine (SVC)":
        clf = SVC(probability=True, random_state=seed)
    else:
        clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_tr, y_tr)
    return clf

clf = train_model(model_choice, X_train, y_train, random_seed)

# Probability and Threshold control for binary tasks
threshold = 0.50
if is_binary and hasattr(clf, "predict_proba"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎚️ Decision Threshold")
    threshold = st.sidebar.slider(
        "Positive Decision Threshold (τ):",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05
    )
    y_probs = clf.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)
else:
    y_preds = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1] if is_binary and hasattr(clf, "predict_proba") else None

# Header
st.markdown('<div class="main-title">Confusion Matrix Diagnostic Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive error analysis, normalized confusion heatmaps, per-class precision/recall decomposition, and decision boundary diagnostics.</div>', unsafe_allow_html=True)

# ----------------- METRICS COMPUTATIONS -----------------
cm_raw = confusion_matrix(y_test, y_preds)
accuracy = accuracy_score(y_test, y_preds)
macro_f1 = f1_score(y_test, y_preds, average='weighted', zero_division=0)
kappa = cohen_kappa_score(y_test, y_preds)
mcc = matthews_corrcoef(y_test, y_preds)

# Top KPI Summary Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#10b981;">{accuracy * 100:.1f}%</div><div class="stat-lbl">Overall Accuracy</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#38bdf8;">{macro_f1 * 100:.1f}%</div><div class="stat-lbl">Weighted F1-Score</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#a855f7;">{kappa:.3f}</div><div class="stat-lbl">Cohen's Kappa (κ)</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:#f59e0b;">{mcc:.3f}</div><div class="stat-lbl">Matthews Corr (MCC)</div></div>""", unsafe_allow_html=True)
with c5:
    total_errors = int((y_test != y_preds).sum())
    err_color = "#ef4444" if total_errors > 0 else "#10b981"
    st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color:{err_color};">{total_errors}</div><div class="stat-lbl">Misclassified Samples</div></div>""", unsafe_allow_html=True)

st.markdown("")

# ----------------- TABS INTERFACE -----------------
tab_matrix, tab_per_class, tab_scatter, tab_tradeoff, tab_custom = st.tabs([
    "🎯 Interactive Confusion Matrix",
    "📊 Per-Class Decomposition",
    "🪐 Error Space Inspection (2D/3D)",
    "🎚️ Threshold & Trade-off Curves",
    "🧪 Custom Matrix Scenario Sandbox"
])

# ----------------- TAB 1: INTERACTIVE CONFUSION MATRIX -----------------
with tab_matrix:
    col_m1, col_m2 = st.columns([3, 2])
    
    with col_m1:
        st.subheader("Confusion Matrix Visualization")
        
        # Calculate normalized matrix
        normalize_arg = None if norm_mode.startswith("None") else norm_mode.split()[0]
        cm_display = confusion_matrix(y_test, y_preds, normalize=normalize_arg)
        
        # Text annotations
        fmt = ".1f" if normalize_arg is None else ".2%"
        
        fig_cm = px.imshow(
            cm_display,
            x=[f"Pred: {t}" for t in target_names],
            y=[f"Actual: {t}" for t in target_names],
            text_auto=fmt,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class", color="Rate / Count")
        )
        fig_cm.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=440,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        st.subheader("Matrix Anatomy & Insights")
        
        if is_binary:
            tn, fp, fn, tp = cm_raw.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            st.markdown(f"""
            <div style="background:#1e293b; border-radius:12px; padding:16px; border:1px solid #334155; margin-bottom:12px;">
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; font-size:0.85rem;">
                    <div style="background:rgba(16, 185, 129, 0.15); border:1px solid #10b981; padding:8px; border-radius:8px;">
                        <span style="color:#6ee7b7; font-weight:bold;">True Negative (TN):</span><br>
                        <span style="font-size:1.2rem; font-family:monospace; color:#f8fafc;">{tn}</span> ({tnr*100:.1f}% Specificity)
                    </div>
                    <div style="background:rgba(239, 68, 68, 0.15); border:1px solid #ef4444; padding:8px; border-radius:8px;">
                        <span style="color:#fca5a5; font-weight:bold;">False Positive (FP / Type I):</span><br>
                        <span style="font-size:1.2rem; font-family:monospace; color:#f8fafc;">{fp}</span> (False Alarm)
                    </div>
                    <div style="background:rgba(239, 68, 68, 0.15); border:1px solid #ef4444; padding:8px; border-radius:8px;">
                        <span style="color:#fca5a5; font-weight:bold;">False Negative (FN / Type II):</span><br>
                        <span style="font-size:1.2rem; font-family:monospace; color:#f8fafc;">{fn}</span> (Missed Positive)
                    </div>
                    <div style="background:rgba(16, 185, 129, 0.15); border:1px solid #10b981; padding:8px; border-radius:8px;">
                        <span style="color:#6ee7b7; font-weight:bold;">True Positive (TP):</span><br>
                        <span style="font-size:1.2rem; font-family:monospace; color:#f8fafc;">{tp}</span> ({tpr*100:.1f}% Sensitivity)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            - **Sensitivity / Recall (TPR)**: `{tpr * 100:.2f}%`
            - **Specificity (TNR)**: `{tnr * 100:.2f}%`
            - **Precision (PPV)**: `{ppv * 100:.2f}%`
            - **Negative Predictive Value (NPV)**: `{npv * 100:.2f}%`
            """)
        else:
            st.markdown("#### Diagonal vs Off-Diagonal Analysis")
            diagonal_sum = np.diag(cm_raw).sum()
            off_diagonal_sum = cm_raw.sum() - diagonal_sum
            
            st.metric("Correct Predictions (Diagonal)", f"{diagonal_sum} / {cm_raw.sum()}")
            st.metric("Confusion Errors (Off-Diagonal)", f"{off_diagonal_sum}")
            
            # Find worst confusion pair
            off_diag_matrix = cm_raw.copy()
            np.fill_diagonal(off_diag_matrix, 0)
            if off_diag_matrix.max() > 0:
                max_err_idx = np.unravel_index(np.argmax(off_diag_matrix), off_diag_matrix.shape)
                actual_name = target_names[max_err_idx[0]]
                pred_name = target_names[max_err_idx[1]]
                count = off_diag_matrix[max_err_idx]
                st.warning(f"⚠️ **Primary Source of Confusion**: Actual **{actual_name}** misclassified as **{pred_name}** ({count} instances).")
            else:
                st.success("🎉 Perfect Classification! Zero off-diagonal errors detected on this test fold.")

# ----------------- TAB 2: PER-CLASS DECOMPOSITION -----------------
with tab_per_class:
    st.subheader("Per-Class Precision, Recall, and Specificity")
    
    report_dict = classification_report(y_test, y_preds, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T.iloc[:len(target_names)]
    
    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        fig_bar = px.bar(
            report_df[['precision', 'recall', 'f1-score']].reset_index(),
            x='index',
            y=['precision', 'recall', 'f1-score'],
            barmode='group',
            labels=dict(index="Class", value="Score", variable="Metric"),
            color_discrete_map={
                "precision": "#0ea5e9",
                "recall": "#10b981",
                "f1-score": "#8b5cf6"
            }
        )
        fig_bar.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font=dict(color="#e2e8f0"),
            height=360,
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_p2:
        st.markdown("#### Per-Class Diagnostic Summary")
        for i, name in enumerate(target_names):
            row_total = cm_raw[i].sum()
            correct = cm_raw[i, i]
            acc = correct / row_total if row_total > 0 else 0
            st.markdown(f"""
            <div style="background:#1e293b; border-radius:8px; padding:10px 14px; margin-bottom:8px; border-left:4px solid #10b981;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-weight:600; color:#f8fafc;">{name.capitalize()}</span>
                    <span style="font-family:monospace; color:#38bdf8;">{acc*100:.1f}% Recall</span>
                </div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:2px;">
                    {correct} of {row_total} test cases classified correctly
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 3: ERROR SPACE INSPECTION -----------------
with tab_scatter:
    st.subheader("Error Space Inspection: Spotting Where Misclassifications Occur")
    st.caption("Inspect which test points fell outside the decision boundary in feature space.")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_x = st.selectbox("Feature for X-Axis:", range(len(feature_names)), format_func=lambda i: feature_names[i], index=0)
    with col_f2:
        f_y = st.selectbox("Feature for Y-Axis:", [i for i in range(len(feature_names)) if i != f_x], format_func=lambda i: feature_names[i], index=1)

    scatter_df = pd.DataFrame({
        feature_names[f_x]: X_test[:, f_x],
        feature_names[f_y]: X_test[:, f_y],
        "Actual Class": [target_names[i] for i in y_test],
        "Predicted Class": [target_names[i] for i in y_preds],
        "Result": ["Correct" if y_test[i] == y_preds[i] else "Misclassified (Error)" for i in range(len(y_test))]
    })

    fig_err = px.scatter(
        scatter_df,
        x=feature_names[f_x],
        y=feature_names[f_y],
        color="Result",
        symbol="Actual Class",
        color_discrete_map={"Correct": "#38bdf8", "Misclassified (Error)": "#ef4444"},
        hover_data=["Actual Class", "Predicted Class"]
    )
    fig_err.update_traces(marker=dict(size=8, line=dict(width=1, color='#ffffff')))
    fig_err.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#e2e8f0"),
        height=450,
        margin=dict(l=10, r=10, t=20, b=20)
    )
    st.plotly_chart(fig_err, use_container_width=True)

# ----------------- TAB 4: THRESHOLD & TRADEOFFS -----------------
with tab_tradeoff:
    st.subheader("Decision Threshold & Trade-off Dynamics")
    
    if is_binary and y_probs is not None:
        fpr, tpr, roc_thresh = roc_curve(y_test, y_probs)
        roc_auc_val = auc(fpr, tpr)
        
        prec, rec, pr_thresh = precision_recall_curve(y_test, y_probs)
        
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            st.markdown("#### ROC Curve with Operating Threshold")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc_val:.3f})', line=dict(color='#06b6d4', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='#64748b', dash='dash')))
            
            # Highlight current operating point
            current_tpr = (cm_raw[1, 1]) / (cm_raw[1, 1] + cm_raw[1, 0]) if (cm_raw[1, 1] + cm_raw[1, 0]) > 0 else 0
            current_fpr = (cm_raw[0, 1]) / (cm_raw[0, 1] + cm_raw[0, 0]) if (cm_raw[0, 1] + cm_raw[0, 0]) > 0 else 0
            fig_roc.add_trace(go.Scatter(
                x=[current_fpr], y=[current_tpr],
                mode='markers+text',
                marker=dict(size=12, color='#facc15'),
                text=[f"τ = {threshold:.2f}"],
                textposition='bottom right',
                name='Operating Point'
            ))
            fig_roc.update_layout(
                paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19", font=dict(color="#e2e8f0"),
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Recall)",
                height=350, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with c_tr2:
            st.markdown("#### Precision-Recall Trade-off Curve")
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name='PR Curve', line=dict(color='#a855f7', width=3)))
            fig_pr.update_layout(
                paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19", font=dict(color="#e2e8f0"),
                xaxis_title="Recall", yaxis_title="Precision",
                height=350, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.info("Switch to the Binary task (Breast Cancer Diagnostic) in the sidebar to activate interactive probability threshold curves!")

# ----------------- TAB 5: CUSTOM MATRIX SCENARIO SANDBOX -----------------
with tab_custom:
    st.subheader("🧪 Theoretical Confusion Matrix Sandbox")
    st.caption("Input custom matrix counts to simulate clinical, financial, or cyber-security error budgets:")

    col_cs1, col_cs2 = st.columns([2, 3])
    with col_cs1:
        sb_tp = st.number_input("Custom True Positives (TP):", value=95, min_value=0)
        sb_fp = st.number_input("Custom False Positives (FP):", value=15, min_value=0)
        sb_fn = st.number_input("Custom False Negatives (FN):", value=5, min_value=0)
        sb_tn = st.number_input("Custom True Negatives (TN):", value=885, min_value=0)
        
        cost_fp = st.number_input("Cost per False Positive ($):", value=50, step=10)
        cost_fn = st.number_input("Cost per False Negative ($):", value=500, step=50)

    with col_cs2:
        total_custom = sb_tp + sb_fp + sb_fn + sb_tn
        c_acc = (sb_tp + sb_tn) / total_custom if total_custom > 0 else 0
        c_prec = sb_tp / (sb_tp + sb_fp) if (sb_tp + sb_fp) > 0 else 0
        c_rec = sb_tp / (sb_tp + sb_fn) if (sb_tp + sb_fn) > 0 else 0
        c_f1 = 2 * (c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0
        total_cost = (sb_fp * cost_fp) + (sb_fn * cost_fn)
        
        st.markdown("#### Simulated Business & Safety Metrics")
        k1, k2, k3 = st.columns(3)
        k1.metric("Simulated Accuracy", f"{c_acc * 100:.2f}%")
        k2.metric("Simulated Precision", f"{c_prec * 100:.2f}%")
        k3.metric("Simulated Recall", f"{c_rec * 100:.2f}%")
        
        st.metric("Total Cost of Errors", f"${total_cost:,.2f}", delta=f"-${sb_fn * cost_fn:,.2f} from missed detections", delta_color="inverse")
        
        custom_cm_data = np.array([[sb_tn, sb_fp], [sb_fn, sb_tp]])
        fig_custom_cm = px.imshow(
            custom_cm_data,
            x=["Pred: Neg", "Pred: Pos"],
            y=["Actual: Neg", "Actual: Pos"],
            text_auto=True,
            color_continuous_scale="Purples"
        )
        fig_custom_cm.update_layout(paper_bgcolor="#0b0f19", height=240, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_custom_cm, use_container_width=True)

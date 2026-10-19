import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Logistic Regression Classifier",
    page_icon="⚖️",
    layout="wide"
)

# Realistic feature labels for binary classification (Loan / Credit Default scenario)
FEATURE_LABELS = [
    "Debt-to-Income Ratio",
    "Credit Utilization (%)",
    "Payment History Score",
    "Annual Income ($k)",
    "Open Credit Lines",
    "Delinquent Accounts",
    "Employment Duration (yrs)",
    "Derogatory Public Records",
    "Recent Credit Inquiries",
    "Savings Balance ($k)"
]

@st.cache_data
def generate_and_split_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        flip_y=0.03,
        random_state=42
    )
    df = pd.DataFrame(X, columns=FEATURE_LABELS)
    df["Target"] = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_LABELS], df["Target"], test_size=0.25, stratify=df["Target"], random_state=42
    )
    return df, X_train, X_test, y_train, y_test

@st.cache_resource
def train_logistic_pipeline(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=500))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

df_full, X_train, X_test, y_train, y_test = generate_and_split_data()
model = train_logistic_pipeline(X_train, y_train)

# Model probabilities on test set
test_probs = model.predict_proba(X_test)[:, 1]
auc_val = roc_auc_score(y_test, test_probs)

# --- Sidebar Controls ---
st.sidebar.header("Interactive Controls")

# Decision threshold adjustment
threshold = st.sidebar.slider(
    "Classification Threshold (τ)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Default is 0.50. Lower thresholds prioritize Recall (catching all positives); higher thresholds prioritize Precision."
)

st.sidebar.divider()
st.sidebar.subheader("Live Single-Record Scorer")

sample_inputs = {}
# Give user sliders for top 5 most impactful features
clf = model.named_steps["classifier"]
top_indices = np.argsort(np.abs(clf.coef_[0]))[::-1][:5]

for idx in top_indices:
    name = FEATURE_LABELS[idx]
    mean_val = float(df_full[name].mean())
    sample_inputs[name] = st.sidebar.slider(
        name,
        min_value=-3.0,
        max_value=3.0,
        value=round(mean_val, 2),
        step=0.1
    )

# Fill unselected features with median
full_sample = [sample_inputs.get(col, float(df_full[col].median())) for col in FEATURE_LABELS]
sample_prob = model.predict_proba([full_sample])[0, 1]
sample_pred = int(sample_prob >= threshold)

# --- Main Dashboard ---
st.title("⚖️ Logistic Regression: Binary Classification Explorer")
st.caption("Inspect decision boundaries, probability calibration, odds ratios, and threshold tuning in real-time.")

# Top KPIs for test set at selected threshold
y_pred_thresh = (test_probs >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred_thresh)
prec = precision_score(y_test, y_pred_thresh, zero_division=0)
rec = recall_score(y_test, y_pred_thresh, zero_division=0)
f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("ROC-AUC Score", f"{auc_val:.4f}", help="Threshold-independent model discriminative ability.")
kpi2.metric("Accuracy", f"{acc * 100:.1f}%")
kpi3.metric("Precision", f"{prec * 100:.1f}%")
kpi4.metric("Recall", f"{rec * 100:.1f}%")
kpi5.metric("F1-Score", f"{f1 * 100:.1f}%")

st.divider()

# Row 1: Real-time prediction card & Feature Importances
c_pred, c_coef = st.columns([1, 1])

with c_pred:
    st.subheader("🎯 Live Inference Result")
    outcome_label = "Class 1 (High Risk / Positive)" if sample_pred == 1 else "Class 0 (Low Risk / Negative)"
    badge_color = "red" if sample_pred == 1 else "green"
    
    st.markdown(
        f"""
        <div style="background-color: {'#fee2e2' if sample_pred == 1 else '#dcfce7'}; 
                    padding: 20px; border-radius: 8px; border-left: 5px solid {'#ef4444' if sample_pred == 1 else '#22c55e'};">
            <h4 style="margin: 0; color: {'#991b1b' if sample_pred == 1 else '#166534'};">
                {outcome_label}
            </h4>
            <p style="margin: 8px 0 0 0; font-size: 15px;">
                Calculated Probability: <strong>{sample_prob * 100:.2f}%</strong> &nbsp;|&nbsp; Operating Threshold: <strong>{threshold * 100:.0f}%</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Probability progress bar
    st.progress(float(sample_prob))
    st.caption("Left: 0.0 (Class 0) ── Threshold Indicator ── Right: 1.0 (Class 1)")

with c_coef:
    st.subheader("📊 Learned Coefficients (Log-Odds)")
    coef_df = pd.DataFrame({
        "Feature": FEATURE_LABELS,
        "Coefficient": clf.coef_[0],
        "Odds Ratio (e^β)": np.exp(clf.coef_[0])
    }).sort_values("Coefficient", key=abs, ascending=True)
    
    fig_bar, ax_bar = plt.subplots(figsize=(6.5, 3.8), dpi=100)
    colors = ["#ef4444" if c > 0 else "#3b82f6" for c in coef_df["Coefficient"]]
    ax_bar.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax_bar.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax_bar.set_xlabel("Weight (Positive = increases probability)")
    ax_bar.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_bar)

st.divider()

# Row 2: ROC Curve & Confusion Matrix
c_roc, c_cm = st.columns([1, 1])

with c_roc:
    st.subheader("📈 ROC Curve & Operating Point")
    fpr, tpr, roc_thresh = roc_curve(y_test, test_probs)
    
    # Find closest threshold index
    closest_idx = np.argmin(np.abs(roc_thresh - threshold))
    
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.2), dpi=100)
    ax_roc.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Guess")
    
    # Highlight current operating point
    ax_roc.scatter([fpr[closest_idx]], [tpr[closest_idx]], color="#dc2626", s=100, zorder=5, 
                   label=f"Current Threshold ({threshold:.2f})")
    
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_roc)

with c_cm:
    st.subheader(f"🗂️ Confusion Matrix (Threshold = {threshold:.2f})")
    cm = confusion_matrix(y_test, y_pred_thresh)
    
    fig_cm, ax_cm = plt.subplots(figsize=(4.8, 4.2), dpi=100)
    im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax_cm.set_yticklabels(["Actual 0", "Actual 1"])
    
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax_cm.text(j, i, f"{val}\n({val/len(y_test)*100:.1f}%)", ha="center", va="center", color=color, fontsize=12)
            
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig_cm)

with st.expander("Inspect Raw Test Set & Predictions"):
    results_df = X_test.copy()
    results_df["Actual"] = y_test
    results_df["Predicted_Probability"] = np.round(test_probs, 4)
    results_df["Predicted_Class"] = y_pred_thresh
    st.dataframe(results_df, use_container_width=True)

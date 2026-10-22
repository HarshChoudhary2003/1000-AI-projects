"""
Naive Bayes Spam & Phishing Detection Studio
============================================
An elite, interactive Streamlit workbench for text classification, threat detection,
and Explainable AI (XAI) using Naive Bayes probabilistic models.

Features:
- Real & Expandable Corpus: 1,200+ Curated Phishing, Financial Scams, SMS & Workplace Emails
- Multi-Model Naive Bayes Engine: Multinomial, Bernoulli, and Complement Naive Bayes
- Live Threat Detection & Explainable AI: Token-level log-odds attribution & visual highlighting
- Hyperparameter Studio: Laplace/Lidstone smoothing (α), TF-IDF vs Count Vectorizer, n-grams
- Diagnostic Analytics: Interactive Confusion Matrix, ROC-AUC, Precision-Recall, Threshold Trade-offs
- Batch Threat Scanner: Bulk text analysis & CSV threat export
- Mathematical Deep Dive: Step-by-step logarithmic probability walkthrough
- Reproducible Python Code Generator & Model Pipeline Exporter
"""

import io
import re
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)

from dataset import generate_expanded_corpus, CURATED_DATA


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Naive Bayes Spam Studio | AI Security Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System (Titanium Obsidian & Champagne Gold)
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
    .studio-card-danger {
        border-left: 5px solid #EF4444;
    }
    .studio-card-success {
        border-left: 5px solid #10B981;
    }
    
    /* Token Highlight Badges */
    .token-spam {
        background: rgba(239, 68, 68, 0.25);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
        margin: 2px 2px;
    }
    .token-ham {
        background: rgba(16, 185, 129, 0.22);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
        margin: 2px 2px;
    }
    .token-neutral {
        color: #CBD5E1;
        padding: 2px 4px;
        display: inline-block;
    }
    
    /* Code Snippet Containers */
    .code-callout {
        background: #05070A;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #38BDF8;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        font-weight: 700;
        font-size: 0.92rem;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper & Training Functions
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_corpus(dataset_choice, corpus_size, spam_ratio, random_seed):
    """Loads or synthesizes the training and validation corpus."""
    if dataset_choice == "Curated Real Corpus (1,200+ samples)":
        return generate_expanded_corpus(target_size=corpus_size, spam_ratio=spam_ratio, random_seed=random_seed)
    elif dataset_choice == "Base Handcrafted Seed Corpus (60 samples)":
        df = pd.DataFrame(CURATED_DATA, columns=["text", "label"])
        df["source"] = "curated_base"
        return df
    else:
        # Fallback to standard generated corpus
        return generate_expanded_corpus(target_size=corpus_size, spam_ratio=spam_ratio, random_seed=random_seed)

@st.cache_resource(show_spinner=False)
def train_naive_bayes_model(df, model_type, vectorizer_type, alpha, ngram_range, use_stop_words, sublinear_tf):
    """Trains the end-to-end vectorizer and Naive Bayes pipeline."""
    X = df["text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 1. Feature Extractor
    stop_words_arg = 'english' if use_stop_words else None
    
    if vectorizer_type == "TF-IDF Vectorizer":
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words_arg,
            sublinear_tf=sublinear_tf,
            min_df=1
        )
    else:
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words_arg,
            min_df=1
        )
        
    # 2. Classifier Engine
    if model_type == "Multinomial Naive Bayes":
        classifier = MultinomialNB(alpha=alpha)
    elif model_type == "Bernoulli Naive Bayes":
        classifier = BernoulliNB(alpha=alpha, binarize=0.0)
    else: # Complement Naive Bayes
        classifier = ComplementNB(alpha=alpha)
        
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Validation Evaluation
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.50).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "vocab_size": len(pipeline.named_steps['vectorizer'].vocabulary_),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    
    return pipeline, metrics, (X_train, X_test, y_train, y_test)


def compute_token_attributions(pipeline, text):
    """
    Computes token-level log-odds ratios to explain why a message was flagged as spam or ham.
    Log-Odds = log(P(w | Spam) / P(w | Ham))
    """
    vec = pipeline.named_steps['vectorizer']
    clf = pipeline.named_steps['classifier']
    
    feature_names = vec.get_feature_names_out()
    vocab = vec.vocabulary_
    
    # Feature log probabilities per class: shape (2, n_features)
    if hasattr(clf, "feature_log_prob_"):
        ham_log_probs = clf.feature_log_prob_[0]
        spam_log_probs = clf.feature_log_prob_[1]
    elif hasattr(clf, "feature_count_"):
        # Smoothed counts calculation
        smoothed = clf.feature_count_ + clf.alpha
        smoothed_totals = smoothed.sum(axis=1, keepdims=True)
        log_probs = np.log(smoothed / smoothed_totals)
        ham_log_probs = log_probs[0]
        spam_log_probs = log_probs[1]
    else:
        return []
    
    # Tokenize input message using basic word boundaries
    raw_tokens = re.findall(r'\b[a-zA-Z0-9_\$]+\b', text)
    token_analysis = []
    
    for raw_token in raw_tokens:
        token_lower = raw_token.lower()
        if token_lower in vocab:
            idx = vocab[token_lower]
            p_spam_log = spam_log_probs[idx]
            p_ham_log = ham_log_probs[idx]
            log_odds = p_spam_log - p_ham_log
            
            token_analysis.append({
                "token": raw_token,
                "in_vocab": True,
                "log_odds": log_odds,
                "direction": "Spam" if log_odds > 0.3 else ("Ham" if log_odds < -0.3 else "Neutral"),
                "odds_ratio": math.exp(np.clip(log_odds, -10, 10))
            })
        else:
            token_analysis.append({
                "token": raw_token,
                "in_vocab": False,
                "log_odds": 0.0,
                "direction": "Unseen",
                "odds_ratio": 1.0
            })
            
    return token_analysis


# ==============================================================================
# Sidebar: Controls & Model Architecture Studio
# ==============================================================================
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 12px;">
    <h2 style="margin: 0; font-size: 1.4rem; color: #F59E0B; font-family: 'Outfit', sans-serif;">🛡️ Spam & Threat Studio</h2>
    <span style="font-size: 0.75rem; color: #94A3B8;">Project #0006 • Probabilistic NLP Suite</span>
</div>
""", unsafe_allow_html=True)

# 1. Dataset Configuration
st.sidebar.markdown("### 🗂️ 1. Corpus Dataset")
dataset_choice = st.sidebar.selectbox(
    "Corpus Source",
    [
        "Curated Real Corpus (1,200+ samples)",
        "Base Handcrafted Seed Corpus (60 samples)",
        "Upload Custom CSV File"
    ],
    index=0
)

custom_df = None
if dataset_choice == "Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Requires 'text' & 'label' columns)", type=["csv"])
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(custom_df)} rows from custom CSV!")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")

if dataset_choice == "Curated Real Corpus (1,200+ samples)":
    corpus_size = st.sidebar.slider("Total Corpus Size", min_value=300, max_value=2500, value=1200, step=100)
    spam_ratio = st.sidebar.slider("Spam Proportion (%)", min_value=15, max_value=60, value=35, step=5) / 100.0
else:
    corpus_size = 60
    spam_ratio = 0.50

# 2. Model Selection
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 2. Naive Bayes Engine")
model_type = st.sidebar.selectbox(
    "Algorithm Variant",
    [
        "Multinomial Naive Bayes",
        "Complement Naive Bayes",
        "Bernoulli Naive Bayes"
    ],
    index=0,
    help="Multinomial: Term frequencies; Complement: Tuned for imbalanced text; Bernoulli: Binary word occurrence."
)

alpha = st.sidebar.slider(
    "Laplace / Lidstone Smoothing (α)",
    min_value=0.01,
    max_value=3.0,
    value=0.50,
    step=0.05,
    help="Prevents zero-probability trap for unseen words. Higher α = smoother probabilities."
)

# 3. Vectorization & NLP Preprocessing
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔤 3. NLP Vectorization")
vectorizer_type = st.sidebar.radio(
    "Feature Representation",
    ["TF-IDF Vectorizer", "Count Vectorizer (Bag of Words)"],
    index=0
)

ngram_selection = st.sidebar.selectbox(
    "N-Gram Range",
    ["Unigrams only (1, 1)", "Unigrams + Bigrams (1, 2)", "Unigrams to Trigrams (1, 3)"],
    index=1
)
ngram_map = {
    "Unigrams only (1, 1)": (1, 1),
    "Unigrams + Bigrams (1, 2)": (1, 2),
    "Unigrams to Trigrams (1, 3)": (1, 3)
}
ngram_range = ngram_map[ngram_selection]

use_stop_words = st.sidebar.checkbox("Filter English Stop Words", value=True)
sublinear_tf = st.sidebar.checkbox("Sublinear TF Scaling (1 + log(tf))", value=True)

# 4. Decision Threshold
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ 4. Decision Threshold")
decision_threshold = st.sidebar.slider(
    "Spam Classification Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Tune to favor Precision (lower false alarms) or Recall (catch every spam)."
)


# ==============================================================================
# Data Pipeline & Model Training Execution
# ==============================================================================
if custom_df is not None and "text" in custom_df.columns and "label" in custom_df.columns:
    df = custom_df
else:
    df = load_corpus(dataset_choice, corpus_size, spam_ratio, random_seed=42)

pipeline, metrics, (X_train, X_test, y_train, y_test) = train_naive_bayes_model(
    df, model_type, vectorizer_type, alpha, ngram_range, use_stop_words, sublinear_tf
)

# Recalculate metrics based on interactive custom decision threshold
y_test_pred_custom = (metrics["y_pred_proba"] >= decision_threshold).astype(int)
acc = accuracy_score(y_test, y_test_pred_custom)
prec = precision_score(y_test, y_test_pred_custom, zero_division=0)
rec = recall_score(y_test, y_test_pred_custom, zero_division=0)
f1 = f1_score(y_test, y_test_pred_custom, zero_division=0)
fpr, tpr, _ = roc_curve(y_test, metrics["y_pred_proba"])
roc_auc = auc(fpr, tpr)


# ==============================================================================
# Header & Executive Metric Strip
# ==============================================================================
st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="margin: 0; font-size: 2.3rem; font-family: 'Outfit', sans-serif;">
        Naive Bayes Spam & Phishing Classifier <span style="color: #F59E0B;">Studio</span>
    </h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-top: 6px;">
        An industrial-grade probabilistic text classification system with real-time explainability, token attribution, and decision threshold calibration.
    </p>
</div>
""", unsafe_allow_html=True)

# Top 6 KPI Dashboard
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Model Accuracy", f"{acc * 100:.1f}%", help="Overall correct predictions")
kpi2.metric("Spam Precision", f"{prec * 100:.1f}%", help="Of messages flagged as spam, % that truly were")
kpi3.metric("Spam Recall", f"{rec * 100:.1f}%", help="Of all actual spam, % successfully caught")
kpi4.metric("F1-Score", f"{f1 * 100:.1f}%", help="Harmonic mean of precision and recall")
kpi5.metric("ROC-AUC Score", f"{roc_auc:.3f}", help="Area under receiver operating characteristic curve")
kpi6.metric("Vocabulary Size", f"{metrics['vocab_size']:,}", help="Unique token features learned")


# ==============================================================================
# Main Studio Tabs Interface
# ==============================================================================
tab_live, tab_diag, tab_vocab, tab_batch, tab_math, tab_code = st.tabs([
    "🛡️ Live Detector & XAI",
    "📊 Diagnostic Metrics",
    "🔍 Feature Importance",
    "⚡ Batch Threat Scanner",
    "📐 Mathematical Engine",
    "🐍 Reproducible Code & Export"
])


# ------------------------------------------------------------------------------
# TAB 1: Live Detector & Explainable AI (XAI)
# ------------------------------------------------------------------------------
with tab_live:
    st.markdown("### 🎯 Live Inference & Token-Level Log-Odds Explainability")
    st.markdown("Input any custom SMS, email, or phishing lure to test the classifier in real time. Inspect exactly which tokens contributed towards the Spam or Ham decision.")
    
    # Preset Samples for Fast Testing
    preset_cols = st.columns([1, 1, 1, 1, 1])
    preset_choice = None
    
    if preset_cols[0].button("💳 Bank Phishing Alert"):
        st.session_state["live_input"] = "URGENT: Your Chase bank card has been locked due to suspicious activity. Verify identity now at bit.ly/chase-auth to restore funds."
    if preset_cols[1].button("🎰 $1,000,000 Lottery"):
        st.session_state["live_input"] = "CONGRATULATIONS! You have won the $1,000,000 UK National Lottery. Contact claim@uk-lotto-intl.com immediately with your PIN code."
    if preset_cols[2].button("📦 Package Delivery Fee"):
        st.session_state["live_input"] = "USPS: Package #US9827419 is on hold at international sorting hub. Unpaid customs postage fee of $2.85 is required: http://usps-parcel-track.biz"
    if preset_cols[3].button("💼 Sprint Review Meeting"):
        st.session_state["live_input"] = "Hey team, please review the attached slide deck for tomorrow's sprint retrospective meeting at 10:00 AM in Conference Room B."
    if preset_cols[4].button("☕ Casual Coffee Invite"):
        st.session_state["live_input"] = "Hey Alex, are we still catching up for coffee and lunch around 12:30 PM today? Let me know if you want to try the new Italian bistro."

    default_text = st.session_state.get(
        "live_input",
        "URGENT: Your PayPal account has been restricted. Unauthorized payment of $650.00 to Coinbase Inc detected. Click here immediately to dispute: bit.ly/pp-dispute"
    )
    
    user_input = st.text_area(
        "Message Input Box",
        value=default_text,
        height=100,
        placeholder="Type or paste any SMS text or email message here..."
    )
    
    if user_input.strip():
        # Perform Model Inference
        raw_prob_spam = pipeline.predict_proba([user_input])[0][1]
        raw_prob_ham = 1.0 - raw_prob_spam
        is_spam = raw_prob_spam >= decision_threshold
        
        # Display Result Card
        col_res1, col_res2 = st.columns([1.2, 1])
        
        with col_res1:
            if is_spam:
                st.markdown(f"""
                <div class="studio-card studio-card-danger">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <h3 style="margin: 0; color: #EF4444; font-size: 1.4rem;">🚨 THREAT DETECTED: SPAM / PHISHING</h3>
                        <span style="background: rgba(239,68,68,0.2); color: #FCA5A5; padding: 4px 10px; border-radius: 99px; font-weight: 700; font-size: 0.8rem;">
                            CRITICAL RISK
                        </span>
                    </div>
                    <p style="color: #CBD5E1; margin: 10px 0 0 0; font-size: 0.95rem;">
                        This message exhibits strong linguistic markers characteristic of malicious phishing, financial fraud, or unwanted promotional bulk campaigns.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="studio-card studio-card-success">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <h3 style="margin: 0; color: #10B981; font-size: 1.4rem;">✅ VERIFIED SAFE: LEGITIMATE (HAM)</h3>
                        <span style="background: rgba(16,185,129,0.2); color: #6EE7B7; padding: 4px 10px; border-radius: 99px; font-weight: 700; font-size: 0.8rem;">
                            BENIGN
                        </span>
                    </div>
                    <p style="color: #CBD5E1; margin: 10px 0 0 0; font-size: 0.95rem;">
                        This message aligns closely with organic conversational, workplace, or legitimate system transactional patterns.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            # Confidence Breakdown Bar
            st.markdown(f"**Calculated Posterior Probability:**")
            st.progress(float(raw_prob_spam))
            pcol_a, pcol_b = st.columns(2)
            pcol_a.markdown(f"<span style='color: #10B981; font-weight: 700;'>P(Ham): {raw_prob_ham * 100:.2f}%</span>", unsafe_allow_html=True)
            pcol_b.markdown(f"<span style='color: #EF4444; font-weight: 700; text-align: right; display: block;'>P(Spam): {raw_prob_spam * 100:.2f}%</span>", unsafe_allow_html=True)

        with col_res2:
            # Probability Distribution Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=raw_prob_spam * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Spam Threat Index", 'font': {'size': 16, 'color': '#F1F5F9'}},
                number={'suffix': "%", 'font': {'color': '#EF4444' if is_spam else '#10B981', 'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#EF4444" if is_spam else "#10B981", 'thickness': 0.28},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'threshold': {
                        'line': {'color': "#F59E0B", 'width': 4},
                        'thickness': 0.75,
                        'value': decision_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ----------------------------------------------------------------------
        # Explainable AI (XAI): Token-Level Log-Odds Attribution Visualizer
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### 🔬 Explainable AI: Token Attribution Heatmap")
        st.markdown("Words are highlighted according to their **Log-Odds Impact** $\\log \\frac{P(w|\\text{Spam})}{P(w|\\text{Ham})}$. Red words increase spam likelihood; Green words indicate legitimate ham content.")
        
        token_attributions = compute_token_attributions(pipeline, user_input)
        
        # Build HTML Highlight View
        highlighted_html = '<div style="background: #090B0F; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 18px; font-size: 1.05rem; line-height: 2.0;">'
        for item in token_attributions:
            t = item["token"]
            direction = item["direction"]
            odds = item["odds_ratio"]
            
            if direction == "Spam":
                title_tooltip = f"Spam Signal: {odds:.1f}x higher likelihood in spam"
                highlighted_html += f'<span class="token-spam" title="{title_tooltip}">{t}</span> '
            elif direction == "Ham":
                title_tooltip = f"Ham Signal: {1/odds:.1f}x higher likelihood in ham"
                highlighted_html += f'<span class="token-ham" title="{title_tooltip}">{t}</span> '
            else:
                highlighted_html += f'<span class="token-neutral">{t}</span> '
        highlighted_html += '</div>'
        
        st.markdown(highlighted_html, unsafe_allow_html=True)
        
        # Attribution Breakdown Dataframe
        active_tokens = [t for t in token_attributions if t["in_vocab"]]
        if active_tokens:
            with st.expander("🔍 View Detailed Token Log-Odds Multipliers Table", expanded=False):
                df_tokens = pd.DataFrame(active_tokens)
                df_tokens["odds_ratio"] = df_tokens["odds_ratio"].apply(lambda v: f"{v:.2f}x")
                df_tokens["log_odds"] = df_tokens["log_odds"].apply(lambda v: f"{v:+.3f}")
                st.dataframe(
                    df_tokens[["token", "direction", "log_odds", "odds_ratio"]].rename(columns={
                        "token": "Token",
                        "direction": "Tendency",
                        "log_odds": "Log-Likelihood Ratio",
                        "odds_ratio": "Odds Impact Multiplier"
                    }),
                    use_container_width=True,
                    hide_index=True
                )


# ------------------------------------------------------------------------------
# TAB 2: Diagnostic Metrics & Visual Evaluation
# ------------------------------------------------------------------------------
with tab_diag:
    st.markdown("### 📊 Comprehensive Diagnostic Metrics & Performance Curves")
    st.markdown("Evaluate model generalizability across holdout test data using the current decision threshold.")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_test_pred_custom)
        cm_labels = ["Legitimate (Ham)", "Spam"]
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=cm_labels,
            y=cm_labels,
            color_continuous_scale="Tealgrn",
            title=f"Confusion Matrix (Threshold = {decision_threshold:.2f})"
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=340,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Confusion Matrix Interpretations
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0)
        cmi1, cmi2 = st.columns(2)
        cmi1.metric("True Negatives (Ham Caught)", f"{tn:,}")
        cmi2.metric("False Positives (Ham Flagged as Spam)", f"{fp:,}", delta="-Low is Better" if fp == 0 else f"+{fp}")
        cmi3, cmi4 = st.columns(2)
        cmi3.metric("False Negatives (Missed Spam)", f"{fn:,}", delta="-Zero is Ideal" if fn == 0 else f"+{fn}")
        cmi4.metric("True Positives (Spam Caught)", f"{tp:,}")

    with col_d2:
        # ROC Curve & Precision-Recall Curve (Dual Tabs)
        curve_choice = st.radio("Select Diagnostic Curve", ["ROC Curve (AUC)", "Precision-Recall Curve"], horizontal=True)
        
        if curve_choice == "ROC Curve (AUC)":
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})', line=dict(color='#F59E0B', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guessing', line=dict(color='#64748B', dash='dash')))
            fig_roc.update_layout(
                title=f"Receiver Operating Characteristic (ROC)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                height=340,
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            pr_prec, pr_rec, _ = precision_recall_curve(y_test, metrics["y_pred_proba"])
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=pr_rec, y=pr_prec, mode='lines', name='Precision-Recall Curve', line=dict(color='#38BDF8', width=3)))
            fig_pr.update_layout(
                title="Precision vs Recall Curve",
                xaxis_title="Recall (Sensitivity)",
                yaxis_title="Precision (Positive Predictive Value)",
                height=340,
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            
    # Decision Threshold Calibration Trade-off
    st.markdown("---")
    st.markdown("#### 🎚️ Decision Threshold Optimization Curve")
    st.markdown("Simulate how sliding the classification threshold from 0.05 to 0.95 affects Precision, Recall, and F1-Score simultaneously.")
    
    threshold_range = np.linspace(0.05, 0.95, 30)
    sim_precisions, sim_recalls, sim_f1s = [], [], []
    
    for t in threshold_range:
        sim_preds = (metrics["y_pred_proba"] >= t).astype(int)
        sim_precisions.append(precision_score(y_test, sim_preds, zero_division=0))
        sim_recalls.append(recall_score(y_test, sim_preds, zero_division=0))
        sim_f1s.append(f1_score(y_test, sim_preds, zero_division=0))
        
    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(go.Scatter(x=threshold_range, y=sim_precisions, mode='lines+markers', name='Precision', line=dict(color='#38BDF8', width=2.5)))
    fig_tradeoff.add_trace(go.Scatter(x=threshold_range, y=sim_recalls, mode='lines+markers', name='Recall', line=dict(color='#EF4444', width=2.5)))
    fig_tradeoff.add_trace(go.Scatter(x=threshold_range, y=sim_f1s, mode='lines+markers', name='F1-Score', line=dict(color='#10B981', width=3)))
    
    # Mark current threshold
    fig_tradeoff.add_vline(x=decision_threshold, line_width=2, line_dash="dash", line_color="#F59E0B", annotation_text=f"Current: {decision_threshold:.2f}")
    
    fig_tradeoff.update_layout(
        title="Threshold Sensitivity Curve (Precision vs. Recall vs. F1)",
        xaxis_title="Classification Threshold",
        yaxis_title="Score",
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: Feature Importance & Vocabulary Analysis
# ------------------------------------------------------------------------------
with tab_vocab:
    st.markdown("### 🔍 Most Discriminative Features & Vocabulary Dynamics")
    st.markdown("Inspect which words most strongly shift the Bayesian probability towards Spam or Ham.")
    
    vec = pipeline.named_steps['vectorizer']
    clf = pipeline.named_steps['classifier']
    feature_names = np.array(vec.get_feature_names_out())
    
    if hasattr(clf, "feature_log_prob_"):
        ham_log_probs = clf.feature_log_prob_[0]
        spam_log_probs = clf.feature_log_prob_[1]
    elif hasattr(clf, "feature_count_"):
        smoothed = clf.feature_count_ + clf.alpha
        smoothed_totals = smoothed.sum(axis=1, keepdims=True)
        log_probs = np.log(smoothed / smoothed_totals)
        ham_log_probs = log_probs[0]
        spam_log_probs = log_probs[1]
    else:
        ham_log_probs = np.zeros(len(feature_names))
        spam_log_probs = np.zeros(len(feature_names))
        
    diff_log_probs = spam_log_probs - ham_log_probs
    
    # Top 20 Spam & Top 20 Ham tokens
    top_spam_indices = np.argsort(diff_log_probs)[-20:]
    top_ham_indices = np.argsort(diff_log_probs)[:20]
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        df_top_spam = pd.DataFrame({
            "Token": feature_names[top_spam_indices],
            "Log-Odds": diff_log_probs[top_spam_indices]
        }).sort_values("Log-Odds", ascending=True)
        
        fig_spam_words = px.bar(
            df_top_spam,
            x="Log-Odds",
            y="Token",
            orientation="h",
            color="Log-Odds",
            color_continuous_scale="Reds",
            title="Top 20 Strongest Spam Indicators"
        )
        fig_spam_words.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_spam_words, use_container_width=True)
        
    with col_v2:
        df_top_ham = pd.DataFrame({
            "Token": feature_names[top_ham_indices],
            "Log-Odds": -diff_log_probs[top_ham_indices]
        }).sort_values("Log-Odds", ascending=True)
        
        fig_ham_words = px.bar(
            df_top_ham,
            x="Log-Odds",
            y="Token",
            orientation="h",
            color="Log-Odds",
            color_continuous_scale="Mint",
            title="Top 20 Strongest Ham (Legitimate) Indicators"
        )
        fig_ham_words.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ham_words, use_container_width=True)
        
    # Message Length Distribution Comparison
    st.markdown("---")
    st.markdown("#### 📏 Message Character Length Distribution")
    
    df_len = df.copy()
    df_len["char_count"] = df_len["text"].apply(len)
    df_len["word_count"] = df_len["text"].apply(lambda t: len(t.split()))
    df_len["Class"] = df_len["label"].map({0: "Ham (Legitimate)", 1: "Spam / Threat"})
    
    fig_hist = px.histogram(
        df_len,
        x="char_count",
        color="Class",
        barmode="overlay",
        nbins=40,
        color_discrete_map={"Ham (Legitimate)": "#10B981", "Spam / Threat": "#EF4444"},
        title="Character Length Distribution (Spam vs. Ham)"
    )
    fig_hist.update_layout(
        xaxis_title="Character Count per Message",
        yaxis_title="Number of Messages",
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: Batch Threat Scanner & Auditor
# ------------------------------------------------------------------------------
with tab_batch:
    st.markdown("### ⚡ Batch Threat Scanner & Auditor")
    st.markdown("Scan hundreds of incoming messages simultaneously. Automatically triage threats by probability severity score and export results.")
    
    batch_mode = st.radio("Batch Input Mode", ["Paste Multi-line Messages", "Upload CSV File"], horizontal=True)
    
    batch_messages = []
    
    if batch_mode == "Paste Multi-line Messages":
        default_batch = (
            "Congratulations! You won a $1,000 Amazon Gift card! Claim immediately.\n"
            "Hey Sarah, let's schedule our 1-on-1 meeting for Wednesday afternoon.\n"
            "URGENT: Your Wells Fargo debit card has been suspended. Call 800-555-0199 now.\n"
            "Your Uber driver is arriving in 3 minutes.\n"
            "Earn $5,000 daily with our secret AI automated crypto trading robot.\n"
            "Please review the attached quarterly financial statement before 5 PM."
        )
        batch_text = st.text_area("Paste Messages (One per line)", value=default_batch, height=140)
        if batch_text.strip():
            batch_messages = [line.strip() for line in batch_text.strip().split("\n") if line.strip()]
    else:
        batch_file = st.file_uploader("Upload CSV File with 'text' column", type=["csv"], key="batch_uploader")
        if batch_file is not None:
            try:
                b_df = pd.read_csv(batch_file)
                if "text" in b_df.columns:
                    batch_messages = b_df["text"].dropna().tolist()
                    st.success(f"Parsed {len(batch_messages)} messages from CSV!")
                else:
                    st.error("Uploaded CSV must have a 'text' column.")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
                
    if batch_messages:
        batch_probs = pipeline.predict_proba(batch_messages)[:, 1]
        
        batch_results = []
        for msg, prob in zip(batch_messages, batch_probs):
            is_threat = prob >= decision_threshold
            if prob >= 0.85:
                severity = "CRITICAL SPAM"
            elif prob >= decision_threshold:
                severity = "HIGH RISK"
            elif prob >= 0.30:
                severity = "SUSPICIOUS"
            else:
                severity = "LEGITIMATE (HAM)"
                
            batch_results.append({
                "Message Preview": msg[:80] + ("..." if len(msg) > 80 else ""),
                "Spam Probability": f"{prob * 100:.1f}%",
                "Raw Score": prob,
                "Verdict": "SPAM" if is_threat else "HAM",
                "Threat Level": severity
            })
            
        df_batch = pd.DataFrame(batch_results)
        
        # Batch Summary Cards
        n_total = len(df_batch)
        n_spam_caught = len(df_batch[df_batch["Verdict"] == "SPAM"])
        n_ham_passed = n_total - n_spam_caught
        
        b1, b2, b3 = st.columns(3)
        b1.metric("Scanned Messages", f"{n_total:,}")
        b2.metric("Flagged as Spam", f"{n_spam_caught:,} ({(n_spam_caught/n_total)*100:.1f}%)")
        b3.metric("Passed as Legitimate", f"{n_ham_passed:,} ({(n_ham_passed/n_total)*100:.1f}%)")
        
        # Display Batch Table
        st.dataframe(df_batch[["Message Preview", "Verdict", "Threat Level", "Spam Probability"]], use_container_width=True, hide_index=True)
        
        # CSV Export
        csv_buffer = io.StringIO()
        df_batch.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Threat Audit Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name="spam_threat_audit_report.csv",
            mime="text/csv"
        )


# ------------------------------------------------------------------------------
# TAB 5: Mathematical Engine & Formula Calculator
# ------------------------------------------------------------------------------
with tab_math:
    st.markdown("### 📐 Mathematical Foundations of Naive Bayes")
    
    st.markdown("""
    Naive Bayes is a supervised probabilistic classifier based on **Bayes' Theorem** with the foundational **conditional independence assumption**: that every feature (word) is statistically independent of every other feature given the target class.
    """)
    
    st.markdown(r"""
    #### 1. Bayes' Theorem for Text Classification
    $$P(Y = c \mid X) = \frac{P(Y = c) \cdot P(X \mid Y = c)}{P(X)}$$
    Where:
    - $Y \in \{\text{Ham}, \text{Spam}\}$ is the target class.
    - $X = (x_1, x_2, \dots, x_n)$ represents the vector of words in the message.
    - $P(Y = c)$ is the **Prior Probability** (the base rate of spam vs ham).
    - $P(X \mid Y = c)$ is the **Likelihood** of seeing that specific combination of words.
    - $P(X)$ is the evidence normalizing constant: $\sum_{k} P(Y=k) P(X \mid Y=k)$.
    """)
    
    st.markdown(r"""
    #### 2. The "Naive" Independence Assumption
    Assuming all words $x_i$ appear independently given the class:
    $$P(X \mid Y = c) = \prod_{i=1}^{n} P(x_i \mid Y = c)$$
    """)
    
    st.markdown(r"""
    #### 3. Log-Space Computation (Underflow Protection)
    Multiplying hundreds of small probabilities leads to floating-point underflow ($0.00000\dots$). We take the natural logarithm to turn multiplication into addition:
    $$\log P(Y = c \mid X) \propto \log P(Y = c) + \sum_{i=1}^{n} \log P(x_i \mid Y = c)$$
    """)
    
    st.markdown(r"""
    #### 4. Laplace / Lidstone Smoothing ($\alpha$)
    If a word $x_i$ never appeared in a class during training, $P(x_i \mid Y = c) = 0$, which zeros out the entire probability product. Laplace smoothing adds a pseudocount $\alpha$:
    $$\hat{\theta}_{ci} = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot |V|}$$
    Where:
    - $N_{ci}$ is the count of token $i$ in class $c$.
    - $N_c$ is the total token count across all messages in class $c$.
    """ + f"- $|V|$ is the total unique vocabulary size ({metrics['vocab_size']:,} tokens).\n" +
    f"- $\\alpha$ is the smoothing hyperparameter (currently set to **{alpha}**).\n")


# ------------------------------------------------------------------------------
# TAB 6: Reproducible Code & Model Export
# ------------------------------------------------------------------------------
with tab_code:
    st.markdown("### 🐍 Reproducible Python Code Generator & Pipeline Export")
    st.markdown("Generate clean, production-grade Python code that replicates the exact model parameters, preprocessing pipeline, and threshold settings configured above.")
    
    vec_cls = 'TfidfVectorizer' if vectorizer_type == 'TF-IDF Vectorizer' else 'CountVectorizer'
    clf_cls = 'MultinomialNB' if model_type == 'Multinomial Naive Bayes' else ('BernoulliNB' if model_type == 'Bernoulli Naive Bayes' else 'ComplementNB')
    stop_words_code = "'english'" if use_stop_words else "None"
    sublinear_code = f",\n        sublinear_tf={sublinear_tf}" if vectorizer_type == 'TF-IDF Vectorizer' else ""

    py_code = f'''# ==============================================================================
# Naive Bayes Spam Detection Production Pipeline
# Generated from 1,000 AI Projects Showcase Studio (Project #0006)
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import {vec_cls}
from sklearn.naive_bayes import {clf_cls}
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. Load your training corpus (or use the curated dataset)
# df = pd.read_csv("your_spam_corpus.csv")
# X, y = df['text'], df['label']

# 2. Build Pipeline
pipeline = Pipeline([
    ('vectorizer', {vec_cls}(
        ngram_range={ngram_range},
        stop_words={stop_words_code}{sublinear_code}
    )),
    ('classifier', {clf_cls}(
        alpha={alpha}
    ))
])

# 3. Train Pipeline
# pipeline.fit(X_train, y_train)

# 4. Predict with Calibrated Threshold
def predict_threat(text, threshold={decision_threshold}):
    prob_spam = pipeline.predict_proba([text])[0][1]
    is_spam = prob_spam >= threshold
    return {{
        "is_spam": bool(is_spam),
        "spam_probability": float(prob_spam),
        "verdict": "SPAM" if is_spam else "HAM"
    }}

# Example Inference
sample = "URGENT: Your account has been suspended. Verify immediately at bit.ly/secure-login"
# print(predict_threat(sample))
'''
    st.code(py_code, language="python")
    
    st.markdown("---")
    st.markdown("#### 📦 Export Trained Model Pipeline")
    
    model_buffer = io.BytesIO()
    joblib.dump(pipeline, model_buffer)
    model_buffer.seek(0)
    
    st.download_button(
        label="📥 Download Trained Scikit-Learn Pipeline (.joblib)",
        data=model_buffer,
        file_name="spam_naive_bayes_pipeline.joblib",
        mime="application/octet-stream"
    )

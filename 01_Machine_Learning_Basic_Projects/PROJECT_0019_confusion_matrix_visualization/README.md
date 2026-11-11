# Project 019: Confusion matrix visualization

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner  
**Domain:** foundational machine learning  
**Tags:** `Beginner`, `confusion`, `matrix`, `visualization`, `machine`, `learning`, `evaluation`, `streamlit`, `ui`

---

## 📖 Project Overview
This project builds an interactive Confusion Matrix Diagnostic Studio for comprehensive model evaluation. It provides granular insight into classification errors, per-class sensitivity/specificity decomposition, decision threshold dynamics ($\tau$), and financial/clinical error cost analysis. It includes both a full-featured **Streamlit Web Dashboard** and a **Standalone Zero-Dependency Browser UI**.

---

## 🎯 Learning Objectives (What You Will Build)
1. Translate classification predictions into contingency matrices ($C_{i,j}$).
2. Dissect the trade-offs between Type I errors (False Positives) and Type II errors (False Negatives).
3. Compute normalized rates (Recall per true row vs. Precision per predicted column).
4. Evaluate multi-class (Iris / Wine) and binary (Breast Cancer) diagnostic performance across varying decision thresholds.

---

## 🏗️ Implementation Blueprint
### Problem Frame
Evaluate multi-class and binary model classifications beyond basic accuracy metrics.

### Baseline
Logistic Regression trained on the Iris 3-class dataset ($N=150$, $70/30$ split):
- Per-class recall: $\text{Recall}_i = \frac{C_{i,i}}{\sum_j C_{i,j}}$
- Accuracy: $\frac{\sum_i C_{i,i}}{\sum_{i,j} C_{i,j}}$

### Evaluation Metrics
- **Accuracy, Balanced Accuracy**
- **Sensitivity / Recall (TPR)**, **Specificity (TNR)**
- **Precision (PPV)**, **Negative Predictive Value (NPV)**
- **Cohen's Kappa ($\kappa$)**, **Matthews Correlation Coefficient (MCC)**

### Deployment & UI
Interactive Streamlit application (`app.py`) with Plotly visualizers, plus a zero-dependency standalone web dashboard (`index.html`).

---

## 🛠️ Stack & Validation Plan

### Recommended Tools & Libraries
- Python 3.x
- NumPy
- scikit-learn
- pandas
- Streamlit
- Plotly
- Chart.js (Standalone Web UI)

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Optional: Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Baseline CLI
```bash
python main.py
```

### 3. Launch Interactive UIs

#### Option A: Streamlit Diagnostic Studio
```bash
streamlit run app.py
```
*(Or double-click `run_ui.bat` on Windows)*

**Features:**
- 🎯 **Interactive Heatmap**: Normalized by true class, predicted class, or total percentage.
- 📊 **Per-Class Decomposition**: Grouped bar charts for Precision, Recall, and F1 across all classes.
- 🪐 **Error Space Inspection**: 2D scatter plots of test instances highlighting exactly which points fell outside decision boundaries.
- 🎚️ **Threshold & Trade-off Curves**: Interactive decision threshold slider ($\tau \in [0.05, 0.95]$) with real-time ROC and Precision-Recall operating point updates.
- 🧪 **Theoretical Matrix Sandbox**: Custom TP, FP, FN, TN inputs with financial/clinical error cost simulation.

#### Option B: Standalone Web Interface (Zero Dependency)
- Open **[`index.html`](index.html)** directly in any web browser.
- Switch between Iris 3-Class and Medical Binary tasks.
- Click any cell to inspect clinical impact and error classification type.
- Live decision threshold slider with real-time matrix updates.

---

## 📈 Production Hardening & Next Steps
- Integrate Cost-Sensitive Learning ($C_{FN} > C_{FP}$) into loss functions for high-stakes domains (fraud detection, oncology).
- Track Confusion Matrix Drift across production data windows to catch concept drift.
- Automate optimal decision threshold tuning via Youden's $J$ statistic or business profit curves.

---
*Part of the 1,000 AI Projects Implementation Library.*

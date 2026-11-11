# Project 018: Correlation analysis tool

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner  
**Domain:** foundational machine learning  
**Tags:** `Beginner`, `correlation`, `analysis`, `machine`, `learning`, `basic`, `statistics`, `streamlit`, `ui`

---

## 📖 Project Overview
This project builds an end-to-end Correlation Analysis Studio to compute, visualize, and diagnose linear and non-linear relationships across multivariate datasets. It features both a full-featured **Streamlit Web Dashboard** and a **Standalone Zero-Dependency Browser UI**, complete with Pearson, Spearman, and Kendall Tau correlation coefficients, p-value statistical significance matrices, and Variance Inflation Factor (VIF) multicollinearity detection.

---

## 🎯 Learning Objectives (What You Will Build)
1. Differentiate between linear correlation (**Pearson**) and monotonic rank correlation (**Spearman** / **Kendall**).
2. Calculate hypothesis test p-values ($p < 0.05$) to establish whether observed associations are statistically significant.
3. Detect severe multicollinearity ($\text{VIF} \ge 10$) that jeopardizes regression stability.
4. Interact with 2D heatmaps, 3D feature manifolds, and bivariate OLS regression scatter plots.

---

## 🏗️ Implementation Blueprint
### Problem Frame
Evaluate pairwise dependencies and detect collinear feature clusters in tabular data.

### Baseline
Four metrics generated synthetically:
- `Metric_A`: Base linear variable ($x_1$).
- `Metric_B`: Positive correlation ($2.5 x_1 + \epsilon$).
- `Metric_C`: Inverse correlation ($-1.8 x_1 + \epsilon$).
- `Noise`: Uniformly distributed uncorrelated noise.

### Evaluation
- **Pearson $r$**: $r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$
- **Spearman $\rho$**: Pearson correlation applied to variable rank orders.
- **VIF Diagnostics**: $\text{VIF}_i = \frac{1}{1 - R_i^2}$

### Deployment & UI
Interactive Streamlit application (`app.py`) with Plotly 2D/3D visualizers, plus a zero-dependency standalone web dashboard (`index.html`).

---

## 🛠️ Stack & Validation Plan

### Recommended Tools & Libraries
- Python 3.x
- NumPy
- pandas
- scikit-learn
- SciPy
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

#### Option A: Streamlit Correlation Studio
```bash
streamlit run app.py
```
*(Or double-click `run_ui.bat` on Windows)*

**Features:**
- 🔥 **Heatmaps & P-Values**: Interactive diverging correlation heatmaps with significance tests and triangular masking.
- 📈 **Bivariate Scatter & OLS**: Pairwise scatter plots with regression trendlines, $R^2$, slopes, and marginal distributions.
- 🪐 **3D Feature Manifold**: 3-axis rotatable 3D scatter clouds to spot collinear planes.
- ⚠️ **VIF Multicollinearity Diagnostics**: Bar charts flagging high-risk collinear predictors.
- 📁 **CSV Upload & Export**: Upload your own tabular CSVs or export the computed correlation matrix.

#### Option B: Standalone Web Interface (Zero Dependency)
- Open **[`index.html`](index.html)** directly in any web browser.
- Interactive heatmap where clicking any pair displays its scatter plot and regression equation.
- Live data tweaker sliders to adjust slope and noise variance with instantaneous recalculation.

---

## 📈 Production Hardening & Next Steps
- Automate collinear feature dropping during pipeline preprocessing (`VarianceInflationFactorSelector`).
- Integrate non-linear association metrics such as Mutual Information (MI) or Distance Correlation (dCor).
- Scale correlation matrix computations using Polars or CuPy GPU acceleration for high-dimensional feature stores ($D > 1,000$).

---
*Part of the 1,000 AI Projects Implementation Library.*

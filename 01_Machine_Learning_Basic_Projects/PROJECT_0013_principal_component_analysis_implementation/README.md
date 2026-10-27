# Project 013: Principal Component Analysis (PCA) Studio

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner / Intermediate  
**Domain:** Foundational Machine Learning & Dimensionality Reduction  
**Tags:** `PCA`, `Dimensionality Reduction`, `Eigenvalues`, `SVD`, `Biplot`, `Scree Plot`, `Image Compression`, `Streamlit`

---

## 📖 Project Overview

This project delivers an industrial-grade **Principal Component Analysis (PCA) Studio and Computational Engine**. It bridges theoretical linear algebra (Covariance Eigendecomposition vs Thin Singular Value Decomposition) with real-world applications including latent manifold visualization, lossy image compression, feature loadings biplot interpretation, and downstream classifier benchmarking to mitigate the curse of dimensionality.

The application includes both a reproducible command-line pipeline (`main.py`) and a high-aesthetic, interactive Streamlit web dashboard (`app.py`).

---

## 🎯 Key Capabilities & Studio Features

### 1. 🔮 Interactive 2D & 3D Manifold Projections
- **2D Latent Projections (PC1 vs PC2):** Scatter plot with categorical coloring, confidence ellipses, and hover metadata.
- **3D Orbit Space (PC1 vs PC2 vs PC3):** 360-degree interactive rotation exploring latent geometric manifolds.
- **Real-time Presets:** Instant loading for Digits (64D), Breast Cancer Biopsy (30D), Wine Cultivars (13D), Botanical Iris (4D), Swiss Roll (6D), and Sensor Arrays (12D).

### 2. 📊 Scree Plot & Spectral Analysis
- **Dual-Axis Energy Spectrum:** Bar chart of individual explained variance + step-line of cumulative variance.
- **Dimensionality Heuristics:** Automated computation of:
  - **Kaiser-Guttman Criterion:** Retains components with eigenvalues $\lambda \ge 1.0$.
  - **Geometric Scree Elbow Point:** Maximum chord curvature detection.
  - **90% & 95% Cutoff Thresholds:** Minimum components required for variance retention.

### 3. 🧭 Feature Loadings & 2D Biplot Analysis
- **Vector Overlay (Biplot):** Original feature eigenvectors projected as directional arrows into PC space. Length indicates variable importance; angular proximity indicates inter-feature correlation.
- **Loadings Heatmap:** Matrix of correlation coefficients between input features and top principal components.
- **Contribution Rankings:** Top-10 dominant contributing features per principal component.

### 4. 🖼️ Lossy Compression & Image Reconstruction Lab
- **Interactive $k$-Slider:** Reconstruct 8x8 grayscale digit images or continuous feature profiles from $k$ components.
- **Side-by-Side Comparison:** Original vs Reconstructed vs Absolute Residual Error heatmaps.
- **Rate-Distortion Curve:** Dataset Mean Squared Error (MSE) vs number of components $k$.

### 5. 🧮 First-Principles Math & SVD Verification
- **Mathematical Formulations:** Centering $X_c = X - \mu$, Covariance $C = \frac{1}{n-1}X_c^T X_c$, Eigendecomposition $C v = \lambda v$, and Thin SVD $X_c = U \Sigma V^T$ with $\lambda_i = \frac{\sigma_i^2}{n-1}$.
- **Numerical Validation:** Side-by-side numerical verification demonstrating machine-precision equivalence ($|X_{scratch} - X_{sklearn}| < 10^{-13}$).

### 6. ⚡ Downstream ML Benchmark (Curse of Dimensionality)
- Evaluates classifier performance (Logistic Regression / Random Forest) on Raw Features vs PCA-reduced subsets.
- Measures accuracy, macro F1-score, training latency speedup, and memory reduction.

### 7. 📁 Custom Dataset Scanner & Exporter
- Drag-and-drop custom CSV file upload.
- Automatic numeric column detection and missing value handling.
- Instant 2D PCA projection and one-click download for projected coordinates ($Z$) and loading matrices ($V$).

---

## 🏗️ Architecture & Module Organization

```
PROJECT_0013_principal_component_analysis_implementation/
├── app.py              # Interactive Streamlit Web Application (7 Studio Tabs)
├── dataset.py          # Curated benchmark datasets & synthetic manifold generators
├── pca_engine.py       # Algorithmic engine (Scratch SVD, Scree, Biplot, Reconstruction)
├── main.py             # Reproducible CLI pipeline & model serialization
├── notebook.ipynb      # Interactive Jupyter notebook walkthrough
├── requirements.txt    # Pinned production dependencies
└── README.md           # Documentation & user guide
```

---

## 🛠️ Stack & Dependencies

- **Language:** Python 3.10+
- **Decomposition & ML:** `scikit-learn`, `numpy`, `scipy`
- **Data Handling:** `pandas`
- **Visualization:** `plotly`, `matplotlib`
- **Dashboard UI:** `streamlit`
- **Model Serialization:** `joblib`

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

### 2. Launch Interactive Web UI

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) (or the active port reported in your console) in your web browser.

### 3. Run Command-Line Pipeline

```bash
# Run default PCA on 64D Digits dataset
python main.py

# Run on Breast Cancer dataset with 3 components and standardized scaling
python main.py --dataset breast_cancer --components 3 --standardize

# Run first-principles SVD mathematical verification
python main.py --verify-math

# Run downstream classifier benchmark
python main.py --benchmark

# Save fitted PCA pipeline artifact
python main.py --save-model pca_digits_baseline.joblib
```

---

## 📊 Sample CLI Output

```
==============================================================================
🌌  PROJECT 013: PRINCIPAL COMPONENT ANALYSIS (PCA) ENGINE & PIPELINE
==============================================================================

[1/5] Loading benchmark dataset: 'digits'...
      - Samples: 1797
      - Input Dimensions: 64 features
      - Classes: 10 distinct groups

[2/5] Fitting PCA (STANDARD) with k=2 components & scaling='standard'...
      - PC1 Explained Variance:  12.03% (Cumulative:  12.03%)
      - PC2 Explained Variance:   9.56% (Cumulative:  21.59%)
      - Total Variance Preserved in 2D: 21.59%

[3/5] Computing Spectral Energy Distribution & Scree Analysis...
      - Kaiser Criterion (Eigenvalues >= 1.0): 17 components
      - Geometric Scree Elbow Point:          9 components
      - 90% Variance Retention Threshold:     30 components
      - 95% Variance Retention Threshold:     30 components

[4/5] Evaluating Lossy Compression & Signal Reconstruction...
      - Global Dataset Reconstruction MSE: 14.0547
      - Dimension Savings / Compression:   96.9%

[EXTRA] Rigorous First-Principles Verification (Scratch SVD vs Scikit-Learn)...
      - Max Projection Delta:      6.82e-14
      - Max Variance Ratio Delta:  1.67e-16
      - Numerically Identical:     True
==============================================================================
✅  PCA PIPELINE EXECUTION COMPLETED SUCCESSFULLY
==============================================================================
```

---

*Part of the 1,000 AI Projects Implementation Library.*

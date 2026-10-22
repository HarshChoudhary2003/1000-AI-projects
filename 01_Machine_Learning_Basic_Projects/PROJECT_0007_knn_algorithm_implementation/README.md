# Project 007: K-Nearest Neighbors (KNN) Spatial Geometry & Benchmark Studio

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner / Applied  
**Domain:** Foundational Supervised Learning & Non-Parametric Spatial Geometry  
**Tags:** `K-Nearest-Neighbors`, `KNN`, `Voronoi-Diagrams`, `Distance-Metrics`, `Curse-of-Dimensionality`, `Breast-Cancer-Wisconsin`, `Streamlit`, `Plotly`, `Scikit-Learn`

---

## 📖 Project Overview

This project delivers an industrial-grade **K-Nearest Neighbors (KNN) Spatial Geometry Studio** with interactive 2D decision boundary visualization, dynamic nearest neighbor vector tracing, hyperparameter optimization ($K$-Elbow curves), multi-metric distance benchmarking, and an interactive simulation of the **Curse of Dimensionality**.

While seemingly intuitive, KNN embodies core foundational principles of non-parametric supervised learning: lazy instance-based evaluation, metric space geometry, local hypothesis approximation, and sensitivity to feature scaling and high-dimensional sparsity.

This project provides:
1. **Interactive Streamlit Web Studio (`app.py`)**: An interactive visual dashboard featuring 2D continuous decision boundary contour plotting, dynamic vector tracing from an interactive probe point to its exact $K$ closest neighbors, cross-validation $K$-optimization curves, distance metric benchmarks ($L_1, L_2, L_\infty, \text{Cosine}$), feature scaling sensitivity comparisons, the Curse of Dimensionality simulator, live single/batch inference, and one-click `.joblib` model export.
2. **Modular Python CLI Pipeline (`main.py`)**: A standalone, zero-dependency CLI pipeline evaluating the authentic **Wisconsin Breast Cancer diagnostic dataset** (569 patient biopsies, 30 morphological features), benchmarking distance metrics, demonstrating the $+4.9\%$ accuracy lift from feature standardization, discovering optimal $K=7$ via 5-fold cross-validation, and tracing neighbor distances.
3. **End-to-End Jupyter Notebook (`notebook.ipynb`)**: A step-by-step notebook guiding through theoretical formulations, exploratory data analysis, cross-validation elbow curves, metric comparisons, and 2D non-linear manifold partitioning.
4. **Curated Benchmark Provider (`dataset.py`)**: Real-world medical and chemical datasets (Wisconsin Breast Cancer, Wine Cultivar recognition) and synthetic 2D non-linear manifolds (Two Moons, Concentric Circles, Intertwined Spirals, Gaussian Blobs).

---

## 🎯 Key Features & Capabilities

- **Interactive 2D Spatial Boundary & Vector Tracer**:
  - Live 2D continuous decision surface contour rendering with training point scatter.
  - Interactive probe coordinate sliders allowing users to position a test query anywhere on the 2D plane.
  - **Dynamic Vector Tracer**: Draws connecting lines from the probe point to its exact $K$ nearest neighbors, with halo rings highlighting the active voting ensemble.
  - Dynamic Neighbor Inspector Table detailing neighbor rank, geometric distance, feature coordinates, and ground truth label.
- **K-Hyperparameter Optimization & Bias-Variance Analysis**:
  - Full 5-fold cross-validation scan across $K \in [1, 45]$ with standard deviation error ribbons.
  - Automated detection and highlighting of the optimal $K$ neighbor count.
  - Interactive breakdown of the Bias-Variance continuum: Overfitting ($K=1$, high variance) vs. Underfitting ($K \to N$, majority class bias).
- **Comprehensive Distance Metric Benchmark**:
  - **Euclidean ($L_2$)**: Classical straight-line ruler distance.
  - **Manhattan ($L_1$)**: City block grid-aligned distance, more robust to outlier coordinates.
  - **Chebyshev ($L_\infty$)**: Maximum single-dimension coordinate difference.
  - **Minkowski ($L_p$)**: Generalized $L_p$ norm with continuous power parameter $p \in [1, 6]$.
  - **Cosine Distance**: Normalized angular separation invariant to vector magnitude.
- **Feature Scaling Sensitivity Lab**:
  - Side-by-side empirical comparison of raw unscaled data vs. **StandardScaler** ($Z$-score), **MinMaxScaler** ($[0, 1]$), and **RobustScaler** (IQR).
  - Empirically demonstrates why features with large raw variances dominate Euclidean distance calculations without normalization.
- **The Curse of Dimensionality Simulator**:
  - Interactive hypercube distance concentration simulator across dimensions $D \in [2, 400]$.
  - Visual verification of the mathematical proof that $\lim_{D \to \infty} \frac{D_{\max} - D_{\min}}{D_{\min}} = 0$.
- **Real-World High-Dimensional & 2D Manifolds**:
  - Authentic Wisconsin Breast Cancer Diagnostic Dataset (30 features, 569 samples).
  - Wine Chemical Cultivar Recognition (13 features, 3 classes).
  - Synthetic non-linear manifolds: Two Moons, Concentric Circles, Intertwined Spirals.
  - Custom tabular CSV uploader with automatic numeric column detection.
- **Production Pipeline Exporter**:
  - Instant generation of clean, copy-pasteable Python scikit-learn code matching active UI parameters.
  - Serialized pipeline download (`.joblib`) bundling model, scaler, and metadata.

---

## 🧮 Mathematical Formulation

### 1. Minkowski Distance ($L_p$ Metric)

Given two points $x, x' \in \mathbb{R}^D$:

$$d_p(x, x') = \|x - x'\|_p = \left( \sum_{i=1}^D |x_i - x'_i|^p \right)^{\frac{1}{p}}$$

Specializations:
- **Manhattan ($L_1$)**: $d_1(x, x') = \sum_{i=1}^D |x_i - x'_i|$
- **Euclidean ($L_2$)**: $d_2(x, x') = \sqrt{\sum_{i=1}^D (x_i - x'_i)^2}$
- **Chebyshev ($L_\infty$)**: $d_\infty(x, x') = \max_{1 \le i \le D} |x_i - x'_i|$

### 2. Neighbor Voting Schemes

For query point $x$, let $N_K(x)$ denote the set of its $K$ nearest neighbors:

- **Uniform Majority Voting:**
  $$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)$$

- **Inverse Distance Weighted Voting:**
  $$w_i = \frac{1}{d(x, x_i) + \epsilon}$$
  $$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_K(x)} w_i \cdot \mathbb{I}(y_i = c)$$

### 3. The Curse of Dimensionality (Distance Concentration)

Beyer et al. (1999) proved that in uniformly distributed high-dimensional spaces:

$$\lim_{D \to \infty} \frac{D_{\max} - D_{\min}}{D_{\min}} = 0$$

As dimension $D$ increases, the ratio of the distance between the nearest neighbor and the farthest neighbor converges to zero. Every point becomes virtually equidistant from every other point, rendering distance-based classification ineffective without dimensionality reduction (PCA, t-SNE, UMAP).

### 4. Search Complexity

| Algorithm | Index Construction | Query Time | Practical Dimension Limit |
| :--- | :--- | :--- | :--- |
| **Brute Force** | $\mathcal{O}(1)$ | $\mathcal{O}(N \cdot D)$ | Any $D$ |
| **KD-Tree** | $\mathcal{O}(D \cdot N \log N)$ | $\mathcal{O}(D \cdot \log N)$ | $D \le 20$ |
| **Ball-Tree** | $\mathcal{O}(D \cdot N \log N)$ | $\mathcal{O}(D \cdot \log N)$ | $D \le 50$ |
| **HNSW (Approx)** | $\mathcal{O}(N \log N)$ | $\mathcal{O}(\log N)$ | $D \ge 1000$ (Production) |

---

## 🏗️ Project Architecture

```
PROJECT_0007_knn_algorithm_implementation/
├── app.py              # Interactive Streamlit Web Studio (2D Vector Tracer, Tabs, Plots)
├── main.py             # Modular CLI pipeline (Cross-Validation, Benchmarks, Evaluation)
├── dataset.py          # Real-world benchmark loader & 2D manifold generator
├── notebook.ipynb      # Complete, educational Jupyter Notebook
├── requirements.txt    # Production dependencies
└── README.md           # Technical documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Navigate to project directory
cd "01_Machine_Learning_Basic_Projects/PROJECT_0007_knn_algorithm_implementation"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run CLI Pipeline

```bash
python main.py
```

Expected CLI output summary:
```text
==========================================================================
  PROJECT 007: K-NEAREST NEIGHBORS (KNN) SPATIAL GEOMETRY PIPELINE
==========================================================================

[1/5] Loading authentic Wisconsin Breast Cancer diagnostic dataset...
      Instances: 569 | Features: 30 clinical measurements
      Classes: 0: Malignant, 1: Benign

[2/5] Partitioning into stratified train (75%) and holdout test (25%) splits...
      Training instances: 426 | Holdout test instances: 143

      Scaling Impact Check (K=5):
        - Raw Unscaled Features:   93.01% Accuracy
        - StandardScaler Z-Score:  97.90% Accuracy (+4.90% lift)

[3/5] Performing 5-fold cross-validation to determine optimal K neighbors:
      >>> Optimal K selected: K = 7 (Peak CV Accuracy: 96.71%)

[4/5] Benchmarking distance metrics on holdout test set (K=optimal):
  Distance Metric Holdout Accuracy F1-Score
   Euclidean (L2)           97.90%   97.89%
   Manhattan (L1)           96.50%   96.46%
Chebyshev (L-inf)           93.71%   93.69%
  Minkowski (p=3)           97.20%   97.19%
           Cosine           95.10%   95.11%

[5/5] Training Champion Pipeline: StandardScaler + KNN(K=7, metric='euclidean')...
Accuracy:  97.90%
Precision: 97.97%
Recall:    97.90%
F1-Score:  97.89%
```

### 3. Launch Interactive Streamlit Studio

```bash
streamlit run app.py
```

The web dashboard opens at `http://localhost:8501` featuring:
- **🎯 2D Boundary & Neighbor Vectors**: Drag the probe point around the canvas and watch dynamic lines trace to the $K$ closest neighbors in real time.
- **📈 K-Hyperparameter Optimization**: Cross-validation curve mapping overfitting ($K=1$) vs. underfitting ($K \to N$).
- **📊 Diagnostic Analytics**: Interactive Confusion Matrix and multi-class classification reports.
- **⚖️ Scaling & Metric Benchmark**: Visual bar charts comparing raw vs. scaled performance and distance metric accuracies.
- **🌌 Curse of Dimensionality**: Distance concentration curve generator across dimensions $D \in [2, 400]$.
- **⚡ Live Inference Probe**: Real-time slider query probe and batch CSV threat scanner.
- **📐 Mathematical Foundations**: Step-by-step LaTeX formula breakdowns.
- **🐍 Reproducible Code & Export**: Download serialized `.joblib` pipelines and copy generated Python code.

---

## 📈 Production Hardening & Scalability Insights

1. **Exact vs. Approximate Nearest Neighbors (ANN)**: For web-scale vector search ($N > 10^6$, $D > 512$ embedding vectors in LLM RAG pipelines), exact KD-Trees degrade to brute force. Production architectures transition to Approximate Nearest Neighbor graphs like **Hierarchical Navigable Small World (HNSW)** or inverted vector indexes via **FAISS** or **Milvus**.
2. **Metric Learning**: Rather than relying strictly on standard Euclidean distance, domain-specific metric learning algorithms (e.g., Large Margin Nearest Neighbor / LMNN) learn a linear transformation matrix $M$ such that $d_M(x, x') = \sqrt{(x - x')^T M (x - x')}$, pulling same-class neighbors closer and pushing distinct classes apart.

---

*Part of the 1,000 AI Projects Implementation Library.*

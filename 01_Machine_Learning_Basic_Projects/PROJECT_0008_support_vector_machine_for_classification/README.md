# Project 008: Support Vector Machine (SVM) Classification Studio

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Intermediate  
**Domain:** Foundational Supervised Learning & Geometric Margin Optimization  
**Tags:** `SVM`, `Support Vector Machine`, `Classification`, `Kernel Trick`, `RBF`, `Maximum Margin`, `Streamlit`, `Biomedical Diagnostics`

---

## 📖 Executive Summary
**Project 008** implements an industrial-strength, production-grade **Support Vector Machine (SVM) Classification Studio**. It transitions standard textbook SVM classification into a high-performance, interactive diagnostic pipeline equipped with authentic real-world datasets (**Wisconsin Breast Cancer Diagnostic**, **Wine Cultivars**, **Iris**), continuous 2D decision boundary & margin gutter visualizations, multi-kernel tournament benchmarking, dual coefficient diagnostics, cross-validated hyperparameter optimization, and a real-time clinical diagnostic inference engine.

---

## 📐 Mathematical Architecture & Core Principles

A Support Vector Machine seeks a separating hyperplane in a $D$-dimensional feature space:

$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

that maximizes the geometric margin $\gamma = \frac{2}{\|\mathbf{w}\|_2}$ while admitting bounded margin violations via slack variables $\xi_i \ge 0$:

### 1. Soft-Margin Primal Optimization
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
$$\text{subject to } y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i \in \{1, \dots, N\}$$

- **$C$ Parameter:** Governs the trade-off between maximizing the margin width (generalization) and penalizing misclassifications or points within the margin (slack $\xi_i$).

### 2. Dual Formulation & The Support Vector Property
By constructing the Lagrangian with multipliers $\alpha_i \ge 0$ and invoking Karush-Kuhn-Tucker (KKT) stationarity conditions:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$
$$\text{subject to } 0 \le \alpha_i \le C \quad \text{and} \quad \sum_{i=1}^N \alpha_i y_i = 0$$

- **Sparsity:** Only training samples with $\alpha_i > 0$ become **Support Vectors**; all other training points have zero influence on the decision boundary.
- **Margin Width (Linear):** For linear kernels, the exact margin width is $\frac{2}{\|\mathbf{w}\|_2}$.

### 3. The Kernel Trick (Mercer's Theorem)
Input vectors are mapped implicitly into an inner-product Hilbert space $\Phi(\mathbf{x})$:
- **Linear Kernel:** $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$
- **RBF (Gaussian) Kernel:** $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$
- **Polynomial Kernel:** $K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^T \mathbf{x}' + r)^d$
- **Sigmoid Kernel:** $K(\mathbf{x}, \mathbf{x}') = \tanh(\gamma \mathbf{x}^T \mathbf{x}' + r)$

---

## 🎯 Key Features & Capabilities

1. **Authentic Real-World Datasets**:
   - **Wisconsin Breast Cancer Diagnostic (WDBC)**: 569 patient samples, 30 continuous cytologic features, binary classification (Malignant vs Benign).
   - **Wine Chemical Recognition**: 178 samples, 13 chemical features, 3 cultivar classes.
   - **Iris Botanical Benchmark**: 150 samples, 4 features, 3 classes.
   - **Synthetic 2D Manifolds**: Linearly Separable, Two Moons, Concentric Circles, Overlapping Blobs.
   - **Custom CSV Uploader**: Train on user-uploaded tabular data with automated feature detection.

2. **Full Multi-Kernel Support**:
   - Linear, Radial Basis Function (RBF), Polynomial ($d \in [2, 6]$), and Sigmoid.
   - Real-time adjustment of $C$, $\gamma$, $d$, independent term $r$, and balanced class weights.

3. **Geometric Telemetry & Support Vector Diagnostics**:
   - Exact support vector counts and ratio (%) of the training set.
   - Linear hyperplane norm $\|\mathbf{w}\|$ and geometric margin width $2 / \|\mathbf{w}\|$.
   - Identification of margin violations (samples where $y_i f(\mathbf{x}_i) < 1$).

4. **Multi-Tab Interactive Streamlit Studio**:
   - **Tab 1: 📊 Executive Studio & Benchmark**: Key performance indicators, confusion matrix, ROC & PR curves, stratified $K$-fold cross-validation.
   - **Tab 2: 🔮 2D Decision Space & Margin Explorer**: Real-time Plotly contour plot with decision boundary ($f=0$), positive margin ($f=+1$), negative margin ($f=-1$), and highlighted support vectors.
   - **Tab 3: ⚔️ Kernel Battle & Grid Search**: Side-by-side tournament of all 4 kernels and a 2D heatmap matrix of cross-validated accuracy across $(C, \gamma)$.
   - **Tab 4: 🩺 Clinical Diagnostic Inference**: Real-time patient profile simulator with pre-loaded archetype presets, feature sliders, diagnostic confidence gauge, and signed margin distance.
   - **Tab 5: 📚 SVM Theory & Architecture**: Derivation of primal, dual, KKT conditions, and production deployment best practices.

---

## 📊 Benchmark Results (Wisconsin Breast Cancer)

Under a 75/25 stratified train/test split with standard feature scaling:

| Kernel | Holdout Accuracy | F1-Score | ROC-AUC | Support Vectors | SV Ratio | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear** | **98.60%** | **0.9860** | **0.9973** | **34** | **8.0%** | 7.1 ms |
| **RBF (Gaussian)** | **97.90%** | **0.9791** | **0.9969** | 96 | 22.5% | 10.2 ms |
| **Sigmoid** | 95.10% | 0.9510 | 0.9952 | 65 | 15.3% | 7.5 ms |
| **Polynomial ($d=3$)** | 90.21% | 0.8982 | 0.9964 | 136 | 31.9% | 7.8 ms |

- **5-Fold Stratified Cross-Validation (RBF):** $96.71\% \pm 0.47\%$
- **Specificity (True Negative Rate):** $98.11\%$
- **Linear Margin Width ($2/\|\mathbf{w}\|$):** $0.7842$

---

## 🚀 Quickstart Guide

### 1. Installation
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Studio (UI)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Command-Line Pipeline (CLI)
Run the baseline automated evaluation:
```bash
python main.py
```

Run with custom arguments:
```bash
# Evaluate Linear Kernel on Breast Cancer
python main.py --dataset breast_cancer --kernel linear --C 1.0

# Evaluate on Multi-class Wine Dataset
python main.py --dataset wine --kernel rbf --C 10.0 --gamma scale

# Benchmark on Non-linear Two Moons Manifold
python main.py --dataset moons --kernel rbf --C 5.0 --gamma 0.5
```

---

## 📁 Repository Structure
```
PROJECT_0008_support_vector_machine_for_classification/
├── app.py              # Flagship Streamlit interactive application (5 tabs)
├── dataset.py          # Real-world loaders (WDBC, Wine, Iris) & 2D manifolds
├── svm_engine.py       # Core SVM training, diagnostics, margin metrics, & inference
├── main.py             # CLI runner with argparse and rich terminal reporting
├── notebook.ipynb      # End-to-end Jupyter Notebook walkthrough & experiments
├── requirements.txt    # Project dependencies
└── README.md           # Comprehensive project documentation
```

---

*Part of the 1,000 AI Projects Implementation Library.*

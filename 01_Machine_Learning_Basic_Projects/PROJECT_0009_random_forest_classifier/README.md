# Project 009: Random Forest Classifier & Ensemble Studio

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Intermediate  
**Domain:** Foundational Supervised Learning & Bagging Ensembles  
**Tags:** `Random Forest`, `Ensemble Learning`, `Bagging`, `Feature Importance`, `OOB Score`, `Streamlit`, `Decision Trees`

---

## 📖 Executive Summary
**Project 009** implements an industrial-strength, production-grade **Random Forest Classifier & Ensemble Studio**. Moving far beyond simplistic tree scripts, this implementation delivers a complete analytics suite featuring authentic multi-class chemical profiling (**Wine Cultivars**) and biomedical diagnostics (**Wisconsin Breast Cancer**), Out-of-Bag (OOB) error convergence profiling, Gini feature importance ranking with inter-tree variance, individual tree voting telemetry, and an interactive 5-tab Streamlit web application.

---

## 📐 Mathematical Architecture & Core Principles

A **Random Forest** is an ensemble meta-estimator that fits $B$ independent, randomized decision trees on bootstrap sub-samples of the dataset and aggregates their predictions:

$$\hat{y} = \text{mode} \{ T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_B(\mathbf{x}) \}$$

### 1. The Bias-Variance Reduction Theorem in Bagging
For $B$ identically distributed trees, each with variance $\sigma^2$ and positive pairwise correlation $\rho$:

$$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

- **As $B \to \infty$:** The second term $\frac{1 - \rho}{B} \sigma^2 \to 0$.
- **Tree Decorrelation:** By restricting split decisions to a random subspace of $m \approx \sqrt{P}$ features, Random Forests substantially decrease the inter-tree correlation $\rho$, driving total variance down without increasing bias.
- **Law of Large Numbers:** Adding more trees does *not* lead to overfitting; it merely stabilizes the ensemble variance.

### 2. Out-of-Bag (OOB) Error Estimation
Because each bootstrap sample is drawn with replacement:

$$P(\text{Sample Not Selected in Fold}) = \left(1 - \frac{1}{N}\right)^N \xrightarrow[N \to \infty]{} \frac{1}{e} \approx 36.8\%$$

- Approximately $36.8\%$ of training instances are left out of each tree's training sample.
- Each observation is evaluated solely using the subset of trees where it was Out-of-Bag, providing an unbiased validation metric without sacrificing a separate test partition.

### 3. Mean Decrease in Impurity (Gini Feature Importance)
The importance of feature $X_j$ is computed as the total decrease in Gini impurity brought by that feature, averaged across all $B$ trees:

$$I(X_j) = \frac{1}{B} \sum_{b=1}^B \sum_{t \in T_b : v(t) = j} p(t) \Delta i(t)$$

where $\Delta i(t) = i(t) - p_L i(t_L) - p_R i(t_R)$.

---

## 🎯 Key Features & Capabilities

1. **Authentic Real-World Datasets**:
   - **Wine Cultivars Chemical Recognition**: 178 samples, 13 chemical features, 3 cultivar classes.
   - **Wisconsin Breast Cancer Diagnostic**: 569 samples, 30 cytologic features, binary classification.
   - **Iris Botanical Benchmark**: 150 samples, 4 features, 3 classes.
   - **Synthetic 2D Manifolds**: Two Moons, Concentric Circles, Overlapping Blobs, Complex Multimodal.
   - **Custom CSV Uploader**: Train on user-uploaded tabular data with automated feature detection.

2. **Full Hyperparameter Flexibility**:
   - Number of Trees (`n_estimators`: 10 to 300).
   - Maximum Tree Depth (`max_depth`: Unlimited or constrained 1 to 25).
   - Splitting criterion: Gini Impurity, Entropy, Log-loss.
   - Max features per split: $\sqrt{P}$, $\log_2(P)$, or full feature set.
   - Balanced class weighting for handling imbalanced datasets.

3. **Ensemble Telemetry & Forest Structural Diagnostics**:
   - Mean, minimum, and maximum tree depths across the forest.
   - Average leaves per tree and total decision node counts.
   - Out-of-Bag (OOB) accuracy and error rate.
   - Real-time training latency profiling.

4. **Interactive 5-Tab Streamlit Studio**:
   - **Tab 1: 📊 Executive Studio & Benchmark**: KPI metric cards, interactive Confusion Matrix heatmap, ROC/PR curves, and 5-fold cross-validation distribution.
   - **Tab 2: 🌲 Forest Architecture & OOB Lab**: Gini feature importance ranking with error bars, cumulative importance curve, and incremental OOB error convergence profiler.
   - **Tab 3: 🔮 2D Ensemble Space & Consensus**: Plotly 2D decision contours overlaid on continuous voting confidence surfaces.
   - **Tab 4: 🎯 Real-Time Decision & Vote Inspector**: Patient/sample archetype simulator showing exact tree-by-tree vote breakdown (e.g. 88 trees voted Cultivar A, 12 trees voted Cultivar B).
   - **Tab 5: 📚 Random Forest Theory & Bagging**: Bias-variance reduction formulas, OOB mathematical derivation, and production deployment checklists.

---

## 📊 Benchmark Results (Wine Cultivars Dataset)

Under a 75/25 stratified train/test split:

| Metric | Score | Notes |
| :--- | :---: | :--- |
| **Holdout Test Accuracy** | **100.00%** | Flawless classification across all 3 cultivars |
| **Out-of-Bag (OOB) Accuracy** | **96.99%** | Unbiased internal cross-validation (100 trees) |
| **5-Fold Cross-Validation** | **98.52% ± 2.96%** | High stability across independent folds |
| **Weighted F1-Score** | **1.0000** | Perfect harmonic precision-recall balance |
| **Average Tree Depth** | **4.9** | Shallow, fast trees (range: 3 to 7) |
| **Top Feature (Gini MDI)** | **Color Intensity (17.6%)** | Followed closely by Flavanoids (17.4%) and Alcohol (13.8%) |
| **Fit Execution Latency** | **~520 ms** | Parallelized training across all CPU cores |

---

## 🚀 Quickstart Guide

### 1. Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Studio (UI)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Command-Line Pipeline (CLI)
Run baseline automated evaluation:
```bash
python main.py
```

Run with custom arguments:
```bash
# Evaluate on Wisconsin Breast Cancer dataset
python main.py --dataset cancer --trees 150 --max-depth 8

# Evaluate on Non-linear Two Moons manifold
python main.py --dataset moons --trees 100 --criterion entropy

# Change max features and test split
python main.py --dataset wine --trees 200 --max-features log2 --test-size 0.30
```

---

## 📁 Repository Structure
```
PROJECT_0009_random_forest_classifier/
├── app.py              # Flagship Streamlit interactive application (5 tabs)
├── dataset.py          # Real-world loaders (Wine, WDBC, Iris) & 2D manifolds
├── rf_engine.py        # Ensemble training, OOB curve, feature importances, & voting telemetry
├── main.py             # CLI runner with argparse and rich terminal reporting
├── notebook.ipynb      # End-to-end Jupyter Notebook walkthrough & experiments
├── requirements.txt    # Project dependencies
└── README.md           # Comprehensive project documentation
```

---

*Part of the 1,000 AI Projects Implementation Library.*

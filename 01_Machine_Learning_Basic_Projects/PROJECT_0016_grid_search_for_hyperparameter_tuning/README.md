# Project 016: Grid search for hyperparameter tuning

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner  
**Domain:** foundational machine learning  
**Tags:** `Beginner`, `grid`, `search`, `hyperparameter`, `tuning`, `machine`, `streamlit`, `ui`

---

## 📖 Project Overview
This project builds a production-minded hyperparameter optimization workflow using `GridSearchCV` on the classic Breast Cancer Wisconsin Diagnostic dataset. It features both a full-featured **Streamlit Web Dashboard** and a **Standalone Interactive Browser UI** to explore parameter grids, 3D response surfaces, parallel coordinate sweeps, and diagnostic inference.

---

## 🎯 Learning Objectives (What You Will Build)
1. Translate hyperparameter search spaces into systematic cross-validated experiments.
2. Build a reproducible Python baseline using Support Vector Classifiers (SVC) and `GridSearchCV`.
3. Inspect 2D heatmaps, 3D response landscapes, and multi-dimensional parallel coordinate sweeps.
4. Test real-time patient diagnosis classifications on the optimal tuned model.

---

## 🏗️ Implementation Blueprint
### Problem Frame
Optimize regularized decision boundary parameters ($C$, kernel, $\gamma$) for maximum cross-validation performance.

### Baseline
Support Vector Machine (SVC) evaluated across 12 combinations ($3 \times 2 \times 2$) with 3-fold cross-validation.

### Evaluation
Compare Mean CV accuracy vs holdout test accuracy, inspecting overfitting gaps and confusion matrix distributions.

### Deployment & UI
Interactive Streamlit application (`app.py`) with Plotly 3D visualizers, plus a zero-dependency standalone web dashboard (`index.html`).

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

#### Option A: Streamlit Hyperparameter Studio
```bash
streamlit run app.py
```
*(Or double-click `run_ui.bat` on Windows)*

**Features:**
- 🪐 **2D & 3D Parameter Landscapes**: Interactive Plotly heatmaps and 3D response surfaces showing optimization peaks.
- 🔀 **Parallel Coordinates Sweep**: Trace high-dimensional parameter relationships colored by CV score.
- 📋 **Overfitting Analysis**: Side-by-side comparison of training vs validation scores.
- 🎯 **Confusion Matrix & ROC**: Test set evaluations for the best estimator.
- 🧪 **Live Diagnosis Simulator**: Adjust patient diagnostic measurements (radius, texture, area, etc.) to predict Benign vs Malignant.

#### Option B: Standalone Web Interface (Zero Dependency)
- Open **[`index.html`](index.html)** directly in any web browser.
- Animated grid sweep scan simulating worker threads across parameter cells.
- Real-time parameter toggles and dynamic train vs test accuracy curves.

---

## 📈 Production Hardening & Next Steps
- Implement RandomizedSearchCV or Bayesian Optimization (Optuna) for larger dimensional spaces.
- Integrate automated early stopping and learning curves to prevent excessive compute costs.
- Cache intermediate fitted models using MLflow or joblib memory pipelines.

---
*Part of the 1,000 AI Projects Implementation Library.*

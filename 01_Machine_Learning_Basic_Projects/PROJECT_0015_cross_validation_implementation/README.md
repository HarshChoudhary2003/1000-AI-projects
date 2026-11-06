# Project 015: Cross-validation implementation

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner  
**Domain:** foundational machine learning  
**Tags:** `Beginner`, `cross`, `validation`, `machine`, `learning`, `basic`, `3d`, `three.js`, `streamlit`

---

## 📖 Project Overview
This project builds a production-minded cross-validation implementation workflow inside foundational machine learning. It explores standard K-Fold, Stratified K-Fold, and Time Series splits on imbalanced classification datasets, featuring both a hardware-accelerated **Three.js 3D WebGL Visualizer** and a **Streamlit 3D Dashboard**.

---

## 🎯 Learning Objectives (What You Will Build)
1. Translate cross-validation implementation from an idea into a measurable ML task.
2. Build a reproducible Python baseline with clean preprocessing, scoring, and variance analysis.
3. Compare **Stratified K-Fold** vs **Standard K-Fold** on imbalanced data (85:15 class distribution).
4. Inspect training and validation partitions in **real-time 3D** space using WebGL / Three.js and Plotly 3D.

---

## 🏗️ Implementation Blueprint
### Problem Frame
Evaluate classifier stability and variance across partitioned training and validation subsets.

### Baseline
Fit Logistic Regression on synthesized imbalanced feature data ($N=500$, $D=8$, $85\%:15\%$).

### Evaluation
Compute mean accuracy, F1-score, and variance across $K$-folds.

### Deployment & 3D Visualization
Interactive 3D representation of data fold slices, feature space, decision hyperplanes, and metric score podiums.

---

## 🛠️ Stack & Validation Plan

### Recommended Tools & Libraries
- Python 3.x
- NumPy
- scikit-learn
- pandas
- Streamlit
- Plotly
- Three.js (WebGL 3D Engine)

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

### 3. Launch the 3D Interactive UI

#### Option A: Standalone Hardware-Accelerated 3D Studio (Three.js WebGL)
- Open **[`index.html`](index.html)** directly in any web browser.
- **3D Features:**
  - 🧱 **3D Fold Slices**: Watch active test folds levitate upward with holographic wireframes while training folds dock together.
  - 🪐 **3D Feature Space**: 500 interactive samples in 3D with glowing test indicators and a 3D decision boundary plane.
  - 🏛️ **3D Score Podiums**: Extruded 3D cylindrical podiums representing individual fold accuracies, with a glowing mean ($\mu$) laser plane.
  - 🕹️ **Full OrbitControls**: Rotate 360°, pan, zoom, and auto-play fold cycles.

#### Option B: Streamlit 3D ML Dashboard
```bash
streamlit run app.py
```
*(Or double-click `run_ui.bat` on Windows)*

- Interactive parameter controls ($K$ slider, class imbalance weight, model selection).
- Plotly 3D scatter plots with misclassification highlighting and decision surfaces.
- Integrated embedding of the Three.js 3D Studio.

---

## 📈 Production Hardening & Next Steps
- Verify Stratification when working with rare events (fraud detection, medical diagnosis).
- Combine with Nested Cross-Validation for unbiased hyperparameter tuning.
- Add GroupKFold when samples share cluster dependencies (e.g. patients, organizations).

---
*Part of the 1,000 AI Projects Implementation Library.*

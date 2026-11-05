# Project 012: Simple recommendation system

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner  
**Domain:** foundational machine learning  
**Tags:** `Beginner`, `simple`, `recommendation`, `machine`, `learning`, `basic`, `streamlit`, `ui`

---

## 📖 Project Overview
This project builds a production-minded simple collaborative filtering recommendation system using Cosine Similarity. It allows users to compute user-user and item-item similarity matrices, rank unwatched candidate items, and explore recommendations via interactive interfaces.

---

## 🎯 Learning Objectives (What You Will Build)
1. Translate simple recommendation system from an idea into a measurable ML task.
2. Build a reproducible Python baseline with clean preprocessing and evaluation.
3. Compute vector cosine similarities between sparse user rating vectors.
4. Interact with the system through a full Streamlit dashboard and zero-dependency web interface.

---

## 🏗️ Implementation Blueprint
### Signals
Combine behavior, item metadata, and user-item ratings for collaborative filtering.

### Ranking
Compute cosine similarity across rating vectors, then rank candidate items based on top-peer affinity or weighted multi-peer consensus.

### Evaluation
Measure offline ranking metrics and inspect similarity distributions with interactive heatmaps.

### Deployment
Serve recommendations with a Streamlit UI (`app.py`) or a lightweight web dashboard (`index.html`).

---

## 🛠️ Stack & Validation Plan

### Recommended Tools & Libraries
- Python 3.x
- NumPy
- pandas
- scikit-learn
- Streamlit
- Plotly

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

#### Option A: Streamlit Web Dashboard (Python ML App)
```bash
streamlit run app.py
```
*(Or double-click `run_ui.bat` on Windows)*

**Features:**
- 🎯 **Target User Recommendations**: Switch between users and choose between Top-Peer matching or Weighted Consensus.
- ⭐ **Personalized Lab**: Rate movies on the fly and get personalized recommendations immediately.
- 📊 **Heatmaps & Matrices**: Interactive Plotly heatmaps for User-User Cosine Similarity, Item-Item Similarity, and raw ratings.
- ✏️ **Live Matrix Editor**: Edit user ratings, add new users, or introduce new movies dynamically.
- 📖 **Mathematical Breakdown**: Step-by-step visual explanation of vectors and cosine similarity formulas.

#### Option B: Standalone Web Interface (Zero Dependency)
- Open `index.html` directly in any web browser.
- Real-time client-side calculation of cosine similarities, interactive rating sliders, and animated taste affinity graphs.

---

## 📈 Production Hardening & Next Steps
- Replace the starter matrix with large-scale movie datasets (e.g. MovieLens 100K/1M).
- Introduce Matrix Factorization (Truncated SVD / Alternating Least Squares) to handle sparse matrices at scale.
- Implement hybrid filtering combining content metadata (genres, actors) with collaborative behavior.

---
*Part of the 1,000 AI Projects Implementation Library.*

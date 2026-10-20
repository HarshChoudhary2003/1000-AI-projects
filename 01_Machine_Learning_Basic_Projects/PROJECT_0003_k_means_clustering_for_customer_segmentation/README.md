# Project 003: K-Means Clustering for Customer Segmentation

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner  
**Domain:** Foundational Unsupervised Machine Learning  
**Tags:** `K-Means`, `Clustering`, `Customer-Segmentation`, `Unsupervised-Learning`, `Streamlit`, `Plotly`, `Scikit-Learn`

---

## 📖 Project Overview
This project delivers an interactive, end-to-end customer segmentation system powered by **K-Means clustering**. It segments retail customers based on demographic and behavioral indicators such as **Annual Income** and **Spending Score**, translating raw cluster math into actionable business personas (e.g., VIP High Rollers, Affluent Savers, Young Trendsetters, Budget Conscious).

The project includes:
1. **Interactive Streamlit Web Application (`app.py`)**: A modern dashboard with multi-source data ingestion, exploratory data analysis, Elbow & Silhouette evaluation, 2D/3D Plotly visualizations, live customer classification, and CSV export.
2. **Modular Python CLI Pipeline (`main.py`)**: Standalone, reproducible script for feature scaling, K evaluation, model training, and segment profiling.

---

## 🎯 Key Features & Capabilities

- **Multi-Source Data Ingestion**:
  - Built-in realistic Mall Customer dataset generator.
  - Configurable synthetic cluster benchmark (blobs).
  - Custom CSV file uploader with dynamic feature mapping.
- **Exploratory Data Analysis (EDA)**:
  - Key business metrics (customer count, median income, spending score).
  - Interactive distribution histograms with box plots.
  - Correlation heatmaps and feature scatter exploration.
- **Cluster Diagnostics**:
  - **The Elbow Method**: Within-Cluster Sum of Squares (WCSS / Inertia) curve.
  - **Silhouette Analysis**: Silhouette coefficient peak identification for optimal $K$.
- **Interactive Cluster Visualizations**:
  - Dynamic 2D scatter plots with centroid star markers.
  - 3D cluster scatter visualizer for multi-feature spaces (e.g., Age + Income + Spending).
- **Automated Business Persona Mapping**:
  - Automatically translates cluster centroids into business profiles (e.g. *👑 VIP High Rollers*, *💼 Affluent Savers*, *🛍️ Young Trendsetters*, *🪙 Budget Conscious*).
  - Generates tailored retention and promotional strategies for each customer archetype.
- **Live Customer Classifier (Simulator)**:
  - Input sliders to evaluate prospective customer metrics in real time.
  - Instant cluster assignment, persona match, strategy recommendation, and visual positioning on the cluster map.
- **Enriched Data Export**:
  - One-click CSV export with assigned cluster labels and marketing personas.

---

## 🏗️ Project Architecture

```
PROJECT_0003_k_means_clustering_for_customer_segmentation/
│
├── app.py              # Interactive Streamlit Web Application
├── main.py             # Standalone Python CLI ML Pipeline
├── requirements.txt    # Project dependencies
├── notebook.ipynb      # Exploration notebook
└── README.md           # Project documentation and guide
```

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

### 2. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
*Open your web browser at `http://localhost:8501` to interact with the dashboard.*

### 3. Run the CLI Pipeline

```bash
python main.py
```

---

## 📊 Customer Personas Reference Guide

| Segment Persona | Typical Income | Spending Score | Recommended Marketing Playbook |
| :--- | :--- | :--- | :--- |
| **👑 VIP High Rollers** | High ($70k+) | High (60+) | White-glove concierge, invite-only VIP loyalty tiers, luxury exclusives. |
| **💼 Affluent Savers** | High ($70k+) | Low (<40) | Quality guarantees, high-durability propositions, investment-grade items. |
| **🛍️ Young Trendsetters** | Low/Mod (<$45k) | High (60+) | Flash sales, viral trend alerts, Buy-Now-Pay-Later (BNPL) payment plans. |
| **🪙 Budget Conscious** | Low (<$45k) | Low (<40) | Discount codes, bundle savings, clearance events, essentials focus. |
| **⚖️ Mainstream Core** | Moderate | Moderate | Standard rewards points, seasonal newsletters, cross-sell bundles. |

---

## 🛠️ Stack & Technologies
- **Language:** Python 3.11+
- **Machine Learning:** `scikit-learn` (KMeans, StandardScaler, Silhouette Score)
- **Data Manipulation:** `pandas`, `numpy`
- **User Interface:** `streamlit`
- **Interactive Visualizations:** `plotly` (Scatter, Heatmap, 3D Plots)

---
*Part of the 1,000 AI Projects Implementation Library.*

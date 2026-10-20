# Project 004: Decision Tree Classifier for Iris Dataset

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner  
**Domain:** Foundational Supervised Machine Learning  
**Tags:** `Decision-Tree`, `Iris-Dataset`, `Classification`, `Scikit-Learn`, `Streamlit`, `Plotly`, `Botanical-ML`

---

## 📖 Project Overview
This project implements an end-to-end multi-class **Decision Tree Classifier** for Ronald Fisher's historic 1936 Iris dataset. The system trains a transparent, interpretable tree model to classify iris blossoms into three distinct species (*Iris setosa*, *Iris versicolor*, and *Iris virginica*) based on sepal and petal dimensions.

The project provides:
1. **Interactive Streamlit Web Application (`app.py`)**: A modern dashboard providing dataset exploratory analysis, full visual decision tree rendering with Matplotlib, 2D decision boundary surface contours, multi-class confusion matrices, and a live botanical flower specimen classifier.
2. **Modular Python CLI Pipeline (`main.py`)**: A clean, standalone script that loads the dataset, performs stratified train-test splitting, trains the decision tree, evaluates classification metrics, and extracts human-readable text rules.

---

## 🎯 Key Features & Capabilities

- **Interactive Exploratory Data Analysis (EDA)**:
  - Summary statistics for each of the 3 species across 4 morphological measurements.
  - Interactive 2D scatter plots and 3D morphological space visualizer.
  - Feature distributions and boxplots per species.
- **Tree Architecture Visualization**:
  - High-resolution, color-coded Decision Tree architectural diagram rendered with `sklearn.tree.plot_tree`.
  - Exportable text-based rule hierarchy (`export_text`) illustrating precise split thresholds.
  - Feature Importance (Gini importance) ranking highlighting Petal Length and Petal Width dominance.
- **Performance & Decision Boundary Mapping**:
  - Test set metrics: Accuracy, Precision, Recall, and Macro F1-Score.
  - Interactive Confusion Matrix heatmap.
  - **2D Decision Surface Visualization**: Real-time orthogonal step-function decision boundary contours projected over any two selected features.
- **Live Flower Classifier Simulator**:
  - Interactive sliders for Sepal Length, Sepal Width, Petal Length, and Petal Width.
  - One-click presets for typical specimens (*Setosa*, *Versicolor*, *Virginica*).
  - Multi-class probability confidence distribution bars.
  - Dynamic visual placement of specimen on the feature scatter map.
- **Data Export & Reproducibility**:
  - Export test set predictions with correctness indicators as CSV.
  - Clean, copy-pasteable Python training snippet.

---

## 🏗️ Project Architecture

```
PROJECT_0004_decision_tree_classifier_for_iris_dataset/
│
├── app.py              # Interactive Streamlit Web Application
├── main.py             # Standalone Python CLI ML Pipeline
├── requirements.txt    # Project dependencies
├── notebook.ipynb      # Exploration and training notebook
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
*Open your browser at `http://localhost:8501` to use the interactive dashboard.*

### 3. Run the CLI Pipeline

```bash
python main.py
```

---

## 🌸 Botanical Species Reference

| Species | Sepal Length | Sepal Width | Petal Length | Petal Width | Key Distinguishing Feature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Iris Setosa** | ~5.0 cm | ~3.4 cm | ~1.5 cm | ~0.2 cm | Linearly separable with very small, narrow petals. |
| **Iris Versicolor** | ~5.9 cm | ~2.8 cm | ~4.3 cm | ~1.3 cm | Intermediate petal size; slight boundary overlap with Virginica. |
| **Iris Virginica** | ~6.6 cm | ~3.0 cm | ~5.6 cm | ~2.0 cm | Longest and widest petals; deep purple blossoms. |

---

## 🛠️ Stack & Technologies
- **Language:** Python 3.11+
- **Machine Learning:** `scikit-learn` (DecisionTreeClassifier, plot_tree, export_text)
- **Data Manipulation:** `pandas`, `numpy`
- **User Interface:** `streamlit`
- **Visualizations:** `plotly`, `matplotlib`, `seaborn`

---
*Part of the 1,000 AI Projects Implementation Library.*

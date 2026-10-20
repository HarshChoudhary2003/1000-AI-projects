# Project 005: Simple Neural Network from Scratch

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner / Applied  
**Domain:** Foundational Deep Learning & Neural Architectures  
**Tags:** `Neural-Networks`, `Backpropagation`, `From-Scratch`, `NumPy`, `XOR-Problem`, `Streamlit`, `Plotly`, `Optimization`

---

## 📖 Project Overview

This project implements an end-to-end **Multi-Layer Perceptron (MLP) Neural Network completely from scratch** using pure **NumPy** without deep learning libraries (no PyTorch, TensorFlow, or Keras). It demystifies the fundamental engine powering modern artificial intelligence: forward tensor propagation, matrix calculus, analytical backpropagation via the chain rule, and numerical optimization algorithms.

The project provides:
1. **Interactive Streamlit Web Studio (`app.py`)**: A modern dashboard featuring dynamic network topology diagrams with synaptic weight-thickness rendering, real-time loss & accuracy convergence curves, continuous 2D decision boundary surfaces, hidden layer representation space projections, mathematical computational graphs, live continuous inference probes, and weight/data export tools.
2. **Modular Python CLI Pipeline (`main.py`)**: A standalone, zero-dependency script demonstrating forward propagation, analytical gradient computation, and gradient descent convergence on the classic non-linear XOR logic benchmark.

---

## 🎯 Key Features & Capabilities

- **Pure NumPy Vectorized Neural Engine**:
  - Fully vectorized matrix multiplication: forward pass, loss calculation, backpropagation, and parameter updates.
  - Configurable hidden topologies: 1 or 2 hidden layers with 2 to 32 neurons.
  - Modular activation functions and analytical derivatives: **Sigmoid**, **Tanh**, **ReLU**, and **Leaky ReLU**.
  - Loss criteria: **Binary Cross-Entropy (BCE)** and **Mean Squared Error (MSE)**.
  - Scratch optimizers: **Vanilla Gradient Descent**, **SGD + Momentum**, and **Adam Optimizer** implemented from first principles.
  - Weight initializers: **Xavier / Glorot Uniform**, **He Normal**, and Gaussian Normal.
- **Interactive Multi-Dataset Playground**:
  - Classic Boolean logic benchmarks: **XOR Gate**, **AND Gate**, **OR Gate**, and **XNOR Gate**.
  - Continuous 2D non-linear manifolds: **Two Moons**, **Concentric Circles**, **Gaussian Blobs**, and **Intertwined Spirals**.
- **Interactive Network Graph & Synaptic Inspector**:
  - Dynamic visual network topology rendered via Plotly.
  - Edge thickness indicates absolute weight magnitude; color indicates excitatory (positive) or inhibitory (negative) synapse weights.
  - Interactive heatmaps of weight matrices ($W^{[1]}, W^{[2]}, W^{[out]}$) and bias vectors.
- **Optimization Dynamics & Convergence**:
  - Real-time epoch-by-epoch tracking of loss $J(\theta)$ and classification accuracy.
  - Convergence telemetry (initial loss, final loss, percentage reduction, exact solved epoch).
- **2D Continuous Decision Boundary Surface**:
  - High-resolution contour heatmap of output probability $P(\hat{y} = 1 \mid x_1, x_2) \in [0, 1]$.
  - Exact decision boundary threshold line ($P = 0.5$) demonstrating non-linear boundary bending.
- **Hidden Layer Representation Space (Space Warping)**:
  - Visualizing the transformed hidden neuron activations $A^{[1]}$.
  - Empirically shows how the neural network disentangles non-linearly separable inputs into a space where they become linearly separable.
- **Live Continuous Inference Simulator**:
  - Continuous coordinate sliders ($x_1, x_2$) with one-click logic gate presets.
  - Live intermediate activation telemetry ($Z^{[1]}, A^{[1]}, Z^{[out]}, \hat{y}$).
- **Export & Reproducibility**:
  - Download trained model weights and biases as JSON.
  - Export test predictions and correctness indicators as CSV.
  - Instant generation of copy-pasteable pure NumPy training code.

---

## 🧮 Mathematical Formulation

### 1. Vectorized Forward Propagation

For an input matrix $X \in \mathbb{R}^{m \times d_{\text{in}}}$, hidden weights $W^{[1]} \in \mathbb{R}^{d_{\text{in}} \times d_h}$, and biases $b^{[1]} \in \mathbb{R}^{1 \times d_h}$:

$$Z^{[1]} = X W^{[1]} + b^{[1]}$$

$$A^{[1]} = f(Z^{[1]})$$

$$Z^{[out]} = A^{[1]} W^{[out]} + b^{[out]}$$

$$\hat{Y} = A^{[out]} = \sigma(Z^{[out]}) = \frac{1}{1 + e^{-Z^{[out]}}}$$

where $f(\cdot)$ is the selected hidden activation (Sigmoid, Tanh, ReLU, or Leaky ReLU).

### 2. Loss Functions

* **Binary Cross-Entropy (BCE):**
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^m \left[ y^{(i)}\ln(\hat{y}^{(i)}) + (1 - y^{(i)})\ln(1 - \hat{y}^{(i)}) \right]$$

* **Mean Squared Error (MSE):**
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{2m}\sum_{i=1}^m \left( y^{(i)} - \hat{y}^{(i)} \right)^2$$

### 3. Backpropagation via the Chain Rule

* **Output Layer Error Gradient:**
  $$\delta^{[out]} = \frac{\partial \mathcal{L}}{\partial Z^{[out]}} = \frac{1}{m}(\hat{Y} - Y)$$

* **Output Layer Parameter Gradients:**
  $$\frac{\partial \mathcal{L}}{\partial W^{[out]}} = (A^{[1]})^T \delta^{[out]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[out]}} = \sum_{i=1}^m \delta^{[out]}$$

* **Hidden Layer Error Gradient:**
  $$\delta^{[1]} = \left( \delta^{[out]} (W^{[out]})^T \right) \odot f'(Z^{[1]})$$

* **Hidden Layer Parameter Gradients:**
  $$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = X^T \delta^{[1]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \sum_{i=1}^m \delta^{[1]}$$

### 4. Parameter Updates (Gradient Descent)

$$W^{[l]} \leftarrow W^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[l]}}, \quad b^{[l]} \leftarrow b^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[l]}}$$

---

## 🏗️ Project Architecture

```
PROJECT_0005_simple_neural_network_from_scratch/
│
├── app.py              # Interactive Streamlit Web Application & Neural Net Studio
├── main.py             # Standalone Python CLI ML Pipeline (NumPy XOR solver)
├── requirements.txt    # Project dependencies
├── notebook.ipynb      # Step-by-step exploration and training notebook
└── README.md           # Comprehensive project documentation
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

### 2. Launch the Streamlit Web Studio

```bash
streamlit run app.py
```
*Open your browser at `http://localhost:8501` to explore the interactive dashboard.*

### 3. Run the CLI Pipeline

```bash
python main.py
```

Expected terminal output:
```text
=== Simple Neural Network from Scratch (XOR Problem) ===
Final Predictions after 3000 epochs:
  Input: [0 0] | Expected: 0 | Predicted: 0.0125 (Class 0)
  Input: [0 1] | Expected: 1 | Predicted: 0.9854 (Class 1)
  Input: [1 0] | Expected: 1 | Predicted: 0.9849 (Class 1)
  Input: [1 1] | Expected: 0 | Predicted: 0.0163 (Class 0)
```

---

## 📊 Benchmark Performance

| Dataset | Topology | Activation | Optimizer | Accuracy | Epochs to Converge |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XOR Gate** | $2 \to 4 \to 1$ | Sigmoid | Gradient Descent | **100.0%** | ~1,200 |
| **AND Gate** | $2 \to 2 \to 1$ | Sigmoid | Gradient Descent | **100.0%** | ~400 |
| **OR Gate** | $2 \to 2 \to 1$ | Sigmoid | Gradient Descent | **100.0%** | ~350 |
| **XNOR Gate** | $2 \to 4 \to 1$ | Sigmoid | Gradient Descent | **100.0%** | ~1,250 |
| **Two Moons** | $2 \to 8 \to 1$ | Tanh | Adam | **98.8%** | ~800 |
| **Concentric Circles** | $2 \to 8 \to 1$ | Tanh | Adam | **100.0%** | ~750 |
| **Intertwined Spirals** | $2 \to 16 \to 16 \to 1$ | Leaky ReLU | Adam | **95.6%** | ~2,500 |

---

## 💡 Key Takeaways & Deep Learning Foundations

1. **The Minsky & Papert Perceptron Limitation (1969)**: A single-layer perceptron without a hidden layer can only create a linear hyperplane ($w_1 x_1 + w_2 x_2 + b = 0$). For XOR, points $(0,1)$ and $(1,0)$ require output 1 while $(0,0)$ and $(1,1)$ require 0, making linear separation impossible.
2. **Hidden Layer Feature Disentanglement**: Introducing at least one non-linear hidden layer allows the network to project non-linearly separable inputs into a higher-dimensional space where classes become linearly separable.
3. **Activation Function Selection**: Sigmoids saturate at large magnitudes causing vanishing gradients; ReLU and Leaky ReLU maintain constant non-zero gradients for positive activations, speeding up convergence in deeper networks.

---

*Part of the 1,000 AI Projects Implementation Library.*

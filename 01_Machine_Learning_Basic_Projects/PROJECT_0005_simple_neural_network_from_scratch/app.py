"""
Neural Network from Scratch Studio | Interactive ML Explorer
============================================================
A comprehensive, interactive Streamlit application exploring a 
multi-layer Perceptron (MLP) built completely from scratch using pure NumPy.

Features:
  - Dataset generator: XOR, AND, OR, NAND, XNOR logic gates + Moons, Circles, Blobs
  - Pure NumPy Neural Network Engine with modular activations (Sigmoid, Tanh, ReLU, LeakyReLU)
  - Custom optimizers (Gradient Descent, Momentum, Adam) & loss functions (MSE, BCE)
  - Interactive neural graph topology visualizer with dynamic weight-thickness edges
  - Real-time training loss & accuracy convergence trajectory
  - 2D Decision Boundary Surface with probability contours
  - Hidden Layer Feature Representation (space manifold disentangling)
  - Forward & Backprop Mathematical Computation Graph with sample probe
  - Live Inference Simulator with real-time intermediate activation inspection
  - JSON / CSV export and pure Python reproducible code generator
"""

import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_blobs


# ==============================================================================
# Page Configuration & Modern Styling
# ==============================================================================
st.set_page_config(
    page_title="Neural Network from Scratch | Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for crisp typography, glowing metrics, and aesthetic cards
st.markdown("""
<style>
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Concept Card Styling */
    .nn-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .nn-card-accent {
        border-left: 6px solid #6366F1;
    }
    .nn-card-success {
        border-left: 6px solid #10B981;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        font-weight: 600;
    }
    
    /* Math block callouts */
    .math-callout {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        font-family: monospace;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Pure NumPy Neural Network Engine
# ==============================================================================

class Activation:
    """Activation functions and their analytical derivatives."""
    
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_deriv(a):
        # 'a' is post-activation value
        return a * (1.0 - a)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_deriv(a):
        # 'a' is post-activation value
        return 1.0 - a ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def relu_deriv(a):
        # 'a' is post-activation or 'z' is pre-activation
        return (a > 0.0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def leaky_relu_deriv(a, alpha=0.01):
        dx = np.ones_like(a)
        dx[a <= 0] = alpha
        return dx


class ScratchNeuralNetwork:
    """
    Modular 2-Layer or 3-Layer Perceptron implemented completely from scratch using NumPy.
    Supports customizable activations, weight initializers, losses, and optimizers.
    """
    def __init__(
        self,
        input_dim=2,
        hidden_dim=4,
        hidden_layers=1,
        activation_name="sigmoid",
        loss_name="bce",
        optimizer_name="gd",
        init_method="xavier",
        learning_rate=0.5,
        momentum=0.9,
        seed=42
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation_name.lower()
        self.loss_name = loss_name.lower()
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.seed = seed
        
        # Resolve activation functions
        if self.activation_name == "sigmoid":
            self.act_fn = Activation.sigmoid
            self.act_deriv = Activation.sigmoid_deriv
        elif self.activation_name == "tanh":
            self.act_fn = Activation.tanh
            self.act_deriv = Activation.tanh_deriv
        elif self.activation_name == "relu":
            self.act_fn = Activation.relu
            self.act_deriv = Activation.relu_deriv
        elif self.activation_name == "leaky_relu":
            self.act_fn = Activation.leaky_relu
            self.act_deriv = Activation.leaky_relu_deriv
        else:
            self.act_fn = Activation.sigmoid
            self.act_deriv = Activation.sigmoid_deriv

        # Initialize weights and biases
        self._init_weights(init_method)
        
        # Optimizer velocity / moment buffers
        self._init_optimizer_buffers()
        
        # Training logs
        self.loss_history = []
        self.accuracy_history = []

    def _init_weights(self, method):
        np.random.seed(self.seed)
        
        def get_w(n_in, n_out):
            if method == "xavier":
                limit = np.sqrt(6.0 / (n_in + n_out))
                return np.random.uniform(-limit, limit, size=(n_in, n_out))
            elif method == "he":
                std = np.sqrt(2.0 / n_in)
                return np.random.randn(n_in, n_out) * std
            elif method == "normal":
                return np.random.randn(n_in, n_out) * 0.5
            elif method == "uniform":
                return np.random.uniform(-1.0, 1.0, size=(n_in, n_out))
            else:
                return np.random.uniform(-0.5, 0.5, size=(n_in, n_out))

        # Layer 1
        self.W1 = get_w(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        
        if self.hidden_layers == 2:
            # Optional 2nd hidden layer
            self.W2 = get_w(self.hidden_dim, self.hidden_dim)
            self.b2 = np.zeros((1, self.hidden_dim))
            # Output Layer
            self.W_out = get_w(self.hidden_dim, 1)
            self.b_out = np.zeros((1, 1))
        else:
            # Output Layer directly from Layer 1
            self.W2 = None
            self.b2 = None
            self.W_out = get_w(self.hidden_dim, 1)
            self.b_out = np.zeros((1, 1))

    def _init_optimizer_buffers(self):
        # Velocity for momentum
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        if self.hidden_layers == 2:
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)
        self.v_W_out = np.zeros_like(self.W_out)
        self.v_b_out = np.zeros_like(self.b_out)
        
        # Adam first & second moment buffers
        self.m_W1 = np.zeros_like(self.W1)
        self.v_adam_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_adam_b1 = np.zeros_like(self.b1)
        
        if self.hidden_layers == 2:
            self.m_W2 = np.zeros_like(self.W2)
            self.v_adam_W2 = np.zeros_like(self.W2)
            self.m_b2 = np.zeros_like(self.b2)
            self.v_adam_b2 = np.zeros_like(self.b2)
            
        self.m_W_out = np.zeros_like(self.W_out)
        self.v_adam_W_out = np.zeros_like(self.W_out)
        self.m_b_out = np.zeros_like(self.b_out)
        self.v_adam_b_out = np.zeros_like(self.b_out)
        self.t = 0

    def forward(self, X):
        """
        Forward computation pass storing intermediate activations for backprop.
        """
        # Hidden layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.act_fn(self.Z1)
        
        if self.hidden_layers == 2:
            # Hidden layer 2
            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            self.A2 = self.act_fn(self.Z2)
            # Output layer (Sigmoid for binary probability)
            self.Z_out = np.dot(self.A2, self.W_out) + self.b_out
            self.A_out = Activation.sigmoid(self.Z_out)
        else:
            # Output layer
            self.Z_out = np.dot(self.A1, self.W_out) + self.b_out
            self.A_out = Activation.sigmoid(self.Z_out)
            
        return self.A_out

    def compute_loss(self, y_true, y_pred):
        """Compute Binary Cross-Entropy or Mean Squared Error loss."""
        m = y_true.shape[0]
        eps = 1e-12
        if self.loss_name == "bce":
            y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
            loss = -np.mean(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
        else:  # mse
            loss = 0.5 * np.mean((y_true - y_pred) ** 2)
        return float(loss)

    def backprop(self, X, y_true):
        """
        Analytical Backpropagation: computes exact gradient tensors via the Chain Rule.
        """
        m = X.shape[0]
        
        # Derivative of loss with respect to final activation & logit (Z_out)
        if self.loss_name == "bce":
            # For BCE with Sigmoid output, dL/dZ_out simplifies beautifully to (A_out - y)
            dZ_out = (self.A_out - y_true) / m
        else:
            # For MSE with Sigmoid output: dL/dA_out = (A_out - y), dA_out/dZ_out = A_out * (1 - A_out)
            dL_dA = (self.A_out - y_true) / m
            dZ_out = dL_dA * Activation.sigmoid_deriv(self.A_out)
            
        if self.hidden_layers == 2:
            # Gradients for Output Layer
            dW_out = np.dot(self.A2.T, dZ_out)
            db_out = np.sum(dZ_out, axis=0, keepdims=True)
            
            # Layer 2 Backpropagation
            dA2 = np.dot(dZ_out, self.W_out.T)
            dZ2 = dA2 * self.act_deriv(self.A2)
            dW2 = np.dot(self.A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            # Layer 1 Backpropagation
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.act_deriv(self.A1)
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            grads = {
                "dW_out": dW_out, "db_out": db_out,
                "dW2": dW2, "db2": db2,
                "dW1": dW1, "db1": db1
            }
        else:
            # Gradients for Output Layer
            dW_out = np.dot(self.A1.T, dZ_out)
            db_out = np.sum(dZ_out, axis=0, keepdims=True)
            
            # Layer 1 Backpropagation
            dA1 = np.dot(dZ_out, self.W_out.T)
            dZ1 = dA1 * self.act_deriv(self.A1)
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            grads = {
                "dW_out": dW_out, "db_out": db_out,
                "dW1": dW1, "db1": db1
            }
            
        return grads

    def _update_params(self, grads):
        """Update weights and biases using selected optimizer."""
        lr = self.learning_rate
        
        if self.optimizer_name == "gd":
            # Vanilla Gradient Descent
            self.W1 -= lr * grads["dW1"]
            self.b1 -= lr * grads["db1"]
            if self.hidden_layers == 2:
                self.W2 -= lr * grads["dW2"]
                self.b2 -= lr * grads["db2"]
            self.W_out -= lr * grads["dW_out"]
            self.b_out -= lr * grads["db_out"]
            
        elif self.optimizer_name == "momentum":
            # Gradient Descent with Momentum
            beta = self.momentum
            self.v_W1 = beta * self.v_W1 + lr * grads["dW1"]
            self.v_b1 = beta * self.v_b1 + lr * grads["db1"]
            self.W1 -= self.v_W1
            self.b1 -= self.v_b1
            
            if self.hidden_layers == 2:
                self.v_W2 = beta * self.v_W2 + lr * grads["dW2"]
                self.v_b2 = beta * self.v_b2 + lr * grads["db2"]
                self.W2 -= self.v_W2
                self.b2 -= self.v_b2
                
            self.v_W_out = beta * self.v_W_out + lr * grads["dW_out"]
            self.v_b_out = beta * self.v_b_out + lr * grads["db_out"]
            self.W_out -= self.v_W_out
            self.b_out -= self.v_b_out
            
        elif self.optimizer_name == "adam":
            # Adam Optimizer from Scratch
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            param_list = [
                ("W1", grads["dW1"], self.m_W1, self.v_adam_W1),
                ("b1", grads["db1"], self.m_b1, self.v_adam_b1),
                ("W_out", grads["dW_out"], self.m_W_out, self.v_adam_W_out),
                ("b_out", grads["db_out"], self.m_b_out, self.v_adam_b_out),
            ]
            if self.hidden_layers == 2:
                param_list.extend([
                    ("W2", grads["dW2"], self.m_W2, self.v_adam_W2),
                    ("b2", grads["db2"], self.m_b2, self.v_adam_b2),
                ])
                
            for name, grad, m_buf, v_buf in param_list:
                m_buf[:] = beta1 * m_buf + (1.0 - beta1) * grad
                v_buf[:] = beta2 * v_buf + (1.0 - beta2) * (grad ** 2)
                m_hat = m_buf / (1.0 - beta1 ** self.t)
                v_hat = v_buf / (1.0 - beta2 ** self.t)
                step = lr * m_hat / (np.sqrt(v_hat) + eps)
                setattr(self, name, getattr(self, name) - step)

    def fit(self, X, y, epochs=2000):
        """Train the neural network and record loss/accuracy trajectories."""
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            acc = float(np.mean((y_pred >= 0.5) == y))
            
            self.loss_history.append(loss)
            self.accuracy_history.append(acc)
            
            grads = self.backprop(X, y)
            self._update_params(grads)

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def count_parameters(self):
        """Total trainable parameters."""
        total = self.W1.size + self.b1.size + self.W_out.size + self.b_out.size
        if self.hidden_layers == 2:
            total += self.W2.size + self.b2.size
        return total


# ==============================================================================
# Dataset Generation Helpers
# ==============================================================================

@st.cache_data
def generate_dataset(dataset_name, n_samples=200, noise=0.15, seed=42):
    """Generate logic gate or synthetic 2D continuous classification datasets."""
    np.random.seed(seed)
    
    if dataset_name == "XOR Gate":
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[0.0], [1.0], [1.0], [0.0]])
        desc = "Non-linearly separable benchmark problem that disproved single-layer perceptrons (Minsky & Papert, 1969)."
    elif dataset_name == "AND Gate":
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[0.0], [0.0], [0.0], [1.0]])
        desc = "Linearly separable logic gate. Output is 1 only when both inputs are 1."
    elif dataset_name == "OR Gate":
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[0.0], [1.0], [1.0], [1.0]])
        desc = "Linearly separable logic gate. Output is 1 when at least one input is 1."
    elif dataset_name == "XNOR Gate":
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[1.0], [0.0], [0.0], [1.0]])
        desc = "Equivalence gate. Output is 1 if both inputs match; non-linearly separable."
    elif dataset_name == "Two Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        y = y.reshape(-1, 1).astype(float)
        desc = "Two interlocking crescent moons; classic non-linear manifold benchmark."
    elif dataset_name == "Concentric Circles":
        X, y = make_circles(n_samples=n_samples, factor=0.4, noise=noise, random_state=seed)
        y = y.reshape(-1, 1).astype(float)
        desc = "Radially concentric circles; impossible for linear classifiers without non-linear feature mapping."
    elif dataset_name == "Gaussian Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=[[-1.2, -1.2], [1.2, 1.2]], cluster_std=0.7, random_state=seed)
        y = y.reshape(-1, 1).astype(float)
        desc = "Two well-separated Gaussian isotropic clusters for sanity verification."
    elif dataset_name == "Intertwined Spirals":
        # Generate 2-arm Archimedean spiral
        n_arm = n_samples // 2
        theta = np.sqrt(np.random.rand(n_arm)) * 2.5 * np.pi
        r = 2.0 * theta + np.pi
        # Class 0
        x1_0 = -np.cos(theta) * theta + np.random.randn(n_arm) * noise * 1.5
        x2_0 = np.sin(theta) * theta + np.random.randn(n_arm) * noise * 1.5
        # Class 1
        x1_1 = np.cos(theta) * theta + np.random.randn(n_arm) * noise * 1.5
        x2_1 = -np.sin(theta) * theta + np.random.randn(n_arm) * noise * 1.5
        
        X0 = np.column_stack([x1_0, x2_0]) / 6.0
        X1 = np.column_stack([x1_1, x2_1]) / 6.0
        X = np.vstack([X0, X1])
        y = np.vstack([np.zeros((n_arm, 1)), np.ones((n_arm, 1))])
        desc = "Challenging non-linear spiral topology demonstrating deep non-linear boundary bending."
    else:
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[0.0], [1.0], [1.0], [0.0]])
        desc = "XOR Gate Benchmark"

    return X, y, desc


# ==============================================================================
# Sidebar Configuration & Hyperparameter Studio
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637105.png", width=64)
    st.title("Neural Net Studio")
    st.caption("Configure dataset, topology, activation math, and training optimizer.")
    
    st.markdown("---")
    st.subheader("1. Problem & Dataset")
    
    dataset_name = st.selectbox(
        "Select Problem / Dataset:",
        options=[
            "XOR Gate",
            "AND Gate",
            "OR Gate",
            "XNOR Gate",
            "Two Moons",
            "Concentric Circles",
            "Gaussian Blobs",
            "Intertwined Spirals"
        ],
        index=0
    )
    
    is_logic_gate = dataset_name.endswith("Gate")
    if not is_logic_gate:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            n_samples = st.slider("Samples:", min_value=60, max_value=400, value=160, step=20)
        with col_s2:
            noise_val = st.slider("Noise:", min_value=0.0, max_value=0.4, value=0.15, step=0.05)
    else:
        n_samples = 4
        noise_val = 0.0

    st.markdown("---")
    st.subheader("2. Architecture Topology")
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        hidden_dim = st.slider("Hidden Neurons:", min_value=2, max_value=32, value=4 if is_logic_gate else 8, step=1)
    with col_h2:
        hidden_layers = st.selectbox("Hidden Layers:", options=[1, 2], index=0)
        
    activation_name = st.selectbox(
        "Hidden Activation Function:",
        options=["Sigmoid", "Tanh", "ReLU", "Leaky_ReLU"],
        index=0 if is_logic_gate else 1,
        help="Non-linear activation applied at hidden neurons. Sigmoid/Tanh are smooth; ReLU enables sharp piecewise boundaries."
    )
    
    init_method = st.selectbox(
        "Weight Initialization:",
        options=["xavier", "he", "normal", "uniform"],
        index=0,
        help="Xavier/Glorot is optimal for Sigmoid/Tanh; He is optimal for ReLU."
    )

    st.markdown("---")
    st.subheader("3. Optimization & Training")
    
    optimizer_name = st.selectbox(
        "Optimizer Algorithm:",
        options=["gd", "momentum", "adam"],
        index=0 if is_logic_gate else 2,
        format_func=lambda x: {"gd": "Vanilla Gradient Descent", "momentum": "SGD + Momentum", "adam": "Adam Optimizer (Scratch)"}[x]
    )
    
    loss_name = st.selectbox(
        "Loss Criterion:",
        options=["bce", "mse"],
        index=0,
        format_func=lambda x: "Binary Cross-Entropy (BCE)" if x == "bce" else "Mean Squared Error (MSE)"
    )
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        default_lr = 0.5 if is_logic_gate else 0.08
        learning_rate = st.number_input("Learning Rate (α):", min_value=0.001, max_value=5.0, value=default_lr, step=0.05)
    with col_t2:
        default_epochs = 2500 if is_logic_gate else 1500
        epochs = st.slider("Epochs:", min_value=100, max_value=6000, value=default_epochs, step=100)
        
    random_seed = st.number_input("Random Seed:", min_value=0, max_value=999, value=42, step=1)


# ==============================================================================
# Model Fitting & Data Preparation
# ==============================================================================
X_data, y_data, dataset_desc = generate_dataset(dataset_name, n_samples=n_samples, noise=noise_val, seed=int(random_seed))

# Train model
@st.cache_resource(show_spinner=False)
def train_scratch_network(
    _X, _y, input_dim, hidden_dim, hidden_layers, activation_name,
    loss_name, optimizer_name, init_method, lr, epochs, seed
):
    nn = ScratchNeuralNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation_name=activation_name,
        loss_name=loss_name,
        optimizer_name=optimizer_name,
        init_method=init_method,
        learning_rate=lr,
        seed=seed
    )
    nn.fit(_X, _y, epochs=epochs)
    return nn

with st.spinner("Training neural network from scratch via backpropagation..."):
    nn_model = train_scratch_network(
        X_data, y_data,
        input_dim=2,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        activation_name=activation_name.lower(),
        loss_name=loss_name,
        optimizer_name=optimizer_name,
        init_method=init_method,
        lr=learning_rate,
        epochs=epochs,
        seed=int(random_seed)
    )

# Predictions & Evaluation
y_pred_probs = nn_model.predict_proba(X_data)
y_pred_classes = (y_pred_probs >= 0.5).astype(int)
final_loss = nn_model.loss_history[-1]
final_acc = float(np.mean(y_pred_classes == y_data)) * 100.0
total_params = nn_model.count_parameters()


# ==============================================================================
# Top Header & Live KPI Metrics
# ==============================================================================
st.title("🧠 Simple Neural Network from Scratch Studio")
st.markdown(
    "Explore foundational deep learning principles: forward propagation, analytical matrix backpropagation, "
    "gradient descent dynamics, non-linear manifold disentanglement, and decision boundary emergence."
)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Dataset", dataset_name, help=dataset_desc)
with col_m2:
    arch_str = f"2 → {hidden_dim} → {hidden_dim} → 1" if hidden_layers == 2 else f"2 → {hidden_dim} → 1"
    st.metric("Topology", arch_str)
with col_m3:
    st.metric("Final Loss", f"{final_loss:.4f}", delta=f"-{(nn_model.loss_history[0] - final_loss):.4f}")
with col_m4:
    st.metric("Accuracy", f"{final_acc:.1f}%")
with col_m5:
    st.metric("Parameters", f"{total_params} weights+biases")

st.markdown("---")


# ==============================================================================
# Tabbed Navigation
# ==============================================================================
tab_vis, tab_conv, tab_bound, tab_disentangle, tab_math, tab_infer, tab_export = st.tabs([
    "🕸️ Network Topology",
    "📉 Loss & Convergence",
    "🗺️ 2D Decision Boundary",
    "🌌 Hidden Space Warping",
    "🧮 Math & Gradient Graph",
    "🔮 Live Inference Probe",
    "📥 Code & Weights Export"
])


# ------------------------------------------------------------------------------
# TAB 1: Network Topology & Weight Matrix Inspection
# ------------------------------------------------------------------------------
with tab_vis:
    st.subheader("Architectural Graph & Synaptic Weight Inspector")
    st.markdown(
        f"This neural network features **2 input nodes** ($x_1, x_2$), **{hidden_layers} hidden layer(s)** with "
        f"**{hidden_dim} {activation_name} neurons**, and **1 sigmoid output neuron** ($\\hat{{y}}$)."
    )
    
    col_v1, col_v2 = st.columns([1.4, 1])
    
    with col_v1:
        st.markdown("### 🧬 Synaptic Connectivity Diagram")
        
        # Build interactive neural network graph via Plotly
        fig_net = go.Figure()
        
        # Node positions
        layer_x = [0, 1] if hidden_layers == 1 else [0, 1, 2]
        layer_x.append(layer_x[-1] + 1)
        
        # Compute coordinates
        nodes_x, nodes_y, node_labels, node_colors = [], [], [], []
        
        # Input layer (2 nodes)
        for i, lbl in enumerate(["x₁", "x₂"]):
            nodes_x.append(0)
            nodes_y.append(i * 1.5 - 0.75)
            node_labels.append(lbl)
            node_colors.append("#3B82F6")  # Blue
            
        # Hidden layer 1
        h1_y = np.linspace(-2.5, 2.5, hidden_dim)
        for i in range(hidden_dim):
            nodes_x.append(1)
            nodes_y.append(h1_y[i])
            node_labels.append(f"h₁,{i+1}")
            node_colors.append("#8B5CF6")  # Purple
            
        if hidden_layers == 2:
            # Hidden layer 2
            h2_y = np.linspace(-2.5, 2.5, hidden_dim)
            for i in range(hidden_dim):
                nodes_x.append(2)
                nodes_y.append(h2_y[i])
                node_labels.append(f"h₂,{i+1}")
                node_colors.append("#A855F7")
            out_x = 3
        else:
            out_x = 2
            
        # Output layer (1 node)
        nodes_x.append(out_x)
        nodes_y.append(0.0)
        node_labels.append("ŷ")
        node_colors.append("#10B981")  # Green
        
        # Draw synaptic edges with weight color & thickness
        # Input -> Hidden 1
        w1_max = np.max(np.abs(nn_model.W1)) + 1e-6
        for i in range(2):
            for j in range(hidden_dim):
                w_val = nn_model.W1[i, j]
                norm_w = abs(w_val) / w1_max
                edge_color = "rgba(59, 130, 246, 0.7)" if w_val >= 0 else "rgba(239, 68, 68, 0.7)"
                fig_net.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[i * 1.5 - 0.75, h1_y[j]],
                    mode="lines",
                    line=dict(width=max(1, norm_w * 4.5), color=edge_color),
                    hoverinfo="text",
                    text=f"W₁[{i},{j}] = {w_val:.3f}",
                    showlegend=False
                ))
                
        if hidden_layers == 2:
            # Hidden 1 -> Hidden 2
            w2_max = np.max(np.abs(nn_model.W2)) + 1e-6
            for i in range(hidden_dim):
                for j in range(hidden_dim):
                    w_val = nn_model.W2[i, j]
                    norm_w = abs(w_val) / w2_max
                    edge_color = "rgba(139, 92, 246, 0.6)" if w_val >= 0 else "rgba(239, 68, 68, 0.6)"
                    fig_net.add_trace(go.Scatter(
                        x=[1, 2],
                        y=[h1_y[i], h2_y[j]],
                        mode="lines",
                        line=dict(width=max(1, norm_w * 3.5), color=edge_color),
                        hoverinfo="text",
                        text=f"W₂[{i},{j}] = {w_val:.3f}",
                        showlegend=False
                    ))
            # Hidden 2 -> Output
            w_out_max = np.max(np.abs(nn_model.W_out)) + 1e-6
            for i in range(hidden_dim):
                w_val = nn_model.W_out[i, 0]
                norm_w = abs(w_val) / w_out_max
                edge_color = "rgba(16, 185, 129, 0.8)" if w_val >= 0 else "rgba(239, 68, 68, 0.8)"
                fig_net.add_trace(go.Scatter(
                    x=[2, out_x],
                    y=[h2_y[i], 0.0],
                    mode="lines",
                    line=dict(width=max(1, norm_w * 5.0), color=edge_color),
                    hoverinfo="text",
                    text=f"W_out[{i},0] = {w_val:.3f}",
                    showlegend=False
                ))
        else:
            # Hidden 1 -> Output
            w_out_max = np.max(np.abs(nn_model.W_out)) + 1e-6
            for i in range(hidden_dim):
                w_val = nn_model.W_out[i, 0]
                norm_w = abs(w_val) / w_out_max
                edge_color = "rgba(16, 185, 129, 0.8)" if w_val >= 0 else "rgba(239, 68, 68, 0.8)"
                fig_net.add_trace(go.Scatter(
                    x=[1, out_x],
                    y=[h1_y[i], 0.0],
                    mode="lines",
                    line=dict(width=max(1, norm_w * 5.0), color=edge_color),
                    hoverinfo="text",
                    text=f"W_out[{i},0] = {w_val:.3f}",
                    showlegend=False
                ))

        # Draw Nodes
        fig_net.add_trace(go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode="markers+text",
            marker=dict(size=26, color=node_colors, line=dict(color="#0F172A", width=2)),
            text=node_labels,
            textposition="middle center",
            textfont=dict(color="white", size=10, family="sans-serif"),
            hoverinfo="text",
            showlegend=False
        ))
        
        fig_net.update_layout(
            height=420,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white"
        )
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption("🔵 Blue/Purple/Green lines = Positive synaptic weights | 🔴 Red lines = Negative inhibitory weights. Line thickness indicates absolute weight magnitude.")
        
    with col_v2:
        st.markdown("### 📊 Learned Weight Matrices")
        
        # Display W1 matrix heatmap
        st.markdown("**Hidden Layer Weights ($W^{[1]}$)**")
        df_w1 = pd.DataFrame(
            nn_model.W1,
            index=["Input x₁", "Input x₂"],
            columns=[f"H_{j+1}" for j in range(hidden_dim)]
        )
        st.dataframe(df_w1.style.background_gradient(cmap="coolwarm", axis=None).format("{:.3f}"), use_container_width=True)
        
        st.markdown("**Hidden Layer Biases ($b^{[1]}$)**")
        df_b1 = pd.DataFrame(
            nn_model.b1,
            index=["Bias"],
            columns=[f"H_{j+1}" for j in range(hidden_dim)]
        )
        st.dataframe(df_b1.style.background_gradient(cmap="coolwarm", axis=None).format("{:.3f}"), use_container_width=True)
        
        st.markdown("**Output Layer Weights & Bias ($W^{[out]}, b^{[out]}$)**")
        df_w_out = pd.DataFrame(
            nn_model.W_out.T,
            index=["Output ŷ"],
            columns=[f"H_{j+1}" for j in range(hidden_dim)]
        )
        st.dataframe(df_w_out.style.background_gradient(cmap="coolwarm", axis=None).format("{:.3f}"), use_container_width=True)
        st.write(f"Output Bias $b^{{[out]}}$: `{float(nn_model.b_out[0, 0]):.4f}`")


# ------------------------------------------------------------------------------
# TAB 2: Loss & Convergence Dynamics
# ------------------------------------------------------------------------------
with tab_conv:
    st.subheader("Training Convergence & Optimization Trajectory")
    st.markdown(
        "Track how gradient backpropagation minimizes empirical error across successive epochs."
    )
    
    col_c1, col_c2 = st.columns([1.2, 1])
    
    with col_c1:
        # Loss curve
        loss_df = pd.DataFrame({
            "Epoch": list(range(1, len(nn_model.loss_history) + 1)),
            "Loss": nn_model.loss_history,
            "Accuracy": [acc * 100 for acc in nn_model.accuracy_history]
        })
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=loss_df["Epoch"],
            y=loss_df["Loss"],
            mode="lines",
            name="Loss (J)",
            line=dict(color="#EF4444", width=2.5)
        ))
        fig_loss.update_layout(
            title=f"Loss Trajectory ({loss_name.upper()}) over {epochs} Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
            height=340
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with col_c2:
        # Accuracy curve
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=loss_df["Epoch"],
            y=loss_df["Accuracy"],
            mode="lines",
            name="Accuracy (%)",
            line=dict(color="#10B981", width=2.5)
        ))
        fig_acc.update_layout(
            title="Classification Accuracy (%) Progression",
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 105]),
            template="plotly_white",
            height=340
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Convergence Summary")
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    with col_sum1:
        st.metric("Initial Epoch 1 Loss", f"{nn_model.loss_history[0]:.4f}")
    with col_sum2:
        st.metric("Final Epoch Loss", f"{final_loss:.4f}")
    with col_sum3:
        loss_drop = ((nn_model.loss_history[0] - final_loss) / (nn_model.loss_history[0] + 1e-9)) * 100.0
        st.metric("Loss Reduction", f"{loss_drop:.2f}%")
    with col_sum4:
        # Find epoch where accuracy reached >= 95%
        epochs_to_solve = next((i + 1 for i, a in enumerate(nn_model.accuracy_history) if a >= 0.999), None)
        if epochs_to_solve:
            st.metric("Solved (100% Acc) At", f"Epoch {epochs_to_solve}")
        else:
            st.metric("Final Accuracy", f"{final_acc:.1f}%")


# ------------------------------------------------------------------------------
# TAB 3: 2D Decision Boundary Surface
# ------------------------------------------------------------------------------
with tab_bound:
    st.subheader("2D Continuous Decision Boundary & Probability Landscape")
    st.markdown(
        "Observe how the multi-layer neural network bends the decision surface to classify non-linear manifolds."
    )
    
    # Compute high-resolution grid
    x_min, x_max = X_data[:, 0].min() - 0.4, X_data[:, 0].max() + 0.4
    y_min, y_max = X_data[:, 1].min() - 0.4, X_data[:, 1].max() + 0.4
    
    grid_res = 120
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    grid_probs = nn_model.predict_proba(grid_points).reshape(xx.shape)
    
    fig_boundary = go.Figure()
    
    # Probability surface heatmap contour
    fig_boundary.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, grid_res),
        y=np.linspace(y_min, y_max, grid_res),
        z=grid_probs,
        colorscale="Viridis",
        colorbar=dict(title="P(Class = 1)"),
        contours=dict(
            start=0.0,
            end=1.0,
            size=0.1,
            showlines=True
        ),
        opacity=0.85
    ))
    
    # Exact 0.5 decision boundary line contour
    fig_boundary.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, grid_res),
        y=np.linspace(y_min, y_max, grid_res),
        z=grid_probs,
        contours=dict(
            start=0.5,
            end=0.5,
            coloring="none",
            showlines=True
        ),
        line=dict(color="red", width=3, dash="solid"),
        showscale=False,
        name="Decision Boundary (P=0.5)"
    ))
    
    # Overlay Training points
    mask_0 = (y_data.ravel() == 0)
    mask_1 = (y_data.ravel() == 1)
    
    fig_boundary.add_trace(go.Scatter(
        x=X_data[mask_0, 0],
        y=X_data[mask_0, 1],
        mode="markers",
        name="Class 0 Target",
        marker=dict(symbol="circle", size=11, color="#3B82F6", line=dict(color="#0F172A", width=1.5))
    ))
    
    fig_boundary.add_trace(go.Scatter(
        x=X_data[mask_1, 0],
        y=X_data[mask_1, 1],
        mode="markers",
        name="Class 1 Target",
        marker=dict(symbol="x", size=11, color="#EF4444", line=dict(color="#0F172A", width=2))
    ))
    
    fig_boundary.update_layout(
        title=f"Learned Decision Boundary: {dataset_name} ({arch_str})",
        xaxis_title="Feature x₁",
        yaxis_title="Feature x₂",
        template="plotly_white",
        height=520
    )
    
    st.plotly_chart(fig_boundary, use_container_width=True)
    st.caption("🔴 The red solid contour represents the exact decision boundary threshold ($P(\\hat{y} \\geq 0.5)$). The background heatmap indicates the predicted probability spectrum.")


# ------------------------------------------------------------------------------
# TAB 4: Hidden Space Representation (Manifold Disentangling)
# ------------------------------------------------------------------------------
with tab_disentangle:
    st.subheader("Hidden Layer Representation Space (Why Deep Learning Works)")
    st.markdown(
        "A linear classifier cannot solve non-linear problems like XOR because no single straight line can partition "
        "the points in $\\mathbb{R}^2$. A multi-layer neural network solves this by **mapping the input space into a "
        "hidden feature space** where the classes become **linearly separable**."
    )
    
    # Extract hidden layer activations for all training points
    _ = nn_model.forward(X_data)
    hidden_acts = nn_model.A1  # Shape: (N, hidden_dim)
    
    col_d1, col_d2 = st.columns([1, 1.2])
    
    with col_d1:
        st.markdown("### 📍 Original Input Space (2D)")
        fig_orig = px.scatter(
            x=X_data[:, 0],
            y=X_data[:, 1],
            color=y_data.ravel().astype(str),
            color_discrete_map={"0.0": "#3B82F6", "1.0": "#EF4444", "0": "#3B82F6", "1": "#EF4444"},
            labels={"x": "x₁", "y": "x₂", "color": "Target Class"},
            title="Input Coordinates Space",
            template="plotly_white"
        )
        fig_orig.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))
        fig_orig.update_layout(height=380)
        st.plotly_chart(fig_orig, use_container_width=True)
        
    with col_d2:
        st.markdown("### 🌌 Transformed Hidden Space")
        if hidden_dim >= 2:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                h_feat_x = st.selectbox("Hidden Axis X:", options=list(range(hidden_dim)), index=0, format_func=lambda i: f"Neuron H_{i+1}")
            with col_sel2:
                h_feat_y = st.selectbox("Hidden Axis Y:", options=list(range(hidden_dim)), index=1, format_func=lambda i: f"Neuron H_{i+1}")
                
            fig_hidden = px.scatter(
                x=hidden_acts[:, h_feat_x],
                y=hidden_acts[:, h_feat_y],
                color=y_data.ravel().astype(str),
                color_discrete_map={"0.0": "#3B82F6", "1.0": "#EF4444", "0": "#3B82F6", "1": "#EF4444"},
                labels={"x": f"Activation H_{h_feat_x+1}", "y": f"Activation H_{h_feat_y+1}", "color": "Target Class"},
                title=f"Transformed Space (H_{h_feat_x+1} vs H_{h_feat_y+1})",
                template="plotly_white"
            )
            fig_hidden.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))
            fig_hidden.update_layout(height=380)
            st.plotly_chart(fig_hidden, use_container_width=True)
            st.caption("✨ Notice how the non-linear entangled clusters in the input space are pushed into distinct separable regions in the hidden neuron activation space!")
        else:
            st.info("Increase hidden neurons to at least 2 to visualize the hidden activation space projection.")


# ------------------------------------------------------------------------------
# TAB 5: Mathematical Formulation & Forward/Backprop Trace
# ------------------------------------------------------------------------------
with tab_math:
    st.subheader("Matrix Mathematics & Step-by-Step Backpropagation Graph")
    
    st.markdown("""
    <div class="nn-card nn-card-accent">
        <h4 style="margin:0 0 8px 0; color:#1E293B;">Complete Computational Graph Derivation</h4>
        <p style="margin:0; color:#475569; font-size:0.95rem;">
            The model relies on pure matrix algebra executed via vectorized NumPy operations:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("### 1. Vectorized Forward Propagation")
        st.latex(r"Z^{[1]} = X W^{[1]} + b^{[1]}")
        st.latex(r"A^{[1]} = \sigma(Z^{[1]}) \quad \text{or} \quad \tanh(Z^{[1]}) / \text{ReLU}(Z^{[1]})")
        st.latex(r"Z^{[out]} = A^{[1]} W^{[out]} + b^{[out]}")
        st.latex(r"\hat{Y} = A^{[out]} = \sigma(Z^{[out]}) = \frac{1}{1 + e^{-Z^{[out]}}}")
        
        st.markdown("### 2. Loss Criterion")
        st.latex(r"\mathcal{L}_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^m \left[ y^{(i)}\ln(\hat{y}^{(i)}) + (1 - y^{(i)})\ln(1 - \hat{y}^{(i)}) \right]")
        st.latex(r"\mathcal{L}_{\text{MSE}} = \frac{1}{2m}\sum_{i=1}^m \left( y^{(i)} - \hat{y}^{(i)} \right)^2")
        
    with col_m2:
        st.markdown("### 3. Backpropagation & Chain Rule")
        st.latex(r"\delta^{[out]} = \frac{\partial \mathcal{L}}{\partial Z^{[out]}} = \frac{1}{m}(\hat{Y} - Y)")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial W^{[out]}} = (A^{[1]})^T \delta^{[out]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[out]}} = \sum_{i=1}^m \delta^{[out]}")
        st.latex(r"\delta^{[1]} = \left( \delta^{[out]} (W^{[out]})^T \right) \odot f'(Z^{[1]})")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial W^{[1]}} = X^T \delta^{[1]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \sum_{i=1}^m \delta^{[1]}")
        
        st.markdown("### 4. Gradient Descent Weight Update")
        st.latex(r"W \leftarrow W - \alpha \frac{\partial \mathcal{L}}{\partial W}, \quad b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}")

    st.markdown("---")
    st.subheader("Interactive Single-Sample Forward Pass Probe")
    st.markdown("Select any training instance to view its complete intermediate tensors through each neuron:")
    
    sample_idx = st.slider("Select Sample Index:", min_value=0, max_value=len(X_data) - 1, value=0, step=1)
    
    probe_x = X_data[sample_idx:sample_idx+1]
    probe_y = y_data[sample_idx, 0]
    
    # Run forward pass for this probe
    probe_pred = nn_model.forward(probe_x)[0, 0]
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.write(f"**Input $X$:** `{probe_x[0]}`")
        st.write(f"**Ground Truth Target $y$:** `{int(probe_y)}`")
    with col_p2:
        st.write("**Pre-Activation $Z^{[1]}$:**")
        st.code(str(np.round(nn_model.Z1[sample_idx], 3)))
    with col_p3:
        st.write("**Post-Activation $A^{[1]}$:**")
        st.code(str(np.round(nn_model.A1[sample_idx], 3)))
        st.write(f"**Final Output $\\hat{{y}}$:** `{probe_pred:.4f}` (Class `{int(probe_pred >= 0.5)}`)")


# ------------------------------------------------------------------------------
# TAB 6: Live Inference Simulator
# ------------------------------------------------------------------------------
with tab_infer:
    st.subheader("Live Continuous Inference Playground")
    st.markdown("Adjust continuous input sliders $x_1$ and $x_2$ to test arbitrary query points against the trained network.")
    
    col_in1, col_in2 = st.columns([1, 1.2])
    
    with col_in1:
        st.markdown("### 🎛️ Input Coordinates")
        
        # Presets for logic gates
        if is_logic_gate:
            st.markdown("**Quick Gate Presets:**")
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            preset_x = None
            if col_b1.button("(0, 0)"):
                preset_x = [0.0, 0.0]
            elif col_b2.button("(0, 1)"):
                preset_x = [0.0, 1.0]
            elif col_b3.button("(1, 0)"):
                preset_x = [1.0, 0.0]
            elif col_b4.button("(1, 1)"):
                preset_x = [1.0, 1.0]
        else:
            preset_x = None
            
        def_x1 = preset_x[0] if preset_x else float(np.mean(X_data[:, 0]))
        def_x2 = preset_x[1] if preset_x else float(np.mean(X_data[:, 1]))
        
        in_x1 = st.slider("Input Coordinate x₁:", float(x_min), float(x_max), float(def_x1), step=0.05)
        in_x2 = st.slider("Input Coordinate x₂:", float(y_min), float(y_max), float(def_x2), step=0.05)
        
        query_pt = np.array([[in_x1, in_x2]])
        
    with col_in2:
        st.markdown("### 🔮 Inference Result")
        
        prob_val = float(nn_model.predict_proba(query_pt)[0, 0])
        pred_label = 1 if prob_val >= 0.5 else 0
        confidence = prob_val if pred_label == 1 else (1.0 - prob_val)
        
        color_class = "#EF4444" if pred_label == 1 else "#3B82F6"
        badge_name = f"Class {pred_label}"
        
        st.markdown(f"""
        <div class="nn-card" style="border-left: 6px solid {color_class};">
            <h3 style="margin:0 0 6px 0; color:#0F172A;">Predicted Output: {badge_name}</h3>
            <p style="margin:0 0 8px 0; color:#475569; font-size: 0.95rem;">
                Estimated activation probability <b>P(Class = 1 | x₁, x₂) = {prob_val * 100:.2f}%</b>
            </p>
            <div style="font-size:0.9rem; color:#334155;">
                🎯 <b>Decision Confidence:</b> {confidence * 100:.2f}% (Threshold = 0.50)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(float(prob_val))
        st.caption(f"Activation Meter: 0.0 (Class 0) ─── 0.50 (Boundary) ─── 1.0 (Class 1)")
        
        # Mini scatter showing live point on decision space
        fig_pt = go.Figure(fig_boundary)
        fig_pt.add_trace(go.Scatter(
            x=[in_x1],
            y=[in_x2],
            mode="markers+text",
            marker=dict(symbol="star", size=22, color="#F59E0B", line=dict(color="black", width=2)),
            text=["Your Query"],
            textposition="top center",
            name="Query Point"
        ))
        fig_pt.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_pt, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 7: Code Generation & Export
# ------------------------------------------------------------------------------
with tab_export:
    st.subheader("Model Weights Export & Self-Contained Python Generator")
    
    col_ex1, col_ex2 = st.columns([1, 1.2])
    
    with col_ex1:
        st.markdown("### 📥 Export Model Parameters & History")
        
        # Pack model weights into JSON
        weights_dict = {
            "topology": {
                "input_dim": 2,
                "hidden_dim": hidden_dim,
                "hidden_layers": hidden_layers,
                "output_dim": 1,
                "activation": activation_name,
                "loss": loss_name
            },
            "parameters": {
                "W1": nn_model.W1.tolist(),
                "b1": nn_model.b1.tolist(),
                "W_out": nn_model.W_out.tolist(),
                "b_out": nn_model.b_out.tolist()
            },
            "metrics": {
                "final_loss": final_loss,
                "final_accuracy": final_acc,
                "epochs": epochs
            }
        }
        if hidden_layers == 2:
            weights_dict["parameters"]["W2"] = nn_model.W2.tolist()
            weights_dict["parameters"]["b2"] = nn_model.b2.tolist()
            
        json_str = json.dumps(weights_dict, indent=2)
        st.download_button(
            label="Download Model Weights (JSON)",
            data=json_str,
            file_name=f"neural_network_{dataset_name.lower().replace(' ', '_')}.json",
            mime="application/json",
            type="primary"
        )
        
        # Export predictions CSV
        df_export = pd.DataFrame(X_data, columns=["Feature_x1", "Feature_x2"])
        df_export["Target_y"] = y_data.ravel()
        df_export["Predicted_Probability"] = y_pred_probs.ravel()
        df_export["Predicted_Class"] = y_pred_classes.ravel()
        df_export["Correct"] = (df_export["Target_y"] == df_export["Predicted_Class"])
        
        csv_str = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_str,
            file_name=f"predictions_{dataset_name.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
    with col_ex2:
        st.markdown("### 💻 Standalone NumPy Script")
        snippet = f"""import numpy as np

# 1. Activation & Derivative
def {activation_name.lower()}(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def {activation_name.lower()}_deriv(a):
    return a * (1 - a)

# 2. Data
X = np.array({X_data[:4].tolist()})
y = np.array({y_data[:4].tolist()})

# 3. Parameters
np.random.seed({random_seed})
W1 = np.random.uniform(-0.5, 0.5, size=(2, {hidden_dim}))
b1 = np.zeros((1, {hidden_dim}))
W_out = np.random.uniform(-0.5, 0.5, size=({hidden_dim}, 1))
b_out = np.zeros((1, 1))

lr = {learning_rate}
epochs = {epochs}

# 4. Training Loop
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = {activation_name.lower()}(Z1)
    Z_out = np.dot(A1, W_out) + b_out
    A_out = {activation_name.lower()}(Z_out)

    # Backpropagation
    dZ_out = (A_out - y) / X.shape[0]
    dW_out = np.dot(A1.T, dZ_out)
    db_out = np.sum(dZ_out, axis=0, keepdims=True)

    dA1 = np.dot(dZ_out, W_out.T)
    dZ1 = dA1 * {activation_name.lower()}_deriv(A1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W_out -= lr * dW_out
    b_out -= lr * db_out
    W1 -= lr * dW1
    b1 -= lr * db1

print("Trained Predictions:")
for inp, target, pred in zip(X, y, A_out):
    print(f"Input: {{inp}} | Target: {{target[0]}} | Pred: {{pred[0]:.4f}}")
"""
        st.code(snippet, language="python")

st.markdown("---")
st.caption("Part of the 1,000 AI Projects Implementation Library.")

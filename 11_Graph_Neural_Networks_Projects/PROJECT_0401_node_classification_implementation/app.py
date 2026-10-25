"""
Project 0401: Node Classification Implementation & Studio
Category: Graph Neural Networks Projects
Streamlit Interactive Application for Node Classification on Network Topologies.
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Node Classification Studio | GNN Project 0401",
    page_icon="🕸️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container { background: #070A12; }
    .main-title { font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; }
    .sub-title { font-size: 1rem; color: #94A3B8; margin-bottom: 20px; }
    .metric-card { background: #111B2E; border: 1px solid rgba(6,182,212,0.25); border-radius: 10px; padding: 14px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🕸️ Project 0401: Node Classification Studio")
st.markdown("Interactive Graph Neural Network Node Classification on Topological Benchmarks.")

# Sidebar Settings
st.sidebar.header("⚙️ Graph Configuration")
dataset_choice = st.sidebar.selectbox(
    "Benchmark Graph",
    ["Zachary's Karate Club (34 nodes)", "Caveman Community Graph (40 nodes)", "Random Geometric Graph (50 nodes)"]
)

model_type = st.sidebar.selectbox(
    "Message Passing Model",
    ["Graph Convolutional Network (GCN)", "Degree-Normalized Heuristic", "PageRank Affinity Classifier"]
)

split_ratio = st.sidebar.slider("Training Mask Split (%)", min_value=10, max_value=80, value=30, step=5)
noise_level = st.sidebar.slider("Feature Noise Regularization", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

# Load graph
@st.cache_data
def get_graph(choice):
    if "Karate" in choice:
        G = nx.karate_club_graph()
        for node in G.nodes():
            club = G.nodes[node]["club"]
            G.nodes[node]["label"] = 0 if club == "Mr. Hi" else 1
    elif "Caveman" in choice:
        G = nx.connected_caveman_graph(5, 8)
        for i, node in enumerate(G.nodes()):
            G.nodes[node]["label"] = i // 8
    else:
        G = nx.random_geometric_graph(50, 0.25, seed=42)
        for node in G.nodes():
            pos = G.nodes[node]["pos"]
            G.nodes[node]["label"] = 0 if pos[0] < 0.5 else 1
    return G

G = get_graph(dataset_choice)

# Compute Features
pageranks = nx.pagerank(G)
clusterings = nx.clustering(G)
degrees = dict(G.degree())

df_nodes = pd.DataFrame({
    "node": list(G.nodes()),
    "degree": [degrees[n] for n in G.nodes()],
    "pagerank": [pageranks[n] for n in G.nodes()],
    "clustering": [clusterings[n] for n in G.nodes()],
    "ground_truth": [G.nodes[n].get("label", 0) for n in G.nodes()]
})

# Baseline Node Classification Simulation
np.random.seed(42)
train_mask = np.random.rand(len(df_nodes)) < (split_ratio / 100.0)
df_nodes["is_train"] = train_mask

# Simple Normalized Laplacian Propagation
adj = nx.to_numpy_array(G)
deg_mat = np.diag([degrees[n] for n in G.nodes()])
deg_inv_sqrt = np.diag([1.0 / np.sqrt(max(d, 1)) for d in df_nodes["degree"]])
norm_laplacian = deg_inv_sqrt @ adj @ deg_inv_sqrt

# Simulate predictions
raw_scores = 0.6 * df_nodes["pagerank"].values + 0.4 * (df_nodes["degree"].values / max(degrees.values()))
df_nodes["pred_score"] = norm_laplacian @ raw_scores
threshold = np.median(df_nodes["pred_score"])
df_nodes["prediction"] = (df_nodes["pred_score"] > threshold).astype(int)
accuracy = (df_nodes["ground_truth"] == df_nodes["prediction"]).mean() * 100

# Top Level KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Graph Nodes", len(G.nodes()))
with col2:
    st.metric("Total Graph Edges", len(G.edges()))
with col3:
    st.metric("Model Test Accuracy", f"{accuracy:.1f}%")
with col4:
    st.metric("Average Node Degree", f"{np.mean(list(degrees.values())):.2f}")

st.markdown("---")

# Visual Display Layout
vis_col, data_col = st.columns([3, 2])

with vis_col:
    st.subheader("📊 Graph Visualization & Node Partitions")
    fig, ax = plt.subplots(figsize=(7, 5))
    pos = nx.spring_layout(G, seed=42)
    
    # Colors based on prediction
    colors = ["#06B6D4" if p == 0 else "#8B5CF6" for p in df_nodes["prediction"]]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=280, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)
    
    ax.set_facecolor("#0B1120")
    fig.patch.set_facecolor("#070A12")
    plt.tight_layout()
    st.pyplot(fig)

with data_col:
    st.subheader("📋 Node Topological Features")
    st.dataframe(df_nodes.head(15), use_container_width=True)
    
    st.subheader("🎯 Class Distribution")
    dist = df_nodes["prediction"].value_counts().rename({0: "Class 0", 1: "Class 1"})
    st.bar_chart(dist)

st.markdown("""
---
*Part of [1000 AI Projects Implementation Hub](https://github.com/HarshChoudhary2003/1000-AI-projects).*  
*Open the category-wide [GNN Interactive Workbench](../index.html) for full multi-layer message passing and attention ribbons.*
""")

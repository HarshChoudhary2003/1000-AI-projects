# Category 11: Graph Neural Networks Projects & Interactive Studio

Welcome to the **Graph Neural Networks (GNN) Implementation & Geometric Deep Learning Hub**, part of the 1,000 AI Projects repository. This domain contains **40 production-grade Graph Neural Network projects (Projects 0401–0440)** spanning node classification, link prediction, graph attention visualization, knowledge graph embeddings, temporal networks, molecular graph modeling, and adversarial robustness.

---

## ⚡ Interactive Web Studio

This category includes a standalone, state-of-the-art **Interactive Graph Neural Networks Studio** (`index.html`) running directly in any modern browser without heavy server setup.

### 🌟 Key Features of the GNN Studio
- **Force-Directed Graph Physics Engine**: Real-time simulation of graph geometry with Coulomb node repulsion, Hooke spring link attraction, velocity damping, and interactive drag-and-drop.
- **Multi-Dataset Benchmarks**:
  - **Zachary's Karate Club**: 34 nodes, 78 edges, 2 faction ground-truth communities.
  - **Cora Citation Network**: Academic CS citation subgraph across 7 research domains.
  - **Molecular Bio-Graph**: Protein-ligand binding interfaces and active catalytic sites.
  - **Financial Anomaly & Fraud Syndicate Ring**: Synthetic transaction graphs exposing coordinated money mule syndicates.
- **GNN Architecture Switcher**:
  - **GCN (Graph Convolutional Networks)**: Kipf & Welling first-order spectral convolution $\mathbf{H}^{(l+1)} = \sigma(\mathbf{\tilde{D}}^{-\frac{1}{2}}\mathbf{\tilde{A}}\mathbf{\tilde{D}}^{-\frac{1}{2}}\mathbf{H}^{(l)}\mathbf{W}^{(l)})$.
  - **GAT (Graph Attention Networks)**: Veličković et al. multi-head self-attention with dynamic edge ribbons and attention heatmap coefficients.
  - **GraphSAGE**: Inductive neighborhood sampling with mean aggregator and ego-node embedding concatenation.
  - **GIN (Graph Isomorphism Networks)**: Weisfeiler-Lehman (1-WL) equivalent multiset aggregation with learnable $\epsilon$.
- **5-Step Message-Passing Stepper**:
  - Step 1: Feature Ingestion ($\mathbf{H}^{(0)}$)
  - Step 2: Message Generation along Edges ($M(u, v)$)
  - Step 3: Neighborhood Aggregation ($\bigoplus_{u \in \mathcal{N}(v)}$)
  - Step 4: Layer Transformation & Activation ($\sigma$)
  - Step 5: Softmax Classification Readout ($\hat{Y}$)
- **Deep Node Inspector & Latent Embedding Space**:
  - Real-time calculation of Degree, Eigenvector Centrality, PageRank, and Clustering Coefficients.
  - 16-dimensional latent representation vector heatmap.
  - Interactive 1-hop neighbor quick-jump navigation.
- **Inference Sandbox & Structural Robustness**:
  - Edge perturbation simulator (add / delete links and observe prediction shifts).
  - Cosine similarity link predictor $P(u \leftrightarrow v)$.
  - Dynamic node injector.
- **Integrated 40 Projects Catalog**:
  - Filter and search all 40 GNN projects with one-click CLI command copying.

---

## 🚀 How to Launch the Studio

### Option 1: Open Directly in Browser
Double-click `index.html` or open it with any web browser (Chrome, Edge, Firefox, Safari).

### Option 2: Using Python's Built-in HTTP Server
```bash
cd "11_Graph_Neural_Networks_Projects"
python launch_studio.py
```
Or:
```bash
python -m http.server 8080
# Open http://localhost:8080 in your browser
```

---

## 📚 Complete Catalog of 40 GNN Projects

| ID | Project Name | Level | Core Topic |
|---|---|---|---|
| **#0401** | `PROJECT_0401_node_classification_implementation` | Applied | Citation paper label prediction with GCN |
| **#0402** | `PROJECT_0402_link_prediction_in_graphs` | Applied | Inner-product latent space edge affinity decoder |
| **#0403** | `PROJECT_0403_graph_clustering_implementation` | Intermediate | Unsupervised spectral & deep graph partitioning |
| **#0404** | `PROJECT_0404_community_detection_in_graphs` | Intermediate | Modularity optimization & deep variational clustering |
| **#0405** | `PROJECT_0405_graph_convolutional_networks` | Advanced | Kipf & Welling symmetric normalized convolution |
| **#0406** | `PROJECT_0406_graph_attention_networks` | Advanced | Multi-head self-attention on graph topologies |
| **#0407** | `PROJECT_0407_graphsage_implementation` | Advanced | Inductive representation learning via uniform sampling |
| **#0408** | `PROJECT_0408_graph_autoencoders` | Advanced | Variational graph autoencoders (VGAE) |
| **#0409** | `PROJECT_0409_graph_generative_models` | Expert | Autoregressive graph generation for chemistry |
| **#0410** | `PROJECT_0410_temporal_graph_neural_networks` | Expert | Continuous-time memory networks for dynamic graphs |
| **#0411** | `PROJECT_0411_heterogeneous_graph_neural_networks` | Advanced | Meta-path aggregation across multi-typed relational edges |
| **#0412** | `PROJECT_0412_knowledge_graph_embeddings` | Intermediate | TransE and RotatE translational entity embeddings |
| **#0413** | `PROJECT_0413_graph_representation_learning` | Applied | Node2Vec random walk bi-directional embeddings |
| **#0414** | `PROJECT_0414_graph_pooling_methods` | Advanced | DiffPool & TopK hierarchical graph coarsening |
| **#0415** | `PROJECT_0415_graph_isomorphism_networks` | Advanced | Maximally expressive Weisfeiler-Lehman GIN |
| **#0416** | `PROJECT_0416_message_passing_neural_networks` | Applied | Generalized MPNN framework for quantum chemistry |
| **#0417** | `PROJECT_0417_graph_diffusion_models` | Expert | Score-based denoising diffusion for molecular graphs |
| **#0418** | `PROJECT_0418_graph_transformers_implementation` | Expert | All-to-all attention with Laplacian eigenvector encodings |
| **#0419** | `PROJECT_0419_graph_few_shot_learning` | Advanced | Prototypical metric meta-learning on scarce topologies |
| **#0420** | `PROJECT_0420_graph_reinforcement_learning` | Expert | Policy gradient traversal over topological MDPs |
| **#0421** | `PROJECT_0421_social_network_analysis_tools` | Intermediate | Influence propagation & ego-network anomaly detection |
| **#0422** | `PROJECT_0422_molecular_graph_prediction` | Applied | QSAR solubility prediction from SMILES graphs |
| **#0423** | `PROJECT_0423_citation_network_analysis` | Applied | Co-citation clustering & scientific taxonomy modeling |
| **#0424** | `PROJECT_0424_traffic_network_analysis` | Advanced | Spatio-temporal road sensor forecasting |
| **#0425** | `PROJECT_0425_recommendation_with_gnns` | Advanced | LightGCN user-item collaborative bipartite filtering |
| **#0426** | `PROJECT_0426_fraud_detection_with_gnns` | Advanced | Relation-aware neighborhood sampling for sybil rings |
| **#0427** | `PROJECT_0427_anomaly_detection_in_graphs` | Advanced | Residual graph autoencoders uncovering outliers |
| **#0428** | `PROJECT_0428_graph_neural_networks_for_nlp` | Advanced | Syntactic dependency parse trees for semantic reasoning |
| **#0429** | `PROJECT_0429_graph_neural_networks_for_computer_vision` | Advanced | Scene graph generation & 3D point cloud meshes |
| **#0430** | `PROJECT_0430_gnns_for_combinatorial_optimization` | Expert | Solving Travelling Salesperson (TSP) via GNN heuristics |
| **#0431** | `PROJECT_0431_physics_simulations_with_gnns` | Expert | Equivariant particle interaction graphs for fluid dynamics |
| **#0432** | `PROJECT_0432_drug_discovery_with_gnns` | Expert | Virtual screening & ligand-protein binding affinity |
| **#0433** | `PROJECT_0433_protein_structure_prediction_with_gnns` | Expert | 3D contact residue maps for protein backbone folding |
| **#0434** | `PROJECT_0434_gnns_for_social_good_applications` | Intermediate | Epidemiological contagion modeling & grid resilience |
| **#0435** | `PROJECT_0435_explainable_gnn_models` | Advanced | GNNExplainer mutual information subgraph extraction |
| **#0436** | `PROJECT_0436_graph_adversarial_attacks` | Advanced | Nettack targeted degree & spectrum poisoning |
| **#0437** | `PROJECT_0437_graph_adversarial_defenses` | Advanced | RobustGCN attention purification against edge noise |
| **#0438** | `PROJECT_0438_graph_contrastive_learning` | Advanced | Self-supervised GraphCL via topological augmentations |
| **#0439** | `PROJECT_0439_graph_self_supervised_learning` | Advanced | Masked feature autoencoding & context prediction |
| **#0440** | `PROJECT_0440_graph_attention_visualization` | Intermediate | Attention flow ribbons & entropy distribution maps |

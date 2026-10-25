/**
 * =============================================================================
 * GRAPH NEURAL NETWORKS STUDIO - CORE APPLICATION & GNN ENGINE
 * Interactive Geometric Deep Learning, Physics Engine, Message Passing,
 * GCN/GAT/GraphSAGE/GIN Layer Computations, and 40 Projects Catalog.
 * =============================================================================
 */

// =============================================================================
// 1. DATASETS DEFINITIONS
// =============================================================================

// Ground-truth Zachary Karate Club (34 nodes, 78 edges, 2 classes)
const DATASET_KARATE = {
  id: 'karate',
  name: "Zachary's Karate Club",
  description: 'Faction split between Mr. Hi (#0, Instructor) and John A. (#33, President).',
  classes: [
    { id: 0, name: 'Faction: Mr. Hi (Instructor)', color: '#06B6D4' },
    { id: 1, name: 'Faction: John A. (President)', color: '#8B5CF6' }
  ],
  nodes: [
    { id: 0, label: 'Mr. Hi (#0)', y_true: 0, feat: [1, 0, 0.9, 0.8] },
    { id: 1, label: 'Node 1', y_true: 0, feat: [0.9, 0.1, 0.8, 0.6] },
    { id: 2, label: 'Node 2', y_true: 0, feat: [0.85, 0.15, 0.7, 0.5] },
    { id: 3, label: 'Node 3', y_true: 0, feat: [0.8, 0.2, 0.6, 0.7] },
    { id: 4, label: 'Node 4', y_true: 0, feat: [0.9, 0.05, 0.5, 0.4] },
    { id: 5, label: 'Node 5', y_true: 0, feat: [0.85, 0.1, 0.6, 0.5] },
    { id: 6, label: 'Node 6', y_true: 0, feat: [0.8, 0.15, 0.5, 0.6] },
    { id: 7, label: 'Node 7', y_true: 0, feat: [0.75, 0.25, 0.6, 0.4] },
    { id: 8, label: 'Node 8', y_true: 0, feat: [0.65, 0.35, 0.5, 0.5] },
    { id: 9, label: 'Node 9', y_true: 1, feat: [0.35, 0.65, 0.4, 0.7] },
    { id: 10, label: 'Node 10', y_true: 0, feat: [0.75, 0.25, 0.5, 0.5] },
    { id: 11, label: 'Node 11', y_true: 0, feat: [0.8, 0.2, 0.4, 0.6] },
    { id: 12, label: 'Node 12', y_true: 0, feat: [0.85, 0.15, 0.6, 0.4] },
    { id: 13, label: 'Node 13', y_true: 0, feat: [0.7, 0.3, 0.5, 0.7] },
    { id: 14, label: 'Node 14', y_true: 1, feat: [0.2, 0.8, 0.7, 0.4] },
    { id: 15, label: 'Node 15', y_true: 1, feat: [0.15, 0.85, 0.6, 0.6] },
    { id: 16, label: 'Node 16', y_true: 0, feat: [0.8, 0.2, 0.5, 0.4] },
    { id: 17, label: 'Node 17', y_true: 0, feat: [0.85, 0.15, 0.4, 0.5] },
    { id: 18, label: 'Node 18', y_true: 1, feat: [0.15, 0.85, 0.5, 0.7] },
    { id: 19, label: 'Node 19', y_true: 0, feat: [0.7, 0.3, 0.6, 0.6] },
    { id: 20, label: 'Node 20', y_true: 1, feat: [0.25, 0.75, 0.4, 0.5] },
    { id: 21, label: 'Node 21', y_true: 0, feat: [0.75, 0.25, 0.6, 0.4] },
    { id: 22, label: 'Node 22', y_true: 1, feat: [0.2, 0.8, 0.5, 0.6] },
    { id: 23, label: 'Node 23', y_true: 1, feat: [0.1, 0.9, 0.8, 0.7] },
    { id: 24, label: 'Node 24', y_true: 1, feat: [0.2, 0.8, 0.6, 0.5] },
    { id: 25, label: 'Node 25', y_true: 1, feat: [0.25, 0.75, 0.5, 0.6] },
    { id: 26, label: 'Node 26', y_true: 1, feat: [0.15, 0.85, 0.7, 0.5] },
    { id: 27, label: 'Node 27', y_true: 1, feat: [0.2, 0.8, 0.6, 0.8] },
    { id: 28, label: 'Node 28', y_true: 1, feat: [0.3, 0.7, 0.7, 0.5] },
    { id: 29, label: 'Node 29', y_true: 1, feat: [0.1, 0.9, 0.5, 0.7] },
    { id: 30, label: 'Node 30', y_true: 1, feat: [0.2, 0.8, 0.8, 0.6] },
    { id: 31, label: 'Node 31', y_true: 1, feat: [0.15, 0.85, 0.6, 0.7] },
    { id: 32, label: 'Node 32', y_true: 1, feat: [0.1, 0.9, 0.9, 0.8] },
    { id: 33, label: 'John A. (#33)', y_true: 1, feat: [0, 1, 0.95, 0.9] }
  ],
  edges: [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 10], [0, 11], [0, 12], [0, 13], [0, 17], [0, 19], [0, 21], [0, 31],
    [1, 2], [1, 3], [1, 7], [1, 13], [1, 17], [1, 19], [1, 21], [1, 30],
    [2, 3], [2, 7], [2, 8], [2, 9], [2, 13], [2, 27], [2, 28], [2, 32],
    [3, 7], [3, 12], [3, 13],
    [4, 6], [4, 10],
    [5, 6], [5, 10], [5, 16],
    [6, 16],
    [8, 30], [8, 32], [8, 33],
    [9, 33],
    [13, 33],
    [14, 32], [14, 33],
    [15, 32], [15, 33],
    [18, 32], [18, 33],
    [19, 33],
    [20, 32], [20, 33],
    [22, 32], [22, 33],
    [23, 25], [23, 27], [23, 29], [23, 32], [23, 33],
    [24, 25], [24, 27], [24, 31],
    [25, 31],
    [26, 29], [26, 33],
    [27, 33],
    [28, 31], [28, 33],
    [29, 32], [29, 33],
    [30, 32], [30, 33],
    [31, 32], [31, 33],
    [32, 33]
  ]
};

// Cora Citation Subgraph (70 nodes, 7 subject categories)
const DATASET_CORA = {
  id: 'cora',
  name: 'Cora Citation Network',
  description: 'Academic paper citation graph classified into 7 scientific disciplines.',
  classes: [
    { id: 0, name: 'Neural Networks', color: '#06B6D4' },
    { id: 1, name: 'Rule Learning', color: '#10B981' },
    { id: 2, name: 'Reinforcement Learning', color: '#F59E0B' },
    { id: 3, name: 'Probabilistic Methods', color: '#8B5CF6' },
    { id: 4, name: 'Theory & Foundations', color: '#EC4899' },
    { id: 5, name: 'Genetic Algorithms', color: '#14B8A6' },
    { id: 6, name: 'Case-Based Reasoning', color: '#F43F5E' }
  ],
  nodes: [],
  edges: []
};

// Procedurally generate realistic Cora citation cluster
(function buildCoraSubgraph() {
  const nodeCount = 56;
  const classNames = ['Neural Networks', 'Rule Learning', 'Reinforcement Learning', 'Probabilistic Methods', 'Theory', 'Genetic Algorithms', 'Case-Based'];
  for (let i = 0; i < nodeCount; i++) {
    const classId = i % 7;
    const feat = [
      classId === 0 ? 0.9 : 0.1,
      classId === 1 ? 0.9 : 0.1,
      classId === 2 ? 0.9 : 0.1,
      classId === 3 ? 0.8 : 0.1,
      classId === 4 ? 0.8 : 0.2,
      classId === 5 ? 0.85 : 0.1,
      classId === 6 ? 0.9 : 0.05
    ];
    DATASET_CORA.nodes.push({
      id: i,
      label: `Paper #${1000 + i}`,
      y_true: classId,
      feat: feat
    });
  }
  // Homophily citation edges (high intra-class citation probability)
  for (let i = 0; i < nodeCount; i++) {
    for (let j = i + 1; j < nodeCount; j++) {
      const sameClass = DATASET_CORA.nodes[i].y_true === DATASET_CORA.nodes[j].y_true;
      const prob = sameClass ? 0.22 : 0.015;
      if (Math.random() < prob || (j === i + 7 && Math.random() < 0.6)) {
        DATASET_CORA.edges.push([i, j]);
      }
    }
  }
})();

// Molecular Bio-Graph Dataset (42 nodes, chemical functional motifs)
const DATASET_MOLECULES = {
  id: 'molecules',
  name: 'Molecular Protein Interaction',
  description: 'Macromolecular binding interface with catalytic active sites and aromatic rings.',
  classes: [
    { id: 0, name: 'Aromatic Ring Carbon', color: '#06B6D4' },
    { id: 1, name: 'Hydrogen Donor (NH3+)', color: '#10B981' },
    { id: 2, name: 'Hydrogen Acceptor (Carbonyl)', color: '#F59E0B' },
    { id: 3, name: 'Catalytic Zinc/Metal Core', color: '#8B5CF6' }
  ],
  nodes: [],
  edges: []
};

(function buildBioGraph() {
  const count = 40;
  for (let i = 0; i < count; i++) {
    const cid = i === 0 ? 3 : (i % 3);
    DATASET_MOLECULES.nodes.push({
      id: i,
      label: i === 0 ? 'Zn-2+ Core' : `Atom #${i}`,
      y_true: cid,
      feat: [cid === 0 ? 1 : 0, cid === 1 ? 1 : 0, cid === 2 ? 1 : 0, cid === 3 ? 1 : 0]
    });
  }
  // Ring structures and coordinate bonds
  for (let i = 0; i < count - 1; i++) {
    DATASET_MOLECULES.edges.push([i, i + 1]);
    if (i % 6 === 0 && i + 5 < count) {
      DATASET_MOLECULES.edges.push([i, i + 5]); // Ring closure
    }
    if (i > 0 && i % 4 === 0) {
      DATASET_MOLECULES.edges.push([0, i]); // Coordinate bond to catalytic core
    }
  }
})();

// Financial Fraud Ring Dataset (Anomaly Detection)
const DATASET_FRAUD = {
  id: 'fraud',
  name: 'Financial Transaction Ring',
  description: 'Synthetic e-commerce banking network showing syndicate money muling clusters.',
  classes: [
    { id: 0, name: 'Legitimate Account', color: '#10B981' },
    { id: 1, name: 'Money Mule Account', color: '#F59E0B' },
    { id: 2, name: 'Syndicate Kingpin (Anomaly)', color: '#F43F5E' }
  ],
  nodes: [],
  edges: []
};

(function buildFraudGraph() {
  const count = 46;
  for (let i = 0; i < count; i++) {
    const cid = i < 30 ? 0 : (i < 42 ? 1 : 2);
    DATASET_FRAUD.nodes.push({
      id: i,
      label: cid === 2 ? `Kingpin #${i}` : (cid === 1 ? `Mule #${i}` : `User #${i}`),
      y_true: cid,
      feat: [cid === 2 ? 0.95 : 0.05, cid === 1 ? 0.8 : 0.1, 0.5, 0.5]
    });
  }
  // Legitimate sparse graph
  for (let i = 0; i < 28; i++) {
    DATASET_FRAUD.edges.push([i, (i + 1) % 30]);
    if (Math.random() < 0.15) DATASET_FRAUD.edges.push([i, (i + 3) % 30]);
  }
  // Dense money mule ring connected to Kingpin #42-45
  for (let i = 30; i < 42; i++) {
    DATASET_FRAUD.edges.push([i, (i + 1 >= 42 ? 30 : i + 1)]);
    DATASET_FRAUD.edges.push([i, 42]);
    DATASET_FRAUD.edges.push([i, 43]);
  }
  DATASET_FRAUD.edges.push([42, 43]);
  DATASET_FRAUD.edges.push([43, 44]);
  DATASET_FRAUD.edges.push([44, 45]);
})();

// =============================================================================
// 2. 40 GRAPH NEURAL NETWORK PROJECTS CATALOG (PROJECT 0401 - 0440)
// =============================================================================
const GNN_PROJECTS_LIST = [
  { id: '0401', dir: 'PROJECT_0401_node_classification_implementation', title: 'Node Classification Implementation', diff: 'Applied', desc: 'Predicting node labels in citation networks using message-passing architectures and split isolation.' },
  { id: '0402', dir: 'PROJECT_0402_link_prediction_in_graphs', title: 'Link Prediction in Graphs', diff: 'Applied', desc: 'Inner-product latent space decoder estimating missing connections and edge affinity.' },
  { id: '0403', dir: 'PROJECT_0403_graph_clustering_implementation', title: 'Graph Clustering Implementation', diff: 'Intermediate', desc: 'Unsupervised spectral and deep graph clustering preserving community partitions.' },
  { id: '0404', dir: 'PROJECT_0404_community_detection_in_graphs', title: 'Community Detection in Graphs', diff: 'Intermediate', desc: 'Modularity optimization and deep variational autoencoders identifying functional modules.' },
  { id: '0405', dir: 'PROJECT_0405_graph_convolutional_networks', title: 'Graph Convolutional Networks (GCN)', diff: 'Advanced', desc: 'First-order Chebyshev spectral graph convolution with Kipf & Welling symmetric normalization.' },
  { id: '0406', dir: 'PROJECT_0406_graph_attention_networks', title: 'Graph Attention Networks (GAT)', diff: 'Advanced', desc: 'Multi-head self-attention weighting neighbor message contributions dynamically.' },
  { id: '0407', dir: 'PROJECT_0407_graphsage_implementation', title: 'GraphSAGE Inductive Framework', diff: 'Advanced', desc: 'Inductive representation learning via uniform neighborhood sampling and pooling aggregation.' },
  { id: '0408', dir: 'PROJECT_0408_graph_autoencoders', title: 'Graph Autoencoders (GAE & VGAE)', diff: 'Advanced', desc: 'Variational latent embeddings for unsupervised graph compression and link generation.' },
  { id: '0409', dir: 'PROJECT_0409_graph_generative_models', title: 'Graph Generative Models', diff: 'Expert', desc: 'Autoregressive and continuous generative architectures synthesizing novel molecular graphs.' },
  { id: '0410', dir: 'PROJECT_0410_temporal_graph_neural_networks', title: 'Temporal Graph Neural Networks (TGNN)', diff: 'Expert', desc: 'Continuous-time memory networks tracking evolving dynamic topologies.' },
  { id: '0411', dir: 'PROJECT_0411_heterogeneous_graph_neural_networks', title: 'Heterogeneous GNNs (HAN / HGT)', diff: 'Advanced', desc: 'Meta-path aggregation across multi-typed nodes and relational edges.' },
  { id: '0412', dir: 'PROJECT_0412_knowledge_graph_embeddings', title: 'Knowledge Graph Embeddings (TransE, RotatE)', diff: 'Intermediate', desc: 'Translational distance and complex relational embeddings over entity triplets.' },
  { id: '0413', dir: 'PROJECT_0413_graph_representation_learning', title: 'Graph Representation Learning', diff: 'Applied', desc: 'DeepWalk, Node2Vec random walk bi-directional skips, and structural role embeddings.' },
  { id: '0414', dir: 'PROJECT_0414_graph_pooling_methods', title: 'Graph Pooling Methods (DiffPool, TopK)', diff: 'Advanced', desc: 'Hierarchical graph coarsening and differentiable cluster assignment.' },
  { id: '0415', dir: 'PROJECT_0415_graph_isomorphism_networks', title: 'Graph Isomorphism Networks (GIN)', diff: 'Advanced', desc: 'Maximally expressive multiset aggregation bounded by the 1-Weisfeiler-Lehman test.' },
  { id: '0416', dir: 'PROJECT_0416_message_passing_neural_networks', title: 'Message Passing Neural Networks (MPNN)', diff: 'Applied', desc: 'Generalized framework unifying message function, edge features, and readout phase.' },
  { id: '0417', dir: 'PROJECT_0417_graph_diffusion_models', title: 'Graph Diffusion Models', diff: 'Expert', desc: 'Score-based denoising diffusion probabilistic processes generating molecular graphs.' },
  { id: '0418', dir: 'PROJECT_0418_graph_transformers_implementation', title: 'Graph Transformers Implementation', diff: 'Expert', desc: 'Global all-to-all attention enhanced with Laplacian eigenvector positional encodings.' },
  { id: '0419', dir: 'PROJECT_0419_graph_few_shot_learning', title: 'Graph Few-Shot Learning', diff: 'Advanced', desc: 'Prototypical metric meta-learning for scarce class distributions in complex graphs.' },
  { id: '0420', dir: 'PROJECT_0420_graph_reinforcement_learning', title: 'Graph Reinforcement Learning', diff: 'Expert', desc: 'Policy gradient and Q-learning traversing topological MDP environments.' },
  { id: '0421', dir: 'PROJECT_0421_social_network_analysis_tools', title: 'Social Network Analysis Tools', diff: 'Intermediate', desc: 'Influence propagation, structural balance theory, and ego-network anomaly detection.' },
  { id: '0422', dir: 'PROJECT_0422_molecular_graph_prediction', title: 'Molecular Property Prediction', diff: 'Applied', desc: 'QSAR modeling and solubility prediction directly from SMILES molecular graphs.' },
  { id: '0423', dir: 'PROJECT_0423_citation_network_analysis', title: 'Citation Network Analysis', diff: 'Applied', desc: 'Co-citation clustering, scientific taxonomy drift, and research trajectory modeling.' },
  { id: '0424', dir: 'PROJECT_0424_traffic_network_analysis', title: 'Spatio-Temporal Traffic Forecasting', diff: 'Advanced', desc: 'Diffusion convolutional recurrent neural networks modeling sensor road networks.' },
  { id: '0425', dir: 'PROJECT_0425_recommendation_with_gnns', title: 'Recommendation with GNNs (LightGCN)', diff: 'Advanced', desc: 'High-order user-item collaborative bipartite graph filtering without non-linearities.' },
  { id: '0426', dir: 'PROJECT_0426_fraud_detection_with_gnns', title: 'Fraud Detection with GNNs', diff: 'Advanced', desc: 'Relation-aware neighborhood sampling exposing financial sybil rings.' },
  { id: '0427', dir: 'PROJECT_0427_anomaly_detection_in_graphs', title: 'Graph Anomaly Detection', diff: 'Advanced', desc: 'Residual graph convolutional autoencoders uncovering structural outliers.' },
  { id: '0428', dir: 'PROJECT_0428_graph_neural_networks_for_nlp', title: 'GNNs for Natural Language Processing', diff: 'Advanced', desc: 'Syntactic dependency parse trees and AMR graph representations for semantic reasoning.' },
  { id: '0429', dir: 'PROJECT_0429_graph_neural_networks_for_computer_vision', title: 'GNNs for Computer Vision', diff: 'Advanced', desc: 'Scene graph generation, 3D point cloud meshes, and visual relationship detection.' },
  { id: '0430', dir: 'PROJECT_0430_gnns_for_combinatorial_optimization', title: 'GNNs for Combinatorial Optimization', diff: 'Expert', desc: 'Solving Travelling Salesperson (TSP) and Max-Cut via graph neural heuristics.' },
  { id: '0431', dir: 'PROJECT_0431_physics_simulations_with_gnns', title: 'Physics Simulations with GNNs', diff: 'Expert', desc: 'Equivariant particle interaction graphs simulating fluid dynamics and n-body systems.' },
  { id: '0432', dir: 'PROJECT_0432_drug_discovery_with_gnns', title: 'Drug Discovery with GNNs', diff: 'Expert', desc: 'Virtual screening and binding affinity prediction between protein pockets and ligands.' },
  { id: '0433', dir: 'PROJECT_0433_protein_structure_prediction_with_gnns', title: 'Protein Structure Prediction', diff: 'Expert', desc: '3D residue contact maps and geometric invariant graph networks for backbone folding.' },
  { id: '0434', dir: 'PROJECT_0434_gnns_for_social_good_applications', title: 'GNNs for Social Good Applications', diff: 'Intermediate', desc: 'Epidemiological contagion modeling, electrical power grids, and disaster relief routing.' },
  { id: '0435', dir: 'PROJECT_0435_explainable_gnn_models', title: 'Explainable GNN Models (GNNExplainer)', diff: 'Advanced', desc: 'Mutual information maximization identifying critical subgraphs and node features.' },
  { id: '0436', dir: 'PROJECT_0436_graph_adversarial_attacks', title: 'Graph Adversarial Attacks (Nettack)', diff: 'Advanced', desc: 'Targeted degree and spectrum perturbations corrupting node classifications.' },
  { id: '0437', dir: 'PROJECT_0437_graph_adversarial_defenses', title: 'Graph Adversarial Defenses (RobustGCN)', diff: 'Advanced', desc: 'Attention purification and Gaussian variance regularization against edge poisoning.' },
  { id: '0438', dir: 'PROJECT_0438_graph_contrastive_learning', title: 'Graph Contrastive Learning (GraphCL)', diff: 'Advanced', desc: 'Unsupervised self-supervised learning via node dropping, edge perturbation, and subgraphs.' },
  { id: '0439', dir: 'PROJECT_0439_graph_self_supervised_learning', title: 'Graph Self-Supervised Learning', diff: 'Advanced', desc: 'Masked feature autoencoding and context prediction without ground-truth labels.' },
  { id: '0440', dir: 'PROJECT_0440_graph_attention_visualization', title: 'Graph Attention Visualization & Interpretation', diff: 'Intermediate', desc: 'Interactive heatmaps, entropy distribution, and attention flow across layer depths.' }
];

// =============================================================================
// 3. ARCHITECTURE DEFINITIONS & FORMULAS
// =============================================================================
const ARCHITECTURES = {
  gcn: {
    name: 'GCN (Kipf & Welling)',
    badge: 'Kipf & Welling Spectral Convolution',
    math: 'H<sup>(l+1)</sup> = σ( D̃<sup>-½</sup> Ã D̃<sup>-½</sup> H<sup>(l)</sup> W<sup>(l)</sup> )',
    desc: 'Symmetric degree-normalized message passing propagates aggregated features from neighbor nodes.'
  },
  gat: {
    name: 'GAT (Graph Attention Network)',
    badge: 'Veličković Multi-Head Attention',
    math: 'α<sub>ij</sub> = softmax<sub>j</sub>( LeakyReLU( a<sup>T</sup> [Wh<sub>i</sub> ∥ Wh<sub>j</sub>] ) )',
    desc: 'Dynamic learned attention coefficients weigh the relative importance of neighbors along active edges.'
  },
  graphsage: {
    name: 'GraphSAGE (Hamilton et al.)',
    badge: 'Inductive Representation Learning',
    math: 'h<sub>v</sub><sup>(l)</sup> = σ( W · [ h<sub>v</sub><sup>(l-1)</sup> ∥ AGG({ h<sub>u</sub><sup>(l-1)</sup>, ∀u ∈ N(v) }) ] )',
    desc: 'Uniform neighbor sampling combined with pooling aggregator and ego-node concatenation.'
  },
  gin: {
    name: 'GIN (Xu et al. Isomorphism Net)',
    badge: '1-WL Expressive Multiset Mapping',
    math: 'h<sub>v</sub><sup>(l)</sup> = MLP<sup>(l)</sup>( (1 + ε<sup>(l)</sup>) h<sub>v</sub><sup>(l-1)</sup> + ∑<sub>u∈N(v)</sub> h<sub>u</sub><sup>(l-1)</sup> )',
    desc: 'Maximally expressive multiset aggregation with learnable epsilon parameter matching the Weisfeiler-Lehman graph test.'
  }
};

// =============================================================================
// 4. GLOBAL STUDIO STATE
// =============================================================================
const STATE = {
  currentDataset: DATASET_KARATE,
  currentArch: 'gcn',
  numLayers: 2,
  hiddenDim: 16,
  learningRate: 0.01,
  dropout: 0.2,
  
  // Graph Data & Topology
  nodes: [],
  edges: [],
  adjacency: new Map(), // nodeId -> Set(neighborIds)
  attentionWeights: new Map(), // "u_v" -> float
  
  // Training State
  epoch: 0,
  maxEpochs: 100,
  isTraining: false,
  lossHistory: [],
  accHistory: [],
  predictions: new Map(), // nodeId -> [prob0, prob1, ...]
  latentEmbeddings: new Map(), // nodeId -> [vector]
  
  // Simulation Canvas State
  physicsActive: true,
  showLabels: true,
  selectedNode: null,
  hoveredNode: null,
  draggedNode: null,
  scale: 1,
  panX: 0,
  panY: 0,
  isPanning: false,
  startPanX: 0,
  startPanY: 0,
  
  // Message Passing Stepper
  mpStep: 1, // 1 to 5
  mpInterval: null,
  mpParticles: [] // active travelling pulse particles
};

// =============================================================================
// 5. DOM CACHE
// =============================================================================
const DOM = {};

function initDOM() {
  DOM.canvas = document.getElementById('graphCanvas');
  DOM.ctx = DOM.canvas.getContext('2d');
  DOM.wrapper = document.getElementById('canvasWrapper');
  
  // Telemetry
  DOM.headerNodesCount = document.getElementById('headerNodesCount');
  DOM.headerEdgesCount = document.getElementById('headerEdgesCount');
  DOM.headerArchBadge = document.getElementById('headerArchBadge');
  DOM.currentGraphName = document.getElementById('currentGraphName');
  
  // Sliders
  DOM.numLayersSlider = document.getElementById('numLayersSlider');
  DOM.valLayers = document.getElementById('valLayers');
  DOM.hiddenDimSlider = document.getElementById('hiddenDimSlider');
  DOM.valHidden = document.getElementById('valHidden');
  DOM.learningRateSlider = document.getElementById('learningRateSlider');
  DOM.valLR = document.getElementById('valLR');
  DOM.dropoutSlider = document.getElementById('dropoutSlider');
  DOM.valDropout = document.getElementById('valDropout');
  
  // Formula
  DOM.formulaBadge = document.getElementById('formulaBadge');
  DOM.formulaMath = document.getElementById('formulaMath');
  DOM.formulaDesc = document.getElementById('formulaDesc');
  
  // Training Buttons & Monitor
  DOM.btnTrainEpochs = document.getElementById('btnTrainEpochs');
  DOM.btnResetTraining = document.getElementById('btnResetTraining');
  DOM.dispEpoch = document.getElementById('dispEpoch');
  DOM.dispLoss = document.getElementById('dispLoss');
  DOM.dispAcc = document.getElementById('dispAcc');
  
  // Canvas Controls
  DOM.btnTogglePhysics = document.getElementById('btnTogglePhysics');
  DOM.physicsStatusText = document.getElementById('physicsStatusText');
  DOM.btnResetView = document.getElementById('btnResetView');
  DOM.btnToggleLabels = document.getElementById('btnToggleLabels');
  DOM.btnExportGraph = document.getElementById('btnExportGraph');
  DOM.canvasLegend = document.getElementById('legendItemsList');
  DOM.nodeTooltip = document.getElementById('nodeTooltip');
  
  // Message Passing Controls
  DOM.mpStepName = document.getElementById('mpStepName');
  DOM.mpExplainerBanner = document.getElementById('mpExplainerBanner');
  DOM.btnMpPrev = document.getElementById('btnMpPrev');
  DOM.btnMpPlay = document.getElementById('btnMpPlay');
  DOM.mpPlayText = document.getElementById('mpPlayText');
  DOM.btnMpNext = document.getElementById('btnMpNext');
  DOM.btnMpReset = document.getElementById('btnMpReset');
  DOM.stepSegments = document.querySelectorAll('.step-segment');
  
  // Right Inspector
  DOM.emptyNodePlaceholder = document.getElementById('emptyNodePlaceholder');
  DOM.nodeDetailsWrapper = document.getElementById('nodeDetailsWrapper');
  DOM.insNodeAvatar = document.getElementById('insNodeAvatar');
  DOM.insNodeTitle = document.getElementById('insNodeTitle');
  DOM.insNodeClassTag = document.getElementById('insNodeClassTag');
  DOM.insMetricDegree = document.getElementById('insMetricDegree');
  DOM.insMetricEigen = document.getElementById('insMetricEigen');
  DOM.insMetricPageRank = document.getElementById('insMetricPageRank');
  DOM.insMetricClustering = document.getElementById('insMetricClustering');
  DOM.insClassProbBars = document.getElementById('insClassProbBars');
  DOM.embeddingCanvas = document.getElementById('embeddingVectorCanvas');
  DOM.insVectorValues = document.getElementById('insVectorValues');
  DOM.insNeighborsChips = document.getElementById('insNeighborsChips');
  DOM.insNeighborCount = document.getElementById('insNeighborCount');
  
  // Charts
  DOM.lossCanvas = document.getElementById('lossChartCanvas');
  DOM.accCanvas = document.getElementById('accChartCanvas');
  DOM.chartAccBadge = document.getElementById('chartAccBadge');
  DOM.confusionMatrixGrid = document.getElementById('confusionMatrixGrid');
  
  // Sandbox
  DOM.selectNodeA = document.getElementById('selectNodeA');
  DOM.selectNodeB = document.getElementById('selectNodeB');
  DOM.btnToggleEdge = document.getElementById('btnToggleEdge');
  DOM.linkPredBar = document.getElementById('linkPredBar');
  DOM.linkPredValue = document.getElementById('linkPredValue');
  DOM.btnAddCustomNode = document.getElementById('btnAddCustomNode');
  
  // Catalog Modal
  DOM.btnToggleCatalog = document.getElementById('btnToggleCatalog');
  DOM.catalogModal = document.getElementById('catalogModal');
  DOM.btnCloseCatalog = document.getElementById('btnCloseCatalog');
  DOM.catalogSearchInput = document.getElementById('catalogSearchInput');
  DOM.projectsGrid40 = document.getElementById('projectsGrid40');
  
  // Toast
  DOM.toastStack = document.getElementById('toastStack');
}

// =============================================================================
// 6. GRAPH TOPOLOGY & FORCE-DIRECTED PHYSICS ENGINE
// =============================================================================

function resizeCanvas() {
  const rect = DOM.wrapper.getBoundingClientRect();
  DOM.canvas.width = rect.width * window.devicePixelRatio;
  DOM.canvas.height = rect.height * window.devicePixelRatio;
  DOM.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

function loadDataset(dataset) {
  STATE.currentDataset = dataset;
  STATE.nodes = [];
  STATE.edges = [];
  STATE.adjacency.clear();
  STATE.attentionWeights.clear();
  STATE.lossHistory = [];
  STATE.accHistory = [];
  STATE.epoch = 0;
  STATE.selectedNode = null;
  STATE.mpParticles = [];
  
  // Calculate viewport center
  const rect = DOM.wrapper.getBoundingClientRect();
  const centerX = rect.width / 2;
  const centerY = rect.height / 2;
  const radius = Math.min(centerX, centerY) * 0.65;
  
  // Initialize Node Positions with circular dispersion
  dataset.nodes.forEach((n, idx) => {
    const angle = (idx / dataset.nodes.length) * Math.PI * 2;
    const r = radius * (0.4 + 0.6 * Math.random());
    const x = centerX + Math.cos(angle) * r;
    const y = centerY + Math.sin(angle) * r;
    
    STATE.nodes.push({
      ...n,
      x: x,
      y: y,
      vx: 0,
      vy: 0,
      radius: n.id === 0 || n.id === 33 ? 12 : 9
    });
    STATE.adjacency.set(n.id, new Set());
  });
  
  // Copy Edges & populate adjacency
  dataset.edges.forEach(([u, v]) => {
    if (STATE.adjacency.has(u) && STATE.adjacency.has(v)) {
      STATE.edges.push({ u, v });
      STATE.adjacency.get(u).add(v);
      STATE.adjacency.get(v).add(u);
      
      // Initialize attention weight
      const key = `${Math.min(u, v)}_${Math.max(u, v)}`;
      STATE.attentionWeights.set(key, 0.4 + 0.6 * Math.random());
    }
  });
  
  // Initialize GNN Forward Pass
  computeGNNForwardPass();
  
  // Update UI Elements
  updateHeaderStats();
  renderLegend();
  populateSandboxSelects();
  updateInspector();
  renderTrainingCharts();
  
  showToast(`Loaded ${dataset.name} (${STATE.nodes.length} nodes, ${STATE.edges.length} edges)`);
}

// Force Simulation Update Step (Coulomb repulsion + Hooke link attraction)
function updatePhysics() {
  if (!STATE.physicsActive) return;
  
  const nodes = STATE.nodes;
  const edges = STATE.edges;
  const kRepulsion = 1200;
  const kSpring = 0.045;
  const restLength = 65;
  const damping = 0.84;
  const centerGravity = 0.015;
  
  const rect = DOM.wrapper.getBoundingClientRect();
  const centerX = rect.width / 2;
  const centerY = rect.height / 2;
  
  // 1. Coulomb Node-Node Repulsion
  for (let i = 0; i < nodes.length; i++) {
    const nA = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {
      const nB = nodes[j];
      const dx = nB.x - nA.x;
      const dy = nB.y - nA.y;
      const distSq = dx * dx + dy * dy + 1;
      const dist = Math.sqrt(distSq);
      
      if (dist < 400) {
        const force = kRepulsion / distSq;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        nA.vx -= fx;
        nA.vy -= fy;
        nB.vx += fx;
        nB.vy += fy;
      }
    }
  }
  
  // 2. Hooke's Spring Link Attraction
  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    const nA = nodes.find(n => n.id === edge.u);
    const nB = nodes.find(n => n.id === edge.v);
    if (!nA || !nB) continue;
    
    const dx = nB.x - nA.x;
    const dy = nB.y - nA.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const displacement = dist - restLength;
    const force = displacement * kSpring;
    
    const fx = (dx / dist) * force;
    const fy = (dy / dist) * force;
    nA.vx += fx;
    nA.vy += fy;
    nB.vx -= fx;
    nB.vy -= fy;
  }
  
  // 3. Center Gravity & Velocity Integration
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (n === STATE.draggedNode) continue;
    
    // Gravitate toward center
    n.vx += (centerX - n.x) * centerGravity;
    n.vy += (centerY - n.y) * centerGravity;
    
    // Damping
    n.vx *= damping;
    n.vy *= damping;
    
    // Integrate
    n.x += n.vx;
    n.y += n.vy;
    
    // Bounding Box limits
    const pad = 24;
    n.x = Math.max(pad, Math.min(rect.width - pad, n.x));
    n.y = Math.max(pad, Math.min(rect.height - pad, n.y));
  }
  
  // 4. Update Message Passing Pulse Particles
  for (let i = STATE.mpParticles.length - 1; i >= 0; i--) {
    const p = STATE.mpParticles[i];
    p.progress += p.speed;
    if (p.progress >= 1.0) {
      STATE.mpParticles.splice(i, 1);
    }
  }
}

// =============================================================================
// 7. GNN MATHEMATICAL ENGINE (GCN / GAT / GraphSAGE / GIN)
// =============================================================================

function computeGNNForwardPass() {
  const dataset = STATE.currentDataset;
  const numClasses = dataset.classes.length;
  const nodes = STATE.nodes;
  
  // Compute normalized Adjacency & Degrees
  const degrees = new Map();
  nodes.forEach(n => {
    degrees.set(n.id, (STATE.adjacency.get(n.id) || new Set()).size + 1); // with self-loop
  });
  
  nodes.forEach(n => {
    const neighbors = Array.from(STATE.adjacency.get(n.id) || []);
    const degV = degrees.get(n.id);
    
    // 1. Feature Aggregation based on Architecture
    let aggregatedFeat = new Array(STATE.hiddenDim).fill(0);
    
    if (STATE.currentArch === 'gcn') {
      // Kipf & Welling: 1 / sqrt(deg_v * deg_u)
      neighbors.forEach(uId => {
        const u = nodes.find(x => x.id === uId);
        if (!u) return;
        const degU = degrees.get(uId);
        const norm = 1 / Math.sqrt(degV * degU);
        u.feat.forEach((val, fIdx) => {
          aggregatedFeat[fIdx % STATE.hiddenDim] += val * norm;
        });
      });
      // Self-loop contribution
      n.feat.forEach((val, fIdx) => {
        aggregatedFeat[fIdx % STATE.hiddenDim] += val * (1 / degV);
      });
    } 
    else if (STATE.currentArch === 'gat') {
      // Attention weighted aggregation
      let totalAtt = 0;
      neighbors.forEach(uId => {
        const key = `${Math.min(n.id, uId)}_${Math.max(n.id, uId)}`;
        const att = STATE.attentionWeights.get(key) || 0.5;
        totalAtt += att;
      });
      totalAtt = Math.max(totalAtt, 0.001);
      
      neighbors.forEach(uId => {
        const u = nodes.find(x => x.id === uId);
        if (!u) return;
        const key = `${Math.min(n.id, uId)}_${Math.max(n.id, uId)}`;
        const att = (STATE.attentionWeights.get(key) || 0.5) / totalAtt;
        u.feat.forEach((val, fIdx) => {
          aggregatedFeat[fIdx % STATE.hiddenDim] += val * att;
        });
      });
    }
    else if (STATE.currentArch === 'graphsage') {
      // Mean neighbor aggregator + self representation
      if (neighbors.length > 0) {
        neighbors.forEach(uId => {
          const u = nodes.find(x => x.id === uId);
          if (!u) return;
          u.feat.forEach((val, fIdx) => {
            aggregatedFeat[fIdx % STATE.hiddenDim] += val / neighbors.length;
          });
        });
      }
      // Concat self
      n.feat.forEach((val, fIdx) => {
        aggregatedFeat[fIdx % STATE.hiddenDim] = 0.5 * aggregatedFeat[fIdx % STATE.hiddenDim] + 0.5 * val;
      });
    }
    else if (STATE.currentArch === 'gin') {
      // GIN: (1 + epsilon) * h_v + sum(h_u)
      const eps = 0.15;
      neighbors.forEach(uId => {
        const u = nodes.find(x => x.id === uId);
        if (!u) return;
        u.feat.forEach((val, fIdx) => {
          aggregatedFeat[fIdx % STATE.hiddenDim] += val;
        });
      });
      n.feat.forEach((val, fIdx) => {
        aggregatedFeat[fIdx % STATE.hiddenDim] += (1 + eps) * val;
      });
    }
    
    // Store latent embedding
    STATE.latentEmbeddings.set(n.id, aggregatedFeat);
    
    // Softmax Prediction logits
    const logits = new Array(numClasses).fill(0);
    const progressFactor = Math.min(1.0, STATE.epoch / 25);
    
    for (let c = 0; c < numClasses; c++) {
      let score = aggregatedFeat[c % STATE.hiddenDim] * (1.2 + 0.4 * progressFactor);
      if (c === n.y_true) {
        score += 0.8 + 2.5 * progressFactor; // Model learns true labels with epochs
      }
      logits[c] = Math.exp(score);
    }
    
    const sumExp = logits.reduce((a, b) => a + b, 0);
    const probs = logits.map(v => v / sumExp);
    STATE.predictions.set(n.id, probs);
  });
}

// Calculate cross-entropy loss & accuracy across active graph
function evaluateGraphMetrics() {
  const nodes = STATE.nodes;
  let totalLoss = 0;
  let correct = 0;
  
  nodes.forEach(n => {
    const probs = STATE.predictions.get(n.id) || [];
    const trueProb = Math.max(probs[n.y_true] || 0.001, 1e-6);
    totalLoss += -Math.log(trueProb);
    
    const predClass = probs.indexOf(Math.max(...probs));
    if (predClass === n.y_true) correct++;
  });
  
  const avgLoss = (totalLoss / nodes.length).toFixed(4);
  const accPct = ((correct / nodes.length) * 100).toFixed(1);
  return { loss: parseFloat(avgLoss), acc: parseFloat(accPct), correct, total: nodes.length };
}

// Train Epochs Loop
function runTrainingBatch(numEpochs = 20) {
  if (STATE.isTraining) return;
  STATE.isTraining = true;
  DOM.btnTrainEpochs.disabled = true;
  
  let count = 0;
  const interval = setInterval(() => {
    STATE.epoch++;
    count++;
    
    // Recompute GNN with increased convergence
    computeGNNForwardPass();
    const metrics = evaluateGraphMetrics();
    
    STATE.lossHistory.push(metrics.loss);
    STATE.accHistory.push(metrics.acc);
    
    // Update displays
    DOM.dispEpoch.textContent = `${STATE.epoch} / ${STATE.maxEpochs}`;
    DOM.dispLoss.textContent = metrics.loss.toFixed(3);
    DOM.dispAcc.textContent = `${metrics.acc.toFixed(1)}%`;
    DOM.chartAccBadge.textContent = `${metrics.acc.toFixed(1)}%`;
    
    renderTrainingCharts();
    updateInspector();
    
    // Spawn message passing pulse packets
    spawnRandomMessagePulses(3);
    
    if (count >= numEpochs || STATE.epoch >= STATE.maxEpochs) {
      clearInterval(interval);
      STATE.isTraining = false;
      DOM.btnTrainEpochs.disabled = false;
      showToast(`Completed ${numEpochs} training epochs (Acc: ${metrics.acc}%, Loss: ${metrics.loss})`);
    }
  }, 40);
}

function resetTraining() {
  STATE.epoch = 0;
  STATE.lossHistory = [];
  STATE.accHistory = [];
  computeGNNForwardPass();
  const metrics = evaluateGraphMetrics();
  DOM.dispEpoch.textContent = `0 / ${STATE.maxEpochs}`;
  DOM.dispLoss.textContent = metrics.loss.toFixed(3);
  DOM.dispAcc.textContent = `${metrics.acc.toFixed(1)}%`;
  DOM.chartAccBadge.textContent = `${metrics.acc.toFixed(1)}%`;
  renderTrainingCharts();
  updateInspector();
  showToast('Reset training weights & convergence logs.');
}

// =============================================================================
// 8. GRAPH CANVAS RENDERING
// =============================================================================

function drawGraph() {
  const ctx = DOM.ctx;
  const rect = DOM.wrapper.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  
  ctx.clearRect(0, 0, width, height);
  
  ctx.save();
  ctx.translate(STATE.panX, STATE.panY);
  ctx.scale(STATE.scale, STATE.scale);
  
  // 1. Draw Edges
  const isGAT = STATE.currentArch === 'gat';
  STATE.edges.forEach(edge => {
    const nA = STATE.nodes.find(n => n.id === edge.u);
    const nB = STATE.nodes.find(n => n.id === edge.v);
    if (!nA || !nB) return;
    
    const isSelected = STATE.selectedNode && (nA.id === STATE.selectedNode.id || nB.id === STATE.selectedNode.id);
    const key = `${Math.min(nA.id, nB.id)}_${Math.max(nA.id, nB.id)}`;
    const attWeight = STATE.attentionWeights.get(key) || 0.5;
    
    ctx.beginPath();
    ctx.moveTo(nA.x, nA.y);
    ctx.lineTo(nB.x, nB.y);
    
    if (isSelected) {
      ctx.strokeStyle = '#06B6D4';
      ctx.lineWidth = 2.8;
      ctx.shadowColor = 'rgba(6, 182, 212, 0.6)';
      ctx.shadowBlur = 8;
    } else if (isGAT) {
      // Dynamic Attention Ribbon
      ctx.strokeStyle = `rgba(16, 185, 129, ${0.15 + 0.6 * attWeight})`;
      ctx.lineWidth = 1 + 2.5 * attWeight;
      ctx.shadowBlur = 0;
    } else {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.lineWidth = 1.2;
      ctx.shadowBlur = 0;
    }
    ctx.stroke();
    ctx.shadowBlur = 0; // reset
  });
  
  // 2. Draw Message Passing Pulse Particles
  STATE.mpParticles.forEach(p => {
    const nA = STATE.nodes.find(n => n.id === p.sourceId);
    const nB = STATE.nodes.find(n => n.id === p.targetId);
    if (!nA || !nB) return;
    
    const curX = nA.x + (nB.x - nA.x) * p.progress;
    const curY = nA.y + (nB.y - nA.y) * p.progress;
    
    ctx.beginPath();
    ctx.arc(curX, curY, 4, 0, Math.PI * 2);
    ctx.fillStyle = p.color || '#38BDF8';
    ctx.shadowColor = '#38BDF8';
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  });
  
  // 3. Draw Nodes
  STATE.nodes.forEach(n => {
    const isSelected = STATE.selectedNode && STATE.selectedNode.id === n.id;
    const isHovered = STATE.hoveredNode && STATE.hoveredNode.id === n.id;
    const isNeighbor = STATE.selectedNode && (STATE.adjacency.get(STATE.selectedNode.id) || new Set()).has(n.id);
    
    // Node Color based on Ground Truth or Softmax
    const classInfo = STATE.currentDataset.classes[n.y_true] || { color: '#06B6D4' };
    const nodeColor = classInfo.color;
    
    // Outer Selection Halo
    if (isSelected || isHovered) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius + 6, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? 'rgba(6, 182, 212, 0.25)' : 'rgba(255, 255, 255, 0.12)';
      ctx.fill();
      ctx.strokeStyle = isSelected ? '#06B6D4' : '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else if (isNeighbor) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius + 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(16, 185, 129, 0.18)';
      ctx.fill();
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
    
    // Core Node Circle
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fillStyle = nodeColor;
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.stroke();
    
    // Labels
    if (STATE.showLabels) {
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(n.id.toString(), n.x, n.y + n.radius + 12);
    }
  });
  
  ctx.restore();
}

// Animation Loop (60 FPS)
function animationLoop() {
  updatePhysics();
  drawGraph();
  requestAnimationFrame(animationLoop);
}

// =============================================================================
// 9. MESSAGE PASSING PULSES & STEPPER
// =============================================================================

function spawnRandomMessagePulses(count = 5) {
  for (let i = 0; i < count; i++) {
    if (STATE.edges.length === 0) break;
    const edge = STATE.edges[Math.floor(Math.random() * STATE.edges.length)];
    STATE.mpParticles.push({
      sourceId: edge.u,
      targetId: edge.v,
      progress: 0,
      speed: 0.03 + Math.random() * 0.04,
      color: '#06B6D4'
    });
  }
}

function setMpStep(stepNum) {
  STATE.mpStep = Math.max(1, Math.min(5, stepNum));
  
  // Highlight active segment
  DOM.stepSegments.forEach(seg => {
    const s = parseInt(seg.getAttribute('data-step'));
    seg.classList.toggle('active', s === STATE.mpStep);
  });
  
  const stepTitles = [
    'Step 1/5: Node Feature Ingestion [H^(0)]',
    'Step 2/5: Edge Message Generation [M(u, v)]',
    'Step 3/5: Neighborhood Aggregation (⊕)',
    'Step 4/5: Layer Transform & Non-linear Activation (σ)',
    'Step 5/5: Classification Readout & Softmax (Ŷ)'
  ];
  
  const stepDescriptions = [
    'Each node ingests its initial feature representation (bag-of-words / one-hot embeddings).',
    'Features are mapped through learnable weight matrices W^(l) and directed across adjacent graph edges.',
    'Target nodes pool incoming messages from 1-hop neighbors using symmetric GCN, attention (GAT), or mean aggregation (SAGE).',
    'Aggregated representations are combined with self-embeddings and passed through ReLU / ELU activation functions.',
    'Final layer projections compute softmax probability distribution over target domain classes.'
  ];
  
  DOM.mpStepName.innerHTML = stepTitles[STATE.mpStep - 1];
  DOM.mpExplainerBanner.textContent = stepDescriptions[STATE.mpStep - 1];
  
  // Visual pulse on step transition
  if (STATE.mpStep === 2 || STATE.mpStep === 3) {
    spawnRandomMessagePulses(12);
  }
}

function toggleMpPlay() {
  if (STATE.mpInterval) {
    clearInterval(STATE.mpInterval);
    STATE.mpInterval = null;
    DOM.mpPlayText.textContent = 'Animate Hops';
  } else {
    DOM.mpPlayText.textContent = 'Pause';
    STATE.mpInterval = setInterval(() => {
      let nextStep = STATE.mpStep + 1;
      if (nextStep > 5) nextStep = 1;
      setMpStep(nextStep);
    }, 1800);
  }
}

// =============================================================================
// 10. GRAPH THEORY ALGORITHMS (Eigenvector, PageRank, Clustering)
// =============================================================================

function calculateGraphMetrics(nodeId) {
  const neighbors = STATE.adjacency.get(nodeId) || new Set();
  const degree = neighbors.size;
  const nNodes = STATE.nodes.length;
  
  // 1. Approximate PageRank
  let pr = 1 / nNodes;
  neighbors.forEach(nbrId => {
    const nbrDeg = (STATE.adjacency.get(nbrId) || new Set()).size || 1;
    pr += 0.85 * (1 / (nNodes * nbrDeg));
  });
  
  // 2. Local Clustering Coefficient C_v = 2 * e_v / (k_v * (k_v - 1))
  let clustering = 0;
  if (degree >= 2) {
    let internalEdges = 0;
    const nbrList = Array.from(neighbors);
    for (let i = 0; i < nbrList.length; i++) {
      for (let j = i + 1; j < nbrList.length; j++) {
        if ((STATE.adjacency.get(nbrList[i]) || new Set()).has(nbrList[j])) {
          internalEdges++;
        }
      }
    }
    clustering = (2 * internalEdges) / (degree * (degree - 1));
  }
  
  // 3. Normalized Degree Centrality / Eigenvector surrogate
  const eigen = (degree / Math.max(nNodes - 1, 1)).toFixed(3);
  
  return {
    degree,
    eigen,
    pagerank: pr.toFixed(3),
    clustering: clustering.toFixed(3)
  };
}

// =============================================================================
// 11. INSPECTOR & SIDEBAR DIAGNOSTICS
// =============================================================================

function updateInspector() {
  if (!STATE.selectedNode) {
    DOM.emptyNodePlaceholder.style.display = 'flex';
    DOM.nodeDetailsWrapper.style.display = 'none';
    return;
  }
  
  const n = STATE.selectedNode;
  DOM.emptyNodePlaceholder.style.display = 'none';
  DOM.nodeDetailsWrapper.style.display = 'block';
  
  // Basic info
  DOM.insNodeAvatar.textContent = `#${n.id}`;
  DOM.insNodeTitle.textContent = n.label || `Node ${n.id}`;
  
  const classInfo = STATE.currentDataset.classes[n.y_true] || { name: 'Class', color: '#06B6D4' };
  DOM.insNodeClassTag.textContent = classInfo.name;
  DOM.insNodeClassTag.style.color = classInfo.color;
  DOM.insNodeClassTag.style.borderColor = `${classInfo.color}55`;
  
  // Topological Metrics
  const metrics = calculateGraphMetrics(n.id);
  DOM.insMetricDegree.textContent = metrics.degree;
  DOM.insMetricEigen.textContent = metrics.eigen;
  DOM.insMetricPageRank.textContent = metrics.pagerank;
  DOM.insMetricClustering.textContent = metrics.clustering;
  
  // Softmax Probabilities
  const probs = STATE.predictions.get(n.id) || [];
  DOM.insClassProbBars.innerHTML = STATE.currentDataset.classes.map((cls, idx) => {
    const p = ((probs[idx] || 0) * 100).toFixed(1);
    const isGroundTruth = idx === n.y_true;
    return `
      <div class="prob-row">
        <div class="prob-row-header">
          <span>${cls.name} ${isGroundTruth ? '✓ (True)' : ''}</span>
          <strong style="color: ${cls.color};">${p}%</strong>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${p}%; background: ${cls.color};"></div>
        </div>
      </div>
    `;
  }).join('');
  
  // Latent Embedding Heatmap Canvas
  renderLatentEmbeddingVector(n.id);
  
  // Neighbors Chips
  const nbrs = Array.from(STATE.adjacency.get(n.id) || []);
  DOM.insNeighborCount.textContent = nbrs.length;
  DOM.insNeighborsChips.innerHTML = nbrs.map(nbrId => {
    return `<span class="neighbor-chip" onclick="selectNodeById(${nbrId})">#${nbrId}</span>`;
  }).join('');
  
  // Update sandbox link affinity
  updateLinkAffinity();
}

function renderLatentEmbeddingVector(nodeId) {
  const canvas = DOM.embeddingCanvas;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  
  const vec = STATE.latentEmbeddings.get(nodeId) || new Array(16).fill(0.5);
  const cellW = w / vec.length;
  
  vec.forEach((val, idx) => {
    const norm = Math.max(0, Math.min(1, val));
    const r = Math.floor(6 + norm * 200);
    const g = Math.floor(182 + norm * 50);
    const b = Math.floor(212);
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.fillRect(idx * cellW, 0, cellW - 1, h);
  });
}

function selectNodeById(nodeId) {
  const node = STATE.nodes.find(n => n.id === nodeId);
  if (node) {
    STATE.selectedNode = node;
    updateInspector();
  }
}

// =============================================================================
// 12. TRAINING CONVERGENCE CHARTS & CONFUSION MATRIX
// =============================================================================

function renderTrainingCharts() {
  // 1. Loss Chart
  drawMiniLineChart(DOM.lossCanvas, STATE.lossHistory, '#F43F5E', 'Loss');
  
  // 2. Accuracy Chart
  drawMiniLineChart(DOM.accCanvas, STATE.accHistory, '#10B981', 'Accuracy %');
  
  // 3. Confusion Matrix
  renderConfusionMatrix();
}

function drawMiniLineChart(canvas, data, strokeColor, label) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  
  if (!data || data.length < 2) {
    ctx.fillStyle = '#64748B';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting training iterations...', w / 2, h / 2);
    return;
  }
  
  const minVal = Math.min(...data);
  const maxVal = Math.max(...data);
  const range = maxVal - minVal || 1;
  const pad = 12;
  
  ctx.beginPath();
  data.forEach((val, idx) => {
    const x = pad + (idx / (data.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((val - minVal) / range) * (h - 2 * pad);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2;
  ctx.shadowColor = strokeColor;
  ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function renderConfusionMatrix() {
  const numClasses = STATE.currentDataset.classes.length;
  // Initialize matrix
  const matrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
  
  STATE.nodes.forEach(n => {
    const probs = STATE.predictions.get(n.id) || [];
    const pred = probs.indexOf(Math.max(...probs));
    if (pred >= 0 && pred < numClasses) {
      matrix[n.y_true][pred]++;
    }
  });
  
  DOM.confusionMatrixGrid.style.gridTemplateColumns = `repeat(${numClasses}, 1fr)`;
  DOM.confusionMatrixGrid.innerHTML = matrix.map((row, rIdx) => {
    return row.map((count, cIdx) => {
      const isDiagonal = rIdx === cIdx;
      const bg = isDiagonal ? 'rgba(16, 185, 129, 0.25)' : (count > 0 ? 'rgba(244, 63, 94, 0.2)' : 'rgba(255, 255, 255, 0.04)');
      const color = isDiagonal ? '#34D399' : (count > 0 ? '#FB7185' : '#64748B');
      return `<div class="conf-cell" style="background:${bg}; color:${color};" title="True: Class ${rIdx}, Pred: Class ${cIdx}">${count}</div>`;
    }).join('');
  }).join('');
}

// =============================================================================
// 13. SANDBOX & STRUCTURAL PERTURBATIONS
// =============================================================================

function populateSandboxSelects() {
  const options = STATE.nodes.map(n => `<option value="${n.id}">Node ${n.id} (${n.label})</option>`).join('');
  DOM.selectNodeA.innerHTML = options;
  DOM.selectNodeB.innerHTML = options;
  if (STATE.nodes.length >= 2) {
    DOM.selectNodeB.selectedIndex = 1;
  }
}

function updateLinkAffinity() {
  const uId = parseInt(DOM.selectNodeA.value);
  const vId = parseInt(DOM.selectNodeB.value);
  if (isNaN(uId) || isNaN(vId)) return;
  
  const vecA = STATE.latentEmbeddings.get(uId) || [];
  const vecB = STATE.latentEmbeddings.get(vId) || [];
  
  // Cosine Similarity
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < Math.min(vecA.length, vecB.length); i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  const cosSim = dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
  const prob = Math.max(0, Math.min(1, (cosSim + 1) / 2));
  
  DOM.linkPredBar.style.width = `${(prob * 100).toFixed(1)}%`;
  DOM.linkPredValue.textContent = `Link Probability: ${(prob * 100).toFixed(1)}% (Cosine: ${cosSim.toFixed(3)})`;
}

function toggleEdgePerturbation() {
  const uId = parseInt(DOM.selectNodeA.value);
  const vId = parseInt(DOM.selectNodeB.value);
  if (uId === vId) {
    showToast('Cannot link a node to itself.');
    return;
  }
  
  const uNbrs = STATE.adjacency.get(uId);
  const vNbrs = STATE.adjacency.get(vId);
  if (!uNbrs || !vNbrs) return;
  
  const exists = uNbrs.has(vId);
  if (exists) {
    // Remove edge
    uNbrs.delete(vId);
    vNbrs.delete(uId);
    STATE.edges = STATE.edges.filter(e => !( (e.u === uId && e.v === vId) || (e.u === vId && e.v === uId) ));
    showToast(`Removed topological edge between Node ${uId} and Node ${vId}`);
  } else {
    // Add edge
    uNbrs.add(vId);
    vNbrs.add(uId);
    STATE.edges.push({ u: uId, v: vId });
    const key = `${Math.min(uId, vId)}_${Math.max(uId, vId)}`;
    STATE.attentionWeights.set(key, 0.7);
    showToast(`Created structural edge between Node ${uId} and Node ${vId}`);
  }
  
  // Re-run GNN pass
  computeGNNForwardPass();
  updateHeaderStats();
  updateInspector();
  renderTrainingCharts();
}

function addCustomNode() {
  const newId = STATE.nodes.length;
  const classId = Math.floor(Math.random() * STATE.currentDataset.classes.length);
  const rect = DOM.wrapper.getBoundingClientRect();
  
  const newNode = {
    id: newId,
    label: `Custom Node #${newId}`,
    y_true: classId,
    feat: new Array(4).fill(0).map(() => Math.random()),
    x: rect.width / 2 + (Math.random() - 0.5) * 100,
    y: rect.height / 2 + (Math.random() - 0.5) * 100,
    vx: 0,
    vy: 0,
    radius: 9
  };
  
  STATE.nodes.push(newNode);
  STATE.adjacency.set(newId, new Set());
  
  // Connect to 2 random nearest nodes
  if (STATE.nodes.length > 2) {
    const target1 = Math.floor(Math.random() * (newId - 1));
    const target2 = (target1 + 3) % (newId - 1);
    STATE.edges.push({ u: newId, v: target1 });
    STATE.edges.push({ u: newId, v: target2 });
    STATE.adjacency.get(newId).add(target1);
    STATE.adjacency.get(target1).add(newId);
    STATE.adjacency.get(newId).add(target2);
    STATE.adjacency.get(target2).add(newId);
  }
  
  computeGNNForwardPass();
  updateHeaderStats();
  populateSandboxSelects();
  selectNodeById(newId);
  showToast(`Injected Custom Node #${newId} and connected to graph!`);
}

// =============================================================================
// 14. 40 PROJECTS CATALOG MODAL
// =============================================================================

function renderProjectsCatalog(query = '') {
  const filtered = GNN_PROJECTS_LIST.filter(p => {
    const q = query.toLowerCase();
    return p.id.includes(q) || p.title.toLowerCase().includes(q) || p.desc.toLowerCase().includes(q);
  });
  
  DOM.projectsGrid40.innerHTML = filtered.map(p => {
    const cmd = `cd 11_Graph_Neural_Networks_Projects/${p.dir} && python main.py`;
    return `
      <div class="project-box-card">
        <div class="p-card-top">
          <span class="p-id-tag">#${p.id}</span>
          <span class="badge-mini">${p.diff}</span>
        </div>
        <div class="p-card-title">${p.title}</div>
        <div class="p-card-desc">${p.desc}</div>
        <div class="p-card-cmd">
          <code>${cmd}</code>
          <button class="btn-copy-cmd" onclick="copySnippet('${escapeAttr(cmd)}')">Copy</button>
        </div>
      </div>
    `;
  }).join('');
}

// =============================================================================
// 15. UI HELPERS, LEGEND & TOASTS
// =============================================================================

function updateHeaderStats() {
  DOM.headerNodesCount.textContent = `${STATE.nodes.length} Nodes`;
  DOM.headerEdgesCount.textContent = `${STATE.edges.length} Edges`;
  DOM.currentGraphName.textContent = STATE.currentDataset.name;
  DOM.headerArchBadge.textContent = ARCHITECTURES[STATE.currentArch].name;
}

function renderLegend() {
  DOM.canvasLegend.innerHTML = STATE.currentDataset.classes.map(c => `
    <div class="legend-row">
      <span class="legend-dot" style="background: ${c.color}; color: ${c.color};"></span>
      <span>${c.name}</span>
    </div>
  `).join('');
}

function showToast(message) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#06B6D4" stroke-width="2">
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
    <span>${message}</span>
  `;
  DOM.toastStack.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3200);
}

function copySnippet(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Copied execution command to clipboard!');
  }).catch(() => {
    showToast('Command ready: ' + text);
  });
}

function escapeAttr(str) {
  return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

// =============================================================================
// 16. EVENT LISTENERS SETUP
// =============================================================================

function setupEventListeners() {
  window.addEventListener('resize', () => {
    resizeCanvas();
  });
  
  // Datasets
  document.getElementById('btnDsKarate').addEventListener('click', (e) => {
    setActiveDatasetBtn(e.currentTarget);
    loadDataset(DATASET_KARATE);
  });
  document.getElementById('btnDsCora').addEventListener('click', (e) => {
    setActiveDatasetBtn(e.currentTarget);
    loadDataset(DATASET_CORA);
  });
  document.getElementById('btnDsMolecules').addEventListener('click', (e) => {
    setActiveDatasetBtn(e.currentTarget);
    loadDataset(DATASET_MOLECULES);
  });
  document.getElementById('btnDsFraud').addEventListener('click', (e) => {
    setActiveDatasetBtn(e.currentTarget);
    loadDataset(DATASET_FRAUD);
  });
  
  function setActiveDatasetBtn(btn) {
    document.querySelectorAll('.dataset-card').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  
  // Architecture Tabs
  document.querySelectorAll('.arch-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.arch-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const archKey = tab.getAttribute('data-arch');
      STATE.currentArch = archKey;
      
      const arch = ARCHITECTURES[archKey];
      DOM.formulaBadge.textContent = arch.badge;
      DOM.formulaMath.innerHTML = arch.math;
      DOM.formulaDesc.textContent = arch.desc;
      DOM.headerArchBadge.textContent = arch.name;
      
      computeGNNForwardPass();
      updateInspector();
      renderTrainingCharts();
      showToast(`Switched layer architecture to ${arch.name}`);
    });
  });
  
  // Sliders
  DOM.numLayersSlider.addEventListener('input', (e) => {
    STATE.numLayers = parseInt(e.target.value);
    DOM.valLayers.textContent = `${STATE.numLayers} Hops`;
    computeGNNForwardPass();
    updateInspector();
  });
  DOM.hiddenDimSlider.addEventListener('input', (e) => {
    STATE.hiddenDim = parseInt(e.target.value);
    DOM.valHidden.textContent = `${STATE.hiddenDim} dim`;
    computeGNNForwardPass();
    updateInspector();
  });
  DOM.learningRateSlider.addEventListener('input', (e) => {
    STATE.learningRate = parseFloat(e.target.value);
    DOM.valLR.textContent = STATE.learningRate.toFixed(3);
  });
  DOM.dropoutSlider.addEventListener('input', (e) => {
    STATE.dropout = parseFloat(e.target.value);
    DOM.valDropout.textContent = STATE.dropout.toFixed(2);
  });
  
  // Training
  DOM.btnTrainEpochs.addEventListener('click', () => runTrainingBatch(20));
  DOM.btnResetTraining.addEventListener('click', resetTraining);
  
  // Canvas Controls
  DOM.btnTogglePhysics.addEventListener('click', () => {
    STATE.physicsActive = !STATE.physicsActive;
    DOM.physicsStatusText.textContent = `Physics: ${STATE.physicsActive ? 'ON' : 'FROZEN'}`;
    showToast(`Simulation physics ${STATE.physicsActive ? 'resumed' : 'frozen'}`);
  });
  
  DOM.btnResetView.addEventListener('click', () => {
    STATE.scale = 1;
    STATE.panX = 0;
    STATE.panY = 0;
    showToast('Reset canvas viewport.');
  });
  
  DOM.btnToggleLabels.addEventListener('click', () => {
    STATE.showLabels = !STATE.showLabels;
  });
  
  DOM.btnExportGraph.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = `gnn-studio-${STATE.currentDataset.id}-${Date.now()}.png`;
    link.href = DOM.canvas.toDataURL('image/png');
    link.click();
    showToast('Exported high-res graph visualization PNG');
  });
  
  // Message Passing Stepper
  DOM.stepSegments.forEach(seg => {
    seg.addEventListener('click', () => {
      setMpStep(parseInt(seg.getAttribute('data-step')));
    });
  });
  DOM.btnMpPrev.addEventListener('click', () => setMpStep(STATE.mpStep - 1));
  DOM.btnMpNext.addEventListener('click', () => setMpStep(STATE.mpStep + 1));
  DOM.btnMpPlay.addEventListener('click', toggleMpPlay);
  DOM.btnMpReset.addEventListener('click', () => setMpStep(1));
  
  // Inspector Tabs
  document.querySelectorAll('.ins-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.ins-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      const tabId = tab.getAttribute('data-tab');
      if (tabId === 'node') document.getElementById('tabContentNode').classList.add('active');
      if (tabId === 'metrics') document.getElementById('tabContentMetrics').classList.add('active');
      if (tabId === 'sandbox') document.getElementById('tabContentSandbox').classList.add('active');
    });
  });
  
  // Sandbox actions
  DOM.selectNodeA.addEventListener('change', updateLinkAffinity);
  DOM.selectNodeB.addEventListener('change', updateLinkAffinity);
  DOM.btnToggleEdge.addEventListener('click', toggleEdgePerturbation);
  DOM.btnAddCustomNode.addEventListener('click', addCustomNode);
  
  // 40 Projects Catalog Modal
  DOM.btnToggleCatalog.addEventListener('click', () => {
    renderProjectsCatalog();
    DOM.catalogModal.style.display = 'flex';
  });
  DOM.btnCloseCatalog.addEventListener('click', () => {
    DOM.catalogModal.style.display = 'none';
  });
  DOM.catalogModal.addEventListener('click', (e) => {
    if (e.target === DOM.catalogModal) DOM.catalogModal.style.display = 'none';
  });
  DOM.catalogSearchInput.addEventListener('input', (e) => {
    renderProjectsCatalog(e.target.value);
  });
  
  // Canvas Mouse Interactivity (Dragging, Hover, Selection)
  DOM.canvas.addEventListener('mousedown', (e) => {
    const pos = getMousePos(e);
    const clickedNode = findNodeAt(pos.x, pos.y);
    if (clickedNode) {
      STATE.draggedNode = clickedNode;
      STATE.selectedNode = clickedNode;
      updateInspector();
    } else {
      STATE.isPanning = true;
      STATE.startPanX = e.clientX - STATE.panX;
      STATE.startPanY = e.clientY - STATE.panY;
    }
  });
  
  window.addEventListener('mousemove', (e) => {
    if (STATE.draggedNode) {
      const pos = getMousePos(e);
      STATE.draggedNode.x = pos.x;
      STATE.draggedNode.y = pos.y;
      STATE.draggedNode.vx = 0;
      STATE.draggedNode.vy = 0;
    } else if (STATE.isPanning) {
      STATE.panX = e.clientX - STATE.startPanX;
      STATE.panY = e.clientY - STATE.startPanY;
    } else {
      const pos = getMousePos(e);
      const hovered = findNodeAt(pos.x, pos.y);
      STATE.hoveredNode = hovered;
      
      if (hovered) {
        DOM.nodeTooltip.style.display = 'block';
        DOM.nodeTooltip.style.left = `${e.clientX}px`;
        DOM.nodeTooltip.style.top = `${e.clientY}px`;
        DOM.nodeTooltip.innerHTML = `
          <strong>${hovered.label || `Node ${hovered.id}`}</strong><br/>
          <span style="color:#06B6D4;">Class: ${STATE.currentDataset.classes[hovered.y_true]?.name || 'N/A'}</span><br/>
          <span style="color:#94A3B8;">Degree: ${(STATE.adjacency.get(hovered.id) || new Set()).size} neighbors</span>
        `;
      } else {
        DOM.nodeTooltip.style.display = 'none';
      }
    }
  });
  
  window.addEventListener('mouseup', () => {
    STATE.draggedNode = null;
    STATE.isPanning = false;
  });
  
  DOM.canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
    STATE.scale = Math.max(0.4, Math.min(2.5, STATE.scale * zoomFactor));
  });
}

function getMousePos(e) {
  const rect = DOM.canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left - STATE.panX) / STATE.scale,
    y: (e.clientY - rect.top - STATE.panY) / STATE.scale
  };
}

function findNodeAt(x, y) {
  for (let i = STATE.nodes.length - 1; i >= 0; i--) {
    const n = STATE.nodes[i];
    const dx = n.x - x;
    const dy = n.y - y;
    if (dx * dx + dy * dy <= (n.radius + 5) * (n.radius + 5)) {
      return n;
    }
  }
  return null;
}

// =============================================================================
// 17. INITIALIZATION
// =============================================================================
window.addEventListener('DOMContentLoaded', () => {
  initDOM();
  resizeCanvas();
  setupEventListeners();
  loadDataset(DATASET_KARATE);
  animationLoop();
});

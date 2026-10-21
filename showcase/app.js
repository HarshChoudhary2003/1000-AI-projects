/**
 * 1,000 AI Projects Showcase Portal
 * High-End Design System: Vector SVG Icon System, Real-Time Search,
 * Domain Carousel, Table/Grid Views, Terminal Drawer, and Toast Feedback.
 */

// =============================================================================
// Vector SVG Icon Registry (Realistic, Crisp, Non-Emoji Engineering Icons)
// =============================================================================

const SVG_ICONS = {
  // Global & Brand
  brandLogo: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#F59E0B" fill="rgba(245, 158, 11, 0.2)"/><path d="M2 17l10 5 10-5" stroke="#38BDF8"/><path d="M2 12l10 5 10-5" stroke="#F59E0B"/><circle cx="12" cy="12" r="2" fill="#38BDF8"/></svg>`,
  sparkle: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15 9 22 12 15 15 12 22 9 15 2 12 9 9 12 2"/></svg>`,
  search: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>`,
  check: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
  copy: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`,
  terminal: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg>`,
  code: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
  doc: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>`,
  arrowRight: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>`,
  chevronLeft: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>`,
  chevronRight: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>`,

  // Asset Badges
  studio: `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/></svg>`,
  python: `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M11.91 0c-3.15 0-5.07.72-5.07 3.32v2.45h5.17v.75H4.82C2.18 6.52 0 8.04 0 11.23c0 3.2 2.18 4.67 4.82 4.67h1.49v-2.2c0-2.02 1.77-3.72 3.79-3.72h5.16v-.75H10.1v-2.7c0-2.43 1.93-3.72 5.07-3.72 2.94 0 4.9.72 4.9 3.03H17.8c0-1.07-.84-1.84-2.13-1.84h-3.76zM8.3 1.83a.92.92 0 1 1 0 1.84.92.92 0 0 1 0-1.84zm3.79 12.98c-2.02 0-3.79 1.7-3.79 3.72v2.2h1.49c2.64 0 4.82-1.47 4.82-4.67 0-3.19-2.18-4.71-4.82-4.71h-1.49v.74h5.16v2.72zm3.61 7.36a.92.92 0 1 1 0-1.84.92.92 0 0 1 0 1.84z"/></svg>`,
  colab: `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M16.94 4.54A7.47 7.47 0 0 0 12 6.43a7.47 7.47 0 0 0-4.94-1.89C2.94 4.54 0 7.48 0 11.6c0 4.12 2.94 7.06 7.06 7.06a7.47 7.47 0 0 0 4.94-1.89 7.47 7.47 0 0 0 4.94 1.89c4.12 0 7.06-2.94 7.06-7.06 0-4.12-2.94-7.06-7.06-7.06zM7.06 16.32c-2.73 0-4.72-1.99-4.72-4.72s1.99-4.72 4.72-4.72c1.47 0 2.8.63 3.74 1.64-1.2 1.34-1.94 3.12-1.94 5.08 0 1.05.21 2.05.6 2.96-.64.48-1.39.76-2.4.76zm9.88 0c-1.01 0-1.76-.28-2.4-.76.39-.91.6-1.91.6-2.96 0-1.96-.74-3.74-1.94-5.08.94-1.01 2.27-1.64 3.74-1.64 2.73 0 4.72 1.99 4.72 4.72s-1.99 4.72-4.72 4.72z"/></svg>`,
  github: `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>`,

  // 25 Category Precision Domain Vector SVGs
  cat_1: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><circle cx="8" cy="15" r="1.5" fill="currentColor"/><circle cx="12" cy="11" r="1.5" fill="currentColor"/><circle cx="16" cy="9" r="1.5" fill="currentColor"/><circle cx="19" cy="6" r="1.5" fill="currentColor"/><path d="M6 18L19 5" stroke-width="1.8"/></svg>`, // ML Basics (Scatter & Regression Line)
  cat_2: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M12 7.5v3.5M12 11l-6 4.5M12 11l6 4.5"/></svg>`, // ML Intermediate (Decision Tree & Ensemble Splits)
  cat_3: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="4" cy="6" r="1.8"/><circle cx="4" cy="12" r="1.8"/><circle cx="4" cy="18" r="1.8"/><circle cx="12" cy="8" r="1.8"/><circle cx="12" cy="16" r="1.8"/><circle cx="20" cy="12" r="1.8"/><path d="M5.8 6l4.4 2M5.8 6l4.4 10M5.8 12l4.4-4M5.8 12l4.4 4M5.8 18l4.4-10M5.8 18l4.4-2M13.8 8l4.4 4M13.8 16l4.4-4" stroke-width="1.2"/></svg>`, // DL Basics (Multi-layer Perceptron Graph)
  cat_4: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="8" width="9" height="9" rx="1.5"/><rect x="8" y="5" width="9" height="9" rx="1.5"/><rect x="12" y="2" width="9" height="9" rx="1.5"/></svg>`, // DL Intermediate (Convolutional Feature Tensor Maps)
  cat_5: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16M4 11h11M4 16h16M4 21h8"/><polyline points="18 13 21 16 18 19"/></svg>`, // NLP (Token Sequence & Syntax Tree)
  cat_6: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3"/><circle cx="12" cy="12" r="3.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/></svg>`, // Computer Vision (Bounding Box Target Reticle)
  cat_7: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polygon points="12 4 15 11 20 12 15 15 12 20 9 15 4 12 9 11 12 4" fill="currentColor" fill-opacity="0.2"/></svg>`, // RL (Markov Decision Policy Compass)
  cat_8: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 20h18M3 15l4-7 4 8 4-11 5 7"/></svg>`, // Time Series (Temporal Trend & Wave Spectrum)
  cat_9: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" fill="currentColor" fill-opacity="0.15"/><path d="M12 17.5V3"/></svg>`, // Recommender (Collaborative Item Affinity)
  cat_10: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z" fill="currentColor" fill-opacity="0.2"/><circle cx="19" cy="5" r="1.5" fill="currentColor"/><circle cx="5" cy="19" r="1.5" fill="currentColor"/></svg>`, // Gen AI (Diffusion Synthesis Sparkles)
  cat_11: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="5" r="2.5"/><circle cx="18" cy="18" r="2.5"/><circle cx="7" cy="18" r="2.5"/><circle cx="12" cy="12" r="2.5" fill="currentColor"/><line x1="8.3" y1="7.2" x2="10" y2="10.2"/><line x1="16" y1="6.8" x2="13.7" y2="10.2"/><line x1="13.7" y1="13.8" x2="16" y2="16.2"/><line x1="10" y1="13.8" x2="8.3" y2="16.2"/><line x1="8.5" y1="6" x2="15.5" y2="5"/></svg>`, // GNN (Topology Graph Embeddings)
  cat_12: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h3l2-5 3 10 3-8 2 6 2-3h3"/></svg>`, // Healthcare (Biomedical ECG Waveform)
  cat_13: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="6"/><rect x="4" y="6" width="4" height="9" rx="0.5"/><line x1="6" y1="15" x2="6" y2="21"/><line x1="12" y1="5" x2="12" y2="9"/><rect x="10" y="9" width="4" height="7" rx="0.5"/><line x1="12" y1="16" x2="12" y2="19"/><line x1="18" y1="2" x2="18" y2="4"/><rect x="16" y="4" width="4" height="11" rx="0.5"/><line x1="18" y1="15" x2="18" y2="22"/></svg>`, // Finance (Trading Candlestick Chart)
  cat_14: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/><circle cx="9" cy="9" r="1.5" fill="currentColor"/><circle cx="15" cy="15" r="1.5" fill="currentColor"/></svg>`, // Adv NLP (Transformer Attention Weights)
  cat_15: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 21 7 21 17 12 22 3 17 3 7 12 2"/><line x1="12" y1="2" x2="12" y2="22"/><line x1="21" y1="7" x2="12" y2="12"/><line x1="3" y1="7" x2="12" y2="12"/></svg>`, // Adv CV (3D Point Cloud & Spatial Depth Mesh)
  cat_16: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="2" fill="currentColor"/><line x1="12" y1="12" x2="18.5" y2="5.5"/></svg>`, // Adv RL (Autonomous Radar Sensor)
  cat_17: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="19" r="3"/><path d="M7.5 17l6-6"/><circle cx="15" cy="9" r="2.5"/><path d="M17 7.5l3-3"/><circle cx="20.5" cy="3.5" r="1.5"/><line x1="2" y1="22" x2="22" y2="22"/></svg>`, // Robotics (Precision Articulated Robotic Arm)
  cat_18: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="10" x2="3" y2="14"/><line x1="7" y1="6" x2="7" y2="18"/><line x1="11" y1="3" x2="11" y2="21"/><line x1="15" y1="8" x2="15" y2="16"/><line x1="19" y1="5" x2="19" y2="19"/><line x1="23" y1="11" x2="23" y2="13"/></svg>`, // Audio (Acoustic Frequency Spectrogram)
  cat_19: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="10.5" cy="10.5" r="6.5"/><line x1="15.5" y1="15.5" x2="21" y2="21" stroke-width="2.5"/><line x1="8" y1="10.5" x2="13" y2="10.5"/><line x1="10.5" y1="8" x2="10.5" y2="13"/></svg>`, // XAI (SHAP Attribution Inspector Lens)
  cat_20: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="2"/><circle cx="12" cy="12" r="2.5" fill="currentColor"/><line x1="8" y1="1" x2="8" y2="5"/><line x1="12" y1="1" x2="12" y2="5"/><line x1="16" y1="1" x2="16" y2="5"/><line x1="8" y1="19" x2="8" y2="23"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="16" y1="19" x2="16" y2="23"/><line x1="1" y1="8" x2="5" y2="8"/><line x1="1" y1="12" x2="5" y2="12"/><line x1="1" y1="16" x2="5" y2="16"/><line x1="19" y1="8" x2="23" y2="8"/><line x1="19" y1="12" x2="23" y2="12"/><line x1="19" y1="16" x2="23" y2="16"/></svg>`, // Edge AI (Microcontroller Silicon Processor)
  cat_21: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18"/><path d="M5 16l5-5 4 4 6-8"/><polyline points="15 7 20 7 20 12"/></svg>`, // Business (Analytics KPI Growth Chart)
  cat_22: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M3.6 9h16.8M3.6 15h16.8M12 3a14 14 0 0 0 0 18M12 3a14 14 0 0 1 0 18"/></svg>`, // Environmental (Earth Biosphere Satellite Loop)
  cat_23: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L4 5v6.5C4 16.5 7.4 20.8 12 22c4.6-1.2 8-5.5 8-10.5V5l-8-3z"/><circle cx="12" cy="11.5" r="2.5" fill="currentColor"/></svg>`, // Cybersecurity (Intrusion Defense Shield)
  cat_24: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="4.5"/><circle cx="16" cy="8" r="4.5"/><circle cx="12" cy="15" r="4.5"/></svg>`, // Multimodal (Vision-Audio-Text Triplet Matrix)
  cat_25: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="5" r="2"/><circle cx="19" cy="5" r="2"/><circle cx="12" cy="12" r="3"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><line x1="6.5" y1="6.5" x2="10" y2="10"/><line x1="17.5" y1="6.5" x2="14" y2="10"/><line x1="10" y1="14" x2="6.5" y2="17.5"/><line x1="14" y1="14" x2="17.5" y2="17.5"/></svg>`, // Meta & Hybrid (AutoML Evolutionary Architecture)
};

function getCategoryIcon(catId) {
  return SVG_ICONS[`cat_${catId}`] || SVG_ICONS.brandLogo;
}

// =============================================================================
// Application State
// =============================================================================

const STATE = {
  allProjects: [],
  filteredProjects: [],
  categories: [],
  metadata: {},
  searchQuery: '',
  selectedCategory: 'all',
  selectedDifficulty: 'all',
  onlyInteractive: false,
  sortBy: 'id_asc',
  viewMode: 'grid', // 'grid' | 'table'
  currentPage: 1,
  pageSize: 24
};

// DOM References
const DOM = {
  searchInput: document.getElementById('searchInput'),
  searchClearBtn: document.getElementById('searchClearBtn'),
  categorySelect: document.getElementById('categorySelect'),
  difficultySelect: document.getElementById('difficultySelect'),
  onlyInteractiveToggle: document.getElementById('onlyInteractiveToggle'),
  sortSelect: document.getElementById('sortSelect'),
  domainCarousel: document.getElementById('domainCarousel'),
  resultsCount: document.getElementById('resultsCount'),
  projectsContainer: document.getElementById('projectsContainer'),
  paginationWrapper: document.getElementById('paginationWrapper'),
  statTotal: document.getElementById('statTotal'),
  statCategories: document.getElementById('statCategories'),
  statApps: document.getElementById('statApps'),
  modalOverlay: document.getElementById('modalOverlay'),
  modalCard: document.getElementById('modalCard'),
  modalTitleBar: document.getElementById('modalTitleBar'),
  modalClose: document.getElementById('modalClose'),
  modalBody: document.getElementById('modalBody'),
  toastNotice: document.getElementById('toastNotice'),
  toastMsg: document.getElementById('toastMsg'),
  viewGridBtn: document.getElementById('viewGridBtn'),
  viewListBtn: document.getElementById('viewListBtn')
};

// Initialize Application
async function init() {
  try {
    const res = await fetch('projects.json?v=2.2');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    
    STATE.metadata = data.metadata || {};
    STATE.categories = data.categories || [];
    STATE.allProjects = data.projects || [];
    STATE.filteredProjects = [...STATE.allProjects];
    
    updateStatsHeader();
    renderDomainCarousel();
    populateCategoryDropdown();
    attachEventListeners();
    applyFilters();
  } catch (err) {
    console.error('Failed to load projects.json:', err);
    DOM.projectsContainer.innerHTML = `
      <div style="text-align: center; padding: 60px 20px;">
        <h3 style="color: #EF4444; margin-bottom: 12px; font-size: 1.4rem;">Failed to load projects data</h3>
        <p style="color: #94A3B8;">Please ensure <code>projects.json</code> is accessible in the same directory.</p>
      </div>
    `;
  }
}

// Update Top KPI Counters
function updateStatsHeader() {
  if (DOM.statTotal) DOM.statTotal.textContent = STATE.allProjects.length.toLocaleString();
  if (DOM.statCategories) DOM.statCategories.textContent = STATE.categories.length;
  if (DOM.statApps) {
    const appsCount = STATE.allProjects.filter(p => p.has_app).length;
    DOM.statApps.textContent = appsCount;
  }
}

// Render Domain Carousel with Vector Icons
function renderDomainCarousel() {
  if (!DOM.domainCarousel) return;
  
  let html = `
    <button class="domain-pill active" data-cat="all">
      <span style="display:inline-flex; align-items:center;">${SVG_ICONS.sparkle}</span>
      <span>All 1,000 Projects</span>
      <span class="domain-pill-count">${STATE.allProjects.length}</span>
    </button>
  `;
  
  STATE.categories.forEach(cat => {
    html += `
      <button class="domain-pill" data-cat="${cat.id}" style="--cat-color: ${cat.color};">
        <span style="display:inline-flex; align-items:center; color:${cat.color};">${getCategoryIcon(cat.id)}</span>
        <span>${cat.name}</span>
        <span class="domain-pill-count">${cat.count}</span>
      </button>
    `;
  });
  
  DOM.domainCarousel.innerHTML = html;
}

// Populate Category Filter Dropdown
function populateCategoryDropdown() {
  if (!DOM.categorySelect) return;
  DOM.categorySelect.innerHTML = `<option value="all">All 25 AI Domains (${STATE.allProjects.length})</option>`;
  
  STATE.categories.forEach(cat => {
    const opt = document.createElement('option');
    opt.value = cat.id;
    opt.textContent = `${cat.name} (${cat.count})`;
    DOM.categorySelect.appendChild(opt);
  });
}

// Filter and Sort Engine
function applyFilters() {
  const query = STATE.searchQuery.trim().toLowerCase();
  
  STATE.filteredProjects = STATE.allProjects.filter(p => {
    if (query) {
      const matchNum = p.num.toString() === query || p.num.toString().padStart(4, '0') === query;
      const matchTitle = p.title.toLowerCase().includes(query);
      const matchDesc = p.desc.toLowerCase().includes(query);
      const matchTags = p.tags && p.tags.some(t => t.toLowerCase().includes(query));
      const matchCat = p.cat_name.toLowerCase().includes(query);
      
      if (!matchNum && !matchTitle && !matchDesc && !matchTags && !matchCat) {
        return false;
      }
    }
    
    if (STATE.selectedCategory !== 'all') {
      if (p.cat_id !== parseInt(STATE.selectedCategory)) return false;
    }
    
    if (STATE.selectedDifficulty !== 'all') {
      const pDiff = (p.diff || '').toLowerCase();
      if (!pDiff.includes(STATE.selectedDifficulty.toLowerCase())) return false;
    }
    
    if (STATE.onlyInteractive && !p.has_app) {
      return false;
    }
    
    return true;
  });
  
  // Sorting
  STATE.filteredProjects.sort((a, b) => {
    if (STATE.sortBy === 'id_asc') return a.num - b.num;
    if (STATE.sortBy === 'id_desc') return b.num - a.num;
    if (STATE.sortBy === 'title_asc') return a.title.localeCompare(b.title);
    if (STATE.sortBy === 'cat_asc') return a.cat_id - b.cat_id || a.num - b.num;
    return 0;
  });
  
  STATE.currentPage = 1;
  render();
}

// Render Project Grid or Table
function render() {
  const total = STATE.filteredProjects.length;
  DOM.resultsCount.innerHTML = `Showing <b>${total}</b> of <b>${STATE.allProjects.length}</b> projects`;
  
  if (total === 0) {
    DOM.projectsContainer.innerHTML = `
      <div style="text-align: center; padding: 70px 20px; background: rgba(18,24,36,0.5); border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1);">
        <div style="display:inline-flex; align-items:center; justify-content:center; width:56px; height:56px; border-radius:50%; background:rgba(255,255,255,0.05); color:#94A3B8; margin-bottom:16px;">
          ${SVG_ICONS.search}
        </div>
        <h3 style="font-size: 1.3rem; margin-bottom: 8px; color: #fff;">No matching projects found</h3>
        <p style="color: #94A3B8; max-width: 450px; margin: 0 auto 24px; font-size:0.9rem;">Try searching for another keyword, clearing the filters, or switching categories.</p>
        <button class="btn-open-modal" style="background: var(--accent-gold); color: #000; font-weight: 700;" onclick="resetFilters()">Reset All Filters</button>
      </div>
    `;
    DOM.paginationWrapper.innerHTML = '';
    return;
  }
  
  const startIdx = (STATE.currentPage - 1) * STATE.pageSize;
  const endIdx = Math.min(startIdx + STATE.pageSize, total);
  const pageItems = STATE.filteredProjects.slice(startIdx, endIdx);
  
  if (STATE.viewMode === 'table') {
    renderTableView(pageItems);
  } else {
    renderGridView(pageItems);
  }
  
  renderPagination(total);
}

// Render Card Grid
function renderGridView(items) {
  DOM.projectsContainer.innerHTML = `
    <div class="projects-grid">
      ${items.map(p => createCardHTML(p)).join('')}
    </div>
  `;
}

// Render High-Density Table
function renderTableView(items) {
  DOM.projectsContainer.innerHTML = `
    <div class="table-scroll-container">
      <table class="projects-table-view">
        <thead>
          <tr>
            <th>ID</th>
            <th>Project Title</th>
            <th>Domain Category</th>
            <th>Difficulty</th>
            <th>Assets</th>
            <th style="text-align: right;">Action</th>
          </tr>
        </thead>
        <tbody>
          ${items.map(p => {
            const numStr = `#${p.num.toString().padStart(4, '0')}`;
            return `
              <tr onclick="openProjectModal('${p.id}')">
                <td><span class="card-num-badge">${numStr}</span></td>
                <td><b style="color: #fff;">${escapeHTML(p.title)}</b></td>
                <td>
                  <span class="card-cat-badge" style="color:${p.cat_color}; background:${p.cat_color}18; border: 1px solid ${p.cat_color}33;">
                    ${getCategoryIcon(p.cat_id)} ${p.cat_name}
                  </span>
                </td>
                <td><span style="font-size: 0.78rem; color: #CBD5E1;">${p.diff}</span></td>
                <td>
                  <div style="display: flex; gap: 6px;">
                    ${p.has_app ? `<span class="badge-asset has-studio" title="Streamlit Web Studio">${SVG_ICONS.studio} Studio</span>` : ''}
                    <span class="badge-asset" title="Python CLI Pipeline">${SVG_ICONS.python} CLI</span>
                    ${p.has_notebook ? `<span class="badge-asset" title="Google Colab">${SVG_ICONS.colab} Colab</span>` : ''}
                  </div>
                </td>
                <td style="text-align: right;">
                  <button class="btn-open-modal" onclick="event.stopPropagation(); openProjectModal('${p.id}')">
                    Inspect ${SVG_ICONS.arrowRight}
                  </button>
                </td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

// Generate Card HTML with Vector Badges
function createCardHTML(p) {
  const numStr = `#${p.num.toString().padStart(4, '0')}`;
  return `
    <div class="project-card" data-project-id="${p.id}">
      <div class="card-top">
        <span class="card-num-badge">${numStr}</span>
        <span class="card-cat-badge" style="color:${p.cat_color}; background:${p.cat_color}18; border: 1px solid ${p.cat_color}33;">
          ${getCategoryIcon(p.cat_id)} ${p.cat_name}
        </span>
      </div>
      
      <h3 class="card-title" onclick="openProjectModal('${p.id}')">${escapeHTML(p.title)}</h3>
      <p class="card-desc">${escapeHTML(p.desc)}</p>
      
      <div class="card-badges-row">
        ${p.has_app ? `<span class="badge-asset has-studio"><span class="pulse-green"></span> ${SVG_ICONS.studio} Live Studio</span>` : ''}
        ${p.has_script ? `<span class="badge-asset">${SVG_ICONS.python} Python CLI</span>` : ''}
        ${p.has_notebook ? `<span class="badge-asset">${SVG_ICONS.colab} Colab Notebook</span>` : ''}
      </div>
      
      <div class="card-bottom-bar">
        <button class="btn-open-modal" onclick="openProjectModal('${p.id}')">
          ${SVG_ICONS.terminal} Inspect & Run
        </button>
        <div class="action-icon-links">
          <a href="${p.github_url}" target="_blank" rel="noopener noreferrer" class="icon-action-btn" title="View Source on GitHub">
            ${SVG_ICONS.github}
          </a>
          ${p.has_notebook ? `
            <a href="${p.colab_url}" target="_blank" rel="noopener noreferrer" class="icon-action-btn" title="Open in Google Colab">
              ${SVG_ICONS.colab}
            </a>
          ` : ''}
        </div>
      </div>
    </div>
  `;
}

// Generate Pagination Controls
function renderPagination(total) {
  const totalPages = Math.ceil(total / STATE.pageSize);
  if (totalPages <= 1) {
    DOM.paginationWrapper.innerHTML = '';
    return;
  }
  
  let html = `
    <button class="btn-page" ${STATE.currentPage === 1 ? 'disabled' : ''} onclick="goToPage(${STATE.currentPage - 1})">
      ${SVG_ICONS.chevronLeft} <span>Prev</span>
    </button>
  `;
  
  const maxButtons = 7;
  let startPage = Math.max(1, STATE.currentPage - 3);
  let endPage = Math.min(totalPages, startPage + maxButtons - 1);
  if (endPage - startPage < maxButtons - 1) {
    startPage = Math.max(1, endPage - maxButtons + 1);
  }
  
  if (startPage > 1) {
    html += `<button class="btn-page" onclick="goToPage(1)">1</button>`;
    if (startPage > 2) html += `<span style="color:#64748B; padding:0 4px;">...</span>`;
  }
  
  for (let i = startPage; i <= endPage; i++) {
    html += `
      <button class="btn-page ${i === STATE.currentPage ? 'active' : ''}" onclick="goToPage(${i})">
        ${i}
      </button>
    `;
  }
  
  if (endPage < totalPages) {
    if (endPage < totalPages - 1) html += `<span style="color:#64748B; padding:0 4px;">...</span>`;
    html += `<button class="btn-page" onclick="goToPage(${totalPages})">${totalPages}</button>`;
  }
  
  html += `
    <button class="btn-page" ${STATE.currentPage === totalPages ? 'disabled' : ''} onclick="goToPage(${STATE.currentPage + 1})">
      <span>Next</span> ${SVG_ICONS.chevronRight}
    </button>
  `;
  
  DOM.paginationWrapper.innerHTML = html;
}

function goToPage(page) {
  STATE.currentPage = page;
  render();
  window.scrollTo({ top: DOM.projectsContainer.offsetTop - 80, behavior: 'smooth' });
}

// Open Developer Terminal Modal
function openProjectModal(projectId) {
  const p = STATE.allProjects.find(item => item.id === projectId);
  if (!p) return;
  
  DOM.modalTitleBar.textContent = `harsh@1000-ai-projects: ~/${p.path}`;
  
  const cloneCmd = `git clone https://github.com/HarshChoudhary2003/1000-AI-projects.git\ncd "1000-AI-projects/${p.path}"\npip install -r requirements.txt\npython main.py`;
  const streamCmd = p.has_app ? `cd "1000-AI-projects/${p.path}"\nstreamlit run app.py` : '';
  
  DOM.modalBody.innerHTML = `
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
      <span class="card-num-badge">#${p.num.toString().padStart(4, '0')}</span>
      <span class="card-cat-badge" style="color:${p.cat_color}; background:${p.cat_color}18; border: 1px solid ${p.cat_color}33;">
        ${getCategoryIcon(p.cat_id)} ${p.cat_name}
      </span>
      <span class="badge-asset" style="margin-left: auto;">${p.diff}</span>
    </div>
    
    <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.7rem; font-weight: 800; color: #fff; margin-bottom: 12px; line-height: 1.25;">
      ${escapeHTML(p.title)}
    </h2>
    
    <p style="font-size: 0.94rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 22px;">
      ${escapeHTML(p.desc)}
    </p>
    
    <!-- Quickstart Execution Command -->
    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.78rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">
      <span style="display:inline-flex; align-items:center; gap:6px;">${SVG_ICONS.terminal} Standalone CLI Execution</span>
    </div>
    <div class="terminal-code-block">
      <span>${escapeHTML(cloneCmd.replace(/\n/g, ' && '))}</span>
      <button class="btn-terminal-copy" onclick="copySnippet('${escapeAttr(cloneCmd)}')">
        ${SVG_ICONS.copy} Copy Command
      </button>
    </div>
    
    <!-- Interactive Web App Studio -->
    ${p.has_app ? `
      <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.78rem; font-weight: 700; color: #34D399; text-transform: uppercase; letter-spacing: 0.05em;">
        <span style="display:inline-flex; align-items:center; gap:6px;">${SVG_ICONS.studio} Launch Interactive Streamlit Studio</span>
      </div>
      <div class="terminal-code-block studio-code">
        <span>${escapeHTML(streamCmd.replace(/\n/g, ' && '))}</span>
        <button class="btn-terminal-copy" onclick="copySnippet('${escapeAttr(streamCmd)}')">
          ${SVG_ICONS.copy} Copy Command
        </button>
      </div>
    ` : ''}
    
    <!-- Quick Action Launchers -->
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; padding-top: 18px; border-top: 1px solid rgba(255,255,255,0.08);">
      <a href="${p.github_url}" target="_blank" rel="noopener noreferrer" class="btn-open-modal" style="background: var(--accent-gold); color: #000; font-weight: 700; padding: 9px 18px;">
        ${SVG_ICONS.github} View Source on GitHub
      </a>
      ${p.has_notebook ? `
        <a href="${p.colab_url}" target="_blank" rel="noopener noreferrer" class="btn-open-modal" style="padding: 9px 16px;">
          ${SVG_ICONS.colab} Open in Google Colab
        </a>
      ` : ''}
      <a href="${p.github_url}/README.md" target="_blank" rel="noopener noreferrer" class="btn-open-modal" style="padding: 9px 16px;">
        ${SVG_ICONS.doc} Read Architecture Blueprint
      </a>
    </div>
  `;
  
  DOM.modalOverlay.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  DOM.modalOverlay.classList.remove('open');
  document.body.style.overflow = '';
}

// Toast Notification
function showToast(message) {
  if (!DOM.toastNotice) return;
  DOM.toastMsg.textContent = message;
  DOM.toastNotice.classList.add('show');
  setTimeout(() => {
    DOM.toastNotice.classList.remove('show');
  }, 2600);
}

function copySnippet(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast("Command copied to clipboard!");
  }).catch(() => {
    const el = document.createElement('textarea');
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
    showToast("Command copied to clipboard!");
  });
}

function resetFilters() {
  STATE.searchQuery = '';
  STATE.selectedCategory = 'all';
  STATE.selectedDifficulty = 'all';
  STATE.onlyInteractive = false;
  STATE.sortBy = 'id_asc';
  
  if (DOM.searchInput) DOM.searchInput.value = '';
  if (DOM.searchClearBtn) DOM.searchClearBtn.classList.remove('visible');
  if (DOM.categorySelect) DOM.categorySelect.value = 'all';
  if (DOM.difficultySelect) DOM.difficultySelect.value = 'all';
  if (DOM.onlyInteractiveToggle) DOM.onlyInteractiveToggle.checked = false;
  if (DOM.sortSelect) DOM.sortSelect.value = 'id_asc';
  
  document.querySelectorAll('.domain-pill').forEach(btn => btn.classList.remove('active'));
  const allBtn = document.querySelector('.domain-pill[data-cat="all"]');
  if (allBtn) allBtn.classList.add('active');
  
  applyFilters();
}

// Event Listeners
function attachEventListeners() {
  let debounceTimer;
  DOM.searchInput.addEventListener('input', (e) => {
    const val = e.target.value;
    if (DOM.searchClearBtn) {
      if (val) DOM.searchClearBtn.classList.add('visible');
      else DOM.searchClearBtn.classList.remove('visible');
    }
    
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      STATE.searchQuery = val;
      applyFilters();
    }, 180);
  });
  
  if (DOM.searchClearBtn) {
    DOM.searchClearBtn.addEventListener('click', () => {
      DOM.searchInput.value = '';
      DOM.searchClearBtn.classList.remove('visible');
      STATE.searchQuery = '';
      applyFilters();
      DOM.searchInput.focus();
    });
  }
  
  DOM.categorySelect.addEventListener('change', (e) => {
    STATE.selectedCategory = e.target.value;
    document.querySelectorAll('.domain-pill').forEach(b => b.classList.remove('active'));
    const target = document.querySelector(`.domain-pill[data-cat="${e.target.value}"]`);
    if (target) target.classList.add('active');
    applyFilters();
  });
  
  DOM.difficultySelect.addEventListener('change', (e) => {
    STATE.selectedDifficulty = e.target.value;
    applyFilters();
  });
  
  DOM.onlyInteractiveToggle.addEventListener('change', (e) => {
    STATE.onlyInteractive = e.target.checked;
    applyFilters();
  });
  
  DOM.sortSelect.addEventListener('change', (e) => {
    STATE.sortBy = e.target.value;
    applyFilters();
  });
  
  DOM.domainCarousel.addEventListener('click', (e) => {
    const btn = e.target.closest('.domain-pill');
    if (!btn) return;
    
    document.querySelectorAll('.domain-pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    
    const cat = btn.getAttribute('data-cat');
    STATE.selectedCategory = cat;
    DOM.categorySelect.value = cat;
    applyFilters();
  });
  
  DOM.viewGridBtn.addEventListener('click', () => {
    STATE.viewMode = 'grid';
    DOM.viewGridBtn.classList.add('active');
    DOM.viewListBtn.classList.remove('active');
    render();
  });
  
  DOM.viewListBtn.addEventListener('click', () => {
    STATE.viewMode = 'table';
    DOM.viewListBtn.classList.add('active');
    DOM.viewGridBtn.classList.remove('active');
    render();
  });
  
  DOM.modalClose.addEventListener('click', closeModal);
  DOM.modalOverlay.addEventListener('click', (e) => {
    if (e.target === DOM.modalOverlay) closeModal();
  });
  
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
    if (e.key === '/' && document.activeElement !== DOM.searchInput) {
      e.preventDefault();
      DOM.searchInput.focus();
    }
  });
}

function escapeHTML(str) {
  if (!str) return '';
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}

function escapeAttr(str) {
  if (!str) return '';
  return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

document.addEventListener('DOMContentLoaded', init);

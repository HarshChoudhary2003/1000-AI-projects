/**
 * 1000 AI Projects Showcase Portal
 * Pure ES6 Client-side Engine: Instant Search, Multi-Filter, Pagination, Modals
 */

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
  currentPage: 1,
  pageSize: 24
};

// DOM References
const DOM = {
  searchInput: document.getElementById('searchInput'),
  categorySelect: document.getElementById('categorySelect'),
  difficultySelect: document.getElementById('difficultySelect'),
  onlyInteractiveToggle: document.getElementById('onlyInteractiveToggle'),
  sortSelect: document.getElementById('sortSelect'),
  resultsCount: document.getElementById('resultsCount'),
  projectsGrid: document.getElementById('projectsGrid'),
  paginationWrapper: document.getElementById('paginationWrapper'),
  quickPills: document.getElementById('quickPills'),
  statTotal: document.getElementById('statTotal'),
  statCategories: document.getElementById('statCategories'),
  statApps: document.getElementById('statApps'),
  modalOverlay: document.getElementById('modalOverlay'),
  modalCard: document.getElementById('modalCard'),
  modalClose: document.getElementById('modalClose'),
  modalBody: document.getElementById('modalBody')
};

// Initialize Application
async function init() {
  try {
    const res = await fetch('projects.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    
    STATE.metadata = data.metadata || {};
    STATE.categories = data.categories || [];
    STATE.allProjects = data.projects || [];
    STATE.filteredProjects = [...STATE.allProjects];
    
    updateStatsHeader();
    populateCategoryDropdown();
    attachEventListeners();
    applyFilters();
  } catch (err) {
    console.error('Failed to load projects.json:', err);
    DOM.projectsGrid.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 60px 20px;">
        <h3 style="color: #EF4444; margin-bottom: 12px;">⚠️ Failed to load projects data</h3>
        <p style="color: #94A3B8;">Please ensure <code>projects.json</code> exists in the same directory.</p>
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

// Populate Category Filter
function populateCategoryDropdown() {
  if (!DOM.categorySelect) return;
  DOM.categorySelect.innerHTML = `<option value="all">All 25 AI Domains (${STATE.allProjects.length})</option>`;
  
  STATE.categories.forEach(cat => {
    const opt = document.createElement('option');
    opt.value = cat.id;
    opt.textContent = `${cat.icon} ${cat.name} (${cat.count})`;
    DOM.categorySelect.appendChild(opt);
  });
}

// Filter and Sort Engine
function applyFilters() {
  const query = STATE.searchQuery.trim().toLowerCase();
  
  STATE.filteredProjects = STATE.allProjects.filter(p => {
    // Search matching: ID, Number, Title, Description, Tags
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
    
    // Category filter
    if (STATE.selectedCategory !== 'all') {
      if (p.cat_id !== parseInt(STATE.selectedCategory)) return false;
    }
    
    // Difficulty filter
    if (STATE.selectedDifficulty !== 'all') {
      const pDiff = (p.diff || '').toLowerCase();
      if (!pDiff.includes(STATE.selectedDifficulty.toLowerCase())) return false;
    }
    
    // Only interactive Streamlit apps
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
    if (STATE.sortBy === 'title_desc') return b.title.localeCompare(a.title);
    if (STATE.sortBy === 'cat_asc') return a.cat_id - b.cat_id || a.num - b.num;
    return 0;
  });
  
  STATE.currentPage = 1;
  render();
}

// Render Project Grid & Pagination
function render() {
  const total = STATE.filteredProjects.length;
  DOM.resultsCount.innerHTML = `Showing <b>${total}</b> of <b>${STATE.allProjects.length}</b> projects`;
  
  if (total === 0) {
    DOM.projectsGrid.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 80px 20px;">
        <div style="font-size: 3rem; margin-bottom: 14px;">🔍</div>
        <h3 style="font-size: 1.3rem; margin-bottom: 8px; color: #fff;">No matching projects found</h3>
        <p style="color: #94A3B8; max-width: 450px; margin: 0 auto 20px;">Try adjusting your keyword search, clearing filters, or switching categories.</p>
        <button class="btn-primary" onclick="resetFilters()">Reset All Filters</button>
      </div>
    `;
    DOM.paginationWrapper.innerHTML = '';
    return;
  }
  
  // Calculate Pagination Slice
  const startIdx = (STATE.currentPage - 1) * STATE.pageSize;
  const endIdx = Math.min(startIdx + STATE.pageSize, total);
  const pageItems = STATE.filteredProjects.slice(startIdx, endIdx);
  
  DOM.projectsGrid.innerHTML = pageItems.map(p => createCardHTML(p)).join('');
  renderPagination(total);
}

// Generate Card HTML
function createCardHTML(p) {
  const numStr = `#${p.num.toString().padStart(4, '0')}`;
  const diffClass = p.diff.toLowerCase().includes('beginner') ? 'diff-beginner' : 
                   (p.diff.toLowerCase().includes('intermediate') ? 'diff-intermediate' : 'diff-advanced');
                   
  return `
    <div class="project-card" data-project-id="${p.id}">
      <div class="card-header">
        <span class="project-id-badge">${numStr}</span>
        <span class="category-chip" style="color:${p.cat_color}; background:${p.cat_color}18; border: 1px solid ${p.cat_color}33;">
          ${p.cat_icon} ${p.cat_name}
        </span>
      </div>
      
      <h3 class="card-title" onclick="openProjectModal('${p.id}')">${escapeHTML(p.title)}</h3>
      <p class="card-desc">${escapeHTML(p.desc)}</p>
      
      <div class="card-assets">
        ${p.has_app ? '<span class="asset-badge app-ready">🌐 Live Web App</span>' : ''}
        ${p.has_script ? '<span class="asset-badge">🐍 CLI Script</span>' : ''}
        ${p.has_notebook ? '<span class="asset-badge">📓 Colab Notebook</span>' : ''}
      </div>
      
      <div class="card-footer">
        <button class="btn-inspect" onclick="openProjectModal('${p.id}')">
          🔍 Details & Run
        </button>
        <div class="card-links">
          <a href="${p.github_url}" target="_blank" rel="noopener noreferrer" class="icon-btn" title="View Source on GitHub">
            💻
          </a>
          ${p.has_notebook ? `
            <a href="${p.colab_url}" target="_blank" rel="noopener noreferrer" class="icon-btn" title="Open in Google Colab">
              📓
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
    <button class="page-btn" ${STATE.currentPage === 1 ? 'disabled' : ''} onclick="goToPage(${STATE.currentPage - 1})">
      ◀ Prev
    </button>
  `;
  
  // Page number buttons
  const maxButtons = 7;
  let startPage = Math.max(1, STATE.currentPage - 3);
  let endPage = Math.min(totalPages, startPage + maxButtons - 1);
  if (endPage - startPage < maxButtons - 1) {
    startPage = Math.max(1, endPage - maxButtons + 1);
  }
  
  if (startPage > 1) {
    html += `<button class="page-btn" onclick="goToPage(1)">1</button>`;
    if (startPage > 2) html += `<span style="color:#64748B; padding:0 4px;">...</span>`;
  }
  
  for (let i = startPage; i <= endPage; i++) {
    html += `
      <button class="page-btn ${i === STATE.currentPage ? 'active' : ''}" onclick="goToPage(${i})">
        ${i}
      </button>
    `;
  }
  
  if (endPage < totalPages) {
    if (endPage < totalPages - 1) html += `<span style="color:#64748B; padding:0 4px;">...</span>`;
    html += `<button class="page-btn" onclick="goToPage(${totalPages})">${totalPages}</button>`;
  }
  
  html += `
    <button class="page-btn" ${STATE.currentPage === totalPages ? 'disabled' : ''} onclick="goToPage(${STATE.currentPage + 1})">
      Next ▶
    </button>
  `;
  
  DOM.paginationWrapper.innerHTML = html;
}

function goToPage(page) {
  STATE.currentPage = page;
  render();
  window.scrollTo({ top: DOM.projectsGrid.offsetTop - 80, behavior: 'smooth' });
}

// Open Detail Modal
function openProjectModal(projectId) {
  const p = STATE.allProjects.find(item => item.id === projectId);
  if (!p) return;
  
  const cloneCmd = `git clone https://github.com/HarshChoudhary2003/1000-AI-projects.git\ncd "1000-AI-projects/${p.path}"\npip install -r requirements.txt\npython main.py`;
  const streamCmd = p.has_app ? `streamlit run app.py` : '';
  
  DOM.modalBody.innerHTML = `
    <div class="modal-header-badge">
      <span class="project-id-badge">#${p.num.toString().padStart(4, '0')}</span>
      <span class="category-chip" style="color:${p.cat_color}; background:${p.cat_color}18; border: 1px solid ${p.cat_color}33;">
        ${p.cat_icon} ${p.cat_name}
      </span>
      <span class="asset-badge" style="margin-left:auto;">${p.diff}</span>
    </div>
    
    <h2 class="modal-title">${escapeHTML(p.title)}</h2>
    <p class="modal-desc">${escapeHTML(p.desc)}</p>
    
    <div class="code-box-header">
      <span>🚀 Quickstart Terminal Command</span>
    </div>
    <div class="terminal-box">
      <span>${escapeHTML(cloneCmd.replace(/\n/g, ' && '))}</span>
      <button class="btn-copy" onclick="copyText('${escapeAttr(cloneCmd)}')">Copy Command</button>
    </div>
    
    ${p.has_app ? `
      <div class="code-box-header">
        <span>🌐 Launch Interactive Streamlit UI</span>
      </div>
      <div class="terminal-box" style="color: #34D399;">
        <span>${escapeHTML(streamCmd)}</span>
        <button class="btn-copy" onclick="copyText('${escapeAttr(streamCmd)}')">Copy</button>
      </div>
    ` : ''}
    
    <div class="modal-actions">
      <a href="${p.github_url}" target="_blank" rel="noopener noreferrer" class="btn-primary">
        💻 View on GitHub
      </a>
      ${p.has_notebook ? `
        <a href="${p.colab_url}" target="_blank" rel="noopener noreferrer" class="btn-secondary">
          📓 Open in Google Colab
        </a>
      ` : ''}
      <a href="${p.github_url}/README.md" target="_blank" rel="noopener noreferrer" class="btn-secondary">
        📖 Read Full Blueprint
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

function resetFilters() {
  STATE.searchQuery = '';
  STATE.selectedCategory = 'all';
  STATE.selectedDifficulty = 'all';
  STATE.onlyInteractive = false;
  STATE.sortBy = 'id_asc';
  
  if (DOM.searchInput) DOM.searchInput.value = '';
  if (DOM.categorySelect) DOM.categorySelect.value = 'all';
  if (DOM.difficultySelect) DOM.difficultySelect.value = 'all';
  if (DOM.onlyInteractiveToggle) DOM.onlyInteractiveToggle.checked = false;
  if (DOM.sortSelect) DOM.sortSelect.value = 'id_asc';
  
  // reset quick pills
  document.querySelectorAll('.pill-btn').forEach(btn => btn.classList.remove('active'));
  const allBtn = document.querySelector('.pill-btn[data-cat="all"]');
  if (allBtn) allBtn.classList.add('active');
  
  applyFilters();
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    alert('Copied to clipboard!');
  }).catch(() => {
    const el = document.createElement('textarea');
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
    alert('Copied to clipboard!');
  });
}

// Event Listeners
function attachEventListeners() {
  // Search input debounced
  let debounceTimer;
  DOM.searchInput.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      STATE.searchQuery = e.target.value;
      applyFilters();
    }, 200);
  });
  
  // Dropdowns
  DOM.categorySelect.addEventListener('change', (e) => {
    STATE.selectedCategory = e.target.value;
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
  
  // Quick Filter Pills
  DOM.quickPills.addEventListener('click', (e) => {
    const btn = e.target.closest('.pill-btn');
    if (!btn) return;
    
    document.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    
    const cat = btn.getAttribute('data-cat');
    const isInteractive = btn.getAttribute('data-interactive') === 'true';
    
    if (isInteractive) {
      STATE.onlyInteractive = true;
      DOM.onlyInteractiveToggle.checked = true;
      STATE.selectedCategory = 'all';
      DOM.categorySelect.value = 'all';
    } else {
      STATE.onlyInteractive = false;
      DOM.onlyInteractiveToggle.checked = false;
      STATE.selectedCategory = cat;
      DOM.categorySelect.value = cat;
    }
    applyFilters();
  });
  
  // Modal close handlers
  DOM.modalClose.addEventListener('click', closeModal);
  DOM.modalOverlay.addEventListener('click', (e) => {
    if (e.target === DOM.modalOverlay) closeModal();
  });
  
  // Keyboard navigation
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
    if (e.key === '/' && document.activeElement !== DOM.searchInput) {
      e.preventDefault();
      DOM.searchInput.focus();
    }
  });
}

// Utility Helpers
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

// Kickoff
document.addEventListener('DOMContentLoaded', init);

/**
 * 1,000 AI Projects Showcase Portal
 * Senior Staff Developer Engine: Instant Client Search, Domain Carousel,
 * Grid/Table View Switching, Terminal Window Modal, and Toast Notification System.
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
    const res = await fetch('projects.json');
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
        <h3 style="color: #EF4444; margin-bottom: 12px; font-size: 1.4rem;">⚠️ Failed to load projects data</h3>
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

// Render Domain Carousel Pills
function renderDomainCarousel() {
  if (!DOM.domainCarousel) return;
  
  let html = `
    <button class="domain-pill active" data-cat="all">
      <span>🚀</span>
      <span>All 1,000 Projects</span>
      <span class="domain-pill-count">${STATE.allProjects.length}</span>
    </button>
  `;
  
  STATE.categories.forEach(cat => {
    html += `
      <button class="domain-pill" data-cat="${cat.id}" style="--cat-color: ${cat.color};">
        <span>${cat.icon}</span>
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
      <div style="text-align: center; padding: 80px 20px; background: rgba(15,23,42,0.4); border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1);">
        <div style="font-size: 3.5rem; margin-bottom: 16px;">🔍</div>
        <h3 style="font-size: 1.4rem; margin-bottom: 8px; color: #fff;">No matching projects found</h3>
        <p style="color: #94A3B8; max-width: 450px; margin: 0 auto 24px;">Try searching for another keyword, clearing the filters, or switching categories.</p>
        <button class="btn-open-modal" style="background: var(--accent-gold); color: #000; font-weight: 700;" onclick="resetFilters()">Reset All Filters</button>
      </div>
    `;
    DOM.paginationWrapper.innerHTML = '';
    return;
  }
  
  // Pagination Slice
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
    <div style="overflow-x: auto;">
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
                    ${p.cat_icon} ${p.cat_name}
                  </span>
                </td>
                <td><span style="font-size: 0.8rem; color: #CBD5E1;">${p.diff}</span></td>
                <td>
                  <div style="display: flex; gap: 6px;">
                    ${p.has_app ? '<span class="badge-asset has-studio" title="Streamlit Web Studio">🌐 Studio</span>' : ''}
                    <span class="badge-asset">🐍 CLI</span>
                    ${p.has_notebook ? '<span class="badge-asset">📓 Colab</span>' : ''}
                  </div>
                </td>
                <td style="text-align: right;">
                  <button class="btn-open-modal" onclick="event.stopPropagation(); openProjectModal('${p.id}')">
                    Inspect ➔
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

// Generate Card HTML
function createCardHTML(p) {
  const numStr = `#${p.num.toString().padStart(4, '0')}`;
  return `
    <div class="project-card" data-project-id="${p.id}">
      <div class="card-top">
        <span class="card-num-badge">${numStr}</span>
        <span class="card-cat-badge" style="color:${p.cat_color}; background:${p.cat_color}18; border: 1px solid ${p.cat_color}33;">
          ${p.cat_icon} ${p.cat_name}
        </span>
      </div>
      
      <h3 class="card-title" onclick="openProjectModal('${p.id}')">${escapeHTML(p.title)}</h3>
      <p class="card-desc">${escapeHTML(p.desc)}</p>
      
      <div class="card-badges-row">
        ${p.has_app ? '<span class="badge-asset has-studio"><span class="pulse-green"></span> Live Web Studio</span>' : ''}
        ${p.has_script ? '<span class="badge-asset">🐍 Python CLI</span>' : ''}
        ${p.has_notebook ? '<span class="badge-asset">📓 Colab Notebook</span>' : ''}
      </div>
      
      <div class="card-bottom-bar">
        <button class="btn-open-modal" onclick="openProjectModal('${p.id}')">
          ⚡ Inspect & Run
        </button>
        <div class="action-icon-links">
          <a href="${p.github_url}" target="_blank" rel="noopener noreferrer" class="icon-action-btn" title="View Source on GitHub">
            💻
          </a>
          ${p.has_notebook ? `
            <a href="${p.colab_url}" target="_blank" rel="noopener noreferrer" class="icon-action-btn" title="Open in Google Colab">
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
    <button class="btn-page" ${STATE.currentPage === 1 ? 'disabled' : ''} onclick="goToPage(${STATE.currentPage - 1})">
      ◀ Prev
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
      Next ▶
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
        ${p.cat_icon} ${p.cat_name}
      </span>
      <span class="badge-asset" style="margin-left: auto;">${p.diff}</span>
    </div>
    
    <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 800; color: #fff; margin-bottom: 14px; line-height: 1.25;">
      ${escapeHTML(p.title)}
    </h2>
    
    <p style="font-size: 0.96rem; color: #CBD5E1; line-height: 1.6; margin-bottom: 24px;">
      ${escapeHTML(p.desc)}
    </p>
    
    <!-- Quickstart Execution Command -->
    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">
      <span>💻 1-Line Standalone Terminal Execution</span>
    </div>
    <div class="terminal-code-block">
      <span>${escapeHTML(cloneCmd.replace(/\n/g, ' && '))}</span>
      <button class="btn-terminal-copy" onclick="copySnippet('${escapeAttr(cloneCmd)}')">Copy Command</button>
    </div>
    
    <!-- Interactive Web App Studio -->
    ${p.has_app ? `
      <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.8rem; font-weight: 700; color: #34D399; text-transform: uppercase; letter-spacing: 0.05em;">
        <span>🌐 Launch Interactive Streamlit Studio</span>
      </div>
      <div class="terminal-code-block studio-code">
        <span>${escapeHTML(streamCmd.replace(/\n/g, ' && '))}</span>
        <button class="btn-terminal-copy" onclick="copySnippet('${escapeAttr(streamCmd)}')">Copy Command</button>
      </div>
    ` : ''}
    
    <!-- Quick Action Launchers -->
    <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 26px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.08);">
      <a href="${p.github_url}" target="_blank" rel="noopener noreferrer" class="btn-open-modal" style="background: var(--accent-gold); color: #000; font-weight: 700; padding: 10px 20px;">
        💻 View Source on GitHub
      </a>
      ${p.has_notebook ? `
        <a href="${p.colab_url}" target="_blank" rel="noopener noreferrer" class="btn-open-modal" style="padding: 10px 18px;">
          📓 Open in Google Colab
        </a>
      ` : ''}
      <a href="${p.github_url}/README.md" target="_blank" rel="noopener noreferrer" class="btn-open-modal" style="padding: 10px 18px;">
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
    showToast("Command copied to clipboard! ✓");
  }).catch(() => {
    const el = document.createElement('textarea');
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
    showToast("Command copied to clipboard! ✓");
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
  // Search with debounce & clear button
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
  
  // Dropdowns
  DOM.categorySelect.addEventListener('change', (e) => {
    STATE.selectedCategory = e.target.value;
    // sync carousel
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
  
  // Domain Carousel Click
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
  
  // View Mode Buttons (Grid vs Table)
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

/* ============================================
   Beatrice Trainer: frontend application
   Talks to the local FastAPI server (server.py).
   ============================================ */

// ---------- Configuration ----------
const isLocalOrigin = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(window.location.origin);
const defaultApiUrl = isLocalOrigin ? window.location.origin : 'http://localhost:8000';

function readStorage(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}
function writeStorage(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode */ }
}

const CONFIG = {
  apiUrl: readStorage('beatrice_api_url', defaultApiUrl),
  pollInterval: 2000,
  maxLogLines: 400,
};

const PLATFORM = document.documentElement.dataset.platform || 'mac';

// ---------- State ----------
const state = {
  currentPage: 'dashboard',
  serverOnline: null,
  status: null,
  autoConfig: null,
  training: {
    phase: 'idle',          // idle | running | completed | failed | stopped | ended
    active: false,
    speaker: null,
    step: 0,
    totalSteps: 10000,
    loss: null,
    lossHistory: [],
    startTime: null,
    stepsPerSec: null,
    memoryUsageMb: null,
    stopRequested: false,
    error: null,
  },
  speakers: [],
  models: [],
};

const $ = (id) => document.getElementById(id);

// ---------- Icons (inline SVG, no emoji in UI chrome) ----------
const ICONS = {
  info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
  success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
  warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
  mic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>',
  box: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
  file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>',
};

// ---------- Navigation ----------
const PAGES = ['dashboard', 'dataset', 'training', 'models', 'colab', 'settings'];

function navigateTo(page) {
  if (!PAGES.includes(page)) page = 'dashboard';
  state.currentPage = page;

  document.querySelectorAll('.nav-item[data-page]').forEach(item => {
    const active = item.dataset.page === page;
    item.classList.toggle('active', active);
    if (active) item.setAttribute('aria-current', 'page');
    else item.removeAttribute('aria-current');
  });

  document.querySelectorAll('.page-view').forEach(view => {
    view.classList.toggle('active', view.id === `page-${page}`);
  });

  if (window.location.hash !== `#${page}`) {
    history.replaceState(null, '', `#${page}`);
  }
  window.scrollTo(0, 0);

  switch (page) {
    case 'dashboard': refreshDashboard(); break;
    case 'dataset': refreshDataset(); break;
    case 'training': refreshTrainingSpeakers(); requestAnimationFrame(drawLossChart); break;
    case 'models': refreshModels(); break;
    case 'settings': renderSettings(); break;
  }
}

document.querySelectorAll('.nav-item[data-page]').forEach(item => {
  item.addEventListener('click', () => navigateTo(item.dataset.page));
});

window.addEventListener('hashchange', () => {
  const page = window.location.hash.replace('#', '');
  if (PAGES.includes(page) && page !== state.currentPage) navigateTo(page);
});

// ---------- API Helper ----------
function errorDetail(data, status) {
  const detail = data && data.detail;
  if (!detail) return `HTTP ${status}`;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(d => d.msg || JSON.stringify(d)).join('; ');
  return JSON.stringify(detail);
}

async function api(endpoint, options = {}) {
  const url = `${CONFIG.apiUrl}${endpoint}`;
  let response;
  try {
    response = await fetch(url, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
  } catch (networkErr) {
    setServerStatus(false);
    const err = new Error('Cannot reach the trainer server. Is it still running?');
    err.network = true;
    throw err;
  }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const err = new Error(cleanText(errorDetail(data, response.status)));
    err.status = response.status;
    throw err;
  }
  return response.json();
}

// ---------- Server Status ----------
function setServerStatus(online) {
  const changed = state.serverOnline !== online;
  state.serverOnline = online;
  const dot = $('server-status-dot');
  const text = $('server-status-text');
  dot.classList.toggle('online', online);
  dot.classList.toggle('offline', !online);
  text.textContent = online ? 'Server online' : 'Server offline';

  if (!online) {
    $('sys-server').innerHTML = badge('Offline', 'danger');
    const memBadge = $('mem-status-badge');
    memBadge.className = 'badge badge-dot';
    memBadge.textContent = 'Offline';
  }
  return changed;
}

async function checkServerStatus() {
  try {
    const data = await api('/api/status');
    const cameBack = setServerStatus(true) && state.status !== null;
    state.status = data;
    updateSystemInfo(data);
    if (cameBack) refreshDashboard();
    return data;
  } catch {
    setServerStatus(false);
    return null;
  }
}

function computeLabel(compute) {
  switch ((compute || '').toLowerCase()) {
    case 'mps': return 'Apple GPU (MPS)';
    case 'cuda': return 'NVIDIA GPU (CUDA)';
    case 'cpu': return 'CPU only';
    default: return compute ? String(compute).toUpperCase() : 'Unknown';
  }
}

function badge(text, variant = '', withDot = false) {
  const cls = ['badge', variant ? `badge-${variant}` : '', withDot ? 'badge-dot' : ''].filter(Boolean).join(' ');
  return `<span class="${cls}">${escapeHtml(text)}</span>`;
}

function updateSystemInfo(data) {
  if (!data) return;
  $('stat-device').textContent = data.gpu_name || data.device || 'Unknown';
  $('stat-device-sub').textContent = computeLabel(data.compute) + (data.torch_version ? ` · PyTorch ${data.torch_version}` : '');
  $('sys-platform').textContent = data.platform || 'Unknown';
  $('sys-compute').textContent = computeLabel(data.compute);
  $('sys-beatrice').innerHTML = data.beatrice_installed ? badge('Installed', 'success') : badge('Not found', 'danger');
  $('sys-server').innerHTML = badge('Connected', 'success');
  renderSettings();
}

async function refreshSystemInfo() {
  const data = await checkServerStatus();
  if (data) showToast('System info refreshed', 'info');
  else showToast('Cannot reach the trainer server', 'error');
}

// ---------- Dashboard ----------
async function refreshDashboard() {
  if (state.serverOnline === false) return;
  try {
    const [dataset, models] = await Promise.all([
      api('/api/dataset/list').catch(() => null),
      api('/api/models/list').catch(() => null),
    ]);
    if (dataset) state.speakers = dataset.speakers || [];
    if (models) state.models = models.models || [];
    updateDashboardStats();
  } catch {
    // server offline; status dot already shows it
  }
}

function updateDashboardStats() {
  $('stat-speakers').textContent = state.speakers.length;
  $('stat-files').textContent = totalFiles().toLocaleString();
  $('stat-models').textContent = state.models.length;
}

function totalFiles() {
  return state.speakers.reduce((sum, s) => sum + (s.file_count || 0), 0);
}

// ---------- Dataset Management ----------
function setupDropZone() {
  const zone = $('drop-zone');
  const input = $('file-input');
  const isAudio = (f) => /\.(wav|zip)$/i.test(f.name);

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const all = Array.from(e.dataTransfer.files);
    const files = all.filter(isAudio);
    if (files.length) uploadFiles(files);
    if (files.length < all.length) showToast(`Skipped ${all.length - files.length} file(s): only WAV and ZIP files are accepted`, 'warning');
  });

  input.addEventListener('change', () => {
    const files = Array.from(input.files).filter(isAudio);
    if (files.length) uploadFiles(files);
    input.value = '';
  });

  const speakerInput = $('speaker-name');
  speakerInput.addEventListener('input', () => {
    const val = sanitizeSpeaker(speakerInput.value) || 'unnamed';
    $('active-speaker-badge').textContent = `Speaker: ${val}`;
  });
}

function sanitizeSpeaker(name) {
  // Mirrors the server: letters, numbers, dash and underscore only.
  return (name || '').trim().replace(/[^\p{L}\p{N}_-]/gu, '');
}

function getSpeakerName() {
  const raw = $('speaker-name').value;
  const clean = sanitizeSpeaker(raw);
  if (!clean) {
    showToast('Enter a speaker name first', 'warning');
    $('speaker-name').focus();
    return null;
  }
  if (clean !== raw.trim()) {
    $('speaker-name').value = clean;
    showToast(`Speaker name saved as "${clean}"`, 'info');
  }
  return clean;
}

function switchImportMethod(method) {
  ['local', 'hf', 'kaggle'].forEach(m => {
    const tab = $(`tab-method-${m}`);
    const container = $(`import-container-${m}`);
    const active = m === method;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
    container.hidden = !active;
  });
}

function setButtonBusy(btn, busy, busyLabel) {
  if (!btn) return;
  if (busy) {
    if (!btn.dataset.label) btn.dataset.label = btn.innerHTML;
    btn.disabled = true;
    btn.classList.add('is-loading');
    btn.innerHTML = `<span class="spinner"></span>${escapeHtml(busyLabel)}`;
  } else {
    btn.disabled = false;
    btn.classList.remove('is-loading');
    if (btn.dataset.label) btn.innerHTML = btn.dataset.label;
    delete btn.dataset.label;
  }
}

function setProgress(prefix, { show, status, pct, variant } = {}) {
  const container = $(`${prefix}-progress-container`);
  if (!container) return;
  if (show !== undefined) container.hidden = !show;
  if (status !== undefined) $(`${prefix}-status`).textContent = status;
  if (pct !== undefined) {
    const p = Math.max(0, Math.min(100, Math.round(pct)));
    $(`${prefix}-progress-pct`).textContent = `${p}%`;
    $(`${prefix}-progress-bar`).style.width = `${p}%`;
  }
  const bar = $(`${prefix}-progress-bar`);
  bar.classList.toggle('is-error', variant === 'error');
  bar.classList.toggle('is-done', variant === 'done');
}

async function runImport({ kind, endpoint, body, button, label }) {
  const prefix = `${kind}-import`;
  setButtonBusy(button, true, 'Importing...');
  setProgress(prefix, { show: true, status: `Starting ${label} import...`, pct: 0, variant: null });

  let polling = true;
  const poll = async () => {
    while (polling) {
      try {
        const progress = await api('/api/dataset/import-progress');
        if (polling && progress && progress.active) {
          const pct = progress.total_count > 0 ? (progress.extracted_count / progress.total_count) * 100 : 0;
          setProgress(prefix, { status: cleanText(progress.status) || 'Importing...', pct });
        }
      } catch { /* keep trying */ }
      await sleep(800);
    }
  };
  poll();

  try {
    const response = await api(endpoint, { method: 'POST', body: JSON.stringify(body) });
    polling = false;
    setProgress(prefix, { status: 'Done', pct: 100, variant: 'done' });
    showToast(cleanText(response.message) || 'Dataset imported', 'success');
    refreshDataset();
    setTimeout(() => setProgress(prefix, { show: false }), 2500);
    return true;
  } catch (err) {
    polling = false;
    setProgress(prefix, { status: err.message, variant: 'error' });
    showToast(`Import failed: ${err.message}`, 'error');
    return false;
  } finally {
    setButtonBusy(button, false);
  }
}

async function importFromHF() {
  const speaker = getSpeakerName();
  if (!speaker) return;
  const repoId = $('hf-repo-id').value.trim();
  if (!repoId) {
    showToast('Enter a Hugging Face dataset ID or URL', 'warning');
    $('hf-repo-id').focus();
    return;
  }
  await runImport({
    kind: 'hf',
    endpoint: '/api/dataset/import-hf',
    body: { repo_id: repoId, speaker },
    button: $('btn-import-hf'),
    label: 'Hugging Face',
  });
}

async function loadKaggleConfig() {
  try {
    const config = await api('/api/dataset/kaggle-config');
    if (config && config.configured) {
      if (config.username && !$('kaggle-username').value) $('kaggle-username').value = config.username;
      if (config.key_configured) {
        $('kaggle-key').value = '';
        $('kaggle-key').placeholder = 'Saved. Leave blank to keep it';
        $('kaggle-key-hint').textContent = 'Using the key saved in ~/.kaggle/kaggle.json. Enter a new one to replace it.';
      }
    }
  } catch {
    // optional
  }
}

async function importFromKaggle() {
  const speaker = getSpeakerName();
  if (!speaker) return;
  const datasetRef = $('kaggle-dataset-ref').value.trim();
  if (!datasetRef) {
    showToast('Enter a Kaggle dataset reference or URL', 'warning');
    $('kaggle-dataset-ref').focus();
    return;
  }
  const ok = await runImport({
    kind: 'kaggle',
    endpoint: '/api/dataset/import-kaggle',
    body: {
      dataset_ref: datasetRef,
      username: $('kaggle-username').value.trim(),
      key: $('kaggle-key').value.trim(),
      speaker,
    },
    button: $('btn-import-kaggle'),
    label: 'Kaggle',
  });
  if (ok) loadKaggleConfig();
}

async function uploadFiles(files) {
  const speaker = getSpeakerName();
  if (!speaker) return;

  let done = 0;
  let succeeded = 0;
  const failures = [];
  setProgress('upload', { show: true, status: `Uploading ${files.length} file(s) to ${speaker}...`, pct: 0, variant: null });

  for (const file of files) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('speaker', speaker);
      let response;
      try {
        response = await fetch(`${CONFIG.apiUrl}/api/dataset/upload`, { method: 'POST', body: formData });
      } catch {
        setServerStatus(false);
        throw new Error('server unreachable');
      }
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(cleanText(errorDetail(errData, response.status)));
      }
      succeeded++;
    } catch (err) {
      failures.push(`${file.name}: ${err.message}`);
    }
    done++;
    setProgress('upload', { status: `Uploaded ${done} of ${files.length}`, pct: (done / files.length) * 100 });
  }

  if (failures.length) {
    setProgress('upload', { status: `${failures.length} of ${files.length} failed`, variant: 'error' });
    showToast(`Upload failed for ${failures.length} file(s). ${failures[0]}`, 'error');
  } else {
    setProgress('upload', { status: `Uploaded ${files.length} file(s) to ${speaker}`, variant: 'done' });
    showToast(`Uploaded ${files.length} file(s) to ${speaker}`, 'success');
  }
  setTimeout(() => setProgress('upload', { show: false }), 3000);
  if (succeeded > 0) refreshDataset();
}

async function refreshDataset() {
  try {
    const data = await api('/api/dataset/list');
    state.speakers = data.speakers || [];
  } catch {
    // keep last known list
  }
  renderSpeakersList();
  const n = state.speakers.length;
  $('file-count-label').textContent = `${totalFiles().toLocaleString()} files · ${n} speaker${n === 1 ? '' : 's'}`;
  updateDashboardStats();
}

function renderSpeakersList() {
  const container = $('dataset-speakers-list');
  if (!state.speakers.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">${ICONS.mic}</div>
        <h3>No datasets yet</h3>
        <p>Add audio above to create your first speaker.</p>
      </div>`;
    return;
  }

  container.innerHTML = state.speakers.map(sp => `
    <div class="list-row">
      <div class="list-row-main">
        <div class="list-row-icon is-accent">${ICONS.mic}</div>
        <div>
          <div class="list-row-title">${escapeHtml(sp.name)}</div>
          <div class="list-row-meta">${(sp.file_count || 0).toLocaleString()} WAV files${sp.duration ? ` · ${formatDuration(sp.duration)}` : ''}${sp.total_size ? ` · ${formatBytes(sp.total_size)}` : ''}</div>
        </div>
      </div>
      <div class="list-row-actions">
        <button type="button" class="btn btn-sm btn-secondary" data-action="train-speaker" data-name="${escapeAttr(sp.name)}">Train</button>
        <button type="button" class="btn btn-sm btn-danger" data-action="delete-speaker" data-name="${escapeAttr(sp.name)}">${ICONS.trash}Delete</button>
      </div>
    </div>
  `).join('');
}

$('dataset-speakers-list').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-action]');
  if (!btn) return;
  const name = btn.dataset.name;
  if (btn.dataset.action === 'delete-speaker') deleteSpeaker(name);
  if (btn.dataset.action === 'train-speaker') {
    navigateTo('training');
    refreshTrainingSpeakers(name);
  }
});

async function deleteSpeaker(name) {
  const ok = await confirmDialog({
    title: `Delete "${name}"?`,
    message: 'This deletes the speaker and all of its audio files from this computer.',
    confirmLabel: 'Delete speaker',
  });
  if (!ok) return;
  try {
    await api(`/api/dataset/${encodeURIComponent(name)}`, { method: 'DELETE' });
    showToast(`Deleted ${name}`, 'success');
    refreshDataset();
  } catch (err) {
    showToast(`Could not delete ${name}: ${err.message}`, 'error');
  }
}

// ---------- Training: setup ----------
async function refreshTrainingSpeakers(preselect) {
  const select = $('train-speaker');
  const previous = preselect || select.value;
  try {
    const data = await api('/api/dataset/list');
    state.speakers = data.speakers || [];
    select.innerHTML = `<option value="">${state.speakers.length ? 'Select a speaker' : 'No speakers yet: add a dataset first'}</option>` +
      state.speakers.map(s => `<option value="${escapeAttr(s.name)}">${escapeHtml(s.name)} (${(s.file_count || 0).toLocaleString()} files)</option>`).join('');
    if (previous && state.speakers.some(s => s.name === previous)) {
      select.value = previous;
      if (preselect) autoConfigureParams();
    }
  } catch {
    // leave as-is
  }
  if (!state.training.active) checkTrainingStatus();
}

function toggleAdvancedSettings(force) {
  const block = $('advanced-settings-block');
  const btn = $('btn-toggle-advanced');
  const open = typeof force === 'boolean' ? force : block.hidden;
  block.hidden = !open;
  btn.setAttribute('aria-expanded', String(open));
  btn.querySelector('span').textContent = open ? 'Hide advanced settings' : 'Show advanced settings';
}

async function autoConfigureParams(manual = false) {
  const speaker = $('train-speaker').value;
  if (!speaker) {
    $('config-empty-state').hidden = false;
    $('config-tuning-details').hidden = true;
    $('steps-recommended-badge').textContent = 'Auto: -';
    if (manual) showToast('Select a speaker first', 'warning');
    return;
  }

  try {
    const config = await api('/api/train/auto-config');
    state.autoConfig = config;

    $('steps-recommended-badge').textContent = `Auto: ${Number(config.max_steps).toLocaleString()}`;
    $('train-steps').value = config.max_steps;
    $('train-batch').value = config.batch_size;
    $('train-grad-accum').value = config.gradient_accumulation_steps;
    $('train-workers').value = config.num_workers;
    $('train-mixed-precision').checked = !!config.mixed_precision;

    $('config-empty-state').hidden = true;
    $('config-tuning-details').hidden = false;
    $('config-val-device').textContent = computeLabel(config.device);
    $('config-val-batch').textContent = config.batch_size;
    $('config-val-accum').textContent = config.gradient_accumulation_steps;
    $('config-val-effective').textContent = config.effective_batch_size;

    let seconds;
    if (typeof config.estimated_time_minutes === 'number' && config.estimated_time_minutes > 0) {
      seconds = config.estimated_time_minutes * 60;
    } else {
      const rate = config.device === 'cuda' ? 6.0 : (config.device === 'mps' ? 3.5 : 0.8);
      seconds = config.max_steps / rate;
    }
    $('config-val-time').textContent = `~${formatDuration(seconds >= 600 ? Math.round(seconds / 60) * 60 : seconds)}`;

    const notes = (config.optimization_notes || []).map(cleanText).filter(Boolean);
    $('config-opt-notes').innerHTML = notes.map(note => `<div class="note-item">${escapeHtml(note)}</div>`).join('');

    if (manual) showToast('Applied auto-tuned settings', 'success');
  } catch (err) {
    showToast(`Auto-tuning failed: ${err.message}`, 'error');
  }
}

async function pollMemoryUsage() {
  try {
    const data = await api('/api/system/memory');
    const memBadge = $('mem-status-badge');
    memBadge.className = 'badge badge-live badge-dot';
    memBadge.textContent = 'Live';

    if (typeof data.ram_total_gb === 'number' && typeof data.ram_used_gb === 'number') {
      const pct = typeof data.ram_percent === 'number' ? data.ram_percent : (data.ram_used_gb / data.ram_total_gb) * 100;
      $('ram-usage-text').textContent = `${data.ram_used_gb.toFixed(1)} / ${data.ram_total_gb.toFixed(1)} GB · ${Math.round(pct)}%`;
      $('ram-progress-bar').style.width = `${Math.min(100, pct)}%`;
      $('ram-progress-bar').classList.toggle('is-hot', pct >= 90);
    }

    // GPU memory: CUDA reports gpu_* fields, Apple MPS reports mps_* fields.
    let shown = false;
    const total = data.ram_total_gb;
    if (data.gpu_backend === 'cuda' && typeof data.gpu_allocated_gb === 'number') {
      $('gpu-mem-title').textContent = 'GPU memory (CUDA)';
      const gpuTotal = data.gpu_total_gb || 0;
      $('mps-usage-text').textContent = gpuTotal ? `${data.gpu_allocated_gb.toFixed(2)} / ${gpuTotal.toFixed(1)} GB` : `${data.gpu_allocated_gb.toFixed(2)} GB allocated`;
      $('gpu-progress-bar').style.width = gpuTotal ? `${Math.min(100, (data.gpu_allocated_gb / gpuTotal) * 100)}%` : '0%';
      $('mps-driver-text').textContent = 'Allocated by the server process. Training runs in its own process.';
      shown = true;
    } else if (typeof data.mps_allocated_gb === 'number') {
      $('gpu-mem-title').textContent = 'GPU memory (MPS)';
      $('mps-usage-text').textContent = `${data.mps_allocated_gb.toFixed(2)} GB allocated`;
      const driver = typeof data.mps_driver_allocated_gb === 'number' ? data.mps_driver_allocated_gb : null;
      $('gpu-progress-bar').style.width = (driver != null && total) ? `${Math.min(100, (driver / total) * 100)}%` : '0%';
      $('mps-driver-text').textContent = driver != null
        ? `Driver: ${driver.toFixed(2)} GB, server process only. A training run's usage shows in System RAM.`
        : 'Server process only. A training run\'s usage shows in System RAM.';
      shown = true;
    }
    $('mps-mem-container').hidden = !shown;
    $('mps-mem-empty').hidden = shown;
  } catch {
    // offline; status dot already shows it
  }
}

// ---------- Training: run ----------
function trainingHelpFor(message) {
  const m = (message || '').toLowerCase();
  if (m.includes('not installed') || m.includes('not ready') || m.includes('not set up') || m.includes('setup') || (m.includes('not found') && m.includes('trainer'))) {
    return 'The Beatrice trainer package is missing or incomplete. Open <button type="button" class="icon-btn" data-go="settings">Settings</button> and click Download or update trainer, or run the start script again.';
  }
  if (m.includes('no wav')) {
    return 'Add audio for this speaker on the <button type="button" class="icon-btn" data-go="dataset">Dataset</button> page first.';
  }
  if (m.includes('out of memory') || m.includes('oom')) {
    return 'The GPU ran out of memory. Lower the batch size under Advanced settings and try again.';
  }
  return 'The console log below has the full output.';
}

function showTrainingError(message, { title = 'Training failed', hint, variant = 'error' } = {}) {
  $('training-error').className = `alert alert-${variant}`;
  $('training-error-title').textContent = title;
  $('training-error-message').textContent = message || 'Unknown error';
  $('training-error-hint').innerHTML = hint || trainingHelpFor(message);
  $('training-error').hidden = false;
}

function dismissTrainingError() {
  $('training-error').hidden = true;
}

$('training-error').addEventListener('click', (e) => {
  const go = e.target.closest('[data-go]');
  if (go) navigateTo(go.dataset.go);
});

async function startTraining() {
  const speaker = $('train-speaker').value;
  if (!speaker) {
    showToast('Select a speaker first', 'warning');
    $('train-speaker').focus();
    return;
  }

  const steps = parseInt($('train-steps').value, 10) || 10000;
  const batch = parseInt($('train-batch').value, 10) || 4;
  const lr = $('train-lr').value.trim() || '1e-4';
  const gradAccum = parseInt($('train-grad-accum').value, 10) || 1;
  const workersRaw = parseInt($('train-workers').value, 10);
  const workers = Number.isFinite(workersRaw) ? workersRaw : 0;
  const mixedPrecision = $('train-mixed-precision').checked;
  const compileModel = !!(state.autoConfig && state.autoConfig.compile_model);

  if (!Number.isFinite(Number(lr)) || Number(lr) <= 0) {
    showToast('Learning rate must be a positive number, for example 1e-4', 'warning');
    $('train-lr').focus();
    return;
  }

  dismissTrainingError();
  const btn = $('btn-start-training');
  setButtonBusy(btn, true, 'Starting...');
  addLog(`Starting training for ${speaker}: ${steps.toLocaleString()} steps, batch ${batch}, lr ${lr}, accumulation ${gradAccum}`, 'info');

  try {
    await api('/api/train/start', {
      method: 'POST',
      body: JSON.stringify({
        speaker,
        steps,
        batch_size: batch,
        learning_rate: lr,
        gradient_accumulation_steps: gradAccum,
        num_workers: workers,
        mixed_precision: mixedPrecision,
        compile_model: compileModel,
      }),
    });

    Object.assign(state.training, {
      active: true,
      speaker,
      totalSteps: steps,
      step: 0,
      loss: null,
      lossHistory: [],
      startTime: Date.now(),
      stepsPerSec: null,
      memoryUsageMb: null,
      stopRequested: false,
      error: null,
    });
    setTrainingPhase('running');
    updateTrainingUI();
    drawLossChart();
    startTrainingPoll();
    showToast(`Training started for ${speaker}`, 'success');
  } catch (err) {
    if (err.status === 409) {
      showToast('A training run is already in progress', 'warning');
      checkTrainingStatus();
      return;
    }
    const title = err.network ? 'Cannot reach the server' : 'Training could not start';
    showTrainingError(err.message, { title });
    showToast(`${title}: ${err.message}`, 'error');
    addLog(`Error: ${err.message}`, 'error');
  } finally {
    setButtonBusy(btn, false);
  }
}

async function stopTraining() {
  const ok = await confirmDialog({
    title: 'Stop training?',
    message: 'The current run ends now. Checkpoints that were already saved stay in the output folder.',
    confirmLabel: 'Stop training',
  });
  if (!ok) return;
  state.training.stopRequested = true;
  try {
    await api('/api/train/stop', { method: 'POST' });
    finishTraining('stopped');
  } catch (err) {
    state.training.stopRequested = false;
    showToast(`Could not stop training: ${err.message}`, 'error');
  }
}

const PHASES = {
  idle:      { label: 'Idle',     badge: '' },
  running:   { label: 'Running',  badge: 'live' },
  completed: { label: 'Complete', badge: 'success' },
  failed:    { label: 'Failed',   badge: 'danger' },
  stopped:   { label: 'Stopped',  badge: '' },
  ended:     { label: 'Ended early', badge: 'warning' },
};

function setTrainingPhase(phase) {
  state.training.phase = phase;
  const running = phase === 'running';
  state.training.active = running;

  $('btn-start-training').hidden = running;
  $('btn-stop-training').hidden = !running;
  $('nav-training-live').hidden = !running;
  $('training-progress-card').hidden = phase === 'idle';
  $('training-speaker-label').textContent = state.training.speaker ? state.training.speaker : '';

  const info = PHASES[phase] || PHASES.idle;
  const b = $('training-status-badge');
  b.className = ['badge', 'badge-dot', info.badge ? `badge-${info.badge}` : ''].filter(Boolean).join(' ');
  b.textContent = info.label;

  const bar = $('train-progress-bar');
  bar.classList.toggle('is-error', phase === 'failed');
  bar.classList.toggle('is-done', phase === 'completed');
  bar.classList.toggle('is-muted', phase === 'stopped' || phase === 'ended');
  $('train-progress-label').textContent = phase === 'running' ? 'Progress' : info.label;
}

let trainingPollTimer = null;
let trainingPollToken = 0;

function startTrainingPoll() {
  stopTrainingPoll();
  const token = ++trainingPollToken;
  const tick = async () => {
    if (token !== trainingPollToken) return;
    await pollTraining();
    if (token === trainingPollToken && state.training.phase === 'running') {
      trainingPollTimer = setTimeout(tick, CONFIG.pollInterval);
    }
  };
  trainingPollTimer = setTimeout(tick, 600);
}

function stopTrainingPoll() {
  trainingPollToken++;
  if (trainingPollTimer) clearTimeout(trainingPollTimer);
  trainingPollTimer = null;
}

function applyStatus(data) {
  const t = state.training;
  if (data.speaker) t.speaker = data.speaker;
  if (typeof data.total_steps === 'number' && data.total_steps > 0) t.totalSteps = data.total_steps;
  if (typeof data.step === 'number') t.step = data.step;
  if (data.loss != null) t.loss = data.loss;
  t.stepsPerSec = data.steps_per_sec != null ? data.steps_per_sec : t.stepsPerSec;
  t.memoryUsageMb = data.memory_usage_mb != null ? data.memory_usage_mb : t.memoryUsageMb;
  if (typeof data.start_time === 'number' && data.start_time > 0) t.startTime = data.start_time * 1000;
  t.serverPhase = data.phase || null;
  t.serverState = data.state || null;
  t.etaSeconds = typeof data.eta_seconds === 'number' ? data.eta_seconds : null;

  if (data.loss != null && typeof data.step === 'number') {
    const last = t.lossHistory[t.lossHistory.length - 1];
    if (!last || last.step !== data.step) {
      t.lossHistory.push({ step: data.step, loss: data.loss });
      if (t.lossHistory.length > 500) t.lossHistory = t.lossHistory.slice(-500);
    }
  }
}

function statusError(data) {
  const e = data && (data.error || data.error_message);
  if (!e) return null;
  return cleanText(typeof e === 'string' ? e : JSON.stringify(e));
}

async function pollTraining() {
  let data;
  try {
    data = await api('/api/train/status');
  } catch {
    return; // transient; keep polling
  }

  applyStatus(data);
  await refreshConsoleLogs();

  const error = statusError(data);
  if (data.active && !error) {
    updateTrainingUI();
    drawLossChart();
    return;
  }

  if (error) finishTraining('failed', error);
  else if (state.training.stopRequested || data.state === 'stopped') finishTraining('stopped');
  else if (data.completed === true || (data.completed == null && state.training.totalSteps > 0 && state.training.step >= state.training.totalSteps)) finishTraining('completed');
  else finishTraining('ended');
}

function finishTraining(phase, error) {
  stopTrainingPoll();
  const t = state.training;
  t.stopRequested = false;
  t.error = error || null;
  setTrainingPhase(phase);
  updateTrainingUI();
  drawLossChart();

  const at = `step ${t.step.toLocaleString()} of ${t.totalSteps.toLocaleString()}`;
  switch (phase) {
    case 'completed':
      showToast('Training complete. Your model is on the Models page', 'success');
      addLog(`Training complete (${at}). Download the model from the Models page.`, 'success');
      refreshModels();
      break;
    case 'failed':
      showTrainingError(error);
      showToast(`Training failed: ${error}`, 'error');
      addLog(`Training failed at ${at}: ${error}`, 'error');
      break;
    case 'stopped':
      showToast('Training stopped', 'warning');
      addLog(`Training stopped by you at ${at}.`, 'warn');
      break;
    case 'ended':
      showTrainingError(`The trainer exited at ${at} without reporting an error.`, {
        title: 'Training ended early',
        hint: 'Check the console log below for the last messages from the trainer.',
        variant: 'warning',
      });
      showToast('Training ended before the last step', 'warning');
      addLog(`Training ended early at ${at}.`, 'warn');
      break;
  }
}

async function checkTrainingStatus() {
  try {
    const data = await api('/api/train/status');
    const error = statusError(data);
    if (data.active && !error) {
      applyStatus(data);
      if (state.training.phase !== 'running') {
        setTrainingPhase('running');
        startTrainingPoll();
      }
      updateTrainingUI();
      drawLossChart();
    } else if (error && state.training.phase === 'idle') {
      // Show the last failed run once, so a crash while the page was closed isn't silent.
      applyStatus(data);
      state.training.error = error;
      setTrainingPhase('failed');
      updateTrainingUI();
      showTrainingError(error, { title: 'Last training run failed' });
      refreshConsoleLogs();
    }
  } catch {
    // not reachable
  }
}

function logType(line) {
  const l = line.toLowerCase();
  if (/(traceback|error|exception|failed|fatal|killed)/.test(l)) return 'error';
  if (/warn/.test(l)) return 'warn';
  if (/(complete|finished|saved|exported|success)/.test(l)) return 'success';
  return 'info';
}

async function refreshConsoleLogs() {
  try {
    const logData = await api('/api/train/console-logs');
    if (!logData || !Array.isArray(logData.logs) || !logData.logs.length) return;
    const log = $('training-log');
    const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.innerHTML = logData.logs
      .map(line => cleanText(String(line)))
      .filter(line => line.length)
      .slice(-CONFIG.maxLogLines)
      .map(line => `<div class="log-line log-${logType(line)}">${escapeHtml(line)}</div>`)
      .join('');
    if (nearBottom) log.scrollTop = log.scrollHeight;
  } catch {
    // ignore
  }
}

function updateTrainingUI() {
  const { step, totalSteps, loss, startTime, stepsPerSec, phase } = state.training;
  const pct = totalSteps > 0 ? Math.min(100, (step / totalSteps) * 100) : 0;

  $('train-current-step').textContent = step.toLocaleString();
  $('train-total-steps-label').textContent = `/ ${totalSteps.toLocaleString()}`;
  $('train-loss').textContent = loss != null ? Number(loss).toFixed(4) : '-';
  $('train-progress-pct').textContent = `${pct.toFixed(1)}%`;
  $('train-progress-bar').style.width = `${pct}%`;

  const elapsed = startTime ? (Date.now() - startTime) / 1000 : 0;
  const measured = (elapsed > 0 && step > 0) ? step / elapsed : null;
  const rate = stepsPerSec != null ? stepsPerSec : measured;

  $('train-speed').textContent = rate != null && step > 0 ? rate.toFixed(2) : '-';
  if (phase === 'running') {
    const sp = state.training.serverPhase;
    $('train-progress-label').textContent = (state.training.serverState === 'starting' || (step === 0 && (!sp || /start|init|prepar|load/i.test(sp))))
      ? 'Starting the trainer...' : 'Progress';
  }
  if (phase === 'running' && state.training.etaSeconds != null && step > 0) {
    $('train-eta').textContent = formatDuration(state.training.etaSeconds);
  } else if (phase === 'running' && rate && rate > 0 && step > 0) {
    $('train-eta').textContent = formatDuration((totalSteps - step) / rate);
  } else {
    $('train-eta').textContent = phase === 'completed' ? 'Done' : '-';
  }
}

// ---------- Loss Chart (Canvas) ----------
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgba(hex, alpha) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return `rgba(124, 140, 255, ${alpha})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function drawLossChart() {
  const canvas = $('loss-chart');
  if (!canvas) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = rect.width;
  const h = rect.height;
  const pad = { top: 22, right: 22, bottom: 30, left: 58 };
  const signal = cssVar('--signal') || '#7c8cff';
  const textLo = cssVar('--text-lo') || '#82828e';
  const grid = cssVar('--chart-grid') || 'rgba(255,255,255,0.06)';
  const mono = '11px "Geist Mono", ui-monospace, monospace';

  ctx.clearRect(0, 0, w, h);

  const data = state.training.lossHistory;
  if (data.length < 2) {
    ctx.fillStyle = textLo;
    ctx.font = '13px Geist, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(state.training.phase === 'running' ? 'Waiting for the first loss values...' : 'No loss values recorded for this run', w / 2, h / 2);
    return;
  }

  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const minStep = data[0].step;
  const maxStep = data[data.length - 1].step;
  const losses = data.map(d => d.loss);
  let minLoss = Math.min(...losses);
  let maxLoss = Math.max(...losses);
  const span = (maxLoss - minLoss) || Math.abs(maxLoss) || 1;
  minLoss -= span * 0.08;
  maxLoss += span * 0.08;

  const x = (v) => pad.left + ((v - minStep) / ((maxStep - minStep) || 1)) * plotW;
  const y = (v) => pad.top + (1 - (v - minLoss) / ((maxLoss - minLoss) || 1)) * plotH;

  // Grid + y labels
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = textLo;
  ctx.font = mono;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const gy = Math.round(pad.top + (plotH / 4) * i) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, gy);
    ctx.lineTo(w - pad.right, gy);
    ctx.stroke();
    const val = maxLoss - ((maxLoss - minLoss) / 4) * i;
    ctx.fillText(val.toFixed(3), pad.left - 10, gy);
  }

  // Area
  const gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
  gradient.addColorStop(0, hexToRgba(signal, 0.22));
  gradient.addColorStop(1, hexToRgba(signal, 0));
  ctx.beginPath();
  ctx.moveTo(x(data[0].step), h - pad.bottom);
  data.forEach(d => ctx.lineTo(x(d.step), y(d.loss)));
  ctx.lineTo(x(maxStep), h - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Line
  ctx.beginPath();
  data.forEach((d, i) => (i === 0 ? ctx.moveTo(x(d.step), y(d.loss)) : ctx.lineTo(x(d.step), y(d.loss))));
  ctx.strokeStyle = signal;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Last point
  const last = data[data.length - 1];
  ctx.beginPath();
  ctx.arc(x(last.step), y(last.loss), 3.5, 0, Math.PI * 2);
  ctx.fillStyle = signal;
  ctx.fill();

  // X labels
  ctx.fillStyle = textLo;
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  ctx.fillText(`step ${minStep.toLocaleString()}`, pad.left, h - 10);
  ctx.textAlign = 'right';
  ctx.fillText(`step ${maxStep.toLocaleString()}`, w - pad.right, h - 10);
}

// ---------- Models ----------
async function refreshModels() {
  try {
    const data = await api('/api/models/list');
    state.models = data.models || [];
  } catch {
    // keep last known list
  }
  renderModelsList();
  updateDashboardStats();
}

function renderModelsList() {
  const container = $('models-list');
  if (!state.models.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">${ICONS.box}</div>
        <h3>No trained models yet</h3>
        <p>Finished training runs show up here.</p>
      </div>`;
    return;
  }

  // Inference packages first: they're what the voice changer imports.
  const models = [...state.models].sort((a, b) => {
    const pa = a.name.startsWith('paraphernalia_') ? 0 : 1;
    const pb = b.name.startsWith('paraphernalia_') ? 0 : 1;
    return pa - pb || String(b.created || '').localeCompare(String(a.created || ''));
  });

  container.innerHTML = models.map(m => {
    const isPackage = m.name.startsWith('paraphernalia_');
    const displayName = isPackage ? m.name.replace(/^paraphernalia_/, '') : m.name;
    const folder = m.path && m.path.includes('/') ? m.path.split('/')[0] : '';
    const typeBadge = isPackage ? badge('Voice model', 'success') : badge('Checkpoint');
    const meta = [
      folder ? `speaker ${folder}` : '',
      m.size ? formatBytes(m.size) : '',
      m.created ? new Date(m.created).toLocaleString() : '',
    ].filter(Boolean).map(escapeHtml).join(' · ');

    return `
      <div class="list-row">
        <div class="list-row-main">
          <div class="list-row-icon ${isPackage ? 'is-accent' : ''}">${isPackage ? ICONS.box : ICONS.file}</div>
          <div>
            <div class="list-row-title">${escapeHtml(displayName)} ${typeBadge}</div>
            <div class="list-row-meta">${meta}</div>
          </div>
        </div>
        <div class="list-row-actions">
          <a href="${CONFIG.apiUrl}/api/models/download/${encodeURIComponent(m.path)}" class="btn btn-sm ${isPackage ? 'btn-primary' : 'btn-secondary'}" download>${ICONS.download}${isPackage ? 'Download ZIP' : 'Download'}</a>
          <button type="button" class="btn btn-sm btn-danger" data-action="delete-model" data-path="${escapeAttr(m.path)}">${ICONS.trash}Delete</button>
        </div>
      </div>`;
  }).join('');
}

$('models-list').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-action="delete-model"]');
  if (btn) deleteModel(btn.dataset.path);
});

async function deleteModel(path) {
  const ok = await confirmDialog({
    title: 'Delete this model?',
    message: `"${path}" is removed from the output folder. This can't be undone.`,
    confirmLabel: 'Delete model',
  });
  if (!ok) return;
  try {
    await api(`/api/models/${encodeURIComponent(path)}`, { method: 'DELETE' });
    showToast('Model deleted', 'success');
    refreshModels();
  } catch (err) {
    showToast(`Could not delete the model: ${err.message}`, 'error');
  }
}

// ---------- Console Log ----------
function addLog(msg, type = 'info') {
  const log = $('training-log');
  const line = document.createElement('div');
  const cls = type === 'warning' ? 'warn' : type;
  line.className = `log-line log-${cls}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${cleanText(msg)}`;
  log.appendChild(line);
  while (log.children.length > CONFIG.maxLogLines) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

function clearLog() {
  $('training-log').innerHTML = '<div class="log-line log-info">Log cleared.</div>';
}

// ---------- Toasts ----------
function showToast(message, type = 'info') {
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
  toast.innerHTML = `
    <span class="toast-icon">${ICONS[type] || ICONS.info}</span>
    <span class="toast-msg">${escapeHtml(cleanText(message))}</span>
    <button type="button" class="toast-close" aria-label="Dismiss">${ICONS.close}</button>`;

  const remove = () => {
    if (!toast.isConnected) return;
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 200);
  };
  toast.querySelector('.toast-close').addEventListener('click', remove);
  container.appendChild(toast);
  while (container.children.length > 4) container.firstChild.remove();
  setTimeout(remove, type === 'error' ? 8000 : 4000);
}

// ---------- Confirm dialog ----------
function confirmDialog({ title, message, confirmLabel = 'Delete', danger = true }) {
  return new Promise(resolve => {
    const modal = $('confirm-modal');
    const okBtn = $('confirm-ok');
    const cancelBtn = $('confirm-cancel');
    $('confirm-title').textContent = title;
    $('confirm-message').textContent = message;
    okBtn.textContent = confirmLabel;
    okBtn.className = `btn ${danger ? 'btn-danger-solid' : 'btn-primary'}`;
    const lastFocus = document.activeElement;
    modal.hidden = false;
    okBtn.focus();

    const close = (result) => {
      modal.hidden = true;
      okBtn.removeEventListener('click', onOk);
      cancelBtn.removeEventListener('click', onCancel);
      modal.removeEventListener('click', onBackdrop);
      document.removeEventListener('keydown', onKey);
      if (lastFocus && lastFocus.focus) lastFocus.focus();
      resolve(result);
    };
    const onOk = () => close(true);
    const onCancel = () => close(false);
    const onBackdrop = (e) => { if (e.target === modal) close(false); };
    const onKey = (e) => {
      if (e.key === 'Escape') close(false);
      if (e.key === 'Tab') {
        e.preventDefault();
        (document.activeElement === okBtn ? cancelBtn : okBtn).focus();
      }
    };
    okBtn.addEventListener('click', onOk);
    cancelBtn.addEventListener('click', onCancel);
    modal.addEventListener('click', onBackdrop);
    document.addEventListener('keydown', onKey);
  });
}

// ---------- Settings ----------
function renderSettings() {
  const data = state.status;
  const list = $('settings-trainer-dir');
  const b = $('settings-trainer-badge');
  $('api-url').value = CONFIG.apiUrl;
  if (!data) {
    b.className = 'badge';
    b.textContent = state.serverOnline === false ? 'Offline' : 'Checking';
    return;
  }
  b.className = `badge ${data.beatrice_installed ? 'badge-success' : 'badge-danger'}`;
  b.textContent = data.beatrice_installed ? 'Installed' : 'Not installed';
  const rows = [
    ['Status', data.beatrice_installed ? 'Installed' : 'Not installed'],
    ['Datasets', data.data_dir || '-'],
    ['Output', data.output_dir || '-'],
    ['Python', data.python || '-'],
    ['PyTorch', data.torch_version || 'not installed'],
  ];
  list.innerHTML = rows.map(([k, v]) => `<div class="kv-row"><span>${escapeHtml(k)}</span><strong>${escapeHtml(String(v))}</strong></div>`).join('');
}

function testConnection() {
  const url = $('api-url').value.trim().replace(/\/+$/, '');
  if (!/^https?:\/\//.test(url)) {
    showToast('Enter a full URL, for example http://localhost:8000', 'warning');
    return;
  }
  CONFIG.apiUrl = url;
  writeStorage('beatrice_api_url', url);
  checkServerStatus().then(data => {
    if (data) {
      showToast('Connected to the trainer server', 'success');
      refreshDashboard();
    } else {
      showToast(`Cannot reach ${url}`, 'error');
    }
  });
}

async function setupBeatriceTrainer() {
  const btn = $('btn-setup-trainer');
  setButtonBusy(btn, true, 'Downloading...');
  showToast('Downloading the Beatrice trainer from Hugging Face. This can take a few minutes.', 'info');
  try {
    const data = await api('/api/setup/beatrice', { method: 'POST' });
    showToast(cleanText(data.message) || 'Beatrice trainer is ready', 'success');
    await checkServerStatus();
  } catch (err) {
    showToast(`Trainer download failed: ${err.message}`, 'error');
  } finally {
    setButtonBusy(btn, false);
  }
}

async function exportDatasetZip() {
  const btn = $('btn-export-zip');
  setButtonBusy(btn, true, 'Packaging...');
  try {
    let response;
    try {
      response = await fetch(`${CONFIG.apiUrl}/api/dataset/export`);
    } catch {
      setServerStatus(false);
      throw new Error('cannot reach the trainer server');
    }
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(cleanText(errorDetail(errData, response.status)));
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'beatrice_dataset.zip';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast(`Exported beatrice_dataset.zip (${formatBytes(blob.size)})`, 'success');
  } catch (err) {
    showToast(`Export failed: ${err.message}`, 'error');
  } finally {
    setButtonBusy(btn, false);
  }
}

async function clearAllData() {
  const ok = await confirmDialog({
    title: 'Delete all local data?',
    message: 'Every dataset and every trained model on this computer is deleted. This can\'t be undone.',
    confirmLabel: 'Delete everything',
  });
  if (!ok) return;
  try {
    await api('/api/reset', { method: 'POST' });
    showToast('All datasets and models deleted', 'success');
    state.speakers = [];
    state.models = [];
    refreshDashboard();
    refreshDataset();
    refreshModels();
  } catch (err) {
    showToast(`Reset failed: ${err.message}`, 'error');
  }
}

// ---------- Utilities ----------
function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function escapeAttr(str) {
  return escapeHtml(str).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Strips emoji and variation selectors from text that comes from the server,
// so the UI chrome stays emoji-free.
function cleanText(str) {
  return String(str ?? '')
    .replace(/[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}️‍⃣]/gu, '')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '-';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return s ? `${m}m ${s}s` : `${m}m`;
  }
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1073741824).toFixed(2)} GB`;
}

// ---------- Colour mode (Dark / Light / System) ----------
const colorModeQuery = window.matchMedia('(prefers-color-scheme: light)');

function resolveMode(mode) {
  if (mode === 'system') return colorModeQuery.matches ? 'light' : 'dark';
  return mode === 'light' ? 'light' : 'dark';
}

function applyColorMode(mode) {
  document.documentElement.setAttribute('data-color-mode', resolveMode(mode));
  document.querySelectorAll('.color-mode-btn').forEach(btn => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
  requestAnimationFrame(drawLossChart);
}

function setColorMode(mode) {
  writeStorage('beatrice_theme', mode);
  applyColorMode(mode);
}

// Kept for older bookmarks/snippets: flips between dark and light.
function toggleTheme() {
  setColorMode(document.documentElement.getAttribute('data-color-mode') === 'light' ? 'dark' : 'light');
}

colorModeQuery.addEventListener('change', () => {
  if (readStorage('beatrice_theme', 'dark') === 'system') applyColorMode('system');
});

applyColorMode(readStorage('beatrice_theme', 'dark'));

// ---------- Init ----------
function setupMobileNav() {
  const toggleBtn = $('menu-toggle-btn');
  const sidebar = $('sidebar');
  const setOpen = (open) => {
    sidebar.classList.toggle('open', open);
    toggleBtn.classList.toggle('open', open);
    toggleBtn.setAttribute('aria-expanded', String(open));
    toggleBtn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  };
  toggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    setOpen(!sidebar.classList.contains('open'));
  });
  document.addEventListener('click', (e) => {
    if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && !toggleBtn.contains(e.target)) setOpen(false);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar.classList.contains('open')) setOpen(false);
  });
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => { if (window.innerWidth <= 860) setOpen(false); });
  });
}

async function init() {
  setupDropZone();
  setupMobileNav();
  $('api-url').value = CONFIG.apiUrl;

  const initial = window.location.hash.replace('#', '');
  navigateTo(PAGES.includes(initial) ? initial : 'dashboard');

  const status = await checkServerStatus();
  if (status) {
    pollMemoryUsage();
    loadKaggleConfig();
    checkTrainingStatus();
    refreshDashboard();
  }

  setInterval(checkServerStatus, 10000);
  setInterval(() => {
    if (state.serverOnline && !document.hidden) pollMemoryUsage();
  }, 3000);

  if ('ResizeObserver' in window) {
    new ResizeObserver(() => drawLossChart()).observe($('loss-chart').parentElement);
  } else {
    window.addEventListener('resize', drawLossChart);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

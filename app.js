/* ============================================
   Beatrice V2 Trainer — Frontend Application
   ============================================ */

// ---------- Configuration ----------
const defaultApiUrl = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
  ? window.location.origin
  : 'http://localhost:8000';

const CONFIG = {
  apiUrl: localStorage.getItem('beatrice_api_url') || defaultApiUrl,
  pollInterval: 2000,
  maxLogLines: 200,
};

// ---------- State ----------
const state = {
  currentPage: 'dashboard',
  serverOnline: false,
  training: {
    active: false,
    step: 0,
    totalSteps: 10000,
    loss: null,
    lossHistory: [],
    startTime: null,
    stepsPerSec: null,
    memoryUsageMb: null,
  },
  speakers: [],
  models: [],
  uploadQueue: [],
};

// ---------- Navigation ----------
function navigateTo(page) {
  // Update state
  state.currentPage = page;

  // Update nav items
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === page);
  });

  // Update page views
  document.querySelectorAll('.page-view').forEach(view => {
    view.classList.toggle('active', view.id === `page-${page}`);
  });

  // Page-specific init
  switch (page) {
    case 'dashboard': refreshDashboard(); break;
    case 'dataset': refreshDataset(); break;
    case 'training': refreshTrainingSpeakers(); break;
    case 'models': refreshModels(); break;
  }
}

// Set up nav clicks
document.querySelectorAll('.nav-item[data-page]').forEach(item => {
  item.addEventListener('click', () => navigateTo(item.dataset.page));
});

// ---------- API Helper ----------
async function api(endpoint, options = {}) {
  try {
    const url = `${CONFIG.apiUrl}${endpoint}`;
    const response = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      setServerStatus(false);
    }
    throw err;
  }
}

// ---------- Server Status ----------
function setServerStatus(online) {
  state.serverOnline = online;
  const dot = document.getElementById('server-status-dot');
  const text = document.getElementById('server-status-text');
  if (online) {
    dot.classList.remove('offline');
    text.textContent = 'Server Online';
  } else {
    dot.classList.add('offline');
    text.textContent = 'Server Offline';
  }
}

async function checkServerStatus() {
  try {
    const data = await api('/api/status');
    setServerStatus(true);
    updateSystemInfo(data);
    return data;
  } catch {
    setServerStatus(false);
    return null;
  }
}

function updateSystemInfo(data) {
  if (!data) return;
  const el = (id) => document.getElementById(id);
  el('stat-device').textContent = data.device || 'System';
  el('stat-device-sub').textContent = data.compute || 'CPU';
  el('sys-platform').textContent = data.platform || 'System';
  el('sys-compute').textContent = data.compute || 'CPU';
  el('sys-beatrice').innerHTML = data.beatrice_installed 
    ? '<span class="badge badge-success" style="display:inline-flex; align-items:center; gap:4px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Installed</span>' 
    : '<span class="badge badge-danger" style="display:inline-flex; align-items:center; gap:4px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> Not found</span>';
  el('sys-server').innerHTML = '<span class="badge badge-success" style="display:inline-flex; align-items:center; gap:4px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Connected</span>';
}

async function refreshSystemInfo() {
  await checkServerStatus();
  showToast('System info refreshed', 'info');
}

// ---------- Dashboard ----------
async function refreshDashboard() {
  try {
    const status = await checkServerStatus();
    if (!status) return;

    // Update stats
    const dataset = await api('/api/dataset/list').catch(() => ({ speakers: [] }));
    const models = await api('/api/models/list').catch(() => ({ models: [] }));

    state.speakers = dataset.speakers || [];
    state.models = models.models || [];

    document.getElementById('stat-speakers').textContent = state.speakers.length;
    document.getElementById('stat-files').textContent = state.speakers.reduce((sum, s) => sum + (s.file_count || 0), 0);
    document.getElementById('stat-models').textContent = state.models.length;
  } catch {
    // Server likely offline
  }
}

// ---------- Dataset Management ----------
function setupDropZone() {
  const zone = document.getElementById('drop-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.wav') || f.name.endsWith('.zip'));
    if (files.length) uploadFiles(files);
    else showToast('Please drop WAV or ZIP files only', 'warning');
  });

  input.addEventListener('change', () => {
    const files = Array.from(input.files);
    if (files.length) uploadFiles(files);
    input.value = '';
  });

  // Update badges when speaker name changes
  const speakerInput = document.getElementById('speaker-name');
  speakerInput.addEventListener('input', () => {
    const val = speakerInput.value || 'unnamed';
    const activeBadge = document.getElementById('active-speaker-badge');
    if (activeBadge) activeBadge.textContent = `Speaker: ${val}`;
    const uploadBadge = document.getElementById('upload-speaker-badge');
    if (uploadBadge) uploadBadge.textContent = `Speaker: ${val}`;
    const hfBadge = document.getElementById('hf-speaker-badge');
    if (hfBadge) hfBadge.textContent = `Speaker: ${val}`;
    const kaggleBadge = document.getElementById('kaggle-speaker-badge');
    if (kaggleBadge) kaggleBadge.textContent = `Speaker: ${val}`;
  });
}

function switchImportMethod(method) {
  const methods = ['local', 'hf', 'kaggle'];
  methods.forEach(m => {
    const tab = document.getElementById(`tab-method-${m}`);
    const container = document.getElementById(`import-container-${m}`);
    if (tab) tab.classList.toggle('active', m === method);
    if (container) container.style.display = (m === method) ? 'block' : 'none';
  });
}

async function importFromHF() {
  const speaker = document.getElementById('speaker-name').value.trim();
  const repoId = document.getElementById('hf-repo-id').value.trim();
  const btn = document.getElementById('btn-import-hf');
  const progressContainer = document.getElementById('hf-import-progress-container');
  const progressStatus = document.getElementById('hf-import-status');
  const progressPct = document.getElementById('hf-import-progress-pct');
  const progressBar = document.getElementById('hf-import-progress-bar');

  if (!speaker) {
    showToast('Please enter a speaker name first', 'warning');
    return;
  }
  if (!repoId) {
    showToast('Please enter a Hugging Face repository ID or URL', 'warning');
    return;
  }

  showToast(`Starting Hugging Face download for "${repoId}"... Please wait.`, 'info');
  btn.disabled = true;
  btn.textContent = '📥 Importing...';

  if (progressContainer) {
    progressContainer.style.display = 'block';
    progressStatus.textContent = 'Initializing import...';
    progressPct.textContent = '0%';
    progressBar.style.width = '0%';
  }

  let pollInterval = null;
  const startPolling = () => {
    pollInterval = setInterval(async () => {
      try {
        const progress = await api('/api/dataset/import-progress');
        if (progress && progress.active) {
          progressStatus.textContent = progress.status || 'Importing...';
          let percent = 0;
          if (progress.total_count > 0) {
            percent = Math.min(100, Math.round((progress.extracted_count / progress.total_count) * 100));
          }
          progressPct.textContent = `${percent}%`;
          progressBar.style.width = `${percent}%`;
        }
      } catch (e) {
        console.error('Error polling import progress:', e);
      }
    }, 800);
  };

  startPolling();

  try {
    const response = await api('/api/dataset/import-hf', {
      method: 'POST',
      body: JSON.stringify({ repo_id: repoId, speaker: speaker }),
    });
    
    // Set progress elements to 100% on success
    if (progressStatus) progressStatus.textContent = 'Success!';
    if (progressPct) progressPct.textContent = '100%';
    if (progressBar) progressBar.style.width = '100%';
    
    showToast(response.message || 'Successfully imported dataset!', 'success');
    refreshDataset();
  } catch (err) {
    showToast(`Failed to import dataset: ${err.message}`, 'error');
  } finally {
    if (pollInterval) {
      clearInterval(pollInterval);
    }
    btn.disabled = false;
    btn.textContent = '⚡ Import Dataset';
    if (progressContainer) {
      setTimeout(() => {
        progressContainer.style.display = 'none';
      }, 2000);
    }
  }
}

async function loadKaggleConfig() {
  try {
    const config = await api('/api/dataset/kaggle-config');
    if (config && config.configured) {
      const usernameInput = document.getElementById('kaggle-username');
      const keyInput = document.getElementById('kaggle-key');
      if (usernameInput) usernameInput.value = config.username || '';
      if (keyInput && config.key_configured) {
        keyInput.value = '••••••••••••••••';
      }
    }
  } catch (err) {
    console.error('Failed to load Kaggle config:', err);
  }
}

async function importFromKaggle() {
  const speaker = document.getElementById('speaker-name').value.trim();
  const datasetRef = document.getElementById('kaggle-dataset-ref').value.trim();
  const username = document.getElementById('kaggle-username').value.trim();
  const key = document.getElementById('kaggle-key').value.trim();
  const btn = document.getElementById('btn-import-kaggle');
  const progressContainer = document.getElementById('kaggle-import-progress-container');
  const progressStatus = document.getElementById('kaggle-import-status');
  const progressPct = document.getElementById('kaggle-import-progress-pct');
  const progressBar = document.getElementById('kaggle-import-progress-bar');

  if (!speaker) {
    showToast('Please enter a speaker name first', 'warning');
    return;
  }
  if (!datasetRef) {
    showToast('Please enter a Kaggle dataset reference or URL', 'warning');
    return;
  }

  showToast(`Starting Kaggle download for "${datasetRef}"... Please wait.`, 'info');
  btn.disabled = true;
  btn.textContent = '📥 Importing...';

  if (progressContainer) {
    progressContainer.style.display = 'block';
    progressStatus.textContent = 'Initializing Kaggle import...';
    progressPct.textContent = '0%';
    progressBar.style.width = '0%';
  }

  let pollInterval = null;
  const startPolling = () => {
    pollInterval = setInterval(async () => {
      try {
        const progress = await api('/api/dataset/import-progress');
        if (progress && progress.active) {
          progressStatus.textContent = progress.status || 'Importing...';
          let percent = 0;
          if (progress.total_count > 0) {
            percent = Math.min(100, Math.round((progress.extracted_count / progress.total_count) * 100));
          }
          progressPct.textContent = `${percent}%`;
          progressBar.style.width = `${percent}%`;
        }
      } catch (e) {
        console.error('Error polling import progress:', e);
      }
    }, 800);
  };

  startPolling();

  try {
    const response = await api('/api/dataset/import-kaggle', {
      method: 'POST',
      body: JSON.stringify({
        dataset_ref: datasetRef,
        username: username,
        key: key,
        speaker: speaker
      }),
    });
    
    if (progressStatus) progressStatus.textContent = 'Success!';
    if (progressPct) progressPct.textContent = '100%';
    if (progressBar) progressBar.style.width = '100%';
    
    showToast(response.message || 'Successfully imported dataset!', 'success');
    refreshDataset();
    loadKaggleConfig();
  } catch (err) {
    showToast(`Failed to import dataset: ${err.message}`, 'error');
  } finally {
    if (pollInterval) {
      clearInterval(pollInterval);
    }
    btn.disabled = false;
    btn.textContent = '⚡ Import Dataset';
    if (progressContainer) {
      setTimeout(() => {
        progressContainer.style.display = 'none';
      }, 2000);
    }
  }
}

async function uploadFiles(files) {
  const speaker = document.getElementById('speaker-name').value.trim();
  if (!speaker) {
    showToast('Please enter a speaker name first', 'warning');
    return;
  }

  showToast(`Uploading ${files.length} file(s)...`, 'info');

  let successCount = 0;
  for (const file of files) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('speaker', speaker);

      const response = await fetch(`${CONFIG.apiUrl}/api/dataset/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      showToast(data.message, 'success');
      successCount++;
    } catch (err) {
      showToast(`Failed to upload ${file.name}: ${err.message}`, 'error');
    }
  }

  if (successCount > 0) {
    refreshDataset();
  }
}

async function refreshDataset() {
  try {
    const data = await api('/api/dataset/list');
    state.speakers = data.speakers || [];
    renderSpeakersList();
    document.getElementById('file-count-label').textContent =
      `${state.speakers.reduce((s, sp) => s + (sp.file_count || 0), 0)} files across ${state.speakers.length} speakers`;
    
    // Update dashboard stats in DOM if present
    const statSpeakers = document.getElementById('stat-speakers');
    const statFiles = document.getElementById('stat-files');
    if (statSpeakers) statSpeakers.textContent = state.speakers.length;
    if (statFiles) statFiles.textContent = state.speakers.reduce((sum, s) => sum + (s.file_count || 0), 0);
  } catch {
    // Render empty
    renderSpeakersList();
  }
}

function renderSpeakersList() {
  const container = document.getElementById('dataset-speakers-list');
  if (!state.speakers.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
        </div>
        <h3>No datasets uploaded yet</h3>
        <p class="text-muted">Upload WAV files above to create your speaker dataset.</p>
      </div>`;
    return;
  }

  container.innerHTML = state.speakers.map(sp => `
    <div class="model-card" style="display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); margin-bottom: 8px;">
      <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 36px; height: 36px; border-radius: var(--radius-sm); background: var(--accent-subtle); color: var(--accent-text); display: flex; align-items: center; justify-content: center;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
        </div>
        <div>
          <div class="model-name" style="font-weight: 600; color: var(--text-primary); font-size: 14px;">${escapeHtml(sp.name)}</div>
          <div class="model-meta" style="font-size: 12px; color: var(--text-muted);">${sp.file_count || 0} WAV files${sp.duration ? ` • ${formatDuration(sp.duration)}` : ''}</div>
        </div>
      </div>
      <button class="btn btn-sm btn-danger" onclick="deleteSpeaker('${escapeHtml(sp.name)}')">Delete</button>
    </div>
  `).join('');
}

async function deleteSpeaker(name) {
  if (!confirm(`Delete speaker "${name}" and all its audio files?`)) return;
  try {
    await api(`/api/dataset/${name}`, { method: 'DELETE' });
    showToast(`Deleted speaker: ${name}`, 'success');
    refreshDataset();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

// ---------- Training ----------
async function refreshTrainingSpeakers() {
  try {
    const data = await api('/api/dataset/list');
    state.speakers = data.speakers || [];
    const select = document.getElementById('train-speaker');
    select.innerHTML = '<option value="">— Select Speaker —</option>' +
      state.speakers.map(s => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.name)} (${s.file_count} files)</option>`).join('');
  } catch {
    // leave as-is
  }

  // Check if training is active
  try {
    const status = await api('/api/train/status');
    if (status.active) {
      state.training.active = true;
      showTrainingUI(true);
      startTrainingPoll();
    }
  } catch {
    // not training
  }
}

function toggleAdvancedSettings() {
  const block = document.getElementById('advanced-settings-block');
  const btn = document.getElementById('btn-toggle-advanced');
  if (block.style.display === 'none') {
    block.style.display = 'block';
    btn.textContent = '⚙️ Hide Advanced Hardware Settings';
  } else {
    block.style.display = 'none';
    btn.textContent = '⚙️ Show Advanced Hardware Settings';
  }
}

async function autoConfigureParams() {
  const speakerSelect = document.getElementById('train-speaker');
  const speaker = speakerSelect.value;
  if (!speaker) {
    document.getElementById('config-empty-state').style.display = 'block';
    document.getElementById('config-tuning-details').style.display = 'none';
    document.getElementById('steps-recommended-badge').textContent = 'Auto: —';
    return;
  }

  try {
    const config = await api('/api/train/auto-config');
    
    // Update recommended steps badge
    const stepsBadge = document.getElementById('steps-recommended-badge');
    if (stepsBadge) {
      stepsBadge.textContent = `Auto: ${config.max_steps.toLocaleString()}`;
    }
    document.getElementById('train-steps').value = config.max_steps;

    // Set advanced inputs
    document.getElementById('train-batch').value = config.batch_size;
    document.getElementById('train-grad-accum').value = config.gradient_accumulation_steps;
    document.getElementById('train-workers').value = config.num_workers;
    document.getElementById('train-mixed-precision').checked = config.mixed_precision;
    document.getElementById('train-compile').checked = config.compile_model;

    // Show tuning card details
    document.getElementById('config-empty-state').style.display = 'none';
    const details = document.getElementById('config-tuning-details');
    details.style.display = 'flex';
    details.style.flexDirection = 'column';

    document.getElementById('config-val-device').textContent = config.device.toUpperCase();
    document.getElementById('config-val-batch').textContent = config.batch_size;
    document.getElementById('config-val-accum').textContent = config.gradient_accumulation_steps;
    document.getElementById('config-val-effective').textContent = config.effective_batch_size;

    // Time estimate: based on steps and device speed
    const stepsPerSecEst = config.device === 'cuda' ? 6.0 : (config.device === 'mps' ? 3.5 : 1.2);
    document.getElementById('config-val-time').textContent = formatDuration(config.max_steps / stepsPerSecEst);

    // Render notes
    const notesContainer = document.getElementById('config-opt-notes');
    notesContainer.innerHTML = config.optimization_notes.map(note => `
      <div class="text-small" style="padding: 4px var(--space-sm); background: rgba(139, 92, 246, 0.05); border-left: 2px solid var(--accent-primary); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; color: var(--text-secondary); margin-bottom: 4px;">
        ${escapeHtml(note)}
      </div>
    `).join('');

    showToast('Applied hardware auto-tuning parameters.', 'success');
  } catch (err) {
    showToast(`Tuning failed: ${err.message}`, 'error');
  }
}

async function pollMemoryUsage() {
  try {
    const data = await api('/api/system/memory');
    
    // Update RAM Progress Bar and text
    const ramText = document.getElementById('ram-usage-text');
    const ramBar = document.getElementById('ram-progress-bar');
    if (ramText && ramBar) {
      ramText.textContent = `${data.ram_used_gb.toFixed(1)} GB / ${data.ram_total_gb.toFixed(1)} GB (${data.ram_percent}%)`;
      ramBar.style.width = `${data.ram_percent}%`;
    }

    // Update MPS allocations
    const mpsContainer = document.getElementById('mps-mem-container');
    const mpsEmpty = document.getElementById('mps-mem-empty');
    const mpsUsageText = document.getElementById('mps-usage-text');
    const mpsDriverText = document.getElementById('mps-driver-text');

    if (data.mps_allocated_gb !== undefined) {
      if (mpsContainer) mpsContainer.style.display = 'block';
      if (mpsEmpty) mpsEmpty.style.display = 'none';
      if (mpsUsageText) mpsUsageText.textContent = `${data.mps_allocated_gb.toFixed(2)} GB Allocated`;
      if (mpsDriverText) mpsDriverText.textContent = `Driver: ${data.mps_driver_allocated_gb.toFixed(2)} GB`;
    } else {
      if (mpsContainer) mpsContainer.style.display = 'none';
      if (mpsEmpty) mpsEmpty.style.display = 'block';
    }
  } catch (err) {
    // Silent ignore
  }
}

async function startTraining() {
  const speaker = document.getElementById('train-speaker').value;
  if (!speaker) {
    showToast('Please select a speaker', 'warning');
    return;
  }

  const steps = parseInt(document.getElementById('train-steps').value) || 10000;
  const batch = parseInt(document.getElementById('train-batch').value) || 4;
  const lr = document.getElementById('train-lr').value || '1e-4';
  const gradAccum = parseInt(document.getElementById('train-grad-accum').value) || 1;
  const workers = parseInt(document.getElementById('train-workers').value) || 0;
  const mixedPrecision = document.getElementById('train-mixed-precision').checked;
  const compileModel = document.getElementById('train-compile').checked;

  try {
    addLog('Starting training...', 'info');
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
        compile_model: compileModel
      }),
    });

    state.training.active = true;
    state.training.totalSteps = steps;
    state.training.step = 0;
    state.training.lossHistory = [];
    state.training.startTime = Date.now();
    state.training.stepsPerSec = null;
    state.training.memoryUsageMb = null;

    showTrainingUI(true);
    document.getElementById('train-total-steps-label').textContent = `/ ${steps.toLocaleString()}`;
    startTrainingPoll();
    showToast('Training started!', 'success');
    addLog(`Training started: ${speaker}, ${steps} steps, batch=${batch}, lr=${lr}, accum=${gradAccum}`, 'success');
  } catch (err) {
    showToast(`Failed to start training: ${err.message}`, 'error');
    addLog(`Error: ${err.message}`, 'error');
  }
}

async function stopTraining() {
  try {
    await api('/api/train/stop', { method: 'POST' });
    state.training.active = false;
    showTrainingUI(false);
    showToast('Training stopped', 'warning');
    addLog('Training stopped by user', 'warning');
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

function showTrainingUI(active) {
  document.getElementById('btn-start-training').style.display = active ? 'none' : '';
  document.getElementById('btn-stop-training').style.display = active ? '' : 'none';
  document.getElementById('training-progress-card').style.display = active ? '' : 'none';

  const badge = document.getElementById('training-status-badge');
  if (active) {
    badge.textContent = 'Running';
    badge.className = 'badge badge-success';
  } else {
    badge.textContent = 'Stopped';
    badge.className = 'badge badge-warning';
  }
}

let trainingPollId = null;

function startTrainingPoll() {
  if (trainingPollId) clearInterval(trainingPollId);
  trainingPollId = setInterval(pollTraining, CONFIG.pollInterval);
}

async function pollTraining() {
  try {
    const data = await api('/api/train/status');
    if (!data.active) {
      state.training.active = false;
      clearInterval(trainingPollId);
      trainingPollId = null;

      // Update state with final values
      state.training.step = data.step || state.training.totalSteps;
      state.training.loss = data.loss;
      state.training.stepsPerSec = data.steps_per_sec;
      state.training.memoryUsageMb = data.memory_usage_mb;

      showTrainingUI(false);

      const badge = document.getElementById('training-status-badge');
      badge.textContent = 'Complete';
      badge.className = 'badge badge-info';
      document.getElementById('training-progress-card').style.display = '';

      // Update UI to 100% complete
      updateTrainingUI();
      drawLossChart();

      showToast('🎉 Training complete!', 'success');
      addLog('Training completed successfully!', 'success');
      return;
    }

    state.training.step = data.step || 0;
    state.training.loss = data.loss;
    state.training.stepsPerSec = data.steps_per_sec;
    state.training.memoryUsageMb = data.memory_usage_mb;

    if (data.loss != null) {
      state.training.lossHistory.push({ step: data.step, loss: data.loss });
      if (state.training.lossHistory.length > 500) {
        state.training.lossHistory = state.training.lossHistory.slice(-500);
      }
    }

    updateTrainingUI();
    drawLossChart();

    // Fetch and render actual training console logs
    try {
      const logData = await api('/api/train/console-logs');
      if (logData && logData.logs && logData.logs.length) {
        const logContainer = document.getElementById('training-log');
        logContainer.innerHTML = logData.logs.map(line => {
          let type = 'info';
          const lower = line.toLowerCase();
          if (lower.includes('error') || lower.includes('fail') || lower.includes('runtimeerror') || lower.includes('valueerror')) {
            type = 'error';
          } else if (lower.includes('warn') || lower.includes('warning') || lower.includes('fallback')) {
            type = 'warning';
          } else if (lower.includes('step') || lower.includes('loss=') || lower.includes('completed') || lower.includes('success')) {
            type = 'success';
          }
          return `<div class="log-line log-${type}">${escapeHtml(line)}</div>`;
        }).join('');
        logContainer.scrollTop = logContainer.scrollHeight;
      }
    } catch (err) {
      // ignore
    }
  } catch {
    // ignore poll errors
  }
}

function updateTrainingUI() {
  const { step, totalSteps, loss, startTime, stepsPerSec, memoryUsageMb } = state.training;
  const pct = totalSteps > 0 ? Math.min(100, (step / totalSteps) * 100) : 0;

  document.getElementById('train-current-step').textContent = step.toLocaleString();
  document.getElementById('train-loss').textContent = loss != null ? loss.toFixed(4) : '—';
  document.getElementById('train-progress-pct').textContent = `${pct.toFixed(1)}%`;
  document.getElementById('train-progress-bar').style.width = `${pct}%`;

  // Calculate elapsed time and speed rate
  const elapsed = startTime ? (Date.now() - startTime) / 1000 : 0;
  const calculatedRate = (elapsed > 0 && step > 0) ? (step / elapsed) : null;
  const displayRate = stepsPerSec != null ? stepsPerSec : calculatedRate;

  // Update speed & memory
  document.getElementById('train-speed').textContent = displayRate != null ? `${displayRate.toFixed(2)} steps/s` : '—';
  
  const memCard = document.getElementById('train-process-mem-card');
  const memVal = document.getElementById('train-process-mem');
  if (memoryUsageMb != null) {
    if (memCard) memCard.style.display = '';
    if (memVal) memVal.textContent = `${memoryUsageMb.toFixed(1)} MB`;
  } else {
    if (memVal) memVal.textContent = '—';
  }

  // ETA
  if (displayRate && displayRate > 0) {
    const remaining = (totalSteps - step) / displayRate;
    document.getElementById('train-eta').textContent = formatDuration(remaining);
  } else {
    document.getElementById('train-eta').textContent = '—';
  }
}

// ---------- Loss Chart (Canvas) ----------
function drawLossChart() {
  const canvas = document.getElementById('loss-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const rect = canvas.parentElement.getBoundingClientRect();

  canvas.width = rect.width * 2;
  canvas.height = rect.height * 2;
  ctx.scale(2, 2);

  const w = rect.width;
  const h = rect.height;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };

  // Clear
  ctx.clearRect(0, 0, w, h);

  const data = state.training.lossHistory;
  if (data.length < 2) {
    ctx.fillStyle = '#64748b';
    ctx.font = '13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for training data...', w / 2, h / 2);
    return;
  }

  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const minStep = data[0].step;
  const maxStep = data[data.length - 1].step;
  const losses = data.map(d => d.loss);
  const minLoss = Math.min(...losses) * 0.95;
  const maxLoss = Math.max(...losses) * 1.05;

  const xScale = (v) => pad.left + ((v - minStep) / (maxStep - minStep || 1)) * plotW;
  const yScale = (v) => pad.top + (1 - (v - minLoss) / (maxLoss - minLoss || 1)) * plotH;

  // Grid lines
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }

  // Gradient fill
  const gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
  gradient.addColorStop(0, 'rgba(139, 92, 246, 0.2)');
  gradient.addColorStop(1, 'rgba(139, 92, 246, 0)');

  ctx.beginPath();
  ctx.moveTo(xScale(data[0].step), h - pad.bottom);
  data.forEach(d => ctx.lineTo(xScale(d.step), yScale(d.loss)));
  ctx.lineTo(xScale(data[data.length - 1].step), h - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Line
  ctx.beginPath();
  data.forEach((d, i) => {
    const x = xScale(d.step);
    const y = yScale(d.loss);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`Step ${minStep}`, pad.left, h - 8);
  ctx.fillText(`Step ${maxStep}`, w - pad.right, h - 8);

  ctx.textAlign = 'right';
  ctx.fillText(maxLoss.toFixed(3), pad.left - 6, pad.top + 4);
  ctx.fillText(minLoss.toFixed(3), pad.left - 6, h - pad.bottom + 4);
}

// ---------- Models ----------
async function refreshModels() {
  try {
    const data = await api('/api/models/list');
    state.models = data.models || [];
    renderModelsList();
    
    // Update dashboard stats in DOM if present
    const statModels = document.getElementById('stat-models');
    if (statModels) statModels.textContent = state.models.length;
  } catch {
    renderModelsList();
  }
}

function renderModelsList() {
  const container = document.getElementById('models-list');
  if (!state.models.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg>
        </div>
        <h3>No trained models yet</h3>
        <p class="text-muted">Start training a voice model to generate output weights here.</p>
      </div>`;
    return;
  }

  container.innerHTML = state.models.map(m => {
    const isParaphernalia = m.name.startsWith('paraphernalia_');
    const displayName = isParaphernalia ? m.name.replace('paraphernalia_', '') : m.name;
    const typeLabel = isParaphernalia 
      ? '<span class="badge badge-success">Inference Package</span>' 
      : '<span class="badge badge-info">Checkpoint</span>';

    return `
      <div class="model-card" style="display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <div style="width: 36px; height: 36px; border-radius: var(--radius-sm); background: var(--accent-subtle); color: var(--accent-text); display: flex; align-items: center; justify-content: center;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg>
          </div>
          <div>
            <div class="model-name" style="font-weight: 600; color: var(--text-primary); font-size: 14px; display: flex; align-items: center; gap: 8px;">
              ${escapeHtml(displayName)}
              ${typeLabel}
            </div>
            <div class="model-meta" style="font-size: 12px; color: var(--text-muted);">
              ${m.size ? formatBytes(m.size) : 'Unknown size'}
              ${m.created ? ` • ${new Date(m.created).toLocaleString()}` : ''}
            </div>
          </div>
        </div>
        <div style="display: flex; gap: 8px;">
          <a href="${CONFIG.apiUrl}/api/models/download/${encodeURIComponent(m.path)}" class="btn btn-sm btn-primary" download>Download</a>
          <button class="btn btn-sm btn-danger" onclick="deleteModel(decodeURIComponent('${encodeURIComponent(m.path)}'))">Delete</button>
        </div>
      </div>
    `;
  }).join('');
}

async function deleteModel(path) {
  if (!confirm(`Are you sure you want to delete "${path}"? This action cannot be undone.`)) return;
  try {
    await api(`/api/models/${encodeURIComponent(path)}`, { method: 'DELETE' });
    showToast('Model successfully deleted', 'success');
    refreshModels();
  } catch (err) {
    showToast(`Failed to delete model: ${err.message}`, 'error');
  }
}

// ---------- Training Log ----------
function addLog(msg, type = 'info') {
  const log = document.getElementById('training-log');
  const time = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.className = `log-line log-${type}`;
  line.textContent = `[${time}] ${msg}`;
  log.appendChild(line);

  // Trim
  while (log.children.length > CONFIG.maxLogLines) {
    log.removeChild(log.firstChild);
  }

  log.scrollTop = log.scrollHeight;
}

function clearLog() {
  document.getElementById('training-log').innerHTML = '<div class="log-line log-info">[INFO] Log cleared.</div>';
}

// ---------- Toasts ----------
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast';

  const icons = { 
    info: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00b4d8" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>', 
    success: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>', 
    warning: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>', 
    error: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>' 
  };
  const cleanMsg = message.replace(/^[✅❌⚠️ℹ️⚡]\s*/, '');
  toast.innerHTML = `<span style="display:flex;align-items:center;">${icons[type] || icons.info}</span><span>${escapeHtml(cleanMsg)}</span>`;

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 200);
  }, 3500);
}

// ---------- Settings ----------
function testConnection() {
  const url = document.getElementById('api-url').value.trim();
  CONFIG.apiUrl = url;
  localStorage.setItem('beatrice_api_url', url);
  checkServerStatus().then(data => {
    if (data) showToast('Connection successful!', 'success');
    else showToast('Cannot reach server', 'error');
  });
}

async function setupBeatriceTrainer() {
  try {
    showToast('Cloning Beatrice Trainer... this may take a minute.', 'info');
    const data = await api('/api/setup/beatrice', { method: 'POST' });
    showToast(data.message || 'Beatrice Trainer set up!', 'success');
    checkServerStatus();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

async function exportDatasetZip() {
  showToast('Exporting dataset as ZIP...', 'info');
  try {
    const response = await fetch(`${CONFIG.apiUrl}/api/dataset/export`);
    if (!response.ok) throw new Error('Export failed');
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'beatrice_dataset.zip';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Dataset exported!', 'success');
  } catch (err) {
    showToast(`Export failed: ${err.message}`, 'error');
  }
}

function clearAllData() {
  if (!confirm('Are you sure? This will delete all datasets and models.')) return;
  api('/api/reset', { method: 'POST' })
    .then(() => {
      showToast('All data cleared', 'success');
      refreshDashboard();
    })
    .catch(err => showToast(`Error: ${err.message}`, 'error'));
}

// ---------- Utilities ----------
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

// ---------- Theme (Light / Dark Mode) ----------
function initTheme() {
  const savedTheme = localStorage.getItem('beatrice_theme') || 'dark';
  applyTheme(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  applyTheme(newTheme);
  localStorage.setItem('beatrice_theme', newTheme);
  showToast(`Switched to ${newTheme === 'dark' ? 'Dark Mode' : 'Light Mode'}`, 'info');
}

function applyTheme(theme) {
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    const toggleText = document.getElementById('theme-toggle-text');
    if (toggleText) toggleText.textContent = 'Dark Mode';
  } else {
    document.documentElement.removeAttribute('data-theme');
    const toggleText = document.getElementById('theme-toggle-text');
    if (toggleText) toggleText.textContent = 'Light Mode';
  }
}

// Immediately apply theme setting to avoid flash of wrong theme
initTheme();

// ---------- Init ----------
document.addEventListener('DOMContentLoaded', () => {
  setupDatasetEvents();
  setupLossChart();
  setupDropZone();
  checkServerStatus().then(status => {
    if (status) {
      pollMemoryUsage();
      loadKaggleConfig();
    }
  });
  refreshDashboard();

  // Mobile sidebar navigation toggle
  const toggleBtn = document.getElementById('menu-toggle-btn');
  const sidebar = document.getElementById('sidebar');

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.toggle('open');
      toggleBtn.classList.toggle('open');
    });

    // Close sidebar on click outside
    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && e.target !== toggleBtn) {
        sidebar.classList.remove('open');
        toggleBtn.classList.remove('open');
      }
    });

    // Close sidebar when clicking a nav item on mobile
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', () => {
        if (window.innerWidth <= 1024) {
          sidebar.classList.remove('open');
          toggleBtn.classList.remove('open');
        }
      });
    });
  }

  // Periodic status check
  setInterval(checkServerStatus, 10000);

  // Periodic memory check
  setInterval(() => {
    if (state.serverOnline) {
      pollMemoryUsage();
    }
  }, 3000);
});

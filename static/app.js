const state = {
  token: localStorage.getItem('rag-token') || 'dev-local-token',
  docs: [],
  releases: [],
  activeRelease: null,
  session: null,
  lastResponse: null,
  lastTrace: null,
  selectedTraceId: null,
  evalRun: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const randomId = (prefix) => `${prefix}-${(globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`)}`;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function toast(message, isError = false) {
  const node = $('#toast');
  node.textContent = message;
  node.style.borderColor = isError ? 'var(--danger)' : 'var(--lime-deep)';
  node.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove('show'), 3200);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set('Authorization', `Bearer ${state.token}`);
  headers.set('X-Contract-Version', '1.0');
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  if (!response.ok) {
    const error = data.error || {};
    throw new Error(`${error.code || response.status}: ${error.message || 'Request failed'}`);
  }
  return data;
}

function switchView(viewName) {
  $$('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === viewName));
  $$('.view').forEach((view) => view.classList.toggle('active', view.id === `view-${viewName}`));
  if (viewName === 'documents') loadDocuments();
  if (viewName === 'test-sessions') loadSessions();
  if (viewName === 'traces') loadTraces();
  if (viewName === 'releases') loadReleases();
  if (viewName === 'evaluations') loadEvaluations();
}

function statusClass(status) {
  const value = String(status || '').toUpperCase();
  if (['READY', 'READY_FOR_TEST', 'PUBLISHED', 'ROLLED_BACK', 'ANSWERED', 'GOOD', 'ENABLED', 'OK', 'COMPLETED'].includes(value)) return 'status-ready';
  if (['FAILED', 'PROCESS_FAILED', 'ERROR', 'BAD', 'DISABLED'].includes(value)) return 'status-error';
  if (['PROCESSING', 'BUILDING', 'RUNNING', 'TESTING', 'DEGRADED', 'NOT_READY', 'STARTING', 'EVALUATING', 'INDEXING', 'WAITING_FOR_DOCUMENT'].includes(value)) return 'status-warn';
  return 'status-neutral';
}

function stateText(status) {
  const value = String(status || 'UNKNOWN');
  return `<span class="${statusClass(value)}">${escapeHtml(value)}</span>`;
}

async function loadHealth() {
  try {
    const health = await api('/rag-api/v1/health');
    const pill = $('#health-pill');
    pill.textContent = health.status;
    pill.className = `status-pill ${statusClass(health.status)}`;
    state.activeRelease = health.releaseId ? { release_id: health.releaseId, knowledge_version: health.knowledgeVersion } : null;
    $('#release-pill').textContent = health.knowledgeVersion || 'no active release';
    $('#metric-release').textContent = health.knowledgeVersion ? health.knowledgeVersion.slice(-12) : '--';
    $('#metric-release-note').textContent = health.releaseId || 'no release';
    renderProviders(health.providers || {});
  } catch (error) {
    const pill = $('#health-pill');
    pill.textContent = 'OFFLINE';
    pill.className = 'status-pill status-error';
    toast(error.message, true);
  }
}

async function loadSystemStatus() {
  try {
    const data = await api('/rag-admin-api/v1/system/status');
    const phase = data.phase || 'UNKNOWN';
    const progress = data.progress || {};
    const total = Number(progress.total || 0);
    const completed = Number(progress.completed || 0);
    const percent = total ? Math.round((completed / total) * 100) : (phase === 'READY' ? 100 : 0);
    $('#execution-pill').textContent = phase;
    $('#execution-pill').className = `section-chip ${statusClass(phase)}`;
    $('#execution-phase').textContent = phase;
    $('#execution-progress-label').textContent = total ? `${completed} / ${total}` : (phase === 'READY' ? 'complete' : 'pending');
    $('#execution-progress-bar').style.width = `${percent}%`;
    $('#execution-meta-text').textContent = total ? `${percent}% latest evaluation progress${progress.accuracy != null ? ` · ${(progress.accuracy * 100).toFixed(1)}% pass` : ''}` : `${data.counts?.documents || 0} documents · ${data.counts?.chunks || 0} chunks`;
    $('#execution-feedback-text').textContent = `${data.counts?.feedback || 0} reviewer feedback entries`;
  } catch (error) {
    $('#execution-pill').textContent = 'STATUS UNAVAILABLE';
    $('#execution-pill').className = 'section-chip status-error';
  }
}

function renderProviders(providers) {
  const entries = [
    ['embedding', providers.embedding],
    ['reranker', providers.reranker],
    ['generation', providers.generation],
    ['deepAgent', { provider: providers.deepAgent?.framework || 'deepagents', model: providers.deepAgent?.enabled ? 'bounded read-only tools' : 'disabled', ready: providers.deepAgent?.enabled !== false }],
    ['vectorStore', { provider: providers.vectorStore || 'unknown', model: 'local persistence', ready: true }],
  ];
  $('#provider-list').innerHTML = entries.map(([name, item = {}]) => {
    const fallback = item.fallback ? ' · fallback' : '';
    const ready = item.ready === false ? 'not loaded' : (item.ready === true ? 'ready' : 'configured');
    return `<div class="provider-row"><span class="provider-name">${escapeHtml(name)}</span><span class="provider-model">${escapeHtml(item.model || item.provider || 'not configured')}${escapeHtml(fallback)}</span><span class="provider-state ${item.fallback ? 'fallback' : ''}">${escapeHtml(ready)}</span></div>`;
  }).join('');
  $('#agent-badge').textContent = providers.deepAgent?.enabled ? 'bounded agent' : 'agent off';
  $('#api-provider-json').textContent = pretty(providers);
}

async function loadDocuments() {
  try {
    const data = await api('/rag-admin-api/v1/documents');
    state.docs = data.documents || [];
    const ready = state.docs.filter((item) => item.status === 'READY_FOR_TEST');
    const chunks = state.docs.reduce((sum, item) => sum + Number(item.processingResult?.chunkCount || 0), 0);
    $('#metric-documents').textContent = state.docs.length;
    $('#metric-documents-note').textContent = `${ready.length} ready for test`;
    $('#metric-chunks').textContent = chunks;
    $('#metric-chunks-note').textContent = 'current document versions';
    renderDocuments();
    renderCandidateDocs();
  } catch (error) { toast(error.message, true); }
}

function renderDocuments() {
  const node = $('#documents-table');
  if (!state.docs.length) { node.innerHTML = '<div class="empty-state">No documents yet.</div>'; return; }
  node.innerHTML = `<div class="table-row head"><span>Source</span><span>Version</span><span>Status</span><span>Chunks</span><span>Action</span></div>` + state.docs.map((doc) => `<div class="table-row"><span class="strong" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.title)}<small class="muted">${escapeHtml(doc.filename)}</small></span><span>v${doc.documentVersion}</span><span>${stateText(doc.status)}</span><span>${escapeHtml(doc.processingResult?.chunkCount ?? '--')}</span><button class="row-action" data-doc-id="${escapeHtml(doc.documentId)}">Inspect</button></div>`).join('');
  node.querySelectorAll('[data-doc-id]').forEach((button) => button.addEventListener('click', () => inspectDocument(button.dataset.docId)));
}

async function inspectDocument(documentId) {
  try {
    const doc = state.docs.find((item) => item.documentId === documentId);
    const data = await api(`/rag-admin-api/v1/documents/${encodeURIComponent(documentId)}/chunks?version=${doc?.documentVersion || ''}`);
    $('#chunk-title').textContent = doc?.title || documentId;
    $('#chunk-policy-badge').textContent = doc?.chunkPolicy || '--';
    $('#chunk-list').classList.remove('empty-state');
    $('#chunk-list').innerHTML = (data.chunks || []).map((chunk) => `<article class="chunk-item"><div class="chunk-meta"><span>${escapeHtml(chunk.chunk_id)} · #${Number(chunk.ordinal) + 1}</span><span>${escapeHtml(JSON.stringify(chunk.location || {}))}</span></div><p>${escapeHtml(chunk.excerpt || chunk.text)}</p></article>`).join('') || '<div class="empty-state">No chunks.</div>';
    switchView('documents');
  } catch (error) { toast(error.message, true); }
}

function renderCandidateDocs() {
  const node = $('#candidate-docs');
  const ready = state.docs.filter((doc) => doc.status === 'READY_FOR_TEST');
  if (!ready.length) { node.className = 'candidate-list empty-state'; node.textContent = 'No READY_FOR_TEST documents.'; return; }
  node.className = 'candidate-list';
  node.innerHTML = ready.map((doc) => `<label class="candidate-check"><input type="checkbox" value="${escapeHtml(doc.documentId)}" data-version="${doc.documentVersion}"><span>${escapeHtml(doc.title)} <small>v${doc.documentVersion} · ${escapeHtml(doc.chunkPolicy || '')}</small></span></label>`).join('');
}

async function loadSessions() {
  try {
    const data = await api('/rag-admin-api/v1/test-sessions?knowledge_base_id=main-business-kb');
    const sessions = data.sessions || [];
    const select = $('#session-select');
    select.innerHTML = '<option value="">Choose an existing session</option>' + sessions.map((session) => `<option value="${escapeHtml(session.testSessionId)}">${escapeHtml(session.testSessionId)} · ${escapeHtml(session.status)}</option>`).join('');
    const saved = localStorage.getItem('rag-session-id');
    if (saved && sessions.some((item) => item.testSessionId === saved)) {
      select.value = saved;
      await selectSession(saved);
    }
  } catch (error) { toast(error.message, true); }
}

async function selectSession(sessionId) {
  if (!sessionId) {
    state.session = null;
    $('#active-session-title').textContent = 'No session selected';
    $('#active-session-status').textContent = '--';
    $('#active-session-meta').textContent = '';
    $('#test-run').disabled = true;
    $('#release-create').disabled = true;
    $('#test-answer').className = 'test-answer empty-state';
    $('#test-answer').textContent = 'Test answers and reviewer decisions will appear here.';
    return;
  }
  try {
    const data = await api(`/rag-admin-api/v1/test-sessions/${encodeURIComponent(sessionId)}`);
    state.session = data;
    localStorage.setItem('rag-session-id', sessionId);
    renderSession();
  } catch (error) { toast(error.message, true); }
}

async function uploadDocument(event) {
  event.preventDefault();
  const file = $('#upload-file').files[0];
  if (!file) { toast('Choose a file first', true); return; }
  const requestId = randomId('upload');
  const metadata = {
    requestId,
    knowledgeBaseId: 'main-business-kb',
    title: $('#upload-title').value.trim() || undefined,
    category: $('#upload-category').value.trim() || undefined,
    sourceOwner: $('#upload-owner').value.trim() || undefined,
    chunkPolicyOverride: $('#upload-policy').value || undefined,
    uploadedBy: 'webui-operator',
  };
  const form = new FormData();
  form.append('file', file);
  form.append('metadata', JSON.stringify(metadata));
  try {
    const data = await api('/rag-admin-api/v1/documents', { method: 'POST', body: form });
    $('#upload-message').textContent = `${data.documentId} · PROCESSING`;
    toast('Document accepted');
    await loadDocuments();
    pollDocument(data.documentId);
  } catch (error) { $('#upload-message').textContent = error.message; toast(error.message, true); }
}

async function pollDocument(documentId) {
  for (let attempt = 0; attempt < 180; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    try {
      const data = await api(`/rag-admin-api/v1/documents/${encodeURIComponent(documentId)}`);
      $('#upload-message').textContent = `${data.documentId} · ${data.status} · ${data.progress}%`;
      await loadDocuments();
      if (!['PROCESSING', 'UPLOADED'].includes(data.status)) return;
    } catch (error) { $('#upload-message').textContent = error.message; return; }
  }
}

function parseContext(value) {
  return value.split('\n').map((line, index) => {
    const match = line.match(/^\s*(USER|ASSISTANT)\s*:\s*(.*)$/i);
    return match && match[2].trim() ? { role: match[1].toUpperCase(), messageId: `context-${index + 1}`, text: match[2].trim().slice(0, 4000) } : null;
  }).filter(Boolean).slice(-12);
}

function renderAnswer(response) {
  state.lastResponse = response;
  $('#query-result-title').textContent = response.result === 'ANSWERED' ? 'Grounded answer' : 'No grounded answer';
  const pill = $('#query-result-pill');
  pill.textContent = response.result;
  pill.className = `status-pill ${statusClass(response.result)}`;
  const answer = $('#query-answer');
  answer.className = `answer-surface ${response.result === 'NO_ANSWER' ? 'negative' : ''}`;
  answer.textContent = response.answer?.text || response.reasonCode || 'No answer returned.';
  const grounding = response.grounding || {};
  const bandClass = String(grounding.confidenceBand || 'LOW').toLowerCase();
  $('#query-grounding').innerHTML = [`<div class="grounding-stat"><small>Diagnostic confidence</small><strong class="confidence-${bandClass}">${((grounding.confidence || 0) * 100).toFixed(1)}%</strong></div>`, `<div class="grounding-stat"><small>Band</small><strong class="confidence-${bandClass}">${escapeHtml(grounding.confidenceBand || '--')}</strong></div>`, `<div class="grounding-stat"><small>Reranker</small><strong>${escapeHtml(response.meta?.rerankerProvider || '--')}</strong></div>`].join('');
  $('#query-citations').innerHTML = (grounding.sourceReferences || []).map((citation) => `<article class="citation"><div class="citation-head"><strong>${escapeHtml(citation.title)}</strong><span>${escapeHtml(citation.chunkId)} · v${citation.documentVersion}</span></div><p>${escapeHtml(citation.excerpt)}</p><span class="subtle">${escapeHtml(citation.verificationStatus)} · ${escapeHtml(JSON.stringify({ page: citation.page, row: citation.rowStart, sheet: citation.sheet }))}</span></article>`).join('') || '<div class="empty-state">No source references.</div>';
  $('#query-trace-id').textContent = response.traceId || 'no trace';
  $('#query-good').disabled = !response.traceId;
  $('#query-bad').disabled = !response.traceId;
  renderDiagnostics(response);
}

function renderDiagnostics(response) {
  const diagnostics = response.diagnostics || {};
  const variants = diagnostics.variants || [];
  const runs = diagnostics.retrievalRuns || [];
  const candidates = diagnostics.selectedCandidates || [];
  const agent = diagnostics.agent || {};
  const variantHtml = variants.map((item) => `<span class="section-chip">${escapeHtml(typeof item === 'string' ? item : item.text)}</span>`).join(' ');
  const candidateHtml = candidates.slice(0, 16).map((item) => `<div class="candidate-row"><span>${item.rank}</span><span title="${escapeHtml(item.title)}">${escapeHtml(item.title || item.chunk_id)}</span><span class="score">R ${Number(item.reranker_score_normalized || 0).toFixed(3)}</span><span class="score">V ${Number(item.vector_score || 0).toFixed(3)}</span><span class="score">L ${Number(item.lexical_score || 0).toFixed(3)}</span></div>`).join('');
  $('#query-diagnostics').classList.remove('empty-state');
  $('#query-diagnostics').innerHTML = `<div class="diagnostic-section"><h3>Rewrite variants</h3><div class="toggle-row">${variantHtml || '<span class="subtle">none</span>'}</div></div><div class="diagnostic-section"><h3>Evidence gate</h3><pre class="code-block">${escapeHtml(pretty(diagnostics.evidenceGate || {}))}</pre></div><div class="diagnostic-section"><h3>Selected candidates · ${candidates.length}</h3><div class="candidate-table">${candidateHtml || '<span class="subtle">none</span>'}</div></div><div class="diagnostic-section"><h3>Agent / provider</h3><pre class="code-block">${escapeHtml(pretty(agent))}</pre></div><div class="diagnostic-section"><h3>Retrieval runs · ${runs.length}</h3><pre class="code-block">${escapeHtml(pretty(runs.map((run) => ({ query: run.query, retrievedCount: run.retrievedCount, selectedCount: run.selectedCount, vectorStore: run.vectorStore }))))}</pre></div>`;
  $('#query-diagnostics').insertAdjacentHTML('beforeend', '<div id="query-trace-block" class="diagnostic-section"><h3>Trace waterfall</h3><span class="subtle">Loading trace...</span></div>');
  loadTraceDetail(response.traceId);
}

async function runQuery() {
  const question = $('#query-question').value.trim();
  if (!question) { toast('Enter a question', true); return; }
  const topK = Math.max(1, Math.min(30, Number($('#query-topk').value || 8)));
  $('#query-topk').value = topK;
  const isTest = $('#query-scope').value === 'test';
  if (isTest && !state.session) { toast('Create a test session first', true); return; }
  $('#query-run').disabled = true;
  $('#query-result-pill').textContent = 'RUNNING';
  $('#query-result-pill').className = 'status-pill status-warn';
  try {
    const response = await api('/rag-admin-api/v1/debug/query', {
      method: 'POST',
      body: JSON.stringify({
        question,
        knowledgeBaseId: 'main-business-kb',
        testSessionId: isTest ? state.session.testSessionId : undefined,
        context: parseContext($('#query-context').value),
        platform: $('#query-platform').value,
        channelType: $('#query-channel').value,
        maxAnswerChars: 600,
        useRewrite: $('#query-rewrite').checked,
        useAgent: $('#query-agent').checked,
        topK,
      }),
    });
    $('#query-error').textContent = '';
    renderAnswer(response);
    $('#quick-result').textContent = `${response.result} · ${response.meta?.latencyMs || 0} ms`;
  } catch (error) {
    $('#query-error').textContent = error.message;
    $('#query-result-pill').textContent = 'ERROR';
    $('#query-result-pill').className = 'status-pill status-error';
    toast(error.message, true);
  } finally { $('#query-run').disabled = false; }
}

async function quickRun() {
  $('#query-question').value = $('#quick-question').value;
  switchView('query-lab');
  await runQuery();
}

async function runFormalQuery() {
  const question = $('#query-question').value.trim();
  if (!question) { toast('Enter a question', true); return; }
  const context = parseContext($('#query-context').value);
  const requestId = randomId('formal-query');
  const traceId = randomId('business-trace');
  const body = {
    contractVersion: '1.0',
    requestId,
    traceId,
    snapshot: { conversationKey: 'DEBUG:webui', conversationVersion: 0, inputFingerprint: `sha256:${requestId}` },
    source: { platform: $('#query-platform').value, accountId: 'webui', channelType: $('#query-channel').value },
    knowledgeScope: { tenantId: 'local-default', knowledgeBaseId: 'main-business-kb' },
    query: { text: question, messageIds: [], lastMessageAt: new Date().toISOString() },
    context,
    constraints: { language: 'zh-CN', answerFormat: 'PLAIN_TEXT', maxAnswerChars: 600, requireGrounding: true },
  };
  $('#formal-run').disabled = true;
  try {
    const response = await api('/rag-api/v1/query', { method: 'POST', body: JSON.stringify(body), headers: { 'X-Request-Id': requestId } });
    renderAnswer(response);
    toast('Contract query completed');
  } catch (error) { $('#query-error').textContent = error.message; toast(error.message, true); }
  finally { $('#formal-run').disabled = false; }
}

async function loadTraceDetail(traceId) {
  if (!traceId) return;
  try {
    const data = await api(`/rag-admin-api/v1/traces/${encodeURIComponent(traceId)}`);
    state.lastTrace = data;
    if (state.lastResponse?.traceId === traceId) renderQueryTrace(data);
    if ($('#view-traces').classList.contains('active')) renderTraceDetail(data);
  } catch { /* The answer remains usable if telemetry is delayed. */ }
}

function renderQueryTrace(trace) {
  const spans = trace.spans || [];
  const feedback = trace.feedback || [];
  const node = $('#query-trace-block');
  if (!node) return;
  node.innerHTML = `<h3>Trace waterfall · <span class="${statusClass(trace.status)}">${escapeHtml(trace.status)}</span></h3><div class="stage-list">${spans.map((span) => `<div class="stage-row"><span>${escapeHtml(span.name)}</span><small>${escapeHtml(span.stage_type)}</small><small class="stage-status ${statusClass(span.status)}">${escapeHtml(span.status)} · ${Number(span.duration_ms || 0)} ms${span.error_code ? ` · ${escapeHtml(span.error_code)}` : ''}</small></div>`).join('')}</div>${feedback.length ? `<div class="feedback-history">${feedback.map((item) => `<div class="feedback-item ${item.rating === 'BAD' ? 'bad' : ''}">${escapeHtml(item.rating)} · ${escapeHtml(item.note || 'no note')}<small>${escapeHtml(item.reviewer_id || '')}</small></div>`).join('')}</div>` : ''}`;
}

async function sendFeedback(rating) {
  const traceId = state.lastResponse?.traceId;
  if (!traceId) return;
  try {
    await api(`/rag-admin-api/v1/traces/${encodeURIComponent(traceId)}/feedback`, { method: 'POST', body: JSON.stringify({ requestId: randomId('feedback'), rating, note: $('#query-feedback-note').value.trim() || null, reviewerId: 'webui-reviewer' }) });
    toast(`Trace marked ${rating}`);
    await loadTraceDetail(traceId);
  } catch (error) { toast(error.message, true); }
}

async function createSession() {
  const candidates = $$('#candidate-docs input[type="checkbox"]:checked').map((input) => ({ documentId: input.value, documentVersion: Number(input.dataset.version) }));
  if ($('#session-mode').value === 'SINGLE_DOCUMENT' && candidates.length !== 1) { toast('SINGLE_DOCUMENT needs exactly one candidate', true); return; }
  if (!candidates.length && !state.activeRelease) { toast('Select a candidate document before the first release', true); return; }
  try {
    const data = await api('/rag-admin-api/v1/test-sessions', { method: 'POST', body: JSON.stringify({ requestId: randomId('session'), knowledgeBaseId: 'main-business-kb', mode: $('#session-mode').value, candidateDocuments: candidates, operatorId: 'webui-operator' }) });
    state.session = data;
    localStorage.setItem('rag-session-id', data.testSessionId);
    $('#session-select').value = data.testSessionId;
    renderSession();
    toast(`Session ${data.testSessionId} created`);
  } catch (error) { $('#session-message').textContent = error.message; toast(error.message, true); }
}

function renderSession() {
  if (!state.session) return;
  $('#active-session-title').textContent = state.session.testSessionId;
  $('#active-session-status').textContent = state.session.status;
  $('#active-session-meta').innerHTML = `<span>${escapeHtml(state.session.mode)}</span><span>base: ${escapeHtml(state.session.baseReleaseId || 'none')}</span><span>${(state.session.candidateDocuments || []).length} candidate(s)</span>`;
  $('#release-session-note').textContent = `${state.session.testSessionId} · ${state.session.status}`;
  $('#release-create').disabled = false;
  $('#test-run').disabled = false;
  $('#test-answer').className = 'test-answer empty-state';
  $('#test-answer').textContent = 'Test answers and reviewer decisions will appear here.';
  loadSessionAnswers();
}

async function runTestQuery() {
  if (!state.session) return;
  const question = $('#test-question').value.trim();
  if (!question) { toast('Enter a test question', true); return; }
  $('#test-run').disabled = true;
  try {
    const answer = await api(`/rag-admin-api/v1/test-sessions/${encodeURIComponent(state.session.testSessionId)}/query`, { method: 'POST', body: JSON.stringify({ requestId: randomId('test'), question, context: [] }) });
    renderTestAnswer(answer);
    await loadSessionAnswers();
  } catch (error) { toast(error.message, true); } finally { $('#test-run').disabled = false; }
}

function renderTestAnswer(answer) {
  const node = $('#test-answer');
  node.className = 'test-answer';
  node.innerHTML = `<article class="review-answer"><div class="review-answer-head"><span>${escapeHtml(answer.answerId)}</span><span>${stateText(answer.result)} · ${escapeHtml(answer.grounding?.confidenceBand || 'LOW')}</span></div><h3>${escapeHtml(answer.answer?.text || answer.reasonCode || 'NO_ANSWER')}</h3><p>${escapeHtml((answer.sourceReferences || []).map((item) => `${item.title}: ${item.excerpt}`).join('\n\n'))}</p><div class="review-actions"><button class="decision-button enabled" data-answer-id="${escapeHtml(answer.answerId)}" data-session-id="${escapeHtml(answer.testSessionId)}" data-decision="ENABLED">Mark enabled</button><button class="decision-button disabled" data-answer-id="${escapeHtml(answer.answerId)}" data-session-id="${escapeHtml(answer.testSessionId)}" data-decision="DISABLED">Mark disabled</button><button class="text-button" data-trace-link="${escapeHtml(answer.traceId)}">Open trace →</button></div></article>`;
  node.querySelectorAll('[data-decision]').forEach((button) => button.addEventListener('click', () => updateDecision(button.dataset.answerId, button.dataset.decision, button.dataset.sessionId)));
  node.querySelector('[data-trace-link]')?.addEventListener('click', () => { state.selectedTraceId = answer.traceId; switchView('traces'); loadTraceDetail(answer.traceId); });
}

async function updateDecision(answerId, decision, sessionId = state.session?.testSessionId) {
  let reasonCode = null;
  let note = null;
  if (decision === 'DISABLED') {
    reasonCode = window.prompt('Reason code (ANSWER_INACCURATE / SHOULD_HANDOFF / SOURCE_INCORRECT / OTHER)', 'ANSWER_INACCURATE') || 'OTHER';
    note = window.prompt('Reviewer note or expected correction', '') || null;
  }
  try {
    await api(`/rag-admin-api/v1/test-sessions/${encodeURIComponent(sessionId)}/answers/${encodeURIComponent(answerId)}/decision`, { method: 'PUT', body: JSON.stringify({ requestId: randomId('decision'), decision, reasonCode, note, operatorId: 'webui-reviewer' }) });
    toast(`Answer marked ${decision}`);
    await loadSessionAnswers();
  } catch (error) { toast(error.message, true); }
}

async function loadSessionAnswers() {
  if (!state.session) return;
  try {
    const data = await api(`/rag-admin-api/v1/test-sessions/${encodeURIComponent(state.session.testSessionId)}/answers`);
    const answers = data.answers || [];
    $('#session-answers').className = answers.length ? 'answer-history' : 'answer-history empty-state';
    $('#session-answers').innerHTML = answers.map((answer) => `<article class="review-answer"><div class="review-answer-head"><span>${escapeHtml(answer.answer_id)}</span><span>${stateText(answer.decision)} · ${stateText(answer.result)}</span></div><h3>${escapeHtml(answer.question)}</h3><p>${escapeHtml(answer.answer?.text || answer.reason_code || 'NO_ANSWER')}</p><small class="subtle">${escapeHtml(answer.decision_note || '')}</small></article>`).join('') || 'No answers yet.';
  } catch (error) { toast(error.message, true); }
}

async function loadReleases() {
  try {
    const data = await api('/rag-admin-api/v1/releases');
    state.releases = data.releases || [];
    state.activeRelease = data.active || state.activeRelease;
    renderReleases();
  } catch (error) { toast(error.message, true); }
}

function renderReleases() {
  const activeId = state.activeRelease?.release_id;
  $('#releases-table').innerHTML = state.releases.length ? state.releases.map((release) => { const canRollback = ['PUBLISHED', 'ROLLED_BACK'].includes(release.status) && release.releaseId !== activeId; return `<article class="release-row ${release.releaseId === activeId ? 'active' : ''}"><span class="release-id">${escapeHtml(release.releaseId)}<small>${escapeHtml(release.knowledgeVersion || '--')}</small></span><span>${stateText(release.status)}</span><span>${escapeHtml(release.enabledTestCaseCount)} / ${escapeHtml(release.disabledTestCaseCount)}<small>enabled / disabled</small></span><span>${escapeHtml(release.publishedBy || '--')}</span>${release.releaseId === activeId ? '<span class="mini-badge">active</span>' : (canRollback ? `<button class="rollback-button" data-rollback="${escapeHtml(release.releaseId)}">Rollback</button>` : '<span class="subtle">not rollbackable</span>')}</article>`; }).join('') : '<div class="empty-state">No releases yet.</div>';
  $('#releases-table').querySelectorAll('[data-rollback]').forEach((button) => button.addEventListener('click', () => rollback(button.dataset.rollback)));
}

async function createRelease() {
  if (!state.session) { toast('Create a test session first', true); return; }
  $('#release-create').disabled = true;
  try {
    const data = await api('/rag-admin-api/v1/knowledge-bases/main-business-kb/releases', { method: 'POST', body: JSON.stringify({ requestId: randomId('release'), testSessionId: state.session.testSessionId, baseReleaseId: state.session.baseReleaseId || undefined, candidateDocuments: state.session.candidateDocuments || [], publishedBy: 'webui-operator', publishNote: 'Published from Control Room' }) });
    toast(`${data.releaseId} is building`);
    pollRelease(data.releaseId);
  } catch (error) { toast(error.message, true); $('#release-create').disabled = false; }
}

async function pollRelease(releaseId) {
  try {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 750));
      try {
        const data = await api(`/rag-admin-api/v1/releases/${encodeURIComponent(releaseId)}`);
        if (data.status !== 'BUILDING') { toast(`${releaseId}: ${data.status}`, data.status === 'FAILED'); await loadHealth(); await loadReleases(); return; }
      } catch (error) { toast(error.message, true); return; }
    }
    toast('Release polling timed out', true);
  } finally { $('#release-create').disabled = false; }
}

async function rollback(releaseId) {
  if (!window.confirm(`Activate ${releaseId}?`)) return;
  try {
    await api(`/rag-admin-api/v1/releases/${encodeURIComponent(releaseId)}/rollback`, { method: 'POST', body: JSON.stringify({ requestId: randomId('rollback'), targetReleaseId: releaseId, operatorId: 'webui-operator' }) });
    toast(`Rolled back to ${releaseId}`);
    await loadHealth(); await loadReleases();
  } catch (error) { toast(error.message, true); }
}

async function loadTraces() {
  try {
    const data = await api('/rag-admin-api/v1/traces?limit=200');
    const traces = data.traces || [];
    $('#metric-traces').textContent = traces.length;
    $('#traces-table').innerHTML = traces.map((trace) => `<button class="trace-row ${trace.trace_id === state.selectedTraceId ? 'selected' : ''}" data-trace-id="${escapeHtml(trace.trace_id)}"><div class="trace-row-top"><span>${escapeHtml(trace.name)}</span><span class="${statusClass(trace.status)}">${escapeHtml(trace.status)}</span></div><div class="trace-row-meta"><span>${escapeHtml(trace.trace_id)}</span><span>${Number(trace.duration_ms || 0)} ms</span></div></button>`).join('') || '<div class="empty-state">No traces yet.</div>';
    $('#overview-traces').innerHTML = traces.slice(0, 6).map((trace) => `<button class="trace-row" data-trace-id="${escapeHtml(trace.trace_id)}"><div class="trace-row-top"><span>${escapeHtml(trace.name)}</span><span class="${statusClass(trace.status)}">${escapeHtml(trace.status)}</span></div><div class="trace-row-meta"><span>${escapeHtml(trace.trace_id)}</span><span>${Number(trace.duration_ms || 0)} ms</span></div></button>`).join('') || '<div class="empty-state">No traces yet.</div>';
    $$('.trace-row[data-trace-id]').forEach((button) => button.addEventListener('click', () => { state.selectedTraceId = button.dataset.traceId; loadTraceDetail(button.dataset.traceId); if ($('#view-overview').classList.contains('active')) switchView('traces'); }));
    if (state.selectedTraceId) await loadTraceDetail(state.selectedTraceId);
  } catch (error) { toast(error.message, true); }
}

function renderTraceDetail(trace) {
  $('#trace-detail-title').textContent = trace.trace_id;
  $('#trace-detail-status').textContent = trace.status;
  $('#trace-detail-status').className = `status-pill ${statusClass(trace.status)}`;
  $('#trace-detail').classList.remove('empty-state');
  $('#trace-detail').innerHTML = `<div class="diagnostic-section"><h3>${escapeHtml(trace.name)} · ${Number(trace.duration_ms || 0)} ms</h3><pre class="code-block">${escapeHtml(pretty({ requestId: trace.request_id, businessTraceId: trace.business_trace_id, input: trace.input_summary, output: trace.output_summary, attributes: trace.attributes }))}</pre></div><div class="diagnostic-section"><h3>Waterfall</h3><div class="stage-list">${(trace.spans || []).map((span) => `<div class="stage-row"><span>${escapeHtml(span.name)}</span><small>${escapeHtml(span.stage_type)}</small><small class="stage-status ${statusClass(span.status)}">${escapeHtml(span.status)} · ${Number(span.duration_ms || 0)} ms${span.error_code ? ` · ${escapeHtml(span.error_code)}` : ''}</small></div>`).join('')}</div></div><div class="diagnostic-section"><h3>Candidate telemetry · ${(trace.candidates || []).length}</h3><div class="candidate-table">${(trace.candidates || []).slice(0, 30).map((item) => `<div class="candidate-row"><span>${item.rank}</span><span>${escapeHtml(item.payload?.title || item.chunk_id)}</span><span class="score">R ${Number(item.reranker_score_normalized || 0).toFixed(3)}</span><span class="score">V ${Number(item.vector_score || 0).toFixed(3)}</span><span class="score">L ${Number(item.lexical_score || 0).toFixed(3)}</span></div>`).join('')}</div></div><div class="diagnostic-section"><h3>Feedback</h3><div class="feedback-history">${(trace.feedback || []).map((item) => `<div class="feedback-item ${item.rating === 'BAD' ? 'bad' : ''}">${escapeHtml(item.rating)} · ${escapeHtml(item.note || 'no note')}<small>${escapeHtml(item.reviewer_id || '')} · ${escapeHtml(item.created_at || '')}</small></div>`).join('') || '<span class="subtle">No reviewer feedback.</span>'}</div></div>`;
}

async function loadEvaluations() {
  try {
    const data = await api('/rag-admin-api/v1/evaluations');
    const runs = data.evaluations || [];
    $('#eval-table').innerHTML = runs.map((run) => `<button class="evaluation-row" data-eval-id="${escapeHtml(run.eval_run_id)}"><strong>${escapeHtml(run.eval_run_id)}</strong><span>${stateText(run.status)}</span><span>${escapeHtml(run.completed_cases)} / ${escapeHtml(run.total_cases)}</span><span>${run.summary?.accuracy != null ? `${(run.summary.accuracy * 100).toFixed(1)}%` : '--'}</span><small>${escapeHtml(run.created_at || '')}</small></button>`).join('') || '<div class="empty-state">No evaluation runs yet.</div>';
    $$('#eval-table [data-eval-id]').forEach((button) => button.addEventListener('click', () => showEvaluation(button.dataset.evalId)));
    const running = runs.find((run) => run.status === 'RUNNING');
    if (running && !state.evalPolling) { state.evalRun = running.eval_run_id; state.evalPolling = true; pollEvaluation(running.eval_run_id); }
  } catch (error) { toast(error.message, true); }
}

async function runEvaluation() {
  $('#eval-run').disabled = true;
  try {
    const data = await api('/rag-admin-api/v1/evaluations/run', { method: 'POST', body: JSON.stringify({ requestId: randomId('eval'), concurrency: 3, useGeneration: true }) });
    state.evalRun = data.evalRunId;
    state.evalPolling = true;
    $('#eval-summary').textContent = `${data.totalCases} cases running · ${data.evalRunId}`;
    pollEvaluation(data.evalRunId);
  } catch (error) { toast(error.message, true); $('#eval-run').disabled = false; }
}

async function pollEvaluation(evalId) {
  for (let attempt = 0; attempt < 1200; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    try {
      const data = await api(`/rag-admin-api/v1/evaluations/${encodeURIComponent(evalId)}`);
      $('#eval-summary').textContent = data.status === 'COMPLETED' ? `${data.summary?.passed || 0} / ${data.summary?.total || 0} passed · ${((data.summary?.accuracy || 0) * 100).toFixed(1)}%` : `${data.completed_cases || 0} / ${data.total_cases || 0} cases running`;
      if (data.status !== 'RUNNING') { $('#eval-run').disabled = false; state.evalPolling = false; await loadEvaluations(); await showEvaluation(evalId); return; }
    } catch (error) { toast(error.message, true); $('#eval-run').disabled = false; state.evalPolling = false; return; }
  }
  state.evalPolling = false;
  $('#eval-run').disabled = false;
  toast('Evaluation polling timed out', true);
}

async function showEvaluation(evalId) {
  try {
    const data = await api(`/rag-admin-api/v1/evaluations/${encodeURIComponent(evalId)}`);
    $('#eval-summary').textContent = `${data.summary?.passed || 0} / ${data.summary?.total || 0} passed · ${((data.summary?.accuracy || 0) * 100).toFixed(1)}%`;
    const cases = data.cases || [];
    $('#eval-failures').className = cases.length ? 'evaluation-list' : 'evaluation-list empty-state';
    $('#eval-failures').innerHTML = cases.map((item) => `<article class="evaluation-row ${item.passed ? '' : 'failed-case'}"><strong>${escapeHtml(item.case_id)}</strong><span>${item.passed ? '<span class="pass">PASS</span>' : '<span class="status-error">FAIL</span>'}</span><span>${escapeHtml(item.expected_result || '--')} → ${escapeHtml(item.actual_result || '--')}</span><span>${escapeHtml((item.assertions || []).map((a) => a.code).join(', ') || 'none')}</span>${item.trace_id ? `<button class="text-button" data-eval-trace="${escapeHtml(item.trace_id)}">Trace →</button>` : '<small>--</small>'}</article>`).join('') || 'No cases recorded.';
    $('#eval-failures').querySelectorAll('[data-eval-trace]').forEach((button) => button.addEventListener('click', () => { state.selectedTraceId = button.dataset.evalTrace; switchView('traces'); loadTraceDetail(button.dataset.evalTrace); }));
    const failures = cases.filter((item) => !item.passed);
    if (failures.length) toast(`${failures.length} failures available in traces`);
  } catch (error) { toast(error.message, true); }
}

function renderApiRoutes() {
  const routes = [
    ['GET', '/rag-api/v1/health', 'readiness and provider posture'],
    ['POST', '/rag-api/v1/query', 'formal published query'],
    ['POST', '/rag-admin-api/v1/documents', 'multipart ingestion'],
    ['GET', '/rag-admin-api/v1/documents', 'source registry'],
    ['GET', '/rag-admin-api/v1/documents/{documentId}', 'processing status'],
    ['GET', '/rag-admin-api/v1/documents/{documentId}/chunks', 'chunk inspection'],
    ['POST', '/rag-admin-api/v1/test-sessions', 'pin candidate versions'],
    ['GET', '/rag-admin-api/v1/test-sessions', 'restore sessions'],
    ['GET', '/rag-admin-api/v1/test-sessions/{id}', 'session detail'],
    ['POST', '/rag-admin-api/v1/test-sessions/{id}/query', 'pre-release query'],
    ['GET', '/rag-admin-api/v1/test-sessions/{id}/answers', 'answer history'],
    ['PUT', '/rag-admin-api/v1/test-sessions/{id}/answers/{answerId}/decision', 'enable or disable'],
    ['POST', '/rag-admin-api/v1/knowledge-bases/{id}/releases', 'build and publish'],
    ['GET', '/rag-admin-api/v1/releases', 'release history'],
    ['GET', '/rag-admin-api/v1/releases/{id}', 'release status'],
    ['POST', '/rag-admin-api/v1/releases/{id}/rollback', 'atomic rollback'],
    ['POST', '/rag-admin-api/v1/debug/query', 'full debug response'],
    ['POST', '/rag-admin-api/v1/debug/retrieve', 'retrieval and rerank only'],
    ['GET', '/rag-admin-api/v1/system/providers', 'provider posture'],
    ['GET', '/rag-admin-api/v1/system/status', 'execution progress'],
    ['GET', '/rag-admin-api/v1/system/chunk-policies', 'chunk policy profiles'],
    ['GET', '/rag-admin-api/v1/negative-cases', 'active negative gates'],
    ['GET', '/rag-admin-api/v1/traces', 'trace index'],
    ['GET', '/rag-admin-api/v1/traces/{traceId}', 'trace waterfall and candidates'],
    ['POST', '/rag-admin-api/v1/traces/{traceId}/feedback', 'GOOD / BAD reviewer feedback'],
    ['POST', '/rag-admin-api/v1/evaluations/run', '200-case regression matrix'],
  ];
  $('#api-routes').innerHTML = routes.map(([method, path, description]) => `<div class="api-row"><span class="api-method">${method}</span><code>${path}</code><span>${description}</span></div>`).join('');
}

function bindEvents() {
  $$('.nav-item').forEach((item) => item.addEventListener('click', () => switchView(item.dataset.view)));
  $$('[data-go]').forEach((button) => button.addEventListener('click', () => switchView(button.dataset.go)));
  $('#refresh-btn').addEventListener('click', refreshAll);
  $('#documents-refresh').addEventListener('click', loadDocuments);
  $('#traces-refresh').addEventListener('click', loadTraces);
  $('#releases-refresh').addEventListener('click', loadReleases);
  $('#session-refresh').addEventListener('click', loadSessionAnswers);
  $('#eval-refresh').addEventListener('click', loadEvaluations);
  $('#upload-form').addEventListener('submit', uploadDocument);
  $('#upload-file').addEventListener('change', () => { $('#file-name').textContent = $('#upload-file').files[0]?.name || 'Choose a source file'; });
  $('#quick-run').addEventListener('click', quickRun);
  $('#query-run').addEventListener('click', runQuery);
  $('#formal-run').addEventListener('click', runFormalQuery);
  $('#query-good').addEventListener('click', () => sendFeedback('GOOD'));
  $('#query-bad').addEventListener('click', () => sendFeedback('BAD'));
  $('#session-create').addEventListener('click', createSession);
  $('#session-select').addEventListener('change', () => selectSession($('#session-select').value));
  $('#test-run').addEventListener('click', runTestQuery);
  $('#release-create').addEventListener('click', createRelease);
  $('#eval-run').addEventListener('click', runEvaluation);
  $('#token-btn').addEventListener('click', () => { const token = window.prompt('Local bearer token', state.token); if (token) { state.token = token.trim(); localStorage.setItem('rag-token', state.token); refreshAll(); } });
}

async function refreshAll() {
  await Promise.all([loadHealth(), loadSystemStatus(), loadDocuments(), loadSessions(), loadTraces(), loadReleases(), loadEvaluations()]);
}

bindEvents();
renderApiRoutes();
refreshAll();

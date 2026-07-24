let methods = [];
let cardTrackers = [];
const container = document.getElementById('cards-container');

let serverState = {};
let activeModalDir = null;
let activeModalMaxStepRendered = -100;
let staticMode = false;

// Public bucket the training jobs write to; readable (and listable)
// anonymously, so a static page needs no backend at all.
const BUCKET = 'recipe-lanes-nca-jobs';
const BUCKET_BASE = `https://storage.googleapis.com/${BUCKET}/`;
const BUCKET_LIST = `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o?fields=items(name,updated),nextPageToken&maxResults=1000`;

function pad5(n) { return String(n).padStart(5, '0'); }

// ---------------------------------------------------------------------
// Relative-time / recency helpers — mirrors lenia.js's relTime/humanize-
// Duration exactly (kept as a duplicate rather than a shared import since
// dashboard.js is a plain <script>, not a module).
// ---------------------------------------------------------------------

const RECENT_MS = 20 * 60 * 1000;   // 20 minutes

function relTime(iso) {
    if (!iso) return null;
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return null;
    const deltaMs = Date.now() - t;
    const s = deltaMs / 1000;
    if (s < 60) return `${Math.max(0, Math.round(s))}s ago`;
    const m = s / 60;
    if (m < 60) return `${Math.round(m)}m ago`;
    const h = m / 60;
    if (h < 24) return `${h.toFixed(1)}h ago`;
    const d = h / 24;
    return `${d.toFixed(1)}d ago`;
}

function humanizeDuration(ms) {
    if (!Number.isFinite(ms) || ms < 0) return null;
    const h = ms / 3600000;
    if (h < 1) return `${Math.round(ms / 60000)}m`;
    return `${h.toFixed(1)}h`;
}

// Last-snapshot recency line, shared by every card.
function renderLastSnapshot(tr) {
    if (!tr.lastSnapObj) return;
    const iso = (tr.runJson && tr.runJson.updated_at) || tr.updated || null;
    const rel = relTime(iso);
    if (!rel) { tr.lastSnapObj.innerHTML = ''; return; }
    const isRecent = (Date.now() - Date.parse(iso)) < RECENT_MS;
    const cls = isRecent ? 'recent' : 'stale';
    const title = isRecent ? 'recently active' : 'inactive';
    tr.lastSnapObj.innerHTML =
        `<span class="snap-dot ${cls}" title="${title}">&#9679;</span>last snapshot ${rel}`;
}

// Best-effort run.json fetch, used only to enrich the card's status line
// with duration/rate and the last-snapshot line with a precise updated_at.
// Failure is silent — the bucket-listing `updated` timestamp is already a
// perfectly good fallback for both.
async function fetchRunJsonForCard(tr) {
    try {
        const res = await fetch(tr.dir + 'run.json?t=' + Date.now());
        if (!res.ok) return;
        tr.runJson = await res.json();
        renderLastSnapshot(tr);
        updateOverviewUI();   // re-render the status line with duration appended
    } catch (e) { /* run.json not published yet — fine, fall back silently */ }
}

async function listCloudRuns(onPage) {
    const runs = {};
    let pageToken = null;
    do {
        const res = await fetch(BUCKET_LIST + (pageToken ? `&pageToken=${pageToken}` : ''));
        if (!res.ok) throw new Error(`bucket list failed: ${res.status}`);
        const d = await res.json();
        (d.items || []).forEach(({name, updated}) => {
            const i = name.indexOf('/');
            if (i < 0) return;
            const run = name.slice(0, i), fname = name.slice(i + 1);
            if (!runs[run]) runs[run] = {maxStep: -1, hasWeights: false, updated: '', kernelStep: -1};
            if (fname.startsWith('COMP_')) {
                const s = parseInt(fname.slice(5, 10));
                if (!isNaN(s)) runs[run].maxStep = Math.max(runs[run].maxStep, s);
                if (updated && updated > runs[run].updated) runs[run].updated = updated;
            } else if (fname === 'weights.json') {
                runs[run].hasWeights = true;
            } else if (fname.startsWith('KERNEL_')) {
                // Cheap lenia-family detection: KERNEL_#####.png files only
                // exist for nca.train_lenia runs, and we're already paging
                // the whole bucket listing here — no extra fetch needed.
                const s = parseInt(fname.slice(7, 12));
                if (!isNaN(s)) runs[run].kernelStep = Math.max(runs[run].kernelStep, s);
            }
        });
        if (onPage) onPage(runs);   // stream cards page by page
        pageToken = d.nextPageToken;
    } while (pageToken);
    return runs;
}

function cloudMethodsFrom(runs) {
    return Object.keys(runs).sort().map(run => {
        const m = {
            id: 'cloud_' + run,
            title: '☁ ' + run,
            dir: BUCKET_BASE + run + '/',
            desc: 'Vertex AI training run (live from the public bucket)',
            seedType: 'cloud',
            cloud: true,
            updated: runs[run].updated,
            isLenia: runs[run].kernelStep >= 0,
            kernelStep: runs[run].kernelStep
        };
        if (runs[run].hasWeights) m.weights_url = BUCKET_BASE + run + '/weights.json';
        return m;
    });
}

async function staticStatusLoop() {
    try {
        const runs = await listCloudRuns();
        const status = {};
        Object.entries(runs).forEach(([run, v]) => {
            status[BUCKET_BASE + run + '/'] = v.maxStep;
        });
        serverState = status;
        updateOverviewUI();
        updateActiveModalUI();
    } catch (e) {
        console.error('bucket poll failed', e);
    }
    setTimeout(staticStatusLoop, 15000);
}

const seenIds = new Set();
let sortKey = localStorage.getItem('dash_sort') || 'newest';

function addOrUpdateCards(list) {
    let added = false;
    let resort = false;
    list.forEach(m => {
        if (seenIds.has(m.id)) {
            const tr = cardTrackers.find(t => t.id === m.id);
            if (tr) {
                if (m.vertex_state) tr.vertexState = m.vertex_state;
                if (m.updated) tr.updated = m.updated;
                renderLastSnapshot(tr);
            }
            const known = methods.find(x => x.id === m.id);
            if (known) {
                if (m.weights_url) known.weights_url = m.weights_url;
                if (m.updated && m.updated !== known.updated) {
                    known.updated = m.updated;   // keep sortCards()'s source fresh
                    resort = true;
                }
            }
            return;
        }
        seenIds.add(m.id);
        methods.push(m);
        added = true;
        const card = document.createElement('div');
        card.className = 'card';
        card.id = `card_${m.id}`;
        card.onclick = () => openModal(m.title, m.dir, m.desc);
        const leniaBlock = m.isLenia ? `
            <div class="sub-desc" style="margin-top:10px;">learned kernels</div>
            <div class="img-container" style="height:90px;">
                <img loading="lazy" id="kernel_${m.id}" src="${m.dir}KERNEL_${pad5(m.kernelStep)}.png" onerror="this.style.display='none'">
            </div>
            <div style="margin-top:6px;">
                <a href="lenia.html" onclick="event.stopPropagation();" style="color:#4db8ff; font-size:0.85em; text-decoration:none;">open in Lenia lab &rarr;</a>
            </div>
        ` : '';
        card.innerHTML = `
            <h3>${m.title}</h3>
            <div class="side-by-side">
                <div>
                    <div class="sub-desc">Live Target</div>
                    <div class="img-container"><img loading="lazy" id="live_tgt_${m.id}" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" onerror="this.src='${m.dir}target.png'"></div>
                </div>
                <div>
                    <div class="sub-desc">Latest Checkpoint</div>
                    <div class="img-container"><img loading="lazy" id="live_${m.id}" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="></div>
                </div>
            </div>
            ${leniaBlock}
            <div class="status" id="live_status_${m.id}">Loading...</div>
            <div class="last-snapshot" id="lastsnap_${m.id}"></div>
        `;
        container.appendChild(card);
        const tr = {
            id: m.id,
            dir: m.dir,
            updated: m.updated || '',
            cardObj: card,
            imgObj: card.querySelector(`#live_${CSS.escape(m.id)}`),
            tgtObj: card.querySelector(`#live_tgt_${CSS.escape(m.id)}`),
            statusObj: card.querySelector(`#live_status_${CSS.escape(m.id)}`),
            lastSnapObj: card.querySelector(`#lastsnap_${CSS.escape(m.id)}`),
            vertexState: m.vertex_state || null,
            lastKnownStep: -100,
            runJson: null
        };
        cardTrackers.push(tr);
        renderLastSnapshot(tr);   // shows the listing-derived timestamp immediately
        fetchRunJsonForCard(tr);   // then refines it with run.json's updated_at + duration/rate
    });
    if (added || resort) sortCards();
    if (added) initializeDropdown();
}

window.setSort = function(k) {
    sortKey = k;
    localStorage.setItem('dash_sort', k);
    sortCards();
    initializeDropdown();   // dropdown follows the same order
};

function methodComparator(a, b) {
    if (sortKey === 'name') return a.title.localeCompare(b.title);
    // newest first; undated (local) entries last, alphabetical
    return (b.updated || '').localeCompare(a.updated || '')
           || a.title.localeCompare(b.title);
}

function sortCards() {
    const arr = [...methods];
    arr.sort(methodComparator);
    arr.forEach(m => {
        const el = document.getElementById(`card_${m.id}`);
        if (el) container.appendChild(el);   // append = reorder in place
    });
}

async function refreshMethods() {
    try {
        if (staticMode) {
            addOrUpdateCards(cloudMethodsFrom(await listCloudRuns()));
        } else {
            const res = await fetch('/api/methods?t=' + Date.now());
            if (res.ok) addOrUpdateCards(await res.json());
        }
    } catch (e) { console.error('refresh failed', e); }
}

async function bootstrap() {
    const sortSel = document.getElementById('sort-select');
    if (sortSel) sortSel.value = sortKey;
    // Prefer the local orchestration server; fall back to reading the
    // public bucket directly (gh-pages / any static hosting). Cards are
    // streamed in as data arrives rather than waiting for everything.
    try {
        const res = await fetch('/api/methods?t=' + Date.now());
        if (!res.ok) throw new Error(`no backend (${res.status})`);
        addOrUpdateCards(await res.json());
    } catch (e) {
        staticMode = true;
        await listCloudRuns(runs => addOrUpdateCards(cloudMethodsFrom(runs)));
    }

    if (staticMode) {
        staticStatusLoop();
    } else {
        const evtSource = new EventSource("/api/status_stream");
        evtSource.onmessage = function(event) {
            serverState = JSON.parse(event.data);
            updateOverviewUI();
            updateActiveModalUI();
        };
        evtSource.onerror = function() {
            console.error("SSE Connection Error");
        };
    }
    setInterval(refreshMethods, 25000);   // new runs appear without reload
    setInterval(() => cardTrackers.forEach(renderLastSnapshot), 60000);   // re-render cached relTime labels
}

bootstrap().catch(err => console.error("Dashboard bootstrap failed", err));
    
function applyFilters() {
    const filterVal = document.getElementById('seed-filter').value;
    const q = (document.getElementById('search-box')?.value || '').toLowerCase();
    methods.forEach(m => {
        const el = document.getElementById(`card_${m.id}`);
        if (!el) return;
        const seedOk = filterVal === 'all' || filterVal === m.seedType;
        const hay = (m.title + ' ' + (m.desc || '') + ' '
                     + (m.tags || []).join(' ')).toLowerCase();
        const searchOk = !q || hay.includes(q);
        el.style.display = (seedOk && searchOk) ? 'block' : 'none';
    });
}
window.applyFilters = applyFilters;
const applySeedFilter = applyFilters;   // legacy onchange handler

function addCloudWeightOptions(selectBox) {
    // Cloud runs that published weights.json load straight from the bucket,
    // ordered by the same sort as the cards.
    [...methods].sort(methodComparator).forEach(m => {
        if (!m.weights_url) return;
        let opt = document.createElement('option');
        opt.value = m.weights_url;
        opt.innerText = m.title;
        selectBox.appendChild(opt);
    });
}

function initializeDropdown() {
    const selectBox = document.getElementById("interactive-model-select");
    const prev = selectBox.value;   // preserve selection across refreshes
    selectBox.innerHTML = '';
    selectBox.dataset.prev = prev;

    fetch('weights/index.json?t=' + Date.now())
        .then(r => r.json())
        .then(idx => {
            methods.forEach(m => {
                if (m.id === 'diffusion' || m.cloud) return;

                let dirName = m.dir.replace('/', '');
                let lookupName = dirName.replace('snaps_web_', '').replace('snaps_', '');

                if (idx.words.includes(lookupName) || m.id === 'guided' || m.id === 'cloud') {
                    let opt = document.createElement('option');
                    opt.value = dirName;
                    opt.innerText = m.title;
                    selectBox.appendChild(opt);
                }
            });
            addCloudWeightOptions(selectBox);
            if (selectBox.dataset.prev) selectBox.value = selectBox.dataset.prev;
        })
        .catch(err => {
            methods.forEach(m => {
                if (m.id === 'diffusion' || m.cloud) return;
                let opt = document.createElement('option');
                opt.value = m.dir.replace('/', '');
                opt.innerText = m.title;
                selectBox.appendChild(opt);
            });
            addCloudWeightOptions(selectBox);
            if (selectBox.dataset.prev) selectBox.value = selectBox.dataset.prev;
        });
}

// Appends " — loss 0.1067 — 4.6h" (whichever parts run.json actually has)
// to a card's status line, once run.json has been fetched.
function runJsonStatusSuffix(rj) {
    if (!rj) return '';
    const parts = [];
    if (Array.isArray(rj.losses) && rj.losses.length) {
        const last = rj.losses[rj.losses.length - 1];
        const lv = Array.isArray(last) ? Number(last[1])
            : (last && typeof last === 'object' ? Number(last.loss) : NaN);
        if (Number.isFinite(lv)) parts.push(`loss ${lv.toFixed(4)}`);
    }
    if (rj.started_at && rj.updated_at) {
        const s = Date.parse(rj.started_at), u = Date.parse(rj.updated_at);
        if (Number.isFinite(s) && Number.isFinite(u) && u >= s) {
            const dur = humanizeDuration(u - s);
            if (dur) parts.push(dur);
        }
    }
    return parts.length ? ' — ' + parts.join(' — ') : '';
}

function updateOverviewUI() {
    cardTrackers.forEach(ct => {
        const currentHighest = serverState[ct.dir];
        if (currentHighest !== undefined) {
            if (currentHighest > ct.lastKnownStep) {
                ct.lastKnownStep = currentHighest;
                if (currentHighest === -1) {
                    ct.lastStatusText = "No runs logged yet (Waiting on step 0)";
                } else {
                    ct.lastStatusText = `Latest: Step ${currentHighest} / 16000`;
                    let stepStr = String(currentHighest).padStart(5, '0');
                    if(ct.imgObj) {
                        ct.imgObj.onerror = function() { this.src = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs='; };
                        ct.imgObj.src = `${ct.dir}COMP_${stepStr}.png?t=` + Date.now();
                    }
                    if(ct.tgtObj) ct.tgtObj.src = `${ct.dir}TARGET_${stepStr}.png?t=` + Date.now();
                }
            }
            
            if (!ct.cardObj || !ct.statusObj) return;
            
            if (ct.dir.includes('proposed_targets')) {
                ct.cardObj.style.borderColor = '#4db8ff';
                ct.statusObj.style.color = '#4db8ff';
                ct.statusObj.innerText = '(Ready for training)';
                if (ct.imgObj) ct.imgObj.style.display = 'none';
            } else if (ct.vertexState) {
                const done = ct.vertexState === 'SUCCEEDED';
                ct.cardObj.style.borderColor = done ? '#00ff00' : '#ffaa00';
                ct.statusObj.style.color = done ? '#00ff00' : '#ffaa00';
                ct.statusObj.innerText = `${ct.lastStatusText || ''} (Vertex: ${ct.vertexState})`;
            } else if (currentHighest >= 15900) {
                ct.cardObj.style.borderColor = '#00ff00';
                ct.statusObj.style.color = '#00ff00';
                ct.statusObj.innerText = (ct.lastStatusText || '') + ' (DONE)';
            } else if (currentHighest >= 0) {
                ct.cardObj.style.borderColor = '#ffaa00';
                ct.statusObj.style.color = '#ffaa00';
                ct.statusObj.innerText = (ct.lastStatusText || '') + ' (RUNNING)';
            } else {
                ct.statusObj.innerText = (ct.lastStatusText || '');
            }
            ct.statusObj.innerText += runJsonStatusSuffix(ct.runJson);
            renderLastSnapshot(ct);
        }
    });
}

window.openModal = function(title, dir, desc) {
    document.getElementById('modal-title').innerText = title;
    document.getElementById('modal-desc').innerText = desc || "No description available.";
    
    let btnBegin = document.getElementById('btn-begin-training');
    if (dir.includes('proposed_targets')) {
        btnBegin.style.display = 'inline-block';
    } else {
        btnBegin.style.display = 'none';
    }
    
    const gallery = document.getElementById('modal-gallery');
    gallery.innerHTML = '';
    activeModalDir = dir;
    activeModalMaxStepRendered = -100;
    document.getElementById('modal').style.display = 'block';
    updateActiveModalUI();
    fetchNotes(dir);
}

function fetchNotes(dir) {
    if (staticMode) {
        document.getElementById('notes-list').innerHTML =
            '<li><i>Notes need the orchestration server (read-only static site).</i></li>';
        return;
    }
    fetch('/api/notes')
        .then(r => r.json())
        .then(notes_db => {
            const list = document.getElementById('notes-list');
            list.innerHTML = '';
            if (notes_db[dir]) {
                notes_db[dir].forEach(n => {
                    const li = document.createElement('li');
                    li.innerText = new Date(n.timestamp * 1000).toLocaleString() + ": " + n.note;
                    list.appendChild(li);
                });
            } else {
                list.innerHTML = '<li><i>No notes yet.</i></li>';
            }
        });
}

window.submitNote = function(event) {
    event.stopPropagation();
    if (staticMode) return;
    const input = document.getElementById('note-input');
    const note = input.value.trim();
    if(!note || !activeModalDir) return;
    
    fetch('/api/notes', {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({dir: activeModalDir, note: note})
    }).then(() => {
        input.value = '';
        fetchNotes(activeModalDir);
    });
}

window.beginTraining = function(event) {
    event.stopPropagation();
    if(!activeModalDir) return;
    fetch('/api/notes', {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({dir: activeModalDir, note: "[SYSTEM: BEGIN TRAINING]"})
    }).then(() => {
        fetchNotes(activeModalDir);
        let tgtStatus = document.getElementById('live_status_proposed_targets');
        if (tgtStatus) {
            tgtStatus.innerText = "Training Started!";
            tgtStatus.style.animation = "none";
            tgtStatus.style.color = "#55ff55";
        }
    });
}

window.closeModal = function() {
    document.getElementById('modal').style.display = 'none';
    activeModalDir = null;
}

window.handleOverlayClick = function(event) {
    closeModal();
}

window.handleContentClick = function(event) {
    if (event.target.closest('#modal-notes') || event.target.closest('.gallery-item')) {
        event.stopPropagation();
    } else {
        closeModal();
        event.stopPropagation();
    }
}

function updateActiveModalUI() {
    if (!activeModalDir) return;
    const maxStep = serverState[activeModalDir];
    if (maxStep !== undefined && maxStep > activeModalMaxStepRendered) {
        const gallery = document.getElementById('modal-gallery');
        let start = Math.max(0, activeModalMaxStepRendered + 100);
        if (activeModalMaxStepRendered === -100) start = 0;
        
        for (let s = start; s <= maxStep; s += 100) {
            let stepStr = String(s).padStart(5, '0');
            
            const div = document.createElement('div');
            div.className = 'gallery-item';
            div.innerHTML = `
                <img src="${activeModalDir}START_${stepStr}.png" onerror="this.style.display='none'" alt="Start ${s}" title="starting state (post-damage)">
                <img src="${activeModalDir}TARGET_${stepStr}.png" onerror="this.src='${activeModalDir}target.png'; this.onerror=null;" alt="Target Step ${s}" title="target">
                <img src="${activeModalDir}COMP_${stepStr}.png" onerror="this.style.display='none'" alt="Step ${s}" title="model output">
                <img src="${activeModalDir}RECOV_${stepStr}.png" onerror="this.style.display='none'" alt="Recovered ${s}" title="damaged sample after CA">
                <span>Step ${s}</span>
            `;
            gallery.appendChild(div);
        }
        activeModalMaxStepRendered = maxStep;
    }
}

window.prevInteractiveModel = function() {
    const select = document.getElementById("interactive-model-select");
    if (select.options.length > 0) {
        select.selectedIndex = (select.selectedIndex - 1 + select.options.length) % select.options.length;
        if(window.loadInteractiveModel) window.loadInteractiveModel();
    }
}

window.nextInteractiveModel = function() {
    const select = document.getElementById("interactive-model-select");
    if (select.options.length > 0) {
        select.selectedIndex = (select.selectedIndex + 1) % select.options.length;
        if(window.loadInteractiveModel) window.loadInteractiveModel();
    }
}

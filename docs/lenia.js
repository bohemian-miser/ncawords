// Trainable Lenia run gallery — sibling of dashboard.js, scoped to runs
// whose directory name starts with "lenia-". Same public bucket, same
// card/sort/search mechanics and progressive loading as dashboard.js; the
// per-run content is different (COMP pattern snapshots, learned KERNEL
// tile strips, optional COUPLING heatmaps, target.png, run.json).
//
// NOTE: this page only *plays back* pre-rendered PNG snapshots written by
// the training job's WebGL engine. It does not simulate Lenia physics in
// JavaScript — the WebGL engine runs a different update rule than a naive
// JS port would reproduce, so a from-scratch, in-browser Lenia simulator
// matching the trainable variant is left as a separate future task.

let methods = [];
let cardTrackers = [];
const container = document.getElementById('cards-container');

// Public bucket the training jobs write to; readable (and listable)
// anonymously, so a static page needs no backend at all. Same bucket and
// listing endpoint as dashboard.js — only the directory-name prefix filter
// (and the file kinds we look for within each run) differ.
const BUCKET = 'recipe-lanes-nca-jobs';
const BUCKET_BASE = `https://storage.googleapis.com/${BUCKET}/`;
const BUCKET_LIST = `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o?fields=items(name,updated),nextPageToken&maxResults=1000`;
const RUN_PREFIX = 'lenia-';
const BLANK_IMG = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';

function pad5(n) { return String(n).padStart(5, '0'); }

function mergeSteps(dst, src) {
    let changed = false;
    (src || []).forEach(s => {
        if (!dst.includes(s)) { dst.push(s); changed = true; }
    });
    if (changed) dst.sort((a, b) => a - b);
    return changed;
}

async function listLeniaRuns(onPage) {
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
            if (!run.startsWith(RUN_PREFIX)) return;   // Lenia runs only
            if (!runs[run]) {
                runs[run] = {
                    compSteps: [], kernelSteps: [], couplingSteps: [],
                    hasTarget: false, hasRunJson: false, updated: ''
                };
            }
            const r = runs[run];
            if (updated && updated > r.updated) r.updated = updated;
            const m = fname.match(/^(COMP|KERNEL|COUPLING)_(\d+)\.png$/);
            if (m) {
                const step = parseInt(m[2], 10);
                if (!isNaN(step)) {
                    if (m[1] === 'COMP') r.compSteps.push(step);
                    else if (m[1] === 'KERNEL') r.kernelSteps.push(step);
                    else r.couplingSteps.push(step);
                }
            } else if (fname === 'target.png') {
                r.hasTarget = true;
            } else if (fname === 'run.json') {
                r.hasRunJson = true;
            }
        });
        Object.values(runs).forEach(r => {
            r.compSteps.sort((a, b) => a - b);
            r.kernelSteps.sort((a, b) => a - b);
            r.couplingSteps.sort((a, b) => a - b);
        });
        if (onPage) onPage(runs);   // stream cards page by page
        pageToken = d.nextPageToken;
    } while (pageToken);
    return runs;
}

function leniaMethodsFrom(runs) {
    return Object.keys(runs).sort().map(run => ({
        id: 'lenia_' + run,
        title: run,
        dir: BUCKET_BASE + run + '/',
        desc: '',
        tags: [],
        compSteps: runs[run].compSteps,
        kernelSteps: runs[run].kernelSteps,
        couplingSteps: runs[run].couplingSteps,
        hasTarget: runs[run].hasTarget,
        hasRunJson: runs[run].hasRunJson,
        updated: runs[run].updated
    }));
}

const seenIds = new Set();
let sortKey = localStorage.getItem('lenia_sort') || 'newest';

function buildSubtitle(args) {
    if (!args) return '';
    const parts = [];
    if (args.variant) parts.push(args.variant);
    if (args.target) parts.push(args.target);
    if (args.C !== undefined) parts.push(`${args.C}ch`);
    if (args.K !== undefined) parts.push(`${args.K} kernel${args.K === 1 ? '' : 's'}`);
    if (args.params !== undefined) parts.push(`${args.params} params`);
    return parts.join(' · ');
}

function latestLoss(losses) {
    if (!Array.isArray(losses) || losses.length === 0) return null;
    return losses[losses.length - 1];   // [step, loss], assumed step-ascending
}

async function fetchRunJson(tr, m) {
    try {
        const res = await fetch(m.dir + 'run.json?t=' + Date.now());
        if (!res.ok) return;
        const rj = await res.json();
        m.desc = rj.text || '';
        m.tags = rj.tags || [];
        const sub = document.getElementById(`subtitle_${CSS.escape(m.id)}`);
        if (sub) sub.innerText = buildSubtitle(rj.args) || '(no args recorded)';
        tr.runJson = rj;
        renderStatus(tr);
        applyFilters();   // desc/tags just arrived; re-run the search filter
    } catch (e) {
        const sub = document.getElementById(`subtitle_${CSS.escape(m.id)}`);
        if (sub) sub.innerText = '(run.json unavailable)';
    }
}

function renderStatus(tr) {
    const statusObj = tr.statusObj;
    if (!statusObj) return;
    const lastStep = tr.compSteps.length ? tr.compSteps[tr.compSteps.length - 1] : null;
    const rj = tr.runJson;
    const lp = rj ? latestLoss(rj.losses) : null;
    const step = (rj && rj.step !== undefined) ? rj.step : lastStep;
    let text = step !== null && step !== undefined ? `Step ${step}` : 'Waiting for snapshots…';
    if (lp) text += ` — loss ${Number(lp[1]).toFixed(4)}`;
    statusObj.innerText = text;
}

function renderFrame(tr) {
    if (!tr.compSteps.length) return;
    tr.frameIdx = Math.max(0, Math.min(tr.frameIdx, tr.compSteps.length - 1));
    const step = tr.compSteps[tr.frameIdx];
    if (tr.imgObj) {
        tr.imgObj.onerror = function () { this.src = BLANK_IMG; };
        tr.imgObj.src = `${tr.dir}COMP_${pad5(step)}.png`;
    }
    if (tr.scrubObj) tr.scrubObj.value = tr.frameIdx;
    if (tr.frameLabelObj) tr.frameLabelObj.innerText = `step ${step}`;
}

function updateLatestExtras(tr) {
    if (tr.kernelSteps.length && tr.kernelObj) {
        const step = tr.kernelSteps[tr.kernelSteps.length - 1];
        tr.kernelObj.style.display = '';
        tr.kernelObj.onerror = function () { this.style.display = 'none'; };
        tr.kernelObj.src = `${tr.dir}KERNEL_${pad5(step)}.png?t=` + Date.now();
    }
    if (tr.couplingSteps.length && tr.couplingObj) {
        const step = tr.couplingSteps[tr.couplingSteps.length - 1];
        tr.couplingObj.style.display = '';
        tr.couplingObj.onerror = function () { this.style.display = 'none'; };
        tr.couplingObj.src = `${tr.dir}COUPLING_${pad5(step)}.png?t=` + Date.now();
    }
    renderStatus(tr);
}

function updateScrubRange(tr) {
    if (tr.scrubObj) tr.scrubObj.max = Math.max(0, tr.compSteps.length - 1);
}

window.togglePlay = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !tr.compSteps.length) return;
    const btn = tr.playBtn;
    if (tr.playTimer) {
        clearTimeout(tr.playTimer);
        tr.playTimer = null;
        if (btn) btn.innerText = '▶';
        return;
    }
    if (btn) btn.innerText = '⏸';
    const schedule = () => {
        tr.playTimer = setTimeout(() => {
            tr.frameIdx = (tr.frameIdx + 1) % tr.compSteps.length;
            renderFrame(tr);
            schedule();
        }, Math.max(40, 550 - (parseInt(tr.speedObj?.value, 10) || 3) * 50));
    };
    schedule();
};

window.scrubTo = function (id, value) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr) return;
    tr.frameIdx = parseInt(value, 10) || 0;
    renderFrame(tr);
};

function buildCard(m) {
    const card = document.createElement('div');
    card.className = 'card';
    card.id = `card_${m.id}`;
    card.innerHTML = `
        <h3>${m.title}</h3>
        <div class="run-desc" id="subtitle_${m.id}">Loading run.json…</div>
        <div class="img-container" style="height:220px;">
            <img loading="lazy" id="comp_${m.id}" src="${BLANK_IMG}">
        </div>
        <div class="lenia-controls">
            <button id="play_${m.id}" title="Play/pause snapshot animation">&#9654;</button>
            <input type="range" id="scrub_${m.id}" min="0" max="0" value="0" style="width:100px;">
            <span id="frame_${m.id}">step –</span>
            <label style="margin-left:6px;">speed <input type="range" id="speed_${m.id}" min="1" max="10" value="3" style="width:60px;"></label>
        </div>
        <div class="kernel-row">
            <div>
                <div class="sub-desc">Learned kernels</div>
                <div class="img-container"><img loading="lazy" id="kernel_${m.id}" style="display:none;"></div>
            </div>
            <div>
                <div class="sub-desc">Channel coupling</div>
                <div class="img-container"><img loading="lazy" id="coupling_${m.id}" style="display:none;"></div>
            </div>
        </div>
        <div class="target-row">
            <div class="sub-desc">Target</div>
            <div class="img-container"><img loading="lazy" id="target_${m.id}" src="${m.dir}target.png" onerror="this.style.display='none'"></div>
        </div>
        <div class="status" id="status_${m.id}">Loading…</div>
    `;
    container.appendChild(card);

    const esc = CSS.escape(m.id);
    const tr = {
        id: m.id,
        dir: m.dir,
        updated: m.updated || '',
        compSteps: [...m.compSteps],
        kernelSteps: [...m.kernelSteps],
        couplingSteps: [...m.couplingSteps],
        frameIdx: Math.max(0, m.compSteps.length - 1),
        playTimer: null,
        runJson: null,
        cardObj: card,
        imgObj: card.querySelector(`#comp_${esc}`),
        kernelObj: card.querySelector(`#kernel_${esc}`),
        couplingObj: card.querySelector(`#coupling_${esc}`),
        scrubObj: card.querySelector(`#scrub_${esc}`),
        speedObj: card.querySelector(`#speed_${esc}`),
        frameLabelObj: card.querySelector(`#frame_${esc}`),
        statusObj: card.querySelector(`#status_${esc}`),
        playBtn: card.querySelector(`#play_${esc}`)
    };
    tr.playBtn.onclick = () => window.togglePlay(m.id);
    tr.scrubObj.oninput = (e) => window.scrubTo(m.id, e.target.value);

    updateScrubRange(tr);
    renderFrame(tr);
    updateLatestExtras(tr);
    fetchRunJson(tr, m);

    cardTrackers.push(tr);
}

function addOrUpdateCards(list) {
    let added = false;
    let resort = false;
    list.forEach(m => {
        if (seenIds.has(m.id)) {
            const tr = cardTrackers.find(t => t.id === m.id);
            const known = methods.find(x => x.id === m.id);
            if (tr) {
                const gotComp = mergeSteps(tr.compSteps, m.compSteps);
                mergeSteps(tr.kernelSteps, m.kernelSteps);
                mergeSteps(tr.couplingSteps, m.couplingSteps);
                updateScrubRange(tr);
                if (gotComp && !tr.playTimer) {
                    tr.frameIdx = tr.compSteps.length - 1;   // jump to latest
                    renderFrame(tr);
                }
                updateLatestExtras(tr);
            }
            if (known && m.updated && m.updated !== known.updated) {
                known.updated = m.updated;
                resort = true;
            }
            return;
        }
        seenIds.add(m.id);
        methods.push(m);
        added = true;
        buildCard(m);
    });
    if (added || resort) sortCards();
}

window.setSort = function (k) {
    sortKey = k;
    localStorage.setItem('lenia_sort', k);
    sortCards();
};

function methodComparator(a, b) {
    if (sortKey === 'name') return a.title.localeCompare(b.title);
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

function applyFilters() {
    const q = (document.getElementById('search-box')?.value || '').toLowerCase();
    methods.forEach(m => {
        const el = document.getElementById(`card_${m.id}`);
        if (!el) return;
        const hay = (m.title + ' ' + (m.desc || '') + ' '
                     + (m.tags || []).join(' ')).toLowerCase();
        el.style.display = (!q || hay.includes(q)) ? 'block' : 'none';
    });
}
window.applyFilters = applyFilters;

async function refreshRuns() {
    try {
        const runs = await listLeniaRuns();
        addOrUpdateCards(leniaMethodsFrom(runs));
    } catch (e) { console.error('lenia refresh failed', e); }
}

async function bootstrap() {
    const sortSel = document.getElementById('sort-select');
    if (sortSel) sortSel.value = sortKey;
    await listLeniaRuns(runs => addOrUpdateCards(leniaMethodsFrom(runs)));
    setInterval(refreshRuns, 20000);   // new runs / snapshots appear without reload
}

bootstrap().catch(err => console.error('Lenia gallery bootstrap failed', err));

// Trainable Lenia run gallery — sibling of dashboard.js, scoped to runs
// whose directory name starts with "lenia-". Same public bucket, same
// card/sort/search mechanics and progressive loading as dashboard.js; the
// per-run content is different (COMP pattern snapshots, learned KERNEL
// tile strips, optional COUPLING heatmaps, target.png, run.json).
//
// The COMP training-snapshot animation, the KERNEL tile strip and the
// COUPLING heatmap all share one timeline: scrubbing/playing the COMP
// animation re-picks the nearest KERNEL/COUPLING frame at or before the
// current step, so "learned kernels" visibly evolve in sync with training.
//
// Each card also offers a "Run live" widget that fetches the run's
// exported weights.json and steps the actual trained Lenia physics in the
// browser via lenia_engine.js (LeniaCA) — a real from-scratch simulator,
// distinct from the pre-rendered PNG timelapse above it.

import { LeniaCA } from './lenia_engine.js';

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
    // Two-stage listing so we never page the whole bucket:
    //   1. delimiter listing under prefix 'lenia-' -> just the run directory
    //      names (one small request);
    //   2. one per-run listing for its files, streamed as each arrives.
    const runs = {};
    const dirRes = await fetch(
        `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o` +
        `?prefix=${RUN_PREFIX}&delimiter=/&fields=prefixes&maxResults=1000`);
    if (!dirRes.ok) throw new Error(`bucket dir list failed: ${dirRes.status}`);
    const dirs = ((await dirRes.json()).prefixes || []).map(p => p.slice(0, -1));

    await Promise.all(dirs.map(async run => {
        const res = await fetch(
            `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o` +
            `?prefix=${encodeURIComponent(run + '/')}` +
            `&fields=items(name,updated)&maxResults=1000`);
        if (!res.ok) return;
        const d = await res.json();
        const r = {
            compSteps: [], kernelSteps: [], couplingSteps: [],
            hasTarget: false, hasRunJson: false, updated: ''
        };
        (d.items || []).forEach(({name, updated}) => {
            const fname = name.slice(run.length + 1);
            if (updated && updated > r.updated) r.updated = updated;
            const m = fname.match(/^(COMP|KERNEL|COUPLING)_(\d+)\.png$/);
            if (m) {
                const step = parseInt(m[2], 10);
                if (!isNaN(step)) {
                    if (m[1] === 'COMP') r.compSteps.push(step);
                    else if (m[1] === 'KERNEL') r.kernelSteps.push(step);
                    else r.couplingSteps.push(step);
                }
            } else if (fname === 'target.png') r.hasTarget = true;
            else if (fname === 'run.json') r.hasRunJson = true;
        });
        r.compSteps.sort((a, b) => a - b);
        r.kernelSteps.sort((a, b) => a - b);
        r.couplingSteps.sort((a, b) => a - b);
        runs[run] = r;
        if (onPage) onPage(runs);   // stream cards as each run's listing lands
    }));
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

// Nearest step <= target in an ascending-sorted array; falls back to the
// smallest available step if every recorded step is after `target` (e.g.
// the kernel snapshot cadence is coarser than the COMP cadence and hasn't
// produced a frame yet at low steps).
function nearestStepAtOrBelow(steps, target) {
    if (!steps.length) return null;
    let best = null;
    for (const s of steps) {
        if (s <= target) best = s;
        else break;
    }
    return best === null ? steps[0] : best;
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
    renderKernelFrame(tr, step);
}

// Keeps the KERNEL/COUPLING images locked to the same timeline position as
// the COMP animation: whatever step COMP is showing, show the nearest
// kernel/coupling snapshot at or before that step.
function renderKernelFrame(tr, compStep) {
    const kStep = nearestStepAtOrBelow(tr.kernelSteps, compStep);
    if (tr.kernelObj) {
        if (kStep !== null) {
            tr.kernelObj.style.display = '';
            tr.kernelObj.onerror = function () { this.style.display = 'none'; };
            tr.kernelObj.src = `${tr.dir}KERNEL_${pad5(kStep)}.png`;
        } else {
            tr.kernelObj.style.display = 'none';
        }
    }
    if (tr.kernelLabelObj) {
        tr.kernelLabelObj.innerText = kStep !== null
            ? `learned kernels @ step ${kStep}` : 'learned kernels';
    }

    const cStep = nearestStepAtOrBelow(tr.couplingSteps, compStep);
    if (tr.couplingObj) {
        if (cStep !== null) {
            tr.couplingObj.style.display = '';
            tr.couplingObj.onerror = function () { this.style.display = 'none'; };
            tr.couplingObj.src = `${tr.dir}COUPLING_${pad5(cStep)}.png`;
        } else {
            tr.couplingObj.style.display = 'none';
        }
    }
    if (tr.couplingLabelObj) {
        tr.couplingLabelObj.innerText = cStep !== null
            ? `channel coupling @ step ${cStep}` : 'channel coupling';
    }
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
                <div class="sub-desc" id="kernel_label_${m.id}">learned kernels</div>
                <div class="img-container"><img loading="lazy" id="kernel_${m.id}" style="display:none;"></div>
            </div>
            <div>
                <div class="sub-desc" id="coupling_label_${m.id}">channel coupling</div>
                <div class="img-container"><img loading="lazy" id="coupling_${m.id}" style="display:none;"></div>
            </div>
        </div>
        <div class="target-row">
            <div class="sub-desc">Target</div>
            <div class="img-container"><img loading="lazy" id="target_${m.id}" src="${m.dir}target.png" onerror="this.style.display='none'"></div>
        </div>
        <div class="status" id="status_${m.id}">Loading…</div>
        <div class="lenia-live-toggle">
            <button id="livetoggle_${m.id}">&#9654; Run live</button>
        </div>
        <div class="lenia-live" id="live_${m.id}" style="display:none;">
            <div class="sub-desc">live trained physics</div>
            <div class="live-canvas-wrap">
                <canvas id="livecanvas_${m.id}" width="64" height="64"></canvas>
            </div>
            <div class="lenia-controls">
                <button id="liveplay_${m.id}" title="Play/pause live simulation">&#9654;</button>
                <button id="livereset_${m.id}" title="Reset to random noise">Reset noise</button>
                <button id="liveclear_${m.id}" title="Clear to empty">Clear</button>
                <label style="margin-left:6px;">speed <input type="range" id="livespeed_${m.id}" min="0.1" max="10" step="0.1" value="1" style="width:60px;"></label>
            </div>
            <div class="run-desc" id="livestatus_${m.id}"></div>
        </div>
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
        kernelLabelObj: card.querySelector(`#kernel_label_${esc}`),
        couplingLabelObj: card.querySelector(`#coupling_label_${esc}`),
        scrubObj: card.querySelector(`#scrub_${esc}`),
        speedObj: card.querySelector(`#speed_${esc}`),
        frameLabelObj: card.querySelector(`#frame_${esc}`),
        statusObj: card.querySelector(`#status_${esc}`),
        playBtn: card.querySelector(`#play_${esc}`),
        // --- live physics widget state ---
        liveToggleBtn: card.querySelector(`#livetoggle_${esc}`),
        liveSecObj: card.querySelector(`#live_${esc}`),
        liveCanvas: card.querySelector(`#livecanvas_${esc}`),
        livePlayBtn: card.querySelector(`#liveplay_${esc}`),
        liveSpeedObj: card.querySelector(`#livespeed_${esc}`),
        liveStatusObj: card.querySelector(`#livestatus_${esc}`),
        liveCA: null,
        liveImgData: null,
        liveTimer: null,
        liveStepAccum: 0,
        livePaused: false,
        liveDamaging: false
    };
    tr.playBtn.onclick = () => window.togglePlay(m.id);
    tr.scrubObj.oninput = (e) => window.scrubTo(m.id, e.target.value);

    tr.liveCtx = tr.liveCanvas.getContext('2d');
    tr.liveToggleBtn.onclick = () => activateOrCollapseLive(tr);
    tr.livePlayBtn.onclick = () => window.toggleLivePause(m.id);
    tr.liveresetBtn = card.querySelector(`#livereset_${esc}`);
    tr.liveresetBtn.onclick = () => window.liveResetNoise(m.id);
    tr.liveclearBtn = card.querySelector(`#liveclear_${esc}`);
    tr.liveclearBtn.onclick = () => window.liveClear(m.id);
    tr.liveCanvas.addEventListener('mousedown', (e) => { tr.liveDamaging = true; liveDamageAt(tr, e); });
    tr.liveCanvas.addEventListener('mousemove', (e) => { if (tr.liveDamaging) liveDamageAt(tr, e); });
    tr.liveCanvas.addEventListener('mouseup', () => { tr.liveDamaging = false; });
    tr.liveCanvas.addEventListener('mouseleave', () => { tr.liveDamaging = false; });

    updateScrubRange(tr);
    renderFrame(tr);
    renderStatus(tr);
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
                const gotKernel = mergeSteps(tr.kernelSteps, m.kernelSteps);
                const gotCoupling = mergeSteps(tr.couplingSteps, m.couplingSteps);
                updateScrubRange(tr);
                if (gotComp && !tr.playTimer) {
                    tr.frameIdx = tr.compSteps.length - 1;   // jump to latest
                    renderFrame(tr);   // also re-syncs kernel/coupling frame
                } else if ((gotKernel || gotCoupling) && tr.compSteps.length) {
                    // New kernel/coupling snapshots landed without a new COMP
                    // frame (or while paused mid-scrub) — resync at the step
                    // currently on screen rather than jumping the timeline.
                    renderKernelFrame(tr, tr.compSteps[tr.frameIdx]);
                }
                renderStatus(tr);
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

// ---------------------------------------------------------------------
// Live trained-physics widget — steps the actual LeniaCA engine from a
// run's exported weights.json, distinct from the pre-rendered PNG
// timelapse above it in the card. Only one card's simulation runs at a
// time: activating one stops whichever other card's loop was running.
// ---------------------------------------------------------------------

let currentLiveTr = null;   // module-level: the one card currently ticking

function stopLiveLoop(tr) {
    if (tr.liveTimer) { clearTimeout(tr.liveTimer); tr.liveTimer = null; }
    if (tr.livePlayBtn) tr.livePlayBtn.innerText = '▶';
}

function drawLive(tr) {
    if (!tr.liveCA || !tr.liveImgData) return;
    const out = tr.liveCA.readRGBA(tr.liveImgData.data);
    if (out && out !== tr.liveImgData.data) tr.liveImgData.data.set(out);
    tr.liveCtx.putImageData(tr.liveImgData, 0, 0);
}

function runLiveLoop(tr) {
    if (!tr.liveCA || tr.livePaused) return;
    // Fractional speeds (< 1 step/tick) accumulate across ticks, mirroring
    // the dashboard's interactive-widget loop (nca_viewer.js).
    tr.liveStepAccum += parseFloat(tr.liveSpeedObj?.value || '1');
    const steps = Math.floor(tr.liveStepAccum);
    tr.liveStepAccum -= steps;
    for (let i = 0; i < steps; i++) tr.liveCA.step();
    if (steps > 0) drawLive(tr);
    tr.liveTimer = setTimeout(() => runLiveLoop(tr), 30);
}

async function activateOrCollapseLive(tr) {
    const isOpen = tr.liveSecObj.style.display !== 'none' && tr.liveSecObj.style.display !== '';
    if (isOpen) {
        stopLiveLoop(tr);
        tr.liveSecObj.style.display = 'none';
        tr.liveToggleBtn.innerText = '▶ Run live';
        if (currentLiveTr === tr) currentLiveTr = null;
        return;
    }

    // Only one live widget runs at a time.
    if (currentLiveTr && currentLiveTr !== tr) {
        stopLiveLoop(currentLiveTr);
        currentLiveTr.liveSecObj.style.display = 'none';
        currentLiveTr.liveToggleBtn.innerText = '▶ Run live';
    }
    currentLiveTr = tr;

    tr.liveSecObj.style.display = 'block';
    tr.liveToggleBtn.innerText = '▼ Hide live';
    tr.livePaused = false;
    if (tr.livePlayBtn) tr.livePlayBtn.innerText = '⏸';

    if (tr.liveCA) {   // already loaded from a previous activation — resume
        tr.liveStatusObj.innerText = 'live trained physics — click/drag to damage';
        runLiveLoop(tr);
        return;
    }

    tr.liveStatusObj.innerText = 'loading weights…';
    try {
        const res = await fetch(tr.dir + 'weights.json?t=' + Date.now());
        if (!res.ok) {
            tr.liveStatusObj.innerText = 'weights not exported yet';
            return;
        }
        const weights = await res.json();
        tr.liveCA = new LeniaCA(weights, 64);
        tr.liveImgData = tr.liveCtx.createImageData(64, 64);
        tr.liveCanvas.style.imageRendering = 'pixelated';
        tr.liveStepAccum = 0;
        drawLive(tr);
        tr.liveStatusObj.innerText = 'live trained physics — click/drag to damage';
        runLiveLoop(tr);
    } catch (e) {
        console.error('live physics load failed', e);
        tr.liveStatusObj.innerText = 'weights not exported yet';
    }
}

window.toggleLivePause = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !tr.liveCA) return;
    tr.livePaused = !tr.livePaused;
    if (tr.livePlayBtn) tr.livePlayBtn.innerText = tr.livePaused ? '▶' : '⏸';
    if (tr.livePaused) stopLiveLoop(tr); else runLiveLoop(tr);
};

window.liveResetNoise = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !tr.liveCA) return;
    tr.liveCA.reset(true);
    drawLive(tr);
};

window.liveClear = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !tr.liveCA) return;
    tr.liveCA.reset(false);
    drawLive(tr);
};

function liveDamageAt(tr, e) {
    if (!tr.liveCA) return;
    const rect = tr.liveCanvas.getBoundingClientRect();
    const normX = (e.clientX - rect.left) / rect.width;
    const normY = (e.clientY - rect.top) / rect.height;
    tr.liveCA.damage(normX * 64, normY * 64, 6);
    drawLive(tr);
}

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

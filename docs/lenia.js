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

import { LeniaCA } from './lenia_engine.js?v=t0mode';

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
// The original trainable-Lenia campaign wrote run dirs as 'lenia-*'; later
// campaigns ('coupling-weight' sweeps and a planned follow-up) write
// 'cw-*' and 'p2-*' instead. List all three prefixes so the gallery
// doesn't silently miss newer runs.
const RUN_PREFIXES = ['lenia-', 'cw-', 'p2-'];
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
    //   1. one delimiter listing per prefix in RUN_PREFIXES -> just the run
    //      directory names (one small request each, merged into one list);
    //   2. one per-run listing for its files, streamed as each arrives.
    const runs = {};
    const dirLists = await Promise.all(RUN_PREFIXES.map(async prefix => {
        const dirRes = await fetch(
            `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o` +
            `?prefix=${prefix}&delimiter=/&fields=prefixes&maxResults=1000`);
        if (!dirRes.ok) throw new Error(`bucket dir list failed (${prefix}): ${dirRes.status}`);
        return ((await dirRes.json()).prefixes || []).map(p => p.slice(0, -1));
    }));
    const dirs = [...new Set(dirLists.flat())];

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
    // sharedk has ONE kernel by design (kernel + coupling matrix); the K
    // arg is inert for it and would mislabel the card.
    if (args.variant === 'sharedk') parts.push('1 kernel (shared)');
    else if (args.K !== undefined) parts.push(`${args.K} kernel${args.K === 1 ? '' : 's'}`);
    if (args.params !== undefined) parts.push(`${args.params} params`);
    // cw-*/p2-* campaign runs carry extra args the original lenia-* runs
    // didn't; surface whichever of these are present.
    if (args.cond !== undefined) parts.push(`cond:${args.cond}`);
    if (args.scaf_strength !== undefined) parts.push(`scaf ${args.scaf_strength}`);
    if (args.size !== undefined) parts.push(`${args.size}px`);
    if (args.train_init !== undefined) parts.push(args.train_init ? 'train_init' : 'no train_init');
    return parts.join(' · ');
}

function latestLoss(losses) {
    if (!Array.isArray(losses) || losses.length === 0) return null;
    return losses[losses.length - 1];   // [step, loss], assumed step-ascending
}

// loss_rel isn't present on any run we've seen yet, but the training job
// may start emitting it on a trailing history/losses entry — check both
// spots defensively rather than assuming a fixed schema.
function extractLossRel(rj) {
    if (!rj) return null;
    if (Array.isArray(rj.history) && rj.history.length) {
        const last = rj.history[rj.history.length - 1];
        if (last && typeof last === 'object' && last.loss_rel !== undefined) return last.loss_rel;
    }
    if (Array.isArray(rj.losses) && rj.losses.length) {
        const last = rj.losses[rj.losses.length - 1];
        if (last && typeof last === 'object' && !Array.isArray(last) && last.loss_rel !== undefined) return last.loss_rel;
    }
    return null;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

async function fetchRunJson(tr, m) {
    try {
        const res = await fetch(m.dir + 'run.json?t=' + Date.now());
        if (!res.ok) return;
        const rj = await res.json();
        m.desc = rj.text || '';
        m.tags = rj.tags || [];
        m.args = rj.args || null;
        m.losses = Array.isArray(rj.losses) ? rj.losses : [];
        const lossVals = m.losses.map(p => Number(p[1])).filter(Number.isFinite);
        m.finalLoss = lossVals.length ? lossVals[lossVals.length - 1] : null;
        m.minLoss = lossVals.length ? Math.min(...lossVals) : null;
        m.lossRel = extractLossRel(rj);
        const sub = document.getElementById(`subtitle_${CSS.escape(m.id)}`);
        if (sub) sub.innerText = buildSubtitle(rj.args) || '(no args recorded)';
        tr.runJson = rj;
        renderStatus(tr);
        drawSparkline(tr, m);
        updateFilterBounds();
        applyFilters();   // desc/tags/args/loss just arrived; re-run all filters
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

// ---------------------------------------------------------------------
// Loss-curve rendering — shared by the per-card sparkline and the large
// graph in the detail modal. Auto-switches to a log-y axis whenever the
// curve spans more than ~1.5 orders of magnitude, since Lenia losses can
// start near 1 and settle two-plus decades lower.
// ---------------------------------------------------------------------

function computeLossScale(values) {
    const finite = values.filter(v => Number.isFinite(v) && v > 0);
    if (!finite.length) return null;
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    const useLog = min > 0 && (max / min) > 30;
    return { min, max, useLog };
}

function lossY(v, scale, h, padTop, padBottom) {
    const usableH = h - padTop - padBottom;
    const vClamped = Math.max(v, scale.min);
    let t;
    if (scale.useLog) {
        const lo = Math.log(scale.min);
        const hi = Math.log(Math.max(scale.max, scale.min * 1.0001));
        t = (Math.log(vClamped) - lo) / (hi - lo || 1);
    } else {
        t = (vClamped - scale.min) / ((scale.max - scale.min) || 1);
    }
    return padTop + (1 - t) * usableH;
}

function drawLossCurve(canvas, losses, { big = false } = {}) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!Array.isArray(losses) || losses.length === 0) {
        ctx.fillStyle = '#666';
        ctx.font = (big ? '12px' : '9px') + ' sans-serif';
        ctx.fillText('no loss data', 4, h / 2);
        return;
    }
    const values = losses.map(p => Number(p[1]));
    const scale = computeLossScale(values);
    if (!scale) {
        ctx.fillStyle = '#666';
        ctx.font = (big ? '12px' : '9px') + ' sans-serif';
        ctx.fillText('no loss data', 4, h / 2);
        return;
    }
    const padLeft = big ? 46 : 2;
    const padRight = big ? 10 : 2;
    const padTop = big ? 10 : 3;
    const padBottom = big ? 20 : 3;
    const n = losses.length;

    ctx.strokeStyle = '#4db8ff';
    ctx.lineWidth = big ? 1.5 : 1;
    ctx.beginPath();
    losses.forEach((p, i) => {
        const x = padLeft + (n > 1 ? (i / (n - 1)) : 1) * (w - padLeft - padRight);
        const y = lossY(Number(p[1]), scale, h, padTop, padBottom);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // current (final) value as a dot
    const lastX = padLeft + (w - padLeft - padRight);
    const lastY = lossY(values[values.length - 1], scale, h, padTop, padBottom);
    ctx.fillStyle = '#ff9f40';
    ctx.beginPath();
    ctx.arc(lastX, lastY, big ? 3.5 : 2, 0, Math.PI * 2);
    ctx.fill();

    if (big) {
        ctx.fillStyle = '#999';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(scale.max.toFixed(4), padLeft - 6, padTop + 8);
        ctx.fillText(scale.min.toFixed(4), padLeft - 6, h - padBottom);
        ctx.textAlign = 'left';
        ctx.fillStyle = '#666';
        ctx.fillText(scale.useLog ? 'log scale' : 'linear scale', padLeft, h - 4);
    }
}

function drawSparkline(tr, m) {
    if (!tr.sparkCanvas) return;
    drawLossCurve(tr.sparkCanvas, m.losses, { big: false });
    if (tr.sparkLabelObj) {
        tr.sparkLabelObj.innerText = (typeof m.finalLoss === 'number')
            ? m.finalLoss.toFixed(4) : '–';
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
        <h3>
            <span class="lenia-card-title" id="title_${m.id}" title="Click for run details">${m.title}</span>
            <button class="info-btn" id="info_${m.id}" title="Run details">&#9432;</button>
        </h3>
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
                <div class="img-container"><img loading="lazy" id="coupling_${m.id}" style="display:none;" title="channel coupling (red +, blue −, gray 0; older runs: grayscale)"></div>
            </div>
        </div>
        <div class="target-row">
            <div class="target-row-flex">
                <div>
                    <div class="sub-desc">Target</div>
                    <div class="img-container" style="height:90px;width:90px;"><img loading="lazy" id="target_${m.id}" src="${m.dir}target.png" onerror="this.style.display='none'"></div>
                </div>
                <div class="spark-wrap">
                    <div class="sub-desc">loss <span id="spark_label_${m.id}">–</span></div>
                    <canvas id="spark_${m.id}" width="160" height="90"></canvas>
                </div>
            </div>
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
                <button id="liveseed_${m.id}" title="Reset the way training started (seed blob / scaffold)">Seed</button>
                <button id="liveclear_${m.id}" title="Clear ALL channels including the stencil">Clear</button>
                <button id="livestencil_${m.id}" title="Toggle the clamped prepattern channel (shown in blue)" style="display:none;">Stencil: on</button>
                <button id="livechans_${m.id}" title="Show every channel as a grayscale heatmap">Channels</button>
                <label style="margin-left:6px;">speed <input type="range" id="livespeed_${m.id}" min="0.1" max="10" step="0.1" value="1" style="width:60px;"></label>
            </div>
            <div class="run-desc" id="livestatus_${m.id}"></div>
            <div id="livechangrid_${m.id}" style="display:none; gap:4px; flex-wrap:wrap; margin-top:6px;"></div>
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
        sparkCanvas: card.querySelector(`#spark_${esc}`),
        sparkLabelObj: card.querySelector(`#spark_label_${esc}`),
        titleObj: card.querySelector(`#title_${esc}`),
        infoBtn: card.querySelector(`#info_${esc}`),
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
    tr.titleObj.onclick = () => openLeniaModal(m.id);
    tr.infoBtn.onclick = () => openLeniaModal(m.id);
    drawSparkline(tr, m);

    tr.liveCtx = tr.liveCanvas.getContext('2d');
    tr.liveToggleBtn.onclick = () => activateOrCollapseLive(tr);
    tr.livePlayBtn.onclick = () => window.toggleLivePause(m.id);
    tr.liveresetBtn = card.querySelector(`#livereset_${esc}`);
    tr.liveresetBtn.onclick = () => window.liveResetNoise(m.id);
    tr.liveseedBtn = document.getElementById(`liveseed_${m.id}`);
    if (tr.liveseedBtn) tr.liveseedBtn.onclick = () => window.liveSeed(m.id);
    tr.liveclearBtn = card.querySelector(`#liveclear_${esc}`);
    tr.liveclearBtn.onclick = () => window.liveClear(m.id);
    tr.livestencilBtn = document.getElementById(`livestencil_${m.id}`);
    if (tr.livestencilBtn) tr.livestencilBtn.onclick = () => window.liveStencil(m.id);
    tr.livechansBtn = document.getElementById(`livechans_${m.id}`);
    tr.livechanGrid = document.getElementById(`livechangrid_${m.id}`);
    if (tr.livechansBtn) tr.livechansBtn.onclick = () => window.liveChannels(m.id);
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

// ---------------------------------------------------------------------
// Structured filters (channels / kernels / params / final loss), layered
// on top of the existing text search. Bounds for the dropdowns and range
// sliders are derived from whatever run.json args/losses have loaded so
// far and widen as more runs stream in; a card with no run.json yet keeps
// showing normally *unless* some filter has actually been narrowed away
// from its full-range default, at which point unknown-data cards drop out
// (we can't tell if they'd match, so we don't claim they do).
// ---------------------------------------------------------------------

const filterBounds = { paramsMin: null, paramsMax: null, lossMin: null, lossMax: null };
let paramsUserTouched = false;
let lossUserTouched = false;

// Loss range slider positions are 0..1000 mapped log-scale onto
// [filterBounds.lossMin, filterBounds.lossMax] so the two ends of the
// slider stay usable even though final losses can span decades.
function lossSliderToValue(pos) {
    const { lossMin, lossMax } = filterBounds;
    if (lossMin === null || lossMax === null || lossMax <= lossMin) return lossMin || 0;
    const lo = Math.log(Math.max(lossMin, 1e-9));
    const hi = Math.log(Math.max(lossMax, lossMin * 1.0001, 1e-9));
    const t = Math.min(1, Math.max(0, pos / 1000));
    return Math.exp(lo + t * (hi - lo));
}

function fmtLoss(v) { return Number.isFinite(v) ? v.toFixed(4) : '–'; }

function updateParamsLabels() {
    const minInp = document.getElementById('params-min');
    const maxInp = document.getElementById('params-max');
    document.getElementById('params-min-label').innerText = minInp ? minInp.value : '–';
    document.getElementById('params-max-label').innerText = maxInp ? maxInp.value : '–';
}

function updateLossLabels() {
    const minInp = document.getElementById('loss-min');
    const maxInp = document.getElementById('loss-max');
    if (minInp) document.getElementById('loss-min-label').innerText = fmtLoss(lossSliderToValue(parseFloat(minInp.value)));
    if (maxInp) document.getElementById('loss-max-label').innerText = fmtLoss(lossSliderToValue(parseFloat(maxInp.value)));
}

window.onParamsRangeInput = function () {
    paramsUserTouched = true;
    updateParamsLabels();
    applyFilters();
};

window.onLossRangeInput = function () {
    lossUserTouched = true;
    updateLossLabels();
    applyFilters();
};

// Rebuilds the channels/kernels dropdown options and the params/loss range
// bounds from every method's loaded args/losses. Only widens bounds (never
// shrinks), and only snaps slider positions back to the full range while
// the user hasn't touched that slider yet, so an in-progress filter isn't
// clobbered by a later-arriving run.
function updateFilterBounds() {
    const cVals = new Set();
    const kVals = new Set();
    let pMin = null, pMax = null, lMin = null, lMax = null;
    methods.forEach(m => {
        const a = m.args;
        if (a) {
            if (a.C !== undefined) cVals.add(a.C);
            if (a.K !== undefined) kVals.add(a.K);
            if (typeof a.params === 'number') {
                pMin = (pMin === null) ? a.params : Math.min(pMin, a.params);
                pMax = (pMax === null) ? a.params : Math.max(pMax, a.params);
            }
        }
        if (typeof m.finalLoss === 'number' && m.finalLoss > 0) {
            lMin = (lMin === null) ? m.finalLoss : Math.min(lMin, m.finalLoss);
            lMax = (lMax === null) ? m.finalLoss : Math.max(lMax, m.finalLoss);
        }
    });

    const chSel = document.getElementById('filter-channels');
    if (chSel) {
        const cur = chSel.value;
        const sorted = [...cVals].sort((a, b) => a - b);
        chSel.innerHTML = '<option value="any">Any</option>'
            + sorted.map(c => `<option value="${c}">${c}</option>`).join('');
        chSel.value = sorted.some(c => String(c) === cur) ? cur : 'any';
    }
    const kSel = document.getElementById('filter-kernels');
    if (kSel) {
        const cur = kSel.value;
        const sorted = [...kVals].sort((a, b) => a - b);
        kSel.innerHTML = '<option value="any">Any</option>'
            + sorted.map(k => `<option value="${k}">${k}</option>`).join('');
        kSel.value = sorted.some(k => String(k) === cur) ? cur : 'any';
    }

    if (pMin !== null) {
        filterBounds.paramsMin = (filterBounds.paramsMin === null) ? pMin : Math.min(filterBounds.paramsMin, pMin);
        filterBounds.paramsMax = (filterBounds.paramsMax === null) ? pMax : Math.max(filterBounds.paramsMax, pMax);
        const minInp = document.getElementById('params-min');
        const maxInp = document.getElementById('params-max');
        if (minInp && maxInp) {
            minInp.min = maxInp.min = filterBounds.paramsMin;
            minInp.max = maxInp.max = filterBounds.paramsMax;
            if (!paramsUserTouched) {
                minInp.value = filterBounds.paramsMin;
                maxInp.value = filterBounds.paramsMax;
            }
        }
    }
    if (lMin !== null) {
        filterBounds.lossMin = (filterBounds.lossMin === null) ? lMin : Math.min(filterBounds.lossMin, lMin);
        filterBounds.lossMax = (filterBounds.lossMax === null) ? lMax : Math.max(filterBounds.lossMax, lMax);
        if (!lossUserTouched) {
            const minInp = document.getElementById('loss-min');
            const maxInp = document.getElementById('loss-max');
            if (minInp) minInp.value = 0;
            if (maxInp) maxInp.value = 1000;
        }
    }
    updateParamsLabels();
    updateLossLabels();
}

function applyFilters() {
    const q = (document.getElementById('search-box')?.value || '').toLowerCase();
    const chSel = document.getElementById('filter-channels');
    const kSel = document.getElementById('filter-kernels');
    const chVal = chSel ? chSel.value : 'any';
    const kVal = kSel ? kSel.value : 'any';

    const pMinInp = document.getElementById('params-min');
    const pMaxInp = document.getElementById('params-max');
    const lMinInp = document.getElementById('loss-min');
    const lMaxInp = document.getElementById('loss-max');

    let paramsNarrowed = false, pLo = null, pHi = null;
    if (pMinInp && pMaxInp && filterBounds.paramsMin !== null && filterBounds.paramsMax > filterBounds.paramsMin) {
        const a = parseFloat(pMinInp.value), b = parseFloat(pMaxInp.value);
        pLo = Math.min(a, b); pHi = Math.max(a, b);
        paramsNarrowed = pLo > filterBounds.paramsMin || pHi < filterBounds.paramsMax;
    }

    let lossNarrowed = false, lLo = null, lHi = null;
    if (lMinInp && lMaxInp && filterBounds.lossMin !== null && filterBounds.lossMax > filterBounds.lossMin) {
        const posA = parseFloat(lMinInp.value), posB = parseFloat(lMaxInp.value);
        lLo = Math.min(lossSliderToValue(posA), lossSliderToValue(posB));
        lHi = Math.max(lossSliderToValue(posA), lossSliderToValue(posB));
        lossNarrowed = Math.min(posA, posB) > 0 || Math.max(posA, posB) < 1000;
    }

    const structuredActive = chVal !== 'any' || kVal !== 'any' || paramsNarrowed || lossNarrowed;

    methods.forEach(m => {
        const el = document.getElementById(`card_${m.id}`);
        if (!el) return;
        const hay = (m.title + ' ' + (m.desc || '') + ' '
                     + (m.tags || []).join(' ')).toLowerCase();
        let visible = (!q || hay.includes(q));

        if (visible && structuredActive) {
            const args = m.args;
            if (!args) {
                // No run.json yet: can't evaluate a narrowed filter against
                // it, so hide it rather than guess.
                visible = false;
            } else {
                if (chVal !== 'any' && String(args.C) !== chVal) visible = false;
                if (visible && kVal !== 'any' && String(args.K) !== kVal) visible = false;
                if (visible && paramsNarrowed) {
                    const p = args.params;
                    if (typeof p !== 'number' || p < pLo || p > pHi) visible = false;
                }
                if (visible && lossNarrowed) {
                    const fl = m.finalLoss;
                    if (typeof fl !== 'number' || fl < lLo || fl > lHi) visible = false;
                }
            }
        }
        el.style.display = visible ? 'block' : 'none';
    });
}
window.applyFilters = applyFilters;

// ---------------------------------------------------------------------
// Detail popup — built once in lenia.html (#lenia-modal) and repopulated
// per card. Shows every args key/value, tags, loss stats, a large log-y
// loss graph, the target image, and the latest COMP/KERNEL/COUPLING
// snapshots, plus direct links into the bucket.
// ---------------------------------------------------------------------

function setModalImage(imgId, labelId, labelPrefix, src, step) {
    const img = document.getElementById(imgId);
    const label = labelId ? document.getElementById(labelId) : null;
    if (!img) return;
    if (src) {
        img.onerror = function () { this.style.display = 'none'; };
        img.style.display = '';
        img.src = src;
        if (label) label.innerText = step !== null && step !== undefined
            ? `${labelPrefix} @ step ${step}` : labelPrefix;
    } else {
        img.style.display = 'none';
        if (label) label.innerText = labelPrefix;
    }
}

function openLeniaModal(id) {
    const tr = cardTrackers.find(t => t.id === id);
    const m = methods.find(x => x.id === id);
    if (!tr || !m) return;

    document.getElementById('lm-title').innerText = m.title;
    document.getElementById('lm-desc').innerText = m.desc || '(no run.json yet)';

    const argsTbl = document.getElementById('lm-args');
    argsTbl.innerHTML = '';
    if (m.args && Object.keys(m.args).length) {
        Object.keys(m.args).forEach(k => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${escapeHtml(k)}</td><td>${escapeHtml(String(m.args[k]))}</td>`;
            argsTbl.appendChild(row);
        });
    } else {
        argsTbl.innerHTML = '<tr><td style="color:#666;">no args recorded</td></tr>';
    }

    document.getElementById('lm-tags').innerText =
        (m.tags && m.tags.length) ? m.tags.join(', ') : '(none)';

    let statTxt = `final: ${fmtLoss(m.finalLoss)}\nmin: ${fmtLoss(m.minLoss)}`;
    if (m.lossRel !== null && m.lossRel !== undefined) {
        statTxt += `\nloss_rel: ${Number(m.lossRel).toFixed(6)}`;
    }
    document.getElementById('lm-loss-stats').innerText = statTxt;

    document.getElementById('lm-links').innerHTML =
        `<a href="${m.dir}run.json" target="_blank" rel="noopener" style="color:#4db8ff;">run.json</a>`
        + ` &nbsp;·&nbsp; <a href="${m.dir}weights.json" target="_blank" rel="noopener" style="color:#4db8ff;">weights.json</a>`;

    drawLossCurve(document.getElementById('lm-loss-canvas'), m.losses || [], { big: true });

    const targetImg = document.getElementById('lm-target');
    targetImg.onerror = function () { this.style.display = 'none'; };
    targetImg.style.display = '';
    targetImg.src = m.dir + 'target.png';

    const lastComp = tr.compSteps.length ? tr.compSteps[tr.compSteps.length - 1] : null;
    setModalImage('lm-comp', 'lm-comp-label', 'Latest COMP',
        lastComp !== null ? `${tr.dir}COMP_${pad5(lastComp)}.png` : null, lastComp);

    const lastKernel = tr.kernelSteps.length ? tr.kernelSteps[tr.kernelSteps.length - 1] : null;
    setModalImage('lm-kernel', 'lm-kernel-label', 'Latest KERNEL',
        lastKernel !== null ? `${tr.dir}KERNEL_${pad5(lastKernel)}.png` : null, lastKernel);

    const lastCoupling = tr.couplingSteps.length ? tr.couplingSteps[tr.couplingSteps.length - 1] : null;
    setModalImage('lm-coupling', 'lm-coupling-label', 'Latest COUPLING',
        lastCoupling !== null ? `${tr.dir}COUPLING_${pad5(lastCoupling)}.png` : null, lastCoupling);

    document.getElementById('lenia-modal').style.display = 'block';
}
window.openLeniaModal = openLeniaModal;

window.closeLeniaModal = function () {
    document.getElementById('lenia-modal').style.display = 'none';
};

window.handleLeniaOverlayClick = function () {
    window.closeLeniaModal();
};

document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const modalEl = document.getElementById('lenia-modal');
    if (modalEl && modalEl.style.display === 'block') window.closeLeniaModal();
});

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
    renderLiveChannels(tr);   // keep the channel grid in sync when visible
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
        const S = weights.size ?? 64;   // run at the trained grid size
        tr.liveCA = new LeniaCA(weights, S);
        tr.liveCanvas.width = S;
        tr.liveCanvas.height = S;
        tr.liveImgData = tr.liveCtx.createImageData(S, S);
        tr.liveCanvas.style.imageRendering = 'pixelated';
        tr.liveStepAccum = 0;
        drawLive(tr);
        tr.liveStatusObj.innerText = 'live trained physics — click/drag to damage';
        // Seed only makes sense when training didn't start from noise —
        // hide it on noise-trained runs so it can't masquerade as a no-op.
        if (tr.liveseedBtn) {
            const ini = weights.init;
            tr.liveseedBtn.style.display =
                (ini === 'seedblob' || ini === 'scaffold') ? '' : 'none';
        }
        if (tr.liveCA.hasScaffold && tr.livestencilBtn) {
            tr.livestencilBtn.style.display = '';
            // stencil OFF by default — nothing hidden unless you opt in
            tr.liveCA.setScaffold(false);
            tr.livestencilBtn.innerText = 'Stencil: OFF';
            tr.liveStatusObj.innerText =
                'scaffold-conditioned run; stencil is OFF (this physics was trained WITH it — Stencil: on to see)';
            drawLive(tr);
        }
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

window.liveSeed = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !tr.liveCA) return;
    if (tr.liveCA.hasScaffold) {
        tr.liveCA.setScaffold(true);   // Seed restores the trained setup
        if (tr.livestencilBtn) tr.livestencilBtn.innerText = 'Stencil: on';
    }
    tr.liveCA.resetTrained();
    drawLive(tr);
};

window.liveChannels = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !tr.livechanGrid) return;
    const vis = tr.livechanGrid.style.display === 'none';
    tr.livechanGrid.style.display = vis ? 'flex' : 'none';
    if (vis) renderLiveChannels(tr);
};

function renderLiveChannels(tr) {
    if (!tr.liveCA || !tr.livechanGrid ||
        tr.livechanGrid.style.display === 'none') return;
    const ca = tr.liveCA, S = ca.width;
    if (!tr.chanCanvases || tr.chanCanvases.length !== ca.C) {
        tr.livechanGrid.innerHTML = '';
        tr.chanCanvases = [];
        for (let c = 0; c < ca.C; c++) {
            const wrap = document.createElement('div');
            wrap.style.textAlign = 'center';
            const cv = document.createElement('canvas');
            cv.width = S; cv.height = S;
            cv.style.width = '72px'; cv.style.height = '72px';
            cv.style.imageRendering = 'pixelated';
            cv.style.border = '1px solid #4443';
            const lab = document.createElement('div');
            lab.innerText = c === ca.C - 1 && ca.hasScaffold ? `ch ${c} (stencil)` : `ch ${c}`;
            lab.style.fontSize = '0.65rem';
            lab.style.opacity = '0.7';
            wrap.appendChild(cv); wrap.appendChild(lab);
            tr.livechanGrid.appendChild(wrap);
            tr.chanCanvases.push(cv);
        }
    }
    for (let c = 0; c < ca.C; c++) {
        const cv = tr.chanCanvases[c], cctx = cv.getContext('2d');
        const img = cctx.createImageData(S, S);
        const ch = ca.readChannel(c);
        for (let i = 0; i < ch.length; i++) {
            const v = Math.max(0, Math.min(1, ch[i]));
            const g = (1 - v) * 255;
            img.data[i * 4] = g; img.data[i * 4 + 1] = g;
            img.data[i * 4 + 2] = g; img.data[i * 4 + 3] = 255;
        }
        cctx.putImageData(img, 0, 0);
    }
}

window.liveStencil = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !tr.liveCA || !tr.liveCA.hasScaffold) return;
    const on = !tr.liveCA.scaffoldOn;
    tr.liveCA.setScaffold(on);
    if (tr.livestencilBtn) tr.livestencilBtn.innerText = on ? 'Stencil: on' : 'Stencil: OFF';
    if (tr.liveStatusObj) tr.liveStatusObj.innerText = on
        ? 'prepattern clamped every step (blue) — the physics inks this stencil'
        : 'stencil removed — watch whether the pattern survives without it';
    drawLive(tr);
};

window.liveClear = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !tr.liveCA) return;
    // Clear means CLEAR: every channel including the clamped stencil.
    // Re-enable it with the Stencil button or Seed.
    if (tr.liveCA.hasScaffold) {
        tr.liveCA.setScaffold(false);
        if (tr.livestencilBtn) tr.livestencilBtn.innerText = 'Stencil: OFF';
        if (tr.liveStatusObj) tr.liveStatusObj.innerText =
            'fully cleared (stencil off) — Seed or Stencil: on to restore';
    }
    tr.liveCA.reset(false);
    drawLive(tr);
};

function liveDamageAt(tr, e) {
    if (!tr.liveCA) return;
    const rect = tr.liveCanvas.getBoundingClientRect();
    const normX = (e.clientX - rect.left) / rect.width;
    const normY = (e.clientY - rect.top) / rect.height;
    tr.liveCA.damage(normX * tr.liveCA.width, normY * tr.liveCA.height, 6);
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

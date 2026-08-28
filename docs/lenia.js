// Unified run gallery — ONE card format for every training run in the
// public bucket (Lenia-family and NCA-family alike). Per-run content:
// pre-rendered `<TAG>_#####.png` snapshot streams (COMP training frames,
// START start-states, KERNEL tile strips, COUPLING heatmaps, plus any
// other generically-named stream), target.png, run.json.
//
// All of a card's snapshot streams share one timeline: scrubbing/playing
// the primary animation re-picks the nearest secondary frame at or before
// the current step, so e.g. "learned kernels" visibly evolve in sync with
// training.
//
// Each card also offers a "Run live" widget that fetches the run's
// exported weights.json and steps the actual trained physics in the
// browser — LeniaCA (lenia_engine.js) for `kind === 'lenia'` exports,
// createCA (nca.js, WebGL2 with CPU fallback) for NCA-format exports — a
// real from-scratch simulator, distinct from the PNG timelapse above it.

import { LeniaCA } from './lenia_engine.js?v=nostencil';
import { createCA } from './nca.js';
import { initFiddle, openFiddleModal, handleFiddleHash } from './fiddle.js?v=lenia-17';

let methods = [];
let cardTrackers = [];
const container = document.getElementById('cards-container');

// Public bucket the training jobs write to; readable (and listable)
// anonymously, so a static page needs no backend at all. Overridable via
// an optional config.js (gitignored) that sets window.NCA_CONFIG.bucket.
const BUCKET = (window.NCA_CONFIG && window.NCA_CONFIG.bucket) || 'recipe-lanes-nca-jobs';
const BUCKET_BASE = `https://storage.googleapis.com/${BUCKET}/`;

// The "Fiddle" feature (fiddle.js) edits a run's exported weights live and
// shares the result as a '#fiddle=<token>' URL. It owns its own modal, so
// all it needs from here are live lookups into the gallery's state — the
// closures below keep reading the same arrays as they fill in.
initFiddle({
    getMethod: id => methods.find(x => x.id === id),
    getTracker: id => cardTrackers.find(t => t.id === id),
    bucketBase: BUCKET_BASE,
    // Its overlay hides the whole gallery and runs its own engine, so the
    // card that was ticking behind it stops for the modal's lifetime.
    pauseLive: pauseLiveForModal,
    resumeLive: resumeLiveAfterModal
});

// Top-level bucket prefixes that are not run directories.
const EXCLUDE_PREFIXES = ['packages/', 'analysis/', 'weights/', 'docs/'];
const BLANK_IMG = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';

// Snapshot-stream tag handling: every `<TAG>_#####.png` in a run dir is a
// stream. One tag is the card's main animated timeline (COMP > GOL >
// PATTERN, else first alphabetically); up to two more show beside it,
// synced to the same timeline (START, KERNEL, COUPLING preferred, then
// others alphabetically).
const PRIMARY_PREF = ['COMP', 'GOL', 'PATTERN'];
const SECONDARY_PREF = ['START', 'KERNEL', 'COUPLING'];
const TAG_LABELS = {
    START: 'start state',
    KERNEL: 'learned kernels',
    COUPLING: 'channel coupling',
    RECOV: 'post-damage recovery',
    TARGET: 'moving target'
};

function tagLabel(tag) { return TAG_LABELS[tag] || tag.toLowerCase(); }

function pickTags(streams) {
    const tags = Object.keys(streams).filter(t => streams[t] && streams[t].length);
    const primary = PRIMARY_PREF.find(t => tags.includes(t))
        || tags.slice().sort()[0] || null;
    const rest = tags.filter(t => t !== primary);
    const secondary = [
        ...SECONDARY_PREF.filter(t => rest.includes(t)),
        ...rest.filter(t => !SECONDARY_PREF.includes(t)).sort()
    ].slice(0, 2);
    return { primary, secondary };
}

function pad5(n) { return String(n).padStart(5, '0'); }

// ---------------------------------------------------------------------
// Relative-time / recency helpers — shared by the card's "last snapshot"
// line and the detail modal's Started/Last update/Duration/Rate rows.
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

function mergeSteps(dst, src) {
    let changed = false;
    (src || []).forEach(s => {
        if (!dst.includes(s)) { dst.push(s); changed = true; }
    });
    if (changed) dst.sort((a, b) => a - b);
    return changed;
}

async function listLeniaRuns(onPage) {
    // Two-stage listing so we never page the whole bucket in one flat walk:
    //   1. ONE delimiter listing of every top-level directory (prefix=''),
    //      minus the known non-run prefixes;
    //   2. one per-run listing for its files, streamed as each arrives.
    const dirs = [];
    let pageToken = null;
    do {
        const dirRes = await fetch(
            `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o` +
            `?prefix=&delimiter=/&fields=prefixes,nextPageToken&maxResults=1000` +
            (pageToken ? `&pageToken=${pageToken}` : ''));
        if (!dirRes.ok) throw new Error(`bucket dir list failed: ${dirRes.status}`);
        const dd = await dirRes.json();
        (dd.prefixes || []).forEach(p => {
            if (!EXCLUDE_PREFIXES.includes(p)) dirs.push(p.slice(0, -1));
        });
        pageToken = dd.nextPageToken;
    } while (pageToken);

    const runs = {};
    await Promise.all(dirs.map(async run => {
        const res = await fetch(
            `https://storage.googleapis.com/storage/v1/b/${BUCKET}/o` +
            `?prefix=${encodeURIComponent(run + '/')}` +
            `&fields=items(name,updated)&maxResults=1000`);
        if (!res.ok) return;
        const d = await res.json();
        const r = {
            streams: {},   // TAG -> ascending snapshot steps
            hasTarget: false, hasRunJson: false, hasCode: false, updated: ''
        };
        (d.items || []).forEach(({name, updated}) => {
            const fname = name.slice(run.length + 1);
            if (updated && updated > r.updated) r.updated = updated;
            const m = fname.match(/^([A-Z]+)_(\d+)\.png$/);
            if (m) {
                const step = parseInt(m[2], 10);
                if (!isNaN(step)) (r.streams[m[1]] || (r.streams[m[1]] = [])).push(step);
            } else if (fname === 'target.png') r.hasTarget = true;
            else if (fname === 'run.json') r.hasRunJson = true;
            else if (fname === 'code.tgz') r.hasCode = true;
        });
        Object.values(r.streams).forEach(a => a.sort((x, y) => x - y));
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
        streams: runs[run].streams,
        hasTarget: runs[run].hasTarget,
        hasRunJson: runs[run].hasRunJson,
        hasCode: runs[run].hasCode,
        updated: runs[run].updated
    }));
}

const seenIds = new Set();
let sortKey = localStorage.getItem('lenia_sort') || 'newest';

// ---------------------------------------------------------------------
// Scaffold classification — any run trained with a stencil (prepattern)
// clamped into the last channel at any point during training
// ('args.cond === "scaffold"') is deprecated. No more stencil, ever: it
// doesn't matter whether the clamp was persistent-every-step, t0-only, or
// ablated a fraction of the time (scaf_persistent/scaf_ablate/the '-t0'
// name suffix) — every scaffold-conditioned cohort (cw-*, p2-*, p3-*,
// cwt0-*, abl-*, gen-* runs) is retracted alike. Only runs trained with
// cond !== 'scaffold' (e.g. the newer ns-* 'none' runs) are current.
// ---------------------------------------------------------------------

function classifyScaffold(args) {
    if (!args || args.cond !== 'scaffold') return { deprecated: false };
    return { deprecated: true };
}

function buildSubtitle(rj, scaffold) {
    if (!rj) return '';
    const args = rj.args || {};
    const parts = [];
    if (args.variant) parts.push(args.variant);
    if (args.target) parts.push(args.target);
    // Lenia exports say C/K; NCA runs record channel_n/hidden_n (in args
    // and/or at run.json top level).
    const ch = args.C ?? args.channel_n ?? rj.channel_n;
    if (ch !== undefined) parts.push(`${ch}ch`);
    // sharedk has ONE kernel by design (kernel + coupling matrix); the K
    // arg is inert for it and would mislabel the card.
    if (args.variant === 'sharedk') parts.push('1 kernel (shared)');
    else if (args.K !== undefined) parts.push(`${args.K} kernel${args.K === 1 ? '' : 's'}`);
    const hn = args.hidden_n ?? rj.hidden_n;
    if (hn !== undefined) parts.push(`hidden ${hn}`);
    if (args.params !== undefined) parts.push(`${args.params} params`);
    if (rj.seed_type) parts.push(`seed:${rj.seed_type}`);
    // cw-*/p2-* campaign runs carry extra args the original lenia-* runs
    // didn't; surface whichever of these are present.
    if (args.cond === 'scaffold' && scaffold && scaffold.deprecated) {
        parts.push('scaffold:deprecated');
    } else if (args.cond !== undefined) {
        parts.push(`cond:${args.cond}`);
    }
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
        // Provenance / rollout metadata (newer runs only; absent is fine).
        m.codeSha = rj.code_sha || null;
        m.sourceRun = rj.source_run || null;
        m.history = Array.isArray(rj.history) ? rj.history : null;
        const scaffold = classifyScaffold(rj.args);
        m.deprecated = scaffold.deprecated;
        const sub = document.getElementById(`subtitle_${CSS.escape(m.id)}`);
        if (sub) sub.innerText = buildSubtitle(rj, scaffold) || '(no args recorded)';
        applyDeprecatedBadge(tr, m);
        tr.runJson = rj;
        renderStatus(tr);
        drawSparkline(tr, m);
        updateFilterBounds();
        applyFilters();   // desc/tags/args/loss just arrived; re-run all filters
        if (openModalId === m.id) renderModalMeta(m.id);   // modal open mid-fetch
    } catch (e) {
        const sub = document.getElementById(`subtitle_${CSS.escape(m.id)}`);
        if (sub) sub.innerText = '(run.json unavailable)';
    }
}

// Toggles the badge/opacity for a card once its run.json classification is
// known. Visibility (show/hide by default) is handled separately in
// applyFilters, alongside the other structured filters.
function applyDeprecatedBadge(tr, m) {
    if (tr.deprecatedBadgeObj) tr.deprecatedBadgeObj.style.display = m.deprecated ? 'block' : 'none';
    if (tr.cardObj) tr.cardObj.classList.toggle('deprecated-card', !!m.deprecated);
}

function primarySteps(tr) {
    return (tr.primaryTag && tr.streams[tr.primaryTag]) || [];
}

function renderStatus(tr) {
    const statusObj = tr.statusObj;
    if (!statusObj) return;
    const steps = primarySteps(tr);
    const lastStep = steps.length ? steps[steps.length - 1] : null;
    const rj = tr.runJson;
    const lp = rj ? latestLoss(rj.losses) : null;
    const step = (rj && rj.step !== undefined) ? rj.step : lastStep;
    let text = step !== null && step !== undefined ? `Step ${step}` : 'Waiting for snapshots…';
    if (lp) text += ` — loss ${Number(lp[1]).toFixed(4)}`;
    statusObj.innerText = text;
    renderLastSnapshot(tr);
}

// Last-snapshot recency line, shared by every card. Prefers run.json's
// updated_at (server-side "when did the job last write anything"); falls
// back to the bucket listing's max per-file `updated` timestamp when
// run.json hasn't loaded (or doesn't have the field) yet.
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
    const steps = primarySteps(tr);
    if (!steps.length) return;
    tr.frameIdx = Math.max(0, Math.min(tr.frameIdx, steps.length - 1));
    const step = steps[tr.frameIdx];
    if (tr.imgObj) {
        tr.imgObj.onerror = function () { this.src = BLANK_IMG; };
        tr.imgObj.src = `${tr.dir}${tr.primaryTag}_${pad5(step)}.png`;
    }
    if (tr.scrubObj) tr.scrubObj.value = tr.frameIdx;
    if (tr.frameLabelObj) tr.frameLabelObj.innerText = `step ${step}`;
    renderSecondaryFrames(tr, step);
}

// Re-derives which up-to-two secondary streams a card shows (they can only
// appear over time as snapshots land) and keeps its two slots assigned.
function assignSecondaryTags(tr) {
    const { primary, secondary } = pickTags(tr.streams);
    tr.primaryTag = primary;
    tr.secTags = secondary;
    tr.secSlots.forEach((slot, i) => {
        const tag = secondary[i] || null;
        if (slot.tag !== tag && slot.imgObj) slot.imgObj.src = BLANK_IMG;
        slot.tag = tag;
        if (slot.wrapObj) slot.wrapObj.style.visibility = tag ? '' : 'hidden';
        if (slot.labelObj) slot.labelObj.innerText = tag ? tagLabel(tag) : '';
    });
    if (tr.secRowObj) {
        tr.secRowObj.style.display = secondary.length ? '' : 'none';
    }
}

// Keeps the secondary images (START/KERNEL/COUPLING/…) locked to the same
// timeline position as the primary animation: whatever step the primary is
// showing, show the nearest secondary snapshot at or before that step.
function renderSecondaryFrames(tr, primaryStep) {
    tr.secSlots.forEach(slot => {
        if (!slot.tag) return;
        const s = nearestStepAtOrBelow(tr.streams[slot.tag] || [], primaryStep);
        if (slot.imgObj) {
            if (s !== null) {
                slot.imgObj.style.display = '';
                slot.imgObj.onerror = function () { this.style.display = 'none'; };
                slot.imgObj.src = `${tr.dir}${slot.tag}_${pad5(s)}.png`;
            } else {
                slot.imgObj.style.display = 'none';
            }
        }
        if (slot.labelObj) {
            slot.labelObj.innerText = s !== null
                ? `${tagLabel(slot.tag)} @ step ${s}` : tagLabel(slot.tag);
        }
    });
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
    if (tr.scrubObj) tr.scrubObj.max = Math.max(0, primarySteps(tr).length - 1);
}

window.togglePlay = function (id) {
    const tr = cardTrackers.find(t => t.id === id);
    if (!tr || !primarySteps(tr).length) return;
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
            tr.frameIdx = (tr.frameIdx + 1) % primarySteps(tr).length;
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
        <div class="deprecated-badge" id="deprecated_${m.id}" style="display:none;">DEPRECATED — stencil-trained (see ledger)</div>
        <h3>
            <span class="lenia-card-title" id="title_${m.id}" title="Click for run details">${m.title}</span>
            <button class="info-btn" id="info_${m.id}" title="Run details">&#9432;</button>
            <button class="fiddle-btn" id="fiddle_${m.id}" title="Fiddle with this run's trained weights">Fiddle</button>
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
        <div class="kernel-row" id="secrow_${m.id}" style="display:none;">
            <div id="sec0_wrap_${m.id}">
                <div class="sub-desc" id="sec0_label_${m.id}"></div>
                <div class="img-container"><img loading="lazy" id="sec0_${m.id}" style="display:none;"></div>
            </div>
            <div id="sec1_wrap_${m.id}">
                <div class="sub-desc" id="sec1_label_${m.id}"></div>
                <div class="img-container"><img loading="lazy" id="sec1_${m.id}" style="display:none;"></div>
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
        <div class="last-snapshot" id="lastsnap_${m.id}"></div>
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
                <button id="liveclear_${m.id}" title="Clear all channels">Clear</button>
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
        streams: Object.fromEntries(
            Object.entries(m.streams).map(([t, s]) => [t, [...s]])),
        primaryTag: null,
        secTags: [],
        frameIdx: Number.MAX_SAFE_INTEGER,   // clamped to latest in renderFrame
        playTimer: null,
        runJson: null,
        cardObj: card,
        imgObj: card.querySelector(`#comp_${esc}`),
        secRowObj: card.querySelector(`#secrow_${esc}`),
        secSlots: [0, 1].map(i => ({
            tag: null,
            wrapObj: card.querySelector(`#sec${i}_wrap_${esc}`),
            imgObj: card.querySelector(`#sec${i}_${esc}`),
            labelObj: card.querySelector(`#sec${i}_label_${esc}`)
        })),
        scrubObj: card.querySelector(`#scrub_${esc}`),
        speedObj: card.querySelector(`#speed_${esc}`),
        frameLabelObj: card.querySelector(`#frame_${esc}`),
        statusObj: card.querySelector(`#status_${esc}`),
        lastSnapObj: card.querySelector(`#lastsnap_${esc}`),
        playBtn: card.querySelector(`#play_${esc}`),
        sparkCanvas: card.querySelector(`#spark_${esc}`),
        sparkLabelObj: card.querySelector(`#spark_label_${esc}`),
        titleObj: card.querySelector(`#title_${esc}`),
        infoBtn: card.querySelector(`#info_${esc}`),
        fiddleBtn: card.querySelector(`#fiddle_${esc}`),
        deprecatedBadgeObj: card.querySelector(`#deprecated_${esc}`),
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
    tr.fiddleBtn.onclick = () => openFiddleModal(m.id);
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
    tr.livechansBtn = document.getElementById(`livechans_${m.id}`);
    tr.livechanGrid = document.getElementById(`livechangrid_${m.id}`);
    if (tr.livechansBtn) tr.livechansBtn.onclick = () => window.liveChannels(m.id);
    tr.liveCanvas.addEventListener('mousedown', (e) => { tr.liveDamaging = true; liveDamageAt(tr, e); });
    tr.liveCanvas.addEventListener('mousemove', (e) => { if (tr.liveDamaging) liveDamageAt(tr, e); });
    tr.liveCanvas.addEventListener('mouseup', () => { tr.liveDamaging = false; });
    tr.liveCanvas.addEventListener('mouseleave', () => { tr.liveDamaging = false; });

    assignSecondaryTags(tr);
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
                if (m.updated && m.updated > (tr.updated || '')) tr.updated = m.updated;
                const prevPrimary = tr.primaryTag;
                let gotPrimary = false, gotSecondary = false;
                Object.entries(m.streams).forEach(([tag, steps]) => {
                    if (!tr.streams[tag]) tr.streams[tag] = [];
                    const got = mergeSteps(tr.streams[tag], steps);
                    if (tag === prevPrimary) gotPrimary = got;
                    else if (got) gotSecondary = true;
                });
                assignSecondaryTags(tr);   // a whole new stream may have appeared
                updateScrubRange(tr);
                const steps = primarySteps(tr);
                if ((gotPrimary || tr.primaryTag !== prevPrimary) && !tr.playTimer) {
                    tr.frameIdx = steps.length - 1;   // jump to latest
                    renderFrame(tr);   // also re-syncs secondary frames
                } else if (gotSecondary && steps.length) {
                    // New secondary snapshots landed without a new primary
                    // frame (or while paused mid-scrub) — resync at the step
                    // currently on screen rather than jumping the timeline.
                    renderSecondaryFrames(tr, steps[Math.min(tr.frameIdx, steps.length - 1)]);
                }
                renderStatus(tr);
            }
            if (known) {
                known.hasCode = known.hasCode || m.hasCode;
                if (m.updated && m.updated !== known.updated) {
                    known.updated = m.updated;
                    resort = true;
                }
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
            const ch = a.C ?? a.channel_n;   // lenia args say C; NCA say channel_n
            if (ch !== undefined) cVals.add(ch);
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
    const showDeprecated = !!document.getElementById('filter-show-deprecated')?.checked;
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

        // Deprecated (stencil-trained) runs are hidden by default; a
        // card without run.json yet is never treated as deprecated.
        if (visible && m.deprecated && !showDeprecated) visible = false;

        if (visible && structuredActive) {
            const args = m.args;
            if (!args) {
                // No run.json yet: can't evaluate a narrowed filter against
                // it, so hide it rather than guess.
                visible = false;
            } else {
                if (chVal !== 'any' && String(args.C ?? args.channel_n) !== chVal) visible = false;
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

let openModalId = null;

// Started / Last update / Duration / Rate / Progress rows, derived from
// run.json's started_at/updated_at/step/steps_total. Any row whose inputs
// are missing is simply omitted rather than shown with placeholder dashes.
function renderModalMeta(id) {
    const box = document.getElementById('lm-timing');
    if (!box) return;
    const tr = cardTrackers.find(t => t.id === id);
    const rj = tr && tr.runJson;
    if (!rj) { box.innerHTML = '<div class="lm-meta-row" style="color:#666;">(run.json not loaded yet)</div>'; return; }

    const startedIso = rj.started_at;
    const updatedIso = rj.updated_at;
    const startedMs = startedIso ? Date.parse(startedIso) : NaN;
    const updatedMs = updatedIso ? Date.parse(updatedIso) : NaN;
    const rows = [];

    if (Number.isFinite(startedMs)) {
        rows.push(`<span class="lm-meta-label">Started:</span> ${new Date(startedMs).toLocaleString()}`);
    }
    if (Number.isFinite(updatedMs)) {
        const rel = relTime(updatedIso);
        rows.push(`<span class="lm-meta-label">Last update:</span> ${new Date(updatedMs).toLocaleString()}${rel ? ` (${rel})` : ''}`);
    }
    let durationMs = null;
    if (Number.isFinite(startedMs) && Number.isFinite(updatedMs) && updatedMs >= startedMs) {
        durationMs = updatedMs - startedMs;
        const dur = humanizeDuration(durationMs);
        if (dur) rows.push(`<span class="lm-meta-label">Duration:</span> ${dur}`);
    }
    if (durationMs !== null && typeof rj.step === 'number' && durationMs > 0) {
        const ratePerHour = rj.step / (durationMs / 3600000);
        rows.push(`<span class="lm-meta-label">Rate:</span> ${ratePerHour.toFixed(0)} steps/hour`);
    }
    if (typeof rj.step === 'number' && typeof rj.steps_total === 'number') {
        rows.push(`<span class="lm-meta-label">Progress:</span> ${rj.step}/${rj.steps_total}`);
    }

    box.innerHTML = rows.length
        ? rows.map(r => `<div class="lm-meta-row">${r}</div>`).join('')
        : '<div class="lm-meta-row" style="color:#666;">(no timing data recorded)</div>';
    renderModalProvenance(id);
}

// Provenance rows — code snapshot, continued-from run, rollout schedule.
// All of these are newer run.json fields; anything missing is omitted and
// the whole section hides when empty.
function renderModalProvenance(id) {
    const wrap = document.getElementById('lm-prov-wrap');
    const box = document.getElementById('lm-prov');
    if (!wrap || !box) return;
    const m = methods.find(x => x.id === id);
    const rows = [];
    if (m && m.codeSha) {
        let row = `<span class="lm-meta-label">Code:</span> <code>${escapeHtml(String(m.codeSha).slice(0, 12))}</code>`;
        if (m.hasCode) {
            row += ` — <a href="${m.dir}code.tgz" style="color:#4db8ff;">code snapshot</a>`;
        }
        rows.push(row);
    }
    if (m && m.sourceRun) {
        const run = String(m.sourceRun);
        rows.push(`<span class="lm-meta-label">Continues:</span> `
            + `<a href="#" onclick="openRunPopup('${escapeHtml(run)}'); return false;" `
            + `style="color:#4db8ff;">${escapeHtml(run)}</a>`);
    }
    if (m && Array.isArray(m.history)) {
        const vals = m.history
            .map(h => (h && typeof h === 'object') ? Number(h.ca_steps) : NaN)
            .filter(Number.isFinite);
        if (vals.length) {
            const latest = vals[vals.length - 1];
            const min = Math.min(...vals), max = Math.max(...vals);
            const range = (min === max) ? `${latest}` : `${min}–${max}`;
            rows.push(`<span class="lm-meta-label">Rollout:</span> `
                + `${range} steps/iter (latest ${latest})`);
        }
    }
    wrap.style.display = rows.length ? '' : 'none';
    box.innerHTML = rows.map(r => `<div class="lm-meta-row">${r}</div>`).join('');
}

// Opens another run's popup by its bucket directory name — used by the
// "Continues <run>" provenance link. Falls back to the bucket dir listing
// if that run has no card (e.g. filtered out of the current bucket).
window.openRunPopup = function (run) {
    const id = 'lenia_' + run;
    if (methods.some(x => x.id === id)) {
        openLeniaModal(id);
    } else {
        window.open(BUCKET_BASE + run + '/', '_blank', 'noopener');
    }
};

function openLeniaModal(id) {
    const tr = cardTrackers.find(t => t.id === id);
    const m = methods.find(x => x.id === id);
    if (!tr || !m) return;
    openModalId = id;

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

    // Latest frame of the primary stream plus each secondary stream, in the
    // modal's three generic image slots.
    const modalTags = [tr.primaryTag, ...(tr.secTags || [])];
    ['lm-comp', 'lm-kernel', 'lm-coupling'].forEach((slotId, i) => {
        const tag = modalTags[i] || null;
        const steps = tag ? (tr.streams[tag] || []) : [];
        const last = steps.length ? steps[steps.length - 1] : null;
        setModalImage(slotId, slotId + '-label',
            tag ? `Latest ${tag}` : '',
            (tag && last !== null) ? `${tr.dir}${tag}_${pad5(last)}.png` : null, last);
    });

    renderModalMeta(id);
    document.getElementById('lenia-modal').style.display = 'block';
}
window.openLeniaModal = openLeniaModal;

window.closeLeniaModal = function () {
    document.getElementById('lenia-modal').style.display = 'none';
    openModalId = null;
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
    try {
        for (let i = 0; i < steps; i++) tr.liveCA.step();
    } catch (e) {
        // A widget whose engine died (e.g. a lost WebGL context) degrades to
        // a stopped card instead of killing the loop with no explanation.
        console.error('lenia: live step failed', e);
        stopLiveLoop(tr);
        tr.livePaused = true;
        tr.liveStatusObj.innerText = 'simulation stopped — ' + e.message;
        return;
    }
    if (steps > 0) drawLive(tr);
    tr.liveTimer = setTimeout(() => runLiveLoop(tr), 30);
}

// The fiddle modal covers the gallery completely and steps its own engine;
// only one simulation should be running, so the ticking card pauses for as
// long as that modal is open and resumes exactly where it left off.
let fiddlePausedTr = null;

function pauseLiveForModal() {
    if (!currentLiveTr || currentLiveTr.livePaused || !currentLiveTr.liveTimer) return;
    fiddlePausedTr = currentLiveTr;
    stopLiveLoop(fiddlePausedTr);
}

function resumeLiveAfterModal() {
    const tr = fiddlePausedTr;
    fiddlePausedTr = null;
    // Not if the user collapsed/paused/switched cards while the modal was up.
    if (!tr || tr !== currentLiveTr || tr.livePaused || !tr.liveCA) return;
    if (tr.livePlayBtn) tr.livePlayBtn.innerText = '⏸';
    runLiveLoop(tr);
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

        // Engine selection by weights.json content: `kind === 'lenia'` is a
        // trainable-Lenia export (lenia_engine.js); anything with the NCA
        // schema (fc0_w / channel_n+hidden_n, see nca.js) steps through
        // createCA (WebGL2, CPU fallback). Both engines share the
        // readRGBA/step/reset/damage/readChannel API.
        let W, H;
        if (weights.kind === 'lenia') {
            const S = weights.size ?? 64;   // run at the trained grid size
            tr.liveCA = new LeniaCA(weights, S);
            tr.liveEngine = 'lenia';
            W = S; H = S;
        } else if (weights.fc0_w ||
                   (weights.channel_n !== undefined && weights.hidden_n !== undefined)) {
            const { ca } = createCA(weights);
            tr.liveCA = ca;
            tr.liveEngine = 'nca';
            W = ca.width; H = ca.height;
            // Start the way training started (noise-trained runs reseed
            // with noise, everything else places the trained seed(s)).
            ca.reset(weights.seedType === 'noise');
        } else {
            tr.liveStatusObj.innerText = 'unrecognized weights.json format';
            return;
        }
        tr.liveCanvas.width = W;
        tr.liveCanvas.height = H;
        tr.liveImgData = tr.liveCtx.createImageData(W, H);
        tr.liveCanvas.style.imageRendering = 'pixelated';
        // Preserve aspect for non-square NCA grids (word models are wide).
        tr.liveCanvas.style.width = '256px';
        tr.liveCanvas.style.height = `${Math.round(256 * H / W)}px`;
        tr.liveStepAccum = 0;
        drawLive(tr);
        tr.liveStatusObj.innerText = 'live trained physics — click/drag to damage';
        // Per-engine button support: Seed only makes sense when training
        // didn't start from noise (it would masquerade as a no-op); Clear
        // (zero all channels) only exists on the Lenia engine — an NCA
        // reset always re-places its seed.
        if (tr.liveseedBtn) {
            const seeded = tr.liveEngine === 'lenia'
                ? (weights.init === 'seedblob' || weights.init === 'scaffold')
                : weights.seedType !== 'noise';
            tr.liveseedBtn.style.display = seeded ? '' : 'none';
        }
        if (tr.liveclearBtn) {
            tr.liveclearBtn.style.display = tr.liveEngine === 'lenia' ? '' : 'none';
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
    // Lenia: replay the recorded training init recipe. NCA: reset(false)
    // places the trained seed(s).
    if (tr.liveEngine === 'lenia') tr.liveCA.resetTrained();
    else tr.liveCA.reset(false);
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
    const ca = tr.liveCA, W = ca.width, H = ca.height;
    const nCh = ca.channel_n;
    if (!tr.chanCanvases || tr.chanCanvases.length !== nCh) {
        tr.livechanGrid.innerHTML = '';
        tr.chanCanvases = [];
        for (let c = 0; c < nCh; c++) {
            const wrap = document.createElement('div');
            wrap.style.textAlign = 'center';
            const cv = document.createElement('canvas');
            cv.width = W; cv.height = H;
            cv.style.width = '72px'; cv.style.height = `${Math.round(72 * H / W)}px`;
            cv.style.imageRendering = 'pixelated';
            cv.style.border = '1px solid #4443';
            const lab = document.createElement('div');
            lab.innerText = `ch ${c}`;
            lab.style.fontSize = '0.65rem';
            lab.style.opacity = '0.7';
            wrap.appendChild(cv); wrap.appendChild(lab);
            tr.livechanGrid.appendChild(wrap);
            tr.chanCanvases.push(cv);
        }
    }
    for (let c = 0; c < nCh; c++) {
        const cv = tr.chanCanvases[c], cctx = cv.getContext('2d');
        const img = cctx.createImageData(W, H);
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
    tr.liveCA.damage(normX * tr.liveCA.width, normY * tr.liveCA.height, 6);
    drawLive(tr);
}

async function refreshRuns() {
    try {
        const runs = await listLeniaRuns();
        addOrUpdateCards(leniaMethodsFrom(runs));
    } catch (e) { console.error('lenia refresh failed', e); }
}

// Cheap re-render of the already-fetched relative-time labels — no network
// activity, just rewrites text nodes so "3m ago" keeps ticking between the
// real 20s bucket refreshes.
function refreshTimeLabels() {
    cardTrackers.forEach(renderLastSnapshot);
    if (openModalId && document.getElementById('lenia-modal')?.style.display === 'block') {
        renderModalMeta(openModalId);
    }
}

async function bootstrap() {
    const sortSel = document.getElementById('sort-select');
    if (sortSel) sortSel.value = sortKey;
    await listLeniaRuns(runs => addOrUpdateCards(leniaMethodsFrom(runs)));
    // A '#fiddle=<token>' URL reopens someone's edited weights; the cards
    // exist by now, so fiddle.js can resolve the run it names.
    handleFiddleHash().catch(err => console.error('fiddle hash failed', err));
    // The unified gallery lists every run dir in the bucket (hundreds of
    // per-run listings per sweep), so poll less aggressively than the old
    // prefix-scoped gallery did.
    setInterval(refreshRuns, 60000);   // new runs / snapshots appear without reload
    setInterval(refreshTimeLabels, 60000);   // re-render cached relTime labels
}

bootstrap().catch(err => console.error('Lenia gallery bootstrap failed', err));

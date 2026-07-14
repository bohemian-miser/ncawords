// lecture.js — full-screen presenter view.
//
// One trained word model grows a whole string on one grid (see API.md).
// The point of this page: pace that growth across the wall-clock length of a
// talk. Steps are accumulated fractionally against real elapsed time (rAF
// timestamps), so 120 steps can be spread over 60 minutes without drift and
// without assuming 60 fps.
//
// Scrubbing is implemented with a snapshot history of `ca.state` (Float32Array,
// [C,H,W]). Scrubbing back is an exact state restore; scrubbing forward past
// the newest snapshot re-simulates, and the CA is stochastic (fire_rate 0.5),
// so re-simulated frames are NOT reproductions of what was shown before.

import { createCA } from './nca.js';

/* ------------------------------------------------------------------ *
 *  Tunables                                                           *
 * ------------------------------------------------------------------ */

const PREFERRED_WORD = 'COMP6441';   // if weights/word_COMP6441.json exists
const DEFAULT_TARGET_STEPS = 120;    // "fully grown" — ?steps=N overrides
const IDLE_STEPS_PER_SEC = 8;        // pace once growth is complete (stays alive / heals)
const DAMAGE_RADIUS = 6;
const HISTORY_BUDGET_BYTES = 96 << 20;   // ~96 MB ceiling for the snapshot ring
const HISTORY_MIN_SNAPS = 48;
const HISTORY_MAX_SNAPS = 400;
const MAX_STEPS_PER_FRAME = 6;       // clamp catch-up bursts (tab was hidden, etc.)
const CHROME_IDLE_MS = 3000;

const params = new URLSearchParams(location.search);
const TARGET_STEPS = Math.max(
  1, parseInt(params.get('steps'), 10) || DEFAULT_TARGET_STEPS);

const $ = (id) => document.getElementById(id);
const els = {
  stage: $('stage'), paper: $('paper'), canvas: $('ca-canvas'),
  error: $('error'), errorMsg: $('error-msg'),
  chrome: $('chrome'), status: $('status'),
  stModel: $('st-model'), stStep: $('st-step'), stPct: $('st-pct'), stEngine: $('st-engine'),
  help: $('help'), helpClose: $('help-close'),
  timeline: $('timeline'), tlFill: $('tl-fill'), tlKnob: $('tl-knob'),
  play: $('btn-play'), reset: $('btn-reset'), back: $('btn-back'), fwd: $('btn-fwd'),
  damage: $('btn-damage'), theme: $('btn-theme'), full: $('btn-full'), helpBtn: $('btn-help'),
  mode: $('mode'), duration: $('duration'), speed: $('speed'), speedLabel: $('speed-label'),
  fieldDuration: $('field-duration'), fieldSpeed: $('field-speed'),
};

/* ------------------------------------------------------------------ *
 *  Engine state                                                       *
 * ------------------------------------------------------------------ */

let ca = null;
let engineMode = 'cpu';
let ctx = null;
let imgData = null;
let W = 1, H = 1, C = 1;

let step = 0;              // CA steps applied to the current state
let maxStep = 0;           // furthest point of the current trajectory
let playing = false;       // starts PAUSED — the presenter starts the talk
let stepAcc = 0;
let lastT = 0;
let rafId = null;
let scrubTarget = null;    // pending scrub destination (applied incrementally)

/* ------------------------------------------------------------------ *
 *  Snapshot history                                                   *
 * ------------------------------------------------------------------ */

const history = {
  snaps: [],      // [{ step, data: Float32Array }], ascending by step
  stride: 1,      // steps between snapshots (doubles when the cap is hit)
  max: HISTORY_MAX_SNAPS,
  bytes: 0,
};

function historyInit() {
  const snapBytes = C * W * H * 4;
  history.max = Math.max(
    HISTORY_MIN_SNAPS,
    Math.min(HISTORY_MAX_SNAPS, Math.floor(HISTORY_BUDGET_BYTES / snapBytes)));
  history.bytes = snapBytes;
  historyClear();
}

function historyClear() {
  history.snaps.length = 0;
  history.stride = 1;
}

function readState() {
  // CPUCA.state is the live buffer; GLCA.state reads back a fresh array.
  return Float32Array.from(ca.state);
}

// Write a snapshot back into the engine. CPUCA exposes its live Float32Array;
// GLCA keeps state in RGBA32F textures, so we re-upload them. Both paths are
// verified once at boot (verifyRestore) — a failure demotes the page to CPU.
function writeState(snap) {
  if (engineMode === 'gl') { glWriteState(snap); return; }
  ca.state.set(snap);
}

function glWriteState(snap) {
  const gl = ca.gl;
  const plane = W * H;
  const T = Math.ceil(C / 4);
  const texs = ca._sets[ca._cur].texs;
  const buf = new Float32Array(plane * 4);
  for (let t = 0; t < T; t++) {
    buf.fill(0);
    for (let lane = 0; lane < 4; lane++) {
      const c = 4 * t + lane;
      if (c >= C) break;
      const off = c * plane;
      for (let i = 0; i < plane; i++) buf[i * 4 + lane] = snap[off + i];
    }
    gl.bindTexture(gl.TEXTURE_2D, texs[t]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA, gl.FLOAT, buf);
  }
}

// Round-trip check: state -> writeState -> state must come back identical.
function verifyRestore() {
  try {
    const before = readState();
    ca.step();
    writeState(before);
    const after = readState();
    let maxDiff = 0;
    for (let i = 0; i < before.length; i++) {
      const d = Math.abs(before[i] - after[i]);
      if (d > maxDiff) maxDiff = d;
    }
    return maxDiff < 1e-5;
  } catch (err) {
    console.warn('state restore probe failed:', err);
    return false;
  }
}

function historyRecord() {
  const snaps = history.snaps;
  if (step % history.stride !== 0) return;
  const last = snaps[snaps.length - 1];
  if (last && last.step === step) { last.data.set(readState()); return; }
  if (last && last.step > step) return;             // shouldn't happen; be safe
  snaps.push({ step, data: readState() });
  if (snaps.length > history.max) {
    // Halve the resolution: keep every other snapshot, double the stride.
    const kept = [];
    for (let i = 0; i < snaps.length; i += 2) kept.push(snaps[i]);
    history.snaps = kept;
    history.stride *= 2;
  }
}

// The trajectory diverged at the current step (scrub-back, damage, re-sim):
// everything recorded after `step` is no longer what the CA will do.
function historyTruncate() {
  const snaps = history.snaps;
  while (snaps.length && snaps[snaps.length - 1].step > step) snaps.pop();
  const last = snaps[snaps.length - 1];
  if (last && last.step === step) last.data.set(readState());
  maxStep = Math.max(step, snaps.length ? snaps[snaps.length - 1].step : 0);
}

function nearestSnapAtOrBefore(target) {
  const snaps = history.snaps;
  let best = null;
  for (let i = snaps.length - 1; i >= 0; i--) {
    if (snaps[i].step <= target) { best = snaps[i]; break; }
  }
  return best;
}

function historyBytes() {
  return history.snaps.length * history.bytes;
}

/* ------------------------------------------------------------------ *
 *  Weights                                                            *
 * ------------------------------------------------------------------ */

async function fetchJSON(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

async function loadModel() {
  // 1) the preferred word model, if it's on disk;
  // 2) else the longest word in the index — the most impressive trained model.
  let index = null;
  try {
    index = await fetchJSON('weights/index.json');
  } catch (err) {
    console.warn('weights/index.json:', err);
  }

  // Try the preferred model first even if it is not indexed yet (it may have
  // just landed on disk); then the longest indexed word, then the rest.
  const candidates = [PREFERRED_WORD];
  const words = (index && Array.isArray(index.words)) ? index.words.slice() : [];
  words.sort((a, b) => b.length - a.length || a.localeCompare(b));
  for (const w of words) if (!candidates.includes(w)) candidates.push(w);

  const tried = [];
  for (const text of candidates) {
    try {
      const weights = await fetchJSON(`weights/word_${text}.json`);
      return { text, weights };
    } catch (err) {
      tried.push(text);
    }
  }
  throw new Error(
    tried.length
      ? `Could not load any word model (tried: ${tried.join(', ')}).`
      : 'No word models are listed in weights/index.json.');
}

/* ------------------------------------------------------------------ *
 *  Rendering + layout                                                 *
 * ------------------------------------------------------------------ */

function draw() {
  if (!ca) return;
  if (engineMode === 'gl' && typeof ca.drawTo === 'function') {
    ca.drawTo(ctx);
  } else {
    const out = ca.readRGBA(imgData.data);
    if (out && out !== imgData.data) imgData.data.set(out);
    ctx.putImageData(imgData, 0, 0);
  }
}

// Scale the grid up as far as it goes inside the viewport, aspect preserved,
// integer factor whenever there is room for one.
function layout() {
  if (!ca) return;
  const pad = parseFloat(getComputedStyle(els.paper).paddingLeft) || 0;
  const cs = getComputedStyle(els.stage);
  const margin = (parseFloat(cs.paddingLeft) || 0) * 2;
  const availW = window.innerWidth - margin - 2 * pad - 2;
  const availH = window.innerHeight - margin - 2 * pad - 2;
  let s = Math.min(availW / W, availH / H);
  if (s >= 1) s = Math.floor(s);
  s = Math.max(s, 0.5);
  els.canvas.style.width = `${Math.round(W * s)}px`;
  els.canvas.style.height = `${Math.round(H * s)}px`;
}

/* ------------------------------------------------------------------ *
 *  Stepping / pacing                                                  *
 * ------------------------------------------------------------------ */

function stepsPerSecond() {
  if (step >= TARGET_STEPS) return IDLE_STEPS_PER_SEC;   // grown: keep it alive
  if (els.mode.value === 'free') return parseFloat(els.speed.value) || 10;
  const minutes = parseFloat(els.duration.value) || 20;
  return TARGET_STEPS / (minutes * 60);
}

function advance(n) {
  for (let i = 0; i < n; i++) {
    ca.step();
    step += 1;
    if (step > maxStep) maxStep = step;
    historyRecord();
  }
}

function frame(t) {
  rafId = null;
  if (!lastT) lastT = t;
  let dt = (t - lastT) / 1000;
  lastT = t;
  if (!(dt > 0)) dt = 0;
  if (dt > 1) dt = 1;             // don't binge after a hidden tab / stall

  let dirty = false;

  if (scrubTarget !== null) {
    dirty = applyScrub() || dirty;
  } else if (playing) {
    stepAcc += dt * stepsPerSecond();
    let n = Math.floor(stepAcc);
    if (n > 0) {
      stepAcc -= n;
      if (n > MAX_STEPS_PER_FRAME) n = MAX_STEPS_PER_FRAME;
      advance(n);
      dirty = true;
    }
  }

  if (dirty) { draw(); updateStatus(); }
  loop();
}

function loop() {
  const wants = (playing || scrubTarget !== null) && !document.hidden;
  if (wants && rafId === null) {
    lastT = 0;
    rafId = requestAnimationFrame(frame);
  } else if (!wants && rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  els.play.textContent = playing ? '❚❚ Pause' : '▶ Play';
}

function setPlaying(v) {
  playing = v;
  stepAcc = 0;
  lastT = 0;
  loop();
}

/* ------------------------------------------------------------------ *
 *  Scrubbing                                                          *
 * ------------------------------------------------------------------ */

function requestScrub(target) {
  const clamped = Math.max(0, Math.min(Math.round(target), timelineMax()));
  scrubTarget = clamped;
  if (scrubTarget === step) { scrubTarget = null; updateStatus(); return; }
  loop();
}

// Back: exact restore from the nearest snapshot at/before the target, then
// re-run the remainder. Forward: keep stepping (spread over frames so a long
// jump doesn't freeze the page).
function applyScrub() {
  const target = scrubTarget;
  if (target === null) return false;

  if (target < step) {
    const snap = nearestSnapAtOrBefore(target);
    if (snap) {
      writeState(snap.data);
      step = snap.step;
      historyTruncate();
    } else {
      ca.reset();
      step = 0;
      historyClear();
      historyRecord();
      maxStep = 0;
    }
  }

  let todo = target - step;
  if (todo <= 0) {
    scrubTarget = null;
    loop();
    return true;
  }
  advance(Math.min(todo, MAX_STEPS_PER_FRAME * 2));
  if (step >= target) { scrubTarget = null; loop(); }
  return true;
}

function timelineMax() {
  return Math.max(TARGET_STEPS, maxStep);
}

function nudge(delta) {
  setPlaying(false);
  requestScrub(step + delta);
}

/* ------------------------------------------------------------------ *
 *  Status + timeline UI                                               *
 * ------------------------------------------------------------------ */

function updateStatus() {
  const pct = Math.min(100, Math.round((step / TARGET_STEPS) * 100));
  els.stStep.textContent = `step ${step} / ${TARGET_STEPS}`;
  els.stPct.textContent = step >= TARGET_STEPS ? 'grown ✓' : `${pct}%`;
  const max = timelineMax();
  const f = max ? Math.min(1, step / max) : 0;
  els.tlFill.style.width = `${f * 100}%`;
  els.tlKnob.style.left = `${f * 100}%`;
  els.timeline.setAttribute('aria-valuemax', String(max));
  els.timeline.setAttribute('aria-valuenow', String(step));
}

/* ------------------------------------------------------------------ *
 *  Damage                                                             *
 * ------------------------------------------------------------------ */

function eventToCell(ev) {
  const r = els.canvas.getBoundingClientRect();
  const x = Math.floor(((ev.clientX - r.left) / r.width) * W);
  const y = Math.floor(((ev.clientY - r.top) / r.height) * H);
  return [Math.max(0, Math.min(W - 1, x)), Math.max(0, Math.min(H - 1, y))];
}

function damageAt(x, y) {
  ca.damage(x, y, DAMAGE_RADIUS);
  historyTruncate();      // the trajectory just changed under us
  draw();                 // works while paused, too
  updateStatus();
}

function damageRandom() {
  if (!ca) return;
  const x = Math.floor(W * (0.1 + Math.random() * 0.8));
  const y = Math.floor(H * (0.25 + Math.random() * 0.5));
  damageAt(x, y);
}

/* ------------------------------------------------------------------ *
 *  Chrome auto-hide                                                   *
 * ------------------------------------------------------------------ */

let idleTimer = null;

function wakeChrome() {
  els.chrome.classList.remove('hidden');
  document.body.classList.remove('hide-cursor');
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    els.chrome.classList.add('hidden');
    document.body.classList.add('hide-cursor');
  }, CHROME_IDLE_MS);
}

/* ------------------------------------------------------------------ *
 *  Controls                                                           *
 * ------------------------------------------------------------------ */

function doReset() {
  ca.reset();
  step = 0;
  maxStep = 0;
  stepAcc = 0;
  historyClear();
  historyRecord();
  scrubTarget = null;
  setPlaying(false);
  draw();
  updateStatus();
}

function syncMode() {
  const free = els.mode.value === 'free';
  els.fieldDuration.hidden = free;
  els.fieldSpeed.hidden = !free;
  stepAcc = 0;
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme !== 'light';
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  els.theme.textContent = dark ? 'Dark' : 'Light';
  try { localStorage.setItem('nca-lecture-theme', document.documentElement.dataset.theme); } catch {}
}

function toggleHelp(force) {
  const show = force !== undefined ? force : els.help.hidden;
  els.help.hidden = !show;
  if (show) wakeChrome();
}

function wireControls() {
  els.play.addEventListener('click', () => setPlaying(!playing));
  els.reset.addEventListener('click', doReset);
  els.back.addEventListener('click', () => nudge(-1));
  els.fwd.addEventListener('click', () => nudge(1));
  els.damage.addEventListener('click', damageRandom);
  els.full.addEventListener('click', toggleFullscreen);
  els.theme.addEventListener('click', toggleTheme);
  els.helpBtn.addEventListener('click', () => toggleHelp());
  els.helpClose.addEventListener('click', () => toggleHelp(false));
  els.mode.addEventListener('change', syncMode);
  els.duration.addEventListener('change', () => { stepAcc = 0; });
  els.speed.addEventListener('input', () => {
    els.speedLabel.textContent = `${els.speed.value}/s`;
  });

  // Timeline scrubbing (pointer drag anywhere on the track).
  let scrubbing = false;
  const scrubFromEvent = (ev) => {
    const r = els.timeline.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    requestScrub(f * timelineMax());
  };
  els.timeline.addEventListener('pointerdown', (ev) => {
    scrubbing = true;
    setPlaying(false);
    els.timeline.setPointerCapture(ev.pointerId);
    scrubFromEvent(ev);
    ev.preventDefault();
  });
  els.timeline.addEventListener('pointermove', (ev) => { if (scrubbing) scrubFromEvent(ev); });
  const endScrub = () => { scrubbing = false; };
  els.timeline.addEventListener('pointerup', endScrub);
  els.timeline.addEventListener('pointercancel', endScrub);

  // Canvas damage (click + drag), works paused.
  let dragging = false;
  const hit = (ev) => { const [x, y] = eventToCell(ev); damageAt(x, y); };
  els.canvas.addEventListener('pointerdown', (ev) => {
    dragging = true;
    els.canvas.setPointerCapture(ev.pointerId);
    hit(ev);
    ev.preventDefault();
  });
  els.canvas.addEventListener('pointermove', (ev) => { if (dragging) hit(ev); });
  const endDrag = () => { dragging = false; };
  els.canvas.addEventListener('pointerup', endDrag);
  els.canvas.addEventListener('pointercancel', endDrag);

  // Keyboard.
  window.addEventListener('keydown', (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const tag = (ev.target && ev.target.tagName) || '';
    if (tag === 'SELECT' || tag === 'INPUT') {
      wakeChrome();
      return;              // let the widget have its own keys
    }
    let handled = true;
    switch (ev.key) {
      case ' ': case 'Spacebar': setPlaying(!playing); break;
      case 'ArrowLeft': nudge(-1); break;
      case 'ArrowRight': nudge(1); break;
      default:
        switch (ev.key.toLowerCase()) {
          case 'r': doReset(); break;
          case 'd': damageRandom(); break;
          case 'f': toggleFullscreen(); break;
          case 'h': toggleHelp(); break;
          case 't': toggleTheme(); break;
          default: handled = false;
        }
    }
    if (handled) ev.preventDefault();
    wakeChrome();
  });

  window.addEventListener('mousemove', wakeChrome, { passive: true });
  window.addEventListener('pointerdown', wakeChrome, { passive: true });
  window.addEventListener('resize', layout);
  document.addEventListener('fullscreenchange', layout);
  document.addEventListener('visibilitychange', loop);
}

/* ------------------------------------------------------------------ *
 *  Boot                                                               *
 * ------------------------------------------------------------------ */

function fail(msg) {
  els.errorMsg.textContent = msg;
  els.error.hidden = false;
  els.stage.style.display = 'none';
  els.chrome.classList.add('hidden');
  console.error(msg);
}

async function boot() {
  try {
    const saved = localStorage.getItem('nca-lecture-theme');
    if (saved === 'light' || saved === 'dark') {
      document.documentElement.dataset.theme = saved;
      els.theme.textContent = saved === 'light' ? 'Dark' : 'Light';
    }
  } catch {}

  let model;
  try {
    model = await loadModel();
  } catch (err) {
    fail(String(err.message || err));
    return;
  }

  try {
    const made = createCA(model.weights);
    ca = made.ca;
    engineMode = made.mode;
    W = ca.width; H = ca.height;
    C = model.weights.channel_n;

    // If we cannot put a snapshot back where it came from, scrubbing is a lie:
    // demote to the CPU engine, whose state is a live Float32Array.
    if (!verifyRestore()) {
      console.warn(`state restore unavailable on the "${engineMode}" backend — falling back to CPU`);
      const cpu = createCA(model.weights, { forceCPU: true });
      ca = cpu.ca;
      engineMode = cpu.mode;
      W = ca.width; H = ca.height;
    }
    ca.reset();

    els.canvas.width = W;
    els.canvas.height = H;
    ctx = els.canvas.getContext('2d');
    imgData = ctx.createImageData(W, H);

    historyInit();
    historyRecord();

    els.stModel.textContent = `${model.text}  ${W}×${H}`;
    els.stEngine.textContent = engineMode === 'gl' ? 'webgl2' : 'cpu';
    els.canvas.title = `${model.text} — click or drag to damage`;
    els.speedLabel.textContent = `${els.speed.value}/s`;
  } catch (err) {
    fail(`The model loaded but the CA engine failed to start: ${err.message || err}`);
    return;
  }

  syncMode();
  layout();
  draw();
  updateStatus();
  wireControls();
  wakeChrome();
  loop();

  // Auto-hide the help card after a moment; the presenter can bring it back
  // with H (or the ? button).
  setTimeout(() => toggleHelp(false), 7000);

  // Handy for probes/tests.
  window.__lecture = {
    get step() { return step; },
    get maxStep() { return maxStep; },
    get engineMode() { return engineMode; },
    get target() { return TARGET_STEPS; },
    get grid() { return [W, H]; },
    get snaps() { return history.snaps.length; },
    get stride() { return history.stride; },
    get historyBytes() { return historyBytes(); },
    get scrubbing() { return scrubTarget !== null; },
    stepsPerSecond, requestScrub, advance, draw, updateStatus, damageRandom,
    setPlaying, doReset,
    readState,
    checksum() {
      const s = readState();
      let h = 0;
      for (let i = 0; i < s.length; i++) h = (h * 31 + Math.round(s[i] * 1e4)) | 0;
      return h;
    },
    rgbaChecksum() {
      const px = ca.readRGBA();
      let h = 0;
      for (let i = 0; i < px.length; i++) h = (h * 31 + px[i]) | 0;
      return h;
    },
  };
  window.__lectureReady = true;
}

window.addEventListener('error', (e) => {
  if (!els.error.hidden) return;
  fail(`Script error: ${e.message}`);
});

boot();

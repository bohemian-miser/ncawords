// lecture.js — full-screen view for a talk.
//
// One trained word model grows a whole string on one grid (see API.md).
// Deliberately simple: play/pause, a speed slider, reset, and click-to-wound.
// No timeline, no history buffer — the CA is stochastic, so "rewinding" was
// never honest anyway, and a projector wants fewer moving parts.

import { createCA } from './nca.js';

const PREFERRED = 'COMP6441';
const DAMAGE_RADIUS = 7;
const GROWN_STEPS = 120;          // roughly "fully grown", for the ETA readout

const $ = (id) => document.getElementById(id);
const els = {
  paper: $('paper'), canvas: $('canvas'), error: $('error'),
  play: $('play'), reset: $('reset'), damage: $('damage'),
  speed: $('speed'), speedLabel: $('speed-label'), status: $('status'),
  controls: $('controls'), hint: $('hint'),
};

let ca = null, mode = 'cpu', ctx = null, imgData = null;
let playing = false;          // start paused: nothing grows until you say so
let steps = 0;
let stepAcc = 0;
let last = 0;

/* ---------- speed: slider 0..100 -> steps/sec, log-scaled ---------- */
// 0.02 steps/s (a 120-step growth stretched over ~100 min) up to 30 steps/s.
const MIN_SPS = 0.02, MAX_SPS = 30;
const sliderToSps = (v) =>
  MIN_SPS * Math.pow(MAX_SPS / MIN_SPS, v / 100);

function stepsPerSec() {
  return sliderToSps(Number(els.speed.value));
}

function describeSpeed(sps) {
  const mins = GROWN_STEPS / sps / 60;
  if (mins >= 1.5) return `${sps.toFixed(2)}/s · grows in ~${Math.round(mins)} min`;
  if (mins >= 0.5) return `${sps.toFixed(2)}/s · grows in ~1 min`;
  return `${sps.toFixed(1)}/s · grows in ~${Math.max(1, Math.round(mins * 60))}s`;
}

function updateSpeedLabel() {
  els.speedLabel.textContent = describeSpeed(stepsPerSec());
}

/* ---------- loading ---------- */

async function loadWeights() {
  // Prefer COMP6441; otherwise the longest word model available.
  try {
    const r = await fetch(`weights/word_${PREFERRED}.json`, { cache: 'no-cache' });
    if (r.ok) return await r.json();
  } catch { /* fall through to the index */ }

  const idx = await fetch('weights/index.json', { cache: 'no-cache' });
  if (!idx.ok) throw new Error('weights/index.json not found');
  const words = (await idx.json()).words || [];
  if (!words.length) throw new Error('no word models are trained yet');
  const pick = words.reduce((a, b) => (b.length > a.length ? b : a));
  const r = await fetch(`weights/word_${pick}.json`, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`weights/word_${pick}.json not found`);
  return await r.json();
}

function fitCanvas() {
  if (!ca) return;
  // Integer upscale so pixels stay crisp, as large as the viewport allows.
  const availW = window.innerWidth * 0.9;
  const availH = window.innerHeight * 0.62;
  const scale = Math.max(1, Math.floor(
    Math.min(availW / ca.width, availH / ca.height)));
  els.canvas.style.width = `${ca.width * scale}px`;
  els.canvas.style.height = `${ca.height * scale}px`;
}

function draw() {
  if (!ca) return;
  if (mode === 'gl' && typeof ca.drawTo === 'function') {
    ca.drawTo(ctx);
  } else {
    const out = ca.readRGBA(imgData.data);
    if (out && out !== imgData.data) imgData.data.set(out);
    ctx.putImageData(imgData, 0, 0);
  }
}

function setStatus(text) {
  els.status.textContent = text;
}

function refreshStatus() {
  const grown = Math.min(100, Math.round((steps / GROWN_STEPS) * 100));
  setStatus(`${ca.text || ''} · step ${steps} · ${grown}% grown · ${mode}`);
}

/* ---------- loop: real elapsed time, so slow speeds are honest ---------- */

function frame(t) {
  if (!playing) return;
  const dt = Math.min(1, (t - last) / 1000 || 0);   // clamp hidden-tab bursts
  last = t;
  stepAcc += dt * stepsPerSec();
  let n = Math.floor(stepAcc);
  if (n > 0) {
    stepAcc -= n;
    n = Math.min(n, 8);
    for (let i = 0; i < n; i++) ca.step();
    steps += n;
    draw();
    refreshStatus();
  }
  requestAnimationFrame(frame);
}

function setPlaying(on) {
  playing = on;
  els.play.textContent = on ? 'Pause' : 'Play';
  if (on) {
    last = performance.now();
    requestAnimationFrame(frame);
  }
}

/* ---------- interaction ---------- */

function eventToCell(ev) {
  const r = els.canvas.getBoundingClientRect();
  return [
    Math.max(0, Math.min(ca.width - 1,
      Math.floor(((ev.clientX - r.left) / r.width) * ca.width))),
    Math.max(0, Math.min(ca.height - 1,
      Math.floor(((ev.clientY - r.top) / r.height) * ca.height))),
  ];
}

function wireInteraction() {
  let dragging = false;
  const wound = (ev) => {
    const [x, y] = eventToCell(ev);
    ca.damage(x, y, DAMAGE_RADIUS);
    draw();                       // visible immediately, even while paused
  };
  els.canvas.addEventListener('pointerdown', (ev) => {
    dragging = true;
    els.canvas.setPointerCapture(ev.pointerId);
    wound(ev);
    ev.preventDefault();
  });
  els.canvas.addEventListener('pointermove', (ev) => { if (dragging) wound(ev); });
  const stop = () => { dragging = false; };
  els.canvas.addEventListener('pointerup', stop);
  els.canvas.addEventListener('pointercancel', stop);

  els.play.addEventListener('click', () => setPlaying(!playing));
  els.reset.addEventListener('click', doReset);
  els.damage.addEventListener('click', damageRandom);
  els.speed.addEventListener('input', updateSpeedLabel);

  window.addEventListener('resize', fitCanvas);

  window.addEventListener('keydown', (ev) => {
    if (ev.key === ' ') { ev.preventDefault(); setPlaying(!playing); }
    else if (ev.key === 'r' || ev.key === 'R') doReset();
    else if (ev.key === 'd' || ev.key === 'D') damageRandom();
    else if (ev.key === 'f' || ev.key === 'F') {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen?.();
    }
  });
}

function doReset() {
  ca.reset();
  steps = 0;
  stepAcc = 0;
  draw();
  refreshStatus();
}

function damageRandom() {
  const x = Math.floor(ca.width * (0.1 + Math.random() * 0.8));
  const y = Math.floor(ca.height * (0.3 + Math.random() * 0.4));
  ca.damage(x, y, DAMAGE_RADIUS + 2);
  draw();
}

/* ---------- boot ---------- */

async function boot() {
  let weights;
  try {
    weights = await loadWeights();
  } catch (err) {
    els.paper.hidden = true;
    els.error.hidden = false;
    els.error.textContent = `Could not load a model: ${err.message}`;
    setStatus('');
    return;
  }

  // Force the CPU engine. This grid is 216x32 and runs at a few steps per
  // second, which costs a CPU almost nothing — while the WebGL path can only
  // be tested on real hardware (headless browsers have no float render
  // targets, so it silently falls back to CPU there anyway). Not worth
  // risking a blank projector for a speedup nobody needs.
  const made = createCA(weights, { forceCPU: true });
  ca = made.ca;
  mode = made.mode;
  ca.text = weights.text || weights.char || '';

  els.canvas.width = ca.width;
  els.canvas.height = ca.height;
  ctx = els.canvas.getContext('2d');
  imgData = ctx.createImageData(ca.width, ca.height);

  fitCanvas();
  ca.reset();
  draw();
  updateSpeedLabel();
  refreshStatus();
  wireInteraction();
}

boot();

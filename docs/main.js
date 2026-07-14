// Growing Neural Cellular Automata for Text — page driver.
// Engine contract: see API.md. The CA engine itself lives in ./nca.js.

import { createCA } from './nca.js';

/* ------------------------------------------------------------------ *
 *  Constants & shared state                                           *
 * ------------------------------------------------------------------ */

const MAX_LEN = 16;
const DAMAGE_RADIUS = 6;

const $ = (id) => document.getElementById(id);

const els = {
  input: $('hero-text'),
  hint: $('hero-hint'),
  row: $('letter-row'),
  play: $('btn-play'),
  step: $('btn-step'),
  reset: $('btn-reset'),
  damage: $('btn-damage'),
  speed: $('speed'),
  speedLabel: $('speed-label'),
  counter: $('step-counter'),
  mode: $('engine-mode'),
  hero: $('hero'),
  wordWrap: $('word-figure-wrap'),
  wordSection: $('word-section'),
  wordPills: $('word-pills'),
  wordCanvas: $('word-canvas'),
  wordNote: $('word-note'),
};

let availableChars = new Set();   // chars with trained weights
let availableWords = [];          // words with trained whole-word models
let indexLoadFailed = false;

const weightCache = new Map();    // cache key -> Promise<weights json>

let letters = [];                 // [{kind:'ca'|'gap'|'missing', char, ca, ...}]
let buildId = 0;                  // guards async rebuilds

let word = null;                  // whole-word CA unit (shares the letter shape)
let wordSelectId = 0;             // guards async word switches

let playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let speed = 1;                    // CA steps per animation frame (0.25 .. 4)
let stepAcc = 0;
let heroVisible = true;
let wordVisible = false;
let rafId = null;

/* ------------------------------------------------------------------ *
 *  Weights loading                                                    *
 * ------------------------------------------------------------------ */

const hex4 = (ch) => ch.codePointAt(0).toString(16).padStart(4, '0');

async function loadIndex() {
  try {
    const res = await fetch('weights/index.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    availableChars = new Set(data.chars || []);
    availableWords = Array.isArray(data.words) ? data.words : [];
  } catch (err) {
    indexLoadFailed = true;
    console.warn('Could not load weights/index.json:', err);
  }
}

function fetchWeights(key, url) {
  if (!weightCache.has(key)) {
    const p = fetch(url).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${key}`);
      return res.json();
    });
    // Drop failed fetches from the cache so a later retry can succeed
    // (e.g. weights that are still being trained/written).
    p.catch(() => weightCache.delete(key));
    weightCache.set(key, p);
    return p;
  }
  return weightCache.get(key);
}

const loadWeights = (char) => fetchWeights(char, `weights/${hex4(char)}.json`);
const loadWordWeights = (text) =>
  fetchWeights(`word:${text}`, `weights/word_${text}.json`);

/* ------------------------------------------------------------------ *
 *  Letter cells                                                       *
 * ------------------------------------------------------------------ */

function sanitize(text) {
  return text.toUpperCase().replace(/[^A-Z ]/g, '').slice(0, MAX_LEN);
}

function makeGap() {
  const el = document.createElement('div');
  el.className = 'letter-gap';
  return { kind: 'gap', el };
}

function makeMissing(char) {
  const el = document.createElement('div');
  el.className = 'letter-cell missing';
  el.textContent = char;
  el.title = `"${char}" has no trained weights yet`;
  return { kind: 'missing', char, el };
}

function makeLetter(char) {
  const el = document.createElement('div');
  el.className = 'letter-cell';
  el.title = `${char} — click/drag to damage, double-click to reset`;

  const canvas = document.createElement('canvas');
  el.appendChild(canvas);

  const letter = {
    kind: 'ca', char, el, canvas,
    ctx: null, ca: null, mode: null, w: 40, h: 40,
    imgData: null, steps: 0, ready: false,
  };

  attachPointerHandlers(letter);
  return letter;
}

// Boot (or re-boot) a CA into a unit — used by hero letters and the word figure.
function initUnitCA(unit, weights) {
  const { ca, mode } = createCA(weights);
  unit.w = typeof ca.width === 'number' ? ca.width : (weights.grid_w ?? weights.grid);
  unit.h = typeof ca.height === 'number' ? ca.height : (weights.grid_h ?? weights.grid);
  unit.canvas.width = unit.w;
  unit.canvas.height = unit.h;
  unit.ctx = unit.canvas.getContext('2d');
  unit.ca = ca;
  unit.mode = mode;
  unit.imgData = unit.ctx.createImageData(unit.w, unit.h);
  unit.steps = 0;
  unit.ready = true;
  ca.reset();
  draw(unit);
}

function draw(unit) {
  if (!unit.ready) return;
  if (unit.mode === 'gl' && typeof unit.ca.drawTo === 'function') {
    unit.ca.drawTo(unit.ctx);
  } else {
    const out = unit.ca.readRGBA(unit.imgData.data);
    if (out && out !== unit.imgData.data) unit.imgData.data.set(out);
    unit.ctx.putImageData(unit.imgData, 0, 0);
  }
}

const caLetters = () => letters.filter((l) => l.kind === 'ca' && l.ready);

// Every steppable unit (for manual controls: step / reset).
function allUnits() {
  const units = caLetters();
  if (word && word.ready) units.push(word);
  return units;
}

// Only units whose figure is currently on screen (for the rAF loop).
function visibleUnits() {
  const units = heroVisible ? caLetters() : [];
  if (wordVisible && word && word.ready) units.push(word);
  return units;
}

/* ------------------------------------------------------------------ *
 *  Rebuilding the row from the text input                             *
 * ------------------------------------------------------------------ */

async function rebuildRow() {
  const id = ++buildId;
  const text = sanitize(els.input.value);
  if (text !== els.input.value) {
    const pos = els.input.selectionStart;
    els.input.value = text;
    if (pos != null) {
      const p = Math.min(pos, text.length);
      try { els.input.setSelectionRange(p, p); } catch { /* not focusable */ }
    }
  }

  letters = [];
  els.row.replaceChildren();

  const missing = new Set();
  for (const char of text) {
    let letter;
    if (char === ' ') {
      letter = makeGap();
    } else if (availableChars.has(char)) {
      letter = makeLetter(char);
    } else {
      letter = makeMissing(char);
      missing.add(char);
    }
    letters.push(letter);
    els.row.appendChild(letter.el);
  }

  updateHint(missing);

  // Load weights (cached after first fetch) and boot each CA.
  await Promise.all(
    letters.filter((l) => l.kind === 'ca').map(async (letter) => {
      try {
        const weights = await loadWeights(letter.char);
        if (id !== buildId) return;            // superseded by a newer rebuild
        initUnitCA(letter, weights);
      } catch (err) {
        console.warn(`Weights for "${letter.char}" failed to load:`, err);
        if (id !== buildId) return;
        const fallback = makeMissing(letter.char);
        letter.el.replaceWith(fallback.el);
        const i = letters.indexOf(letter);
        if (i >= 0) letters[i] = fallback;
      }
    })
  );

  if (id === buildId) updateStatus();
}

function updateHint(missingSet) {
  if (indexLoadFailed) {
    els.hint.textContent =
      'Could not load the trained-weights index — the demo has nothing to grow yet.';
  } else if (missingSet.size > 0) {
    const list = [...missingSet].join(', ');
    els.hint.textContent =
      `Not trained yet, shown as placeholders: ${list}. More letters appear as training finishes.`;
  } else {
    els.hint.textContent = '';
  }
}

/* ------------------------------------------------------------------ *
 *  Damage interaction                                                 *
 * ------------------------------------------------------------------ */

function eventToCell(unit, ev) {
  const rect = unit.canvas.getBoundingClientRect();
  const gx = Math.floor(((ev.clientX - rect.left) / rect.width) * unit.w);
  const gy = Math.floor(((ev.clientY - rect.top) / rect.height) * unit.h);
  return [
    Math.max(0, Math.min(unit.w - 1, gx)),
    Math.max(0, Math.min(unit.h - 1, gy)),
  ];
}

function attachPointerHandlers(unit) {
  let dragging = false;

  const applyDamage = (ev) => {
    if (!unit.ready) return;
    const [x, y] = eventToCell(unit, ev);
    unit.ca.damage(x, y, DAMAGE_RADIUS);
    if (!running()) draw(unit);
  };

  unit.canvas.addEventListener('pointerdown', (ev) => {
    if (!unit.ready) return;
    dragging = true;
    unit.canvas.setPointerCapture(ev.pointerId);
    applyDamage(ev);
    ev.preventDefault();
  });
  unit.canvas.addEventListener('pointermove', (ev) => {
    if (dragging) applyDamage(ev);
  });
  const stop = () => { dragging = false; };
  unit.canvas.addEventListener('pointerup', stop);
  unit.canvas.addEventListener('pointercancel', stop);

  unit.canvas.addEventListener('dblclick', () => {
    if (!unit.ready) return;
    unit.ca.reset();
    unit.steps = 0;
    if (!running()) { draw(unit); updateStatus(); }
  });
}

function damageRandom() {
  const live = caLetters();
  if (live.length === 0) return;
  const letter = live[Math.floor(Math.random() * live.length)];
  // Aim somewhere in the central region, where the glyph actually is.
  const x = Math.floor(letter.w * (0.25 + Math.random() * 0.5));
  const y = Math.floor(letter.h * (0.25 + Math.random() * 0.5));
  letter.ca.damage(x, y, DAMAGE_RADIUS);
  if (!running()) draw(letter);
}

/* ------------------------------------------------------------------ *
 *  Animation loop                                                     *
 * ------------------------------------------------------------------ */

function running() {
  return playing && !document.hidden &&
    (heroVisible || (wordVisible && word && word.ready));
}

function stepUnits(units, n = 1) {
  for (const unit of units) {
    for (let i = 0; i < n; i++) unit.ca.step();
    unit.steps += n;
    draw(unit);
  }
}

function frame() {
  rafId = null;
  if (!running()) return;
  stepAcc += speed;
  const n = Math.floor(stepAcc);
  if (n > 0) {
    stepAcc -= n;
    stepUnits(visibleUnits(), Math.min(n, 8));    // clamp catch-up bursts
    updateStatus();
  }
  rafId = requestAnimationFrame(frame);
}

function syncLoop() {
  if (running() && rafId === null) {
    rafId = requestAnimationFrame(frame);
  } else if (!running() && rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  els.play.textContent = playing ? 'Pause' : 'Play';
}

function updateStatus() {
  const first = caLetters()[0] || (word && word.ready ? word : null);
  els.counter.textContent = first ? String(first.steps) : '0';
  els.mode.textContent = first ? first.mode : '';
}

/* ------------------------------------------------------------------ *
 *  Controls                                                           *
 * ------------------------------------------------------------------ */

els.play.addEventListener('click', () => {
  playing = !playing;
  syncLoop();
});

els.step.addEventListener('click', () => {
  playing = false;
  syncLoop();
  stepUnits(allUnits(), 1);
  updateStatus();
});

els.reset.addEventListener('click', () => {
  for (const unit of allUnits()) {
    unit.ca.reset();
    unit.steps = 0;
    draw(unit);
  }
  stepAcc = 0;
  updateStatus();
});

els.damage.addEventListener('click', damageRandom);

function updateSpeed() {
  speed = Math.pow(2, parseFloat(els.speed.value));
  // Round to a friendly label (slider is log2-spaced, so values like 1.19 occur).
  const nice = speed >= 1 ? speed.toFixed(speed % 1 ? 2 : 0) : speed.toFixed(2);
  els.speedLabel.textContent = `${parseFloat(nice)}×`;
}
els.speed.addEventListener('input', updateSpeed);
updateSpeed();

let inputTimer = null;
els.input.addEventListener('input', () => {
  clearTimeout(inputTimer);
  inputTimer = setTimeout(rebuildRow, 120);
});

/* ------------------------------------------------------------------ *
 *  Visibility management                                              *
 * ------------------------------------------------------------------ */

document.addEventListener('visibilitychange', syncLoop);

if ('IntersectionObserver' in window) {
  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.target === els.hero) heroVisible = entry.isIntersecting;
        else if (entry.target === els.wordWrap) wordVisible = entry.isIntersecting;
      }
      syncLoop();
    },
    { rootMargin: '200px 0px 200px 0px' }
  );
  io.observe(els.hero);
  if (els.wordWrap) io.observe(els.wordWrap);
} else {
  wordVisible = true;   // no observer support: keep everything stepping
}

/* ------------------------------------------------------------------ *
 *  Word figure: one model grows a whole word on one grid              *
 * ------------------------------------------------------------------ */

function buildWordFigure() {
  if (!els.wordWrap || availableWords.length === 0) return;   // stays hidden

  els.wordWrap.hidden = false;
  if (els.wordSection) els.wordSection.hidden = false;

  // One persistent unit object: handlers are attached once to the shared
  // canvas; selecting another word swaps the CA inside the same unit.
  word = {
    kind: 'ca', char: null, canvas: els.wordCanvas,
    ctx: null, ca: null, mode: null, w: 1, h: 1,
    imgData: null, steps: 0, ready: false,
  };
  attachPointerHandlers(word);

  const pills = availableWords.map((text) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'word-pill';
    btn.textContent = text;
    btn.addEventListener('click', () => selectWord(text));
    els.wordPills.appendChild(btn);
    return btn;
  });

  function setActivePill(text) {
    pills.forEach((b) => b.classList.toggle('active', b.textContent === text));
  }
  buildWordFigure.setActivePill = setActivePill;

  selectWord(availableWords[0]);
}

async function selectWord(text) {
  const id = ++wordSelectId;
  buildWordFigure.setActivePill(text);
  els.wordNote.hidden = true;

  try {
    const weights = await loadWordWeights(text);
    if (id !== wordSelectId) return;           // superseded by another click
    word.ready = false;
    initUnitCA(word, weights);
    word.char = text;
    // ~4x upscale, capped by CSS max-width: 100% for small screens.
    els.wordCanvas.style.width = `${word.w * 4}px`;
    els.wordCanvas.title =
      `${text} — one model, one grid. Click/drag to damage, double-click to reset.`;
    syncLoop();
    updateStatus();
  } catch (err) {
    console.warn(`Word model for "${text}" failed to load:`, err);
    if (id !== wordSelectId) return;
    els.wordNote.textContent =
      `The model for "${text}" is not available yet — it may still be training.`;
    els.wordNote.hidden = false;
  }
}

/* ------------------------------------------------------------------ *
 *  OCR verification figure                                            *
 * ------------------------------------------------------------------ */

async function buildOcrFigure() {
  const figure = $('ocr-figure');
  const grid = $('ocr-grid');
  const summary = $('ocr-summary');
  const note = $('ocr-note');
  const caption = figure.querySelector('figcaption');

  let report = null;
  try {
    const res = await fetch('ocr_report.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    report = await res.json();
  } catch {
    // Report not generated yet: collapse the figure to a short note.
    grid.hidden = true;
    summary.hidden = true;
    if (caption) caption.hidden = true;
    note.hidden = false;
    return;
  }

  const results = Array.isArray(report.results) ? report.results : [];
  summary.textContent =
    `Tesseract verdict: ${report.ok} of ${report.total} trained letters read back correctly.`;

  const frag = document.createDocumentFragment();
  for (const r of results) {
    const item = document.createElement('div');
    item.className = `ocr-item ${r.ok ? 'ocr-ok' : 'ocr-bad'}`;

    const img = document.createElement('img');
    img.src = r.img || `grown/${hex4(r.char)}.png`;
    img.alt = `Grown pattern for the letter ${r.char}`;
    img.loading = 'lazy';
    img.width = 64;
    img.height = 64;

    const label = document.createElement('div');
    label.className = 'ocr-label';

    const expected = document.createElement('span');
    expected.className = 'ocr-expected';
    expected.textContent = r.char;

    const read = document.createElement('span');
    read.className = 'ocr-read';
    read.textContent = `→ ${String(r.ocr ?? '').trim() || '∅'}`;
    read.title = 'What Tesseract read';

    const verdict = document.createElement('span');
    verdict.className = 'ocr-verdict';
    verdict.textContent = r.ok ? '✓' : '✗';

    label.append(expected, read, verdict);
    item.append(img, label);
    frag.appendChild(item);
  }
  grid.replaceChildren(frag);
}

/* ------------------------------------------------------------------ *
 *  Boot                                                               *
 * ------------------------------------------------------------------ */

async function boot() {
  await loadIndex();
  buildWordFigure();
  await rebuildRow();
  syncLoop();
  buildOcrFigure();
}

boot();

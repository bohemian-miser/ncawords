// demo_modulation.js — "channel B modulates channel A's kernel"
//
// A hand-built, 2-channel illustration of what the dyn1/dynwave trainable-
// Lenia variants learn end-to-end: a cell's growth kernel is not fixed, it
// is *composed* per cell, per step, from the local state. Here channel B is
// a control field the user paints by hand, and it blends two independently
// trained "multik" physics (hex, dots) into channel A's update:
//
//   dxHex[i]  = sum_k h_k * growth(conv(A, kernel_k), mu_k, sg_k)   -- hex physics
//   dxDots[i] = sum_k h_k * growth(conv(A, kernel_k), mu_k, sg_k)   -- dots physics
//   dx[i]     = (1 - B[i]) * dxHex[i] + B[i] * dxDots[i]
//   A[i]     += dt * dx[i]                                          -- leaky-clamped
//
// Painting B never touches A: it only ever changes which trained kernel set
// governs A's update at that cell.

const HEX_URL = "https://storage.googleapis.com/recipe-lanes-nca-jobs/lenia-multik-hex/weights.json";
const DOTS_URL = "https://storage.googleapis.com/recipe-lanes-nca-jobs/lenia-multik-dots/weights.json";

const S = 96; // grid side
const N = S * S;

// ---------------------------------------------------------------- state --

const A = new Float32Array(N);
const B = new Float32Array(N);
const dxHex = new Float32Array(N);
const dxDots = new Float32Array(N);
const uBuf = new Float32Array(N);

function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

function resetA() {
  for (let i = 0; i < N; i++) A[i] = Math.random() * 0.6;
}

function bRamp() {
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++)
      B[y * S + x] = x / S;
}

function bFill(v) { B.fill(v); }

resetA();
bRamp();

// ------------------------------------------------------------- physics --

async function loadPhysics(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  const w = await res.json();
  return {
    kernels: w.kernels.map((k) => Float32Array.from(k.flat())),
    mu: w.mu,
    sg: w.sg,
    h: w.h,
    dt: w.dt,
    ks: w.ks,
    leak: w.leak ?? 0.05,
  };
}

// toroidal cross-correlation of `src` (SxS) with a ks x ks kernel, into `out`
function convolve(src, kernel, ks, out) {
  const r = ks >> 1;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      let acc = 0;
      for (let ky = 0; ky < ks; ky++) {
        const yy = (y + ky - r + S) % S;
        const rowK = ky * ks;
        const rowS = yy * S;
        for (let kx = 0; kx < ks; kx++) {
          const xx = (x + kx - r + S) % S;
          acc += kernel[rowK + kx] * src[rowS + xx];
        }
      }
      out[y * S + x] = acc;
    }
  }
}

function growth(u, mu, sg) {
  const d = (u - mu) / sg;
  return 2 * Math.exp((-d * d) / 2) - 1;
}

// accumulate this physics' growth field for the *current* A into `dxOut`
function computeDx(phys, dxOut) {
  dxOut.fill(0);
  for (let k = 0; k < phys.kernels.length; k++) {
    convolve(A, phys.kernels[k], phys.ks, uBuf);
    const mu = phys.mu[k], sg = phys.sg[k], h = phys.h[k];
    for (let i = 0; i < N; i++) dxOut[i] += h * growth(uBuf[i], mu, sg);
  }
}

let hexPhys = null, dotsPhys = null;

function step() {
  if (!hexPhys || !dotsPhys) return;
  computeDx(hexPhys, dxHex);
  computeDx(dotsPhys, dxDots);
  const dt = hexPhys.dt; // both physics train with dt=0.25; use hex's per spec
  const leak = hexPhys.leak;
  for (let i = 0; i < N; i++) {
    const b = B[i];
    const dx = (1 - b) * dxHex[i] + b * dxDots[i];
    const xn = A[i] + dt * dx;
    const c = clamp01(xn);
    A[i] = c + leak * (xn - c);
  }
  stepCount++;
}

// --------------------------------------------------------------- view ---

const canvas = document.getElementById("demo-canvas");
const ctx = canvas.getContext("2d");
const imgData = ctx.createImageData(S, S);

const viewSelect = document.getElementById("view-select");

function render() {
  const view = viewSelect.value;
  const data = imgData.data;
  for (let i = 0; i < N; i++) {
    const p = i * 4;
    let r, g, bl;
    if (view === "b") {
      const v = Math.round(clamp01(B[i]) * 255);
      r = g = bl = v;
    } else {
      const ac = clamp01(A[i]);
      const base = Math.round((1 - ac) * 255); // ink-on-white
      if (view === "a") {
        r = g = bl = base;
      } else {
        // subtle blue (B=0) -> red (B=1) tint, mostly visible off the ink
        const bc = clamp01(B[i]);
        const tintR = 40 + bc * 180;
        const tintG = 60;
        const tintB = 220 - bc * 180;
        const k = 0.28;
        r = Math.round(base * (1 - k) + tintR * k);
        g = Math.round(base * (1 - k) + tintG * k);
        bl = Math.round(base * (1 - k) + tintB * k);
      }
    }
    data[p] = r; data[p + 1] = g; data[p + 2] = bl; data[p + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

// -------------------------------------------------------------- brush ---

const radiusInput = document.getElementById("brush-radius");
const radiusVal = document.getElementById("brush-radius-val");
const valueInput = document.getElementById("brush-value");
const valueVal = document.getElementById("brush-value-val");
const eraseToggle = document.getElementById("brush-erase");

radiusInput.addEventListener("input", () => {
  radiusVal.textContent = radiusInput.value;
});
valueInput.addEventListener("input", () => {
  valueVal.textContent = Number(valueInput.value).toFixed(2);
});

function paintB(gx, gy, erase) {
  const radius = Number(radiusInput.value);
  const value = erase ? 0 : Number(valueInput.value);
  const r2 = radius * radius;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy > r2) continue;
      const yy = ((gy + dy) % S + S) % S;
      const xx = ((gx + dx) % S + S) % S;
      B[yy * S + xx] = value;
    }
  }
}

let painting = false;
let paintErase = false;

function canvasToGrid(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = S / rect.width;
  const scaleY = S / rect.height;
  const gx = Math.floor((clientX - rect.left) * scaleX);
  const gy = Math.floor((clientY - rect.top) * scaleY);
  return [Math.max(0, Math.min(S - 1, gx)), Math.max(0, Math.min(S - 1, gy))];
}

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("pointerdown", (e) => {
  painting = true;
  paintErase = e.button === 2 || eraseToggle.checked;
  const [gx, gy] = canvasToGrid(e.clientX, e.clientY);
  paintB(gx, gy, paintErase);
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener("pointermove", (e) => {
  if (!painting) return;
  const [gx, gy] = canvasToGrid(e.clientX, e.clientY);
  paintB(gx, gy, paintErase);
});

function stopPainting() { painting = false; }
canvas.addEventListener("pointerup", stopPainting);
canvas.addEventListener("pointerleave", stopPainting);
canvas.addEventListener("pointercancel", stopPainting);

// ------------------------------------------------------------ controls --

const statusEl = document.getElementById("demo-status");
const stepCounterEl = document.getElementById("step-counter");
const playBtn = document.getElementById("btn-play");
const resetBtn = document.getElementById("btn-reset");
const bRampBtn = document.getElementById("btn-b-ramp");
const bZeroBtn = document.getElementById("btn-b-zero");
const bOneBtn = document.getElementById("btn-b-one");
const speedInput = document.getElementById("speed");
const speedLabel = document.getElementById("speed-label");

let playing = true;
let stepCount = 0;
let accumulator = 0;

playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
});

resetBtn.addEventListener("click", () => {
  resetA();
  stepCount = 0;
});

bRampBtn.addEventListener("click", bRamp);
bZeroBtn.addEventListener("click", () => bFill(0));
bOneBtn.addEventListener("click", () => bFill(1));

speedInput.addEventListener("input", () => {
  speedLabel.textContent = `${Number(speedInput.value)}×`;
});

// ------------------------------------------------------------ kernels ---

function drawKernelStrip(canvasEl, kernels, ks) {
  const gap = 2;
  const n = kernels.length;
  const w = n * ks + (n - 1) * gap;
  const h = ks;
  canvasEl.width = w;
  canvasEl.height = h;
  const scale = 6;
  canvasEl.style.width = `${w * scale}px`;
  canvasEl.style.height = `${h * scale}px`;
  const kctx = canvasEl.getContext("2d");
  const img = kctx.createImageData(w, h);
  for (let p = 0; p < w * h; p++) {
    img.data[p * 4] = 230; img.data[p * 4 + 1] = 230; img.data[p * 4 + 2] = 230; img.data[p * 4 + 3] = 255;
  }
  for (let k = 0; k < n; k++) {
    const kern = kernels[k];
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < kern.length; i++) {
      if (kern[i] < min) min = kern[i];
      if (kern[i] > max) max = kern[i];
    }
    const range = max - min || 1;
    const xOff = k * (ks + gap);
    for (let y = 0; y < ks; y++) {
      for (let x = 0; x < ks; x++) {
        const v = (kern[y * ks + x] - min) / range;
        const gray = Math.round(v * 255);
        const px = (y * w + xOff + x) * 4;
        img.data[px] = gray; img.data[px + 1] = gray; img.data[px + 2] = gray; img.data[px + 3] = 255;
      }
    }
  }
  kctx.putImageData(img, 0, 0);
}

// ------------------------------------------------------------- main loop -

function loop() {
  if (playing && hexPhys && dotsPhys) {
    accumulator += Number(speedInput.value);
    while (accumulator >= 1) {
      step();
      accumulator -= 1;
    }
  }
  render();
  stepCounterEl.textContent = String(stepCount);
  requestAnimationFrame(loop);
}

async function init() {
  render(); // paint initial B ramp / A noise immediately
  try {
    [hexPhys, dotsPhys] = await Promise.all([loadPhysics(HEX_URL), loadPhysics(DOTS_URL)]);
    drawKernelStrip(document.getElementById("kernels-hex"), hexPhys.kernels, hexPhys.ks);
    drawKernelStrip(document.getElementById("kernels-dots"), dotsPhys.kernels, dotsPhys.ks);
    statusEl.textContent = "trained physics loaded — running";
  } catch (err) {
    statusEl.textContent = `failed to load trained physics: ${err.message}`;
    console.error(err);
  }
  requestAnimationFrame(loop);
}

init();

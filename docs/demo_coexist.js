// demo_coexist.js — "two organisms, one world: paint which rules apply"
//
// Two independently-trained Growing-NCA models share ONE 16-channel state
// grid. A user-painted control field B picks, per cell, which model's
// learned update rule runs:
//
//   p        = [state, sobelX(state), sobelY(state)]      -- shared perception
//   dxA      = fc1_A(relu(fc0_A(p)))                      -- model A's proposal
//   dxB      = fc1_B(relu(fc0_B(p)))                      -- model B's proposal
//   x       += fire * ((1 - B) * dxA + B * dxB)           -- one shared fire mask
//   alive    = maxpool3x3(alpha) > 0.1 pre & post, else cell zeroed
//
// The step math mirrors nca.js's CPUCA exactly (zero padding at borders,
// same Sobel constants, same blocked perception layout, same alive mask);
// see docs/test/test_coexist.mjs for the parity test against CPUCA.

const BUCKET = "https://storage.googleapis.com/recipe-lanes-nca-jobs";

// ---------------------------------------------------------------------------
// Weight handling — same interpretation as nca.js flattenWeights().
// ---------------------------------------------------------------------------

function flattenWeights(weights) {
  const C = weights.channel_n;
  const HN = weights.hidden_n;
  const C3 = 3 * C;
  // "blocked" fc0_w columns are already [state | sobel_x | sobel_y];
  // "interleaved" (or absent layout) columns are [id0, sx0, sy0, id1, ...]
  // and get reordered to blocked at load time — exactly as nca.js does.
  const blocked = weights.layout === "blocked";
  const w0 = new Float32Array(HN * C3);
  for (let k = 0; k < HN; k++) {
    const row = weights.fc0_w[k];
    if (blocked) {
      for (let j = 0; j < C3; j++) w0[k * C3 + j] = row[j];
    } else {
      for (let c = 0; c < C; c++) {
        w0[k * C3 + c] = row[3 * c];             // identity
        w0[k * C3 + C + c] = row[3 * c + 1];     // sobel x
        w0[k * C3 + 2 * C + c] = row[3 * c + 2]; // sobel y
      }
    }
  }
  const b0 = Float32Array.from(weights.fc0_b);
  const w1 = new Float32Array(C * HN);
  for (let c = 0; c < C; c++) {
    const row = weights.fc1_w[c];
    for (let k = 0; k < HN; k++) w1[c * HN + k] = row[k];
  }
  return { C, HN, C3, w0, b0, w1 };
}

function gridDims(weights) {
  if (weights.kind === "word" || weights.grid == null) {
    return { W: weights.grid_w, H: weights.grid_h };
  }
  return { W: weights.grid, H: weights.grid };
}

// dx for a cell whose whole 3x3 neighborhood is zero (perception = 0):
// dx = fc1 @ relu(fc0_b). Same arithmetic as the main loop (nca.js parity).
function computeDxZero({ C, HN, b0, w1 }) {
  const h = new Float32Array(HN);
  for (let k = 0; k < HN; k++) h[k] = b0[k] > 0 ? b0[k] : 0;
  const dx = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    const off = c * HN;
    for (let k = 0; k < HN; k++) s += w1[off + k] * h[k];
    dx[c] = s;
  }
  return dx;
}

// ---------------------------------------------------------------------------
// BlendedCA — two rule-sets, one state, per-cell blend field B.
// ---------------------------------------------------------------------------

export class BlendedCA {
  constructor(weightsA, weightsB) {
    if (weightsA.channel_n !== weightsB.channel_n) {
      throw new Error(`channel_n mismatch: ${weightsA.channel_n} vs ${weightsB.channel_n}`);
    }
    const dA = gridDims(weightsA), dB = gridDims(weightsB);
    if (dA.W !== dB.W || dA.H !== dB.H) {
      throw new Error(`grid mismatch: ${dA.W}x${dA.H} vs ${dB.W}x${dB.H}`);
    }
    this._W = dA.W;
    this._H = dA.H;
    this.fire_rate = weightsA.fire_rate != null ? weightsA.fire_rate : 0.5;

    this._fA = flattenWeights(weightsA);
    this._fB = flattenWeights(weightsB);
    this._C = this._fA.C;
    this._C3 = this._fA.C3;
    this._dxZeroA = computeDxZero(this._fA);
    this._dxZeroB = computeDxZero(this._fB);

    const plane = this._W * this._H;
    this._plane = plane;
    this._buf = new Float32Array(this._C * plane);
    this._back = new Float32Array(this._C * plane);
    this._pre = new Uint8Array(plane);
    this._post = new Uint8Array(plane);
    this._nz = new Uint8Array(plane);
    this._nbnz = new Uint8Array(plane);
    // Preallocated per-cell work buffers (no allocations inside step()).
    this._p = new Float64Array(this._C3);
    this._hA = new Float64Array(this._fA.HN);
    this._hB = new Float64Array(this._fB.HN);
    this._dxA = new Float64Array(this._C);
    this._dxB = new Float64Array(this._C);
    // The painted control field: 0 -> model A's rule, 1 -> model B's.
    this.B = new Float32Array(plane);
  }

  get state() { return this._buf; }
  get width() { return this._W; }
  get height() { return this._H; }

  clear() { this._buf.fill(0); }

  // Seed: alpha + hidden channels (3..C-1) = 1 at (x, y), as nca.js does.
  seed(x, y) {
    const plane = this._plane, i = y * this._W + x;
    for (let c = 3; c < this._C; c++) this._buf[c * plane + i] = 1.0;
  }

  damage(x, y, r) {
    const W = this._W, H = this._H, plane = this._plane, r2 = r * r;
    const y0 = Math.max(0, Math.ceil(y - r)), y1 = Math.min(H - 1, Math.floor(y + r));
    for (let yy = y0; yy <= y1; yy++) {
      for (let xx = Math.max(0, Math.ceil(x - r)); xx <= Math.min(W - 1, Math.floor(x + r)); xx++) {
        if ((xx - x) * (xx - x) + (yy - y) * (yy - y) <= r2) {
          const i = yy * W + xx;
          for (let c = 0; c < this._C; c++) this._buf[c * plane + i] = 0;
        }
      }
    }
  }

  // maxpool3x3(alpha) > 0.1 -> out (0/1); out-of-bounds neighbors read as 0.
  _aliveMask(buf, out) {
    const W = this._W, H = this._H, plane = this._plane, A = 3 * plane;
    for (let y = 0; y < H; y++) {
      const yu = y > 0, yd = y < H - 1;
      for (let x = 0; x < W; x++) {
        const i = y * W + x, b = A + i;
        let m = buf[b];
        if (x > 0) { const v = buf[b - 1]; if (v > m) m = v; }
        if (x < W - 1) { const v = buf[b + 1]; if (v > m) m = v; }
        if (yu) {
          const bu = b - W;
          let v = buf[bu]; if (v > m) m = v;
          if (x > 0) { v = buf[bu - 1]; if (v > m) m = v; }
          if (x < W - 1) { v = buf[bu + 1]; if (v > m) m = v; }
        }
        if (yd) {
          const bd = b + W;
          let v = buf[bd]; if (v > m) m = v;
          if (x > 0) { v = buf[bd - 1]; if (v > m) m = v; }
          if (x < W - 1) { v = buf[bd + 1]; if (v > m) m = v; }
        }
        out[i] = m > 0.1 ? 1 : 0;
      }
    }
  }

  _nonzeroDilated(buf) {
    const W = this._W, H = this._H, plane = this._plane, C = this._C;
    const nz = this._nz, nb = this._nbnz;
    nz.fill(0);
    for (let c = 0; c < C; c++) {
      const off = c * plane;
      for (let i = 0; i < plane; i++) if (buf[off + i] !== 0) nz[i] = 1;
    }
    for (let y = 0; y < H; y++) {
      const yu = y > 0, yd = y < H - 1;
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        let m = nz[i];
        if (!m && x > 0) m = nz[i - 1];
        if (!m && x < W - 1) m = nz[i + 1];
        if (!m && yu) m = nz[i - W] | (x > 0 ? nz[i - W - 1] : 0) | (x < W - 1 ? nz[i - W + 1] : 0);
        if (!m && yd) m = nz[i + W] | (x > 0 ? nz[i + W - 1] : 0) | (x < W - 1 ? nz[i + W + 1] : 0);
        nb[i] = m ? 1 : 0;
      }
    }
    return nb;
  }

  step(fireRate = null, rand = Math.random) {
    if (fireRate === null || fireRate === undefined) fireRate = this.fire_rate;
    const W = this._W, H = this._H, plane = this._plane;
    const C = this._C, C3 = this._C3;
    const HNA = this._fA.HN, HNB = this._fB.HN;
    const w0A = this._fA.w0, b0A = this._fA.b0, w1A = this._fA.w1;
    const w0B = this._fB.w0, b0B = this._fB.b0, w1B = this._fB.w1;
    const dzA = this._dxZeroA, dzB = this._dxZeroB;
    const cur = this._buf, nxt = this._back;
    const pre = this._pre, post = this._post;
    const p = this._p, hA = this._hA, hB = this._hB;
    const dxA = this._dxA, dxB = this._dxB;
    const Bf = this.B;

    this._aliveMask(cur, pre);
    const nbnz = this._nonzeroDilated(cur);

    for (let y = 0; y < H; y++) {
      const yu = y > 0, yd = y < H - 1;
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        // One shared Bernoulli draw per cell per step — both models see it.
        const fired = rand() <= fireRate;
        if (!fired) {
          for (let c = 0; c < C; c++) nxt[c * plane + i] = cur[c * plane + i];
          continue;
        }
        const bv = Bf[i], av = 1 - bv;
        if (!nbnz[i]) {
          // Entire 3x3 neighborhood is zero: perception is exactly 0.
          for (let c = 0; c < C; c++) {
            nxt[c * plane + i] = cur[c * plane + i] + (av * dzA[c] + bv * dzB[c]);
          }
          continue;
        }
        const xl = x > 0, xr = x < W - 1;
        // Perception (identity, sobelX, sobelY per channel), zero padding.
        for (let c = 0; c < C; c++) {
          const b = c * plane + i;
          const vC = cur[b];
          const vL = xl ? cur[b - 1] : 0, vR = xr ? cur[b + 1] : 0;
          let vUL = 0, vU = 0, vUR = 0, vDL = 0, vD = 0, vDR = 0;
          if (yu) {
            const bu = b - W;
            vU = cur[bu];
            if (xl) vUL = cur[bu - 1];
            if (xr) vUR = cur[bu + 1];
          }
          if (yd) {
            const bd = b + W;
            vD = cur[bd];
            if (xl) vDL = cur[bd - 1];
            if (xr) vDR = cur[bd + 1];
          }
          p[c] = vC;
          p[C + c] = (vUR + 2 * vR + vDR - vUL - 2 * vL - vDL) * 0.125;
          p[2 * C + c] = (vDL + 2 * vD + vDR - vUL - 2 * vU - vUR) * 0.125;
        }
        // Model A: h = relu(fc0 p + b0); dxA = fc1 h.
        for (let k = 0; k < HNA; k++) {
          let s = b0A[k];
          const off = k * C3;
          for (let j = 0; j < C3; j++) s += w0A[off + j] * p[j];
          hA[k] = s > 0 ? s : 0;
        }
        for (let c = 0; c < C; c++) {
          let s = 0;
          const off = c * HNA;
          for (let k = 0; k < HNA; k++) s += w1A[off + k] * hA[k];
          dxA[c] = s;
        }
        // Model B, same perception.
        for (let k = 0; k < HNB; k++) {
          let s = b0B[k];
          const off = k * C3;
          for (let j = 0; j < C3; j++) s += w0B[off + j] * p[j];
          hB[k] = s > 0 ? s : 0;
        }
        for (let c = 0; c < C; c++) {
          let s = 0;
          const off = c * HNB;
          for (let k = 0; k < HNB; k++) s += w1B[off + k] * hB[k];
          dxB[c] = s;
        }
        // Blend by the painted field and apply.
        for (let c = 0; c < C; c++) {
          const b = c * plane + i;
          nxt[b] = cur[b] + (av * dxA[c] + bv * dxB[c]);
        }
      }
    }

    this._aliveMask(nxt, post);
    for (let i = 0; i < plane; i++) {
      if (!(pre[i] && post[i])) {
        for (let c = 0; c < C; c++) nxt[c * plane + i] = 0;
      }
    }
    this._back = cur;
    this._buf = nxt;
  }

  // Ink-on-white RGBA, same as nca.js rgbaFromState.
  readRGBA(out) {
    const plane = this._plane, state = this._buf;
    if (!out) out = new Uint8ClampedArray(plane * 4);
    for (let i = 0; i < plane; i++) {
      let a = state[3 * plane + i];
      if (a < 0) a = 0; else if (a > 1) a = 1;
      const w = 1 - a;
      out[i * 4 + 0] = (w + state[i]) * 255;
      out[i * 4 + 1] = (w + state[plane + i]) * 255;
      out[i * 4 + 2] = (w + state[2 * plane + i]) * 255;
      out[i * 4 + 3] = 255;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Page UI (skipped under Node so the parity test can import BlendedCA).
// ---------------------------------------------------------------------------

if (typeof document !== "undefined" && document.getElementById("demo-canvas")) {
  const canvas = document.getElementById("demo-canvas");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("demo-status");
  const retryBtn = document.getElementById("btn-retry");
  const stepCounterEl = document.getElementById("step-counter");
  const playBtn = document.getElementById("btn-play");
  const seedsBtn = document.getElementById("btn-seeds");
  const bHalvesBtn = document.getElementById("btn-b-halves");
  const bZeroBtn = document.getElementById("btn-b-zero");
  const bOneBtn = document.getElementById("btn-b-one");
  const speedInput = document.getElementById("speed");
  const speedLabel = document.getElementById("speed-label");
  const toolSelect = document.getElementById("tool-select");
  const radiusInput = document.getElementById("brush-radius");
  const radiusVal = document.getElementById("brush-radius-val");
  const valueInput = document.getElementById("brush-value");
  const valueVal = document.getElementById("brush-value-val");
  const eraseToggle = document.getElementById("brush-erase");
  const tintToggle = document.getElementById("tint-toggle");
  const modelASelect = document.getElementById("model-a");
  const modelBSelect = document.getElementById("model-b");

  let ca = null;          // BlendedCA once weights arrive
  let savedB = null;      // painted B survives model reloads (if dims match)
  let imgData = null;
  let rgbaBuf = null;
  let playing = true;
  let stepCount = 0;
  let accumulator = 0;
  let loadToken = 0;      // invalidates stale in-flight loads

  function setStatus(msg, showRetry = false) {
    // The retry button is a child of the status element; keep it.
    statusEl.firstChild.textContent = msg + (showRetry ? " " : "");
    retryBtn.hidden = !showRetry;
  }
  // Ensure a dedicated text node exists before the button.
  if (statusEl.firstChild === retryBtn) {
    statusEl.insertBefore(document.createTextNode(""), retryBtn);
  }

  function bHalves() {
    if (!ca) return;
    const W = ca.width, H = ca.height, B = ca.B;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        B[y * W + x] = x < W / 2 ? 0 : 1;
  }

  function plantSeeds() {
    if (!ca) return;
    ca.clear();
    const W = ca.width, H = ca.height;
    ca.seed(Math.round(W * 0.25), H >> 1); // (16, 32) on a 64x64 grid
    ca.seed(Math.round(W * 0.75), H >> 1); // (48, 32) on a 64x64 grid
    stepCount = 0;
  }

  async function fetchWeights(run) {
    const res = await fetch(`${BUCKET}/${run}/weights.json`, { cache: "no-store" });
    if (res.status === 404 || res.status === 403) {
      const err = new Error(`${run} has no weights yet`);
      err.training = true;
      throw err;
    }
    if (!res.ok) throw new Error(`${run}: HTTP ${res.status}`);
    return res.json();
  }

  async function loadModels() {
    const token = ++loadToken;
    const runA = modelASelect.value, runB = modelBSelect.value;
    setStatus(`loading ${runA} + ${runB}…`);
    if (ca) savedB = ca.B;
    ca = null;
    try {
      const [wA, wB] = await Promise.all([fetchWeights(runA), fetchWeights(runB)]);
      if (token !== loadToken) return; // superseded by a newer selection
      const next = new BlendedCA(wA, wB);
      canvas.width = next.width;
      canvas.height = next.height;
      imgData = ctx.createImageData(next.width, next.height);
      rgbaBuf = new Uint8ClampedArray(next.width * next.height * 4);
      if (savedB && savedB.length === next.B.length) {
        next.B.set(savedB);
      }
      ca = next;
      if (!savedB || savedB.length !== ca.B.length) bHalves();
      savedB = null;
      plantSeeds();
      setStatus(`${runA} (A) + ${runB} (B) loaded — running`);
    } catch (err) {
      if (token !== loadToken) return;
      if (err.training) {
        setStatus(`${err.message} — still training, try again shortly.`, true);
      } else {
        setStatus(`failed to load weights: ${err.message}`, true);
      }
      console.error(err);
    }
  }

  // ------------------------------------------------------------- render --

  function render() {
    if (!ca) return;
    const plane = ca.width * ca.height;
    ca.readRGBA(rgbaBuf);
    const data = imgData.data;
    if (tintToggle.checked) {
      const B = ca.B, k = 0.22;
      for (let i = 0; i < plane; i++) {
        const p = i * 4;
        const bc = B[i] < 0 ? 0 : B[i] > 1 ? 1 : B[i];
        // subtle blue (B=0) -> red (B=1) tint over the ink-on-white image
        const tintR = 40 + bc * 180;
        const tintG = 60;
        const tintB = 220 - bc * 180;
        data[p] = rgbaBuf[p] * (1 - k) + tintR * k;
        data[p + 1] = rgbaBuf[p + 1] * (1 - k) + tintG * k;
        data[p + 2] = rgbaBuf[p + 2] * (1 - k) + tintB * k;
        data[p + 3] = 255;
      }
    } else {
      data.set(rgbaBuf);
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // -------------------------------------------------------------- brush --

  radiusInput.addEventListener("input", () => {
    radiusVal.textContent = radiusInput.value;
  });
  valueInput.addEventListener("input", () => {
    valueVal.textContent = Number(valueInput.value).toFixed(2);
  });

  function paintB(gx, gy, erase) {
    if (!ca) return;
    const W = ca.width, H = ca.height, B = ca.B;
    const radius = Number(radiusInput.value);
    const value = erase ? 0 : Number(valueInput.value);
    const r2 = radius * radius;
    for (let dy = -radius; dy <= radius; dy++) {
      const yy = gy + dy;
      if (yy < 0 || yy >= H) continue;
      for (let dx = -radius; dx <= radius; dx++) {
        const xx = gx + dx;
        if (xx < 0 || xx >= W) continue;
        if (dx * dx + dy * dy > r2) continue;
        B[yy * W + xx] = value;
      }
    }
  }

  function applyTool(gx, gy, erase) {
    if (!ca) return;
    if (toolSelect.value === "damage") {
      ca.damage(gx, gy, 5);
    } else {
      paintB(gx, gy, erase);
    }
  }

  let painting = false;
  let paintErase = false;

  function canvasToGrid(clientX, clientY) {
    const W = ca ? ca.width : canvas.width, H = ca ? ca.height : canvas.height;
    const rect = canvas.getBoundingClientRect();
    const gx = Math.floor((clientX - rect.left) * (W / rect.width));
    const gy = Math.floor((clientY - rect.top) * (H / rect.height));
    return [Math.max(0, Math.min(W - 1, gx)), Math.max(0, Math.min(H - 1, gy))];
  }

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    painting = true;
    paintErase = e.button === 2 || eraseToggle.checked;
    const [gx, gy] = canvasToGrid(e.clientX, e.clientY);
    applyTool(gx, gy, paintErase);
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!painting) return;
    const [gx, gy] = canvasToGrid(e.clientX, e.clientY);
    applyTool(gx, gy, paintErase);
  });

  function stopPainting() { painting = false; }
  canvas.addEventListener("pointerup", stopPainting);
  canvas.addEventListener("pointerleave", stopPainting);
  canvas.addEventListener("pointercancel", stopPainting);

  // ----------------------------------------------------------- controls --

  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.textContent = playing ? "Pause" : "Play";
  });
  seedsBtn.addEventListener("click", plantSeeds);
  bHalvesBtn.addEventListener("click", bHalves);
  bZeroBtn.addEventListener("click", () => { if (ca) ca.B.fill(0); });
  bOneBtn.addEventListener("click", () => { if (ca) ca.B.fill(1); });
  speedInput.addEventListener("input", () => {
    speedLabel.textContent = `${Number(speedInput.value)}×`;
  });
  retryBtn.addEventListener("click", loadModels);
  modelASelect.addEventListener("change", loadModels);
  modelBSelect.addEventListener("change", loadModels);

  // ---------------------------------------------------------- main loop --

  function loop() {
    if (ca && playing) {
      accumulator += Number(speedInput.value);
      while (accumulator >= 1) {
        ca.step();
        stepCount++;
        accumulator -= 1;
      }
    }
    render();
    stepCounterEl.textContent = String(stepCount);
    requestAnimationFrame(loop);
  }

  loadModels();
  requestAnimationFrame(loop);
}

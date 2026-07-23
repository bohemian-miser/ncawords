// demo_steer.js — steering a glider by gradient-modulated kernels.
//
// Channel A runs a trained multik Lenia physics. A static repellent field B
// (Gaussian around a user-placed dot) defines a per-cell steering vector
// s = -grad(B)/|grad(B)| (saturated by gradient strength). Each cell's
// effective kernel is K0 + a*(sx*Kx + sy*Ky), where Kx/Ky are the
// cos/sin-of-stencil-angle dipole weightings of K0 — so the kernel's focus
// leans away from uphill-B and the pattern flees the dot.

const WEIGHTS_URL =
  'https://storage.googleapis.com/recipe-lanes-nca-jobs/lenia-multik-dots/weights.json';

export const S = 96;      // grid side, toroidal
const KS = 15;            // kernel side
const R = 7;              // kernel radius

export class SteerSim {
  constructor(w) {
    this.mu = w.mu;
    this.sg = w.sg;
    this.h = w.h;
    this.dt = w.dt;
    this.leak = w.leak;
    this.nk = w.kernels.length;

    // Flat kernels + dipole basis: Kx = K0*cos(phi), Ky = K0*sin(phi),
    // phi = atan2(dy, dx) over the 15x15 stencil (center 0,0).
    this.K0 = [];
    this.Kx = [];
    this.Ky = [];
    for (let k = 0; k < this.nk; k++) {
      const k0 = new Float32Array(KS * KS);
      const kx = new Float32Array(KS * KS);
      const ky = new Float32Array(KS * KS);
      for (let iy = 0; iy < KS; iy++) {
        for (let ix = 0; ix < KS; ix++) {
          const dy = iy - R;
          const dx = ix - R;
          const phi = Math.atan2(dy, dx);
          const v = w.kernels[k][iy][ix];
          const j = iy * KS + ix;
          k0[j] = v;
          kx[j] = v * Math.cos(phi);
          ky[j] = v * Math.sin(phi);
        }
      }
      this.K0.push(k0);
      this.Kx.push(kx);
      this.Ky.push(ky);
    }

    // Toroidal wrap lookup: wrap[y + ky] = (y + ky - R) mod S.
    this.wrap = new Int32Array(S + KS);
    for (let i = 0; i < S + KS; i++) this.wrap[i] = (((i - R) % S) + S) % S;

    this.A = new Float32Array(S * S);
    this.B = new Float32Array(S * S);
    this.sx = new Float32Array(S * S);
    this.sy = new Float32Array(S * S);
    this.dxTot = new Float32Array(S * S);

    // Circular-mean tables for the toroidal centroid.
    this.cosT = new Float32Array(S);
    this.sinT = new Float32Array(S);
    for (let i = 0; i < S; i++) {
      this.cosT[i] = Math.cos((2 * Math.PI * i) / S);
      this.sinT[i] = Math.sin((2 * Math.PI * i) / S);
    }

    this.a = 0.8;          // steering strength
    this.sigma = 14;       // repellent Gaussian width
    this.dot = { x: 60, y: 48 };
    this.stepCount = 0;

    this.resetBlob();
    this.updateField();
  }

  // Zeros + one disk (radius 5, value 1.0) at (x=30, y=48).
  resetBlob() {
    this.A.fill(0);
    const cx = 30;
    const cy = 48;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        let ddx = Math.abs(x - cx);
        ddx = Math.min(ddx, S - ddx);
        let ddy = Math.abs(y - cy);
        ddy = Math.min(ddy, S - ddy);
        if (ddx * ddx + ddy * ddy <= 25) this.A[y * S + x] = 1.0;
      }
    }
    this.stepCount = 0;
  }

  // Recompute B and the steering field (call when dot or sigma changes).
  updateField() {
    const { B, sx, sy } = this;
    const bx = this.dot.x;
    const by = this.dot.y;
    const twoSig2 = 2 * this.sigma * this.sigma;
    for (let y = 0; y < S; y++) {
      let ddy = Math.abs(y - by);
      ddy = Math.min(ddy, S - ddy);
      for (let x = 0; x < S; x++) {
        let ddx = Math.abs(x - bx);
        ddx = Math.min(ddx, S - ddx);
        B[y * S + x] = Math.exp(-(ddx * ddx + ddy * ddy) / twoSig2);
      }
    }

    // Central-difference toroidal gradient of B.
    const gx = new Float32Array(S * S);
    const gy = new Float32Array(S * S);
    let maxMag = 0;
    for (let y = 0; y < S; y++) {
      const ym = ((y + S - 1) % S) * S;
      const yp = ((y + 1) % S) * S;
      const yr = y * S;
      for (let x = 0; x < S; x++) {
        const xm = (x + S - 1) % S;
        const xp = (x + 1) % S;
        const i = yr + x;
        gx[i] = (B[yr + xp] - B[yr + xm]) * 0.5;
        gy[i] = (B[yp + x] - B[ym + x]) * 0.5;
        const m = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
        if (m > maxMag) maxMag = m;
      }
    }

    // Unit direction, saturated by strength; steer AWAY from the dot.
    const gmax = 0.6 * maxMag;
    for (let i = 0; i < S * S; i++) {
      const m = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
      const wgt = gmax > 0 ? Math.min(1, m / gmax) : 0;
      const inv = 1 / (m + 1e-9);
      sx[i] = -gx[i] * inv * wgt;
      sy[i] = -gy[i] * inv * wgt;
    }
  }

  step() {
    const { A, sx, sy, dxTot, wrap, a } = this;
    dxTot.fill(0);
    for (let k = 0; k < this.nk; k++) {
      const K0 = this.K0[k];
      const Kx = this.Kx[k];
      const Ky = this.Ky[k];
      const mu = this.mu[k];
      const invSg = 1 / this.sg[k];
      const h = this.h[k];
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          let u0 = 0;
          let ux = 0;
          let uy = 0;
          for (let ky = 0; ky < KS; ky++) {
            const row = wrap[y + ky] * S;
            const kb = ky * KS;
            for (let kx = 0; kx < KS; kx++) {
              const v = A[row + wrap[x + kx]];
              const j = kb + kx;
              u0 += v * K0[j];
              ux += v * Kx[j];
              uy += v * Ky[j];
            }
          }
          const i = y * S + x;
          const U = u0 + a * (sx[i] * ux + sy[i] * uy);
          const z = (U - mu) * invSg;
          dxTot[i] += h * (2 * Math.exp(-0.5 * z * z) - 1);
        }
      }
    }
    const dt = this.dt;
    const leak = this.leak;
    for (let i = 0; i < S * S; i++) {
      const v = A[i] + dt * dxTot[i];
      const c = v < 0 ? 0 : v > 1 ? 1 : v;
      A[i] = c + leak * (v - c);
    }
    this.stepCount++;
  }

  // Toroidal (circular-mean) weighted centroid of A; null when empty.
  centroid() {
    const { A, cosT, sinT } = this;
    let tot = 0;
    let cX = 0;
    let sX = 0;
    let cY = 0;
    let sY = 0;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const w = A[y * S + x];
        if (w > 0) {
          tot += w;
          cX += w * cosT[x];
          sX += w * sinT[x];
          cY += w * cosT[y];
          sY += w * sinT[y];
        }
      }
    }
    if (tot < 1e-6) return null;
    const fx = Math.atan2(sX / tot, cX / tot) / (2 * Math.PI);
    const fy = Math.atan2(sY / tot, cY / tot) / (2 * Math.PI);
    return {
      x: ((fx * S) % S + S) % S,
      y: ((fy * S) % S + S) % S,
    };
  }

  // Effective kernel K0_0 + a*(sx*Kx_0 + sy*Ky_0) at a given grid cell
  // (toroidally wrapped, rounded to the nearest cell), min-max normalized
  // to [0,1]. Returns { data, sx, sy }.
  effectiveKernelAtCell(x, y) {
    const cx = ((Math.round(x) % S) + S) % S;
    const cy = ((Math.round(y) % S) + S) % S;
    const i = cy * S + cx;
    const sxc = this.sx[i];
    const syc = this.sy[i];
    const K0 = this.K0[0];
    const Kx = this.Kx[0];
    const Ky = this.Ky[0];
    const out = new Float32Array(KS * KS);
    let lo = Infinity;
    let hi = -Infinity;
    for (let j = 0; j < KS * KS; j++) {
      const v = K0[j] + this.a * (sxc * Kx[j] + syc * Ky[j]);
      out[j] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const span = hi - lo > 1e-12 ? hi - lo : 1;
    for (let j = 0; j < KS * KS; j++) out[j] = (out[j] - lo) / span;
    return { data: out, sx: sxc, sy: syc };
  }

  // Effective kernel at the blob's centroid, min-max normalized to [0,1].
  // Returns { data, sx, sy } or null when the blob is empty.
  effectiveKernelAtCentroid() {
    const c = this.centroid();
    if (!c) return null;
    return this.effectiveKernelAtCell(c.x, c.y);
  }
}

/* ---------- browser UI ---------- */

function main() {
  const canvas = document.getElementById('steer-canvas');
  const ctx = canvas.getContext('2d');
  const kCanvas = document.getElementById('kernel-canvas');
  const kCtx = kCanvas.getContext('2d');
  const kLabel = document.getElementById('kernel-inset-label');
  const status = document.getElementById('demo-status');
  const stepCounter = document.getElementById('step-counter');
  const btnPlay = document.getElementById('btn-play');
  const btnReset = document.getElementById('btn-reset');
  const strength = document.getElementById('strength');
  const strengthVal = document.getElementById('strength-val');
  const sigma = document.getElementById('sigma');
  const sigmaVal = document.getElementById('sigma-val');
  const speed = document.getElementById('speed');
  const speedVal = document.getElementById('speed-val');

  const SCALE = canvas.width / S; // 5x upscale
  const off = document.createElement('canvas');
  off.width = S;
  off.height = S;
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(S, S);
  const kImg = kCtx.createImageData(KS, KS);

  let sim = null;
  let playing = true;
  let stepAcc = 0;
  let frame = 0;

  // Grid-space cursor position while hovering the main canvas, or null
  // when the mouse isn't over it (falls back to blob-centroid preview).
  let hover = null;
  let lastHoverAt = 0;
  const HOVER_THROTTLE_MS = 30;

  function render() {
    const { A, B } = sim;
    const px = img.data;
    for (let i = 0; i < S * S; i++) {
      const v = A[i] < 0 ? 0 : A[i] > 1 ? 1 : A[i];
      const base = 255 * (1 - v); // ink on white
      const al = Math.min(0.35, 0.35 * B[i]); // translucent red overlay
      px[i * 4] = base * (1 - al) + 201 * al;
      px[i * 4 + 1] = base * (1 - al) + 42 * al;
      px[i * 4 + 2] = base * (1 - al) + 42 * al;
      px[i * 4 + 3] = 255;
    }
    offCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

    // The repellent dot.
    ctx.beginPath();
    ctx.arc((sim.dot.x + 0.5) * SCALE, (sim.dot.y + 0.5) * SCALE, 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#c92a2a';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
  }

  function renderKernelInset() {
    let eff;
    if (hover) {
      eff = sim.effectiveKernelAtCell(hover.x, hover.y);
      kLabel.textContent = 'kernel at cursor';
    } else {
      eff = sim.effectiveKernelAtCentroid();
      kLabel.textContent = 'kernel at the blob';
    }
    const px = kImg.data;
    for (let j = 0; j < KS * KS; j++) {
      const v = eff ? eff.data[j] : 0;
      const g = 255 * (1 - v); // ink on white, dark = strong
      px[j * 4] = g;
      px[j * 4 + 1] = g;
      px[j * 4 + 2] = g;
      px[j * 4 + 3] = 255;
    }
    kCtx.putImageData(kImg, 0, 0);
  }

  function loop() {
    if (playing) {
      stepAcc += parseFloat(speed.value);
      while (stepAcc >= 1) {
        sim.step();
        stepAcc -= 1;
      }
    }
    render();
    if (frame % 10 === 0) renderKernelInset();
    frame++;
    stepCounter.textContent = String(sim.stepCount);
    requestAnimationFrame(loop);
  }

  function moveDot(e) {
    const rect = canvas.getBoundingClientRect();
    let gx = ((e.clientX - rect.left) / rect.width) * S;
    let gy = ((e.clientY - rect.top) / rect.height) * S;
    gx = Math.max(0, Math.min(S - 0.001, gx));
    gy = Math.max(0, Math.min(S - 0.001, gy));
    sim.dot.x = gx;
    sim.dot.y = gy;
    sim.updateField();
  }

  // Grid coords under the pointer, for the kernel-inset hover preview.
  // Runs alongside (not instead of) drag handling on the same pointermove.
  function updateHover(e) {
    const now = performance.now();
    if (now - lastHoverAt < HOVER_THROTTLE_MS) return;
    lastHoverAt = now;
    const rect = canvas.getBoundingClientRect();
    let gx = ((e.clientX - rect.left) / rect.width) * S;
    let gy = ((e.clientY - rect.top) / rect.height) * S;
    gx = Math.max(0, Math.min(S - 0.001, gx));
    gy = Math.max(0, Math.min(S - 0.001, gy));
    hover = { x: gx, y: gy };
    if (sim) renderKernelInset();
  }

  let dragging = false;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    moveDot(e);
    e.preventDefault();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (dragging) moveDot(e);
    updateHover(e);
  });
  canvas.addEventListener('pointerup', () => {
    dragging = false;
  });
  canvas.addEventListener('pointercancel', () => {
    dragging = false;
  });
  canvas.addEventListener('pointerleave', () => {
    hover = null;
    if (sim) renderKernelInset();
  });

  btnPlay.addEventListener('click', () => {
    playing = !playing;
    btnPlay.textContent = playing ? 'Pause' : 'Play';
  });
  btnReset.addEventListener('click', () => {
    sim.resetBlob();
  });
  strength.addEventListener('input', () => {
    sim.a = parseFloat(strength.value);
    strengthVal.textContent = sim.a.toFixed(2);
    renderKernelInset();
  });
  sigma.addEventListener('input', () => {
    sim.sigma = parseFloat(sigma.value);
    sigmaVal.textContent = sigma.value;
    sim.updateField();
    renderKernelInset();
  });
  speed.addEventListener('input', () => {
    speedVal.textContent = parseFloat(speed.value).toFixed(2).replace(/\.?0+$/, '') + '×';
  });

  fetch(WEIGHTS_URL)
    .then((r) => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then((w) => {
      sim = new SteerSim(w);
      status.textContent =
        w.kernels.length + ' trained kernels · dt=' + w.dt + ' · leak=' + w.leak;
      renderKernelInset();
      requestAnimationFrame(loop);
    })
    .catch((err) => {
      status.textContent = 'failed to load weights: ' + err.message;
    });
}

if (typeof document !== 'undefined' && document.getElementById('steer-canvas')) {
  main();
}

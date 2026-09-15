// demo_dynkernel.js — Dynamic Kernel NCA: Physics Interpolation & Organism Comparison
//
// Allows comparing two 16-channel NCA models (Model A and Model B) — including
// the new Dynamic Layer-Modulated 5x5 Kernel models, fixed 5x5 kernel models, and
// classic models — side-by-side on ONE shared grid using the spatial interpolation trick:
//
//   dx_A = fc1_A(relu(fc0_A(p_A)))    -- Model A's proposal (with dynamic/fixed kernels)
//   dx_B = fc1_B(relu(fc0_B(p_B)))    -- Model B's proposal (with dynamic/fixed kernels)
//   dx   = (1 - B(x,y)) * dx_A + B(x,y) * dx_B
//   x   += fire_mask * dx
//
// A painted control field B(x,y) in [0, 1] locally decides the mixture of rules.
// Supports spontaneous self-organization from pure noise, seed planting, damage repair,
// and real-time inspector of effective 5x5 dynamic kernels and basis activations.

const SIM_SIZE = 288;
const SIM_PLANE = SIM_SIZE * SIM_SIZE;
const C = 16;
const HN = 128;

// Available models in docs/weights/
const MODEL_REGISTRY = [
  { id: "snaps_lenia_dynkernel5x5_fire_noise", name: "🔥 Fire (Dynamic 5×5 · Noise)", path: "weights/snaps_lenia_dynkernel5x5_fire_noise.json" },
  { id: "snaps_lenia_dynkernel5x5_rocket_noise", name: "🚀 Rocket (Dynamic 5×5 · Noise)", path: "weights/snaps_lenia_dynkernel5x5_rocket_noise.json" },
  { id: "snaps_lenia_dynkernel5x5_gecko_noise", name: "🦎 Gecko (Dynamic 5×5 · Noise)", path: "weights/snaps_lenia_dynkernel5x5_gecko_noise.json" },
  { id: "snaps_lenia_dynkernel5x5_heart_noise", name: "❤️ Heart (Dynamic 5×5 · Noise)", path: "weights/snaps_lenia_dynkernel5x5_heart_noise.json" },
  { id: "snaps_lenia_dynkernel5x5_alien_noise", name: "👽 Alien (Dynamic 5×5 · Noise)", path: "weights/snaps_lenia_dynkernel5x5_alien_noise.json" },
  { id: "snaps_lenia_dynkernel5x5_butterfly_noise", name: "🦋 Butterfly (Dynamic 5×5 · Noise)", path: "weights/snaps_lenia_dynkernel5x5_butterfly_noise.json" },
  { id: "snaps_lenia_dynkernel5x5_hex", name: "⬡ Hex (Dynamic 5×5)", path: "weights/snaps_lenia_dynkernel5x5_hex.json" },
  { id: "snaps_lenia_dynkernel5x5_dots", name: "⚬ Dots (Dynamic 5×5)", path: "weights/snaps_lenia_dynkernel5x5_dots.json" },
  { id: "snaps_lenia_kernel5x5_fire", name: "⚙️ Fire (Fixed 5×5 Kernel)", path: "weights/snaps_lenia_kernel5x5_fire.json" },
  { id: "snaps_lenia_kernel5x5_dots_ring", name: "⚙️ Dots Ring (Fixed 5×5 Kernel)", path: "weights/snaps_lenia_kernel5x5_dots_ring.json" },
  { id: "snaps_lenia_kernel_fire", name: "⚙️ Fire (Fixed 3×3 Kernel)", path: "weights/snaps_lenia_kernel_fire.json" },
  { id: "snaps_lenia_kernel_dots", name: "⚙️ Dots (Fixed 3×3 Kernel)", path: "weights/snaps_lenia_kernel_dots.json" }
];

// Helper to flatten 2D weight matrices into 1D Float32Array for cache-friendly multiplication
function parseWeights(raw) {
  const isDynamic = raw.kind === "dynamic_kernel_nca" || Boolean(raw.basis_kernels);
  const Nk = raw.num_kernels || 2;
  const ks = raw.kernel_size || (raw.kernels ? raw.kernels[0].length : (isDynamic ? 5 : 3));
  const pad = Math.floor(ks / 2);
  const inC = (1 + Nk) * C;
  
  // Flatten fc0_w [HN, inC]
  const fc0_w = new Float32Array(HN * inC);
  for (let k = 0; k < HN; k++) {
    const row = raw.fc0_w[k];
    const off = k * inC;
    for (let j = 0; j < inC; j++) fc0_w[off + j] = row[j];
  }
  const fc0_b = Float32Array.from(raw.fc0_b);
  
  // Flatten fc1_w [C, HN]
  const fc1_w = new Float32Array(C * HN);
  for (let c = 0; c < C; c++) {
    const row = raw.fc1_w[c];
    const off = c * HN;
    for (let k = 0; k < HN; k++) fc1_w[off + k] = row[k];
  }

  const parsed = {
    name: raw.name || "NCA",
    isDynamic,
    Nk,
    ks,
    pad,
    inC,
    fc0_w,
    fc0_b,
    fc1_w,
    seedType: raw.seedType || "seed",
    temperature: raw.temperature || 1.0
  };

  if (isDynamic) {
    parsed.num_basis = raw.num_basis || 4;
    parsed.basis_kernels = raw.basis_kernels; // [Nk, M, ks, ks]
    parsed.w_surround = raw.w_surround;       // [Nk*M, C, 3, 3]
    parsed.b_surround = Float32Array.from(raw.b_surround);
    parsed.w_x_to_k = raw.w_x_to_k || null;
    parsed.b_x_to_k = raw.b_x_to_k ? Float32Array.from(raw.b_x_to_k) : null;
    parsed.w_h_to_k = raw.w_h_to_k || null;
    parsed.b_h_to_k = raw.b_h_to_k ? Float32Array.from(raw.b_h_to_k) : null;

    // Pre-flatten basis kernels [Nk][M] into 1D Float32Arrays for zero-allocation synthesis
    parsed.flatBasis = [];
    for (let k = 0; k < Nk; k++) {
      const basisList = [];
      for (let m = 0; m < parsed.num_basis; m++) {
        const flat = new Float32Array(ks * ks);
        const bkm = raw.basis_kernels[k][m];
        for (let r = 0; r < ks; r++) {
          for (let c = 0; c < ks; c++) {
            flat[r * ks + c] = bkm[r][c];
          }
        }
        basisList.push(flat);
      }
      parsed.flatBasis.push(basisList);
    }
  } else {
    parsed.flatKernels = [];
    const srcK = (raw.kernels && raw.kernels.length >= 2) ? raw.kernels : [
      [[-1/8, 0, 1/8], [-2/8, 0, 2/8], [-1/8, 0, 1/8]],
      [[-1/8, -2/8, -1/8], [0, 0, 0], [1/8, 2/8, 1/8]]
    ];
    parsed.kernels = srcK;
    for (let k = 0; k < Nk; k++) {
      const flat = new Float32Array(ks * ks);
      const kMat = srcK[k];
      for (let r = 0; r < ks; r++) {
        for (let c = 0; c < ks; c++) {
          flat[r * ks + c] = kMat[r][c];
        }
      }
      parsed.flatKernels.push(flat);
    }
  }
  return parsed;
}

// ---------------------------------------------------------------------------
// Blended Dual-Model Engine
// ---------------------------------------------------------------------------

class BlendedDynKernelCA {
  constructor(weightsA, weightsB, size = SIM_SIZE) {
    this.W = size;
    this.H = size;
    this.plane = size * size;
    this.modelA = parseWeights(weightsA);
    this.modelB = parseWeights(weightsB);
    
    this.buf = new Float32Array(C * this.plane);
    this.back = new Float32Array(C * this.plane);
    this.B = new Float32Array(this.plane); // Control field: 0 = Model A, 1 = Model B
    
    // Per-cell scratch buffers to eliminate garbage collection in step()
    this.maxInC = Math.max(this.modelA.inC, this.modelB.inC);
    this.pA = new Float32Array(this.maxInC);
    this.pB = new Float32Array(this.maxInC);
    this.hA = new Float32Array(HN);
    this.hB = new Float32Array(HN);
    this.dxA = new Float32Array(C);
    this.dxB = new Float32Array(C);

    // Scratch buffers for dynamic kernel synthesis (zero allocation in step)
    this.k0A = new Float32Array(25);
    this.k1A = new Float32Array(25);
    this.k0B = new Float32Array(25);
    this.k1B = new Float32Array(25);
    this.scratchLogits = new Float32Array(8);
    this.scratchAlpha = new Float32Array(8);
    
    this.pre = new Uint8Array(this.plane);
    this.post = new Uint8Array(this.plane);
    this.fire_rate = 0.5;
    this.aliveMaskEnabled = true;

    // Global layer pooled activations for dynamic models
    this.x_barA = new Float32Array(C);
    this.h_barA = new Float32Array(HN);
    this.alpha_layerA = new Float32Array(this.modelA.Nk * (this.modelA.num_basis || 4));
    this.hStateA = new Float32Array(HN * this.plane);
    this.hNextA = new Float32Array(HN * this.plane);

    this.x_barB = new Float32Array(C);
    this.h_barB = new Float32Array(HN);
    this.alpha_layerB = new Float32Array(this.modelB.Nk * (this.modelB.num_basis || 4));
    this.hStateB = new Float32Array(HN * this.plane);
    this.hNextB = new Float32Array(HN * this.plane);
  }

  clear() {
    this.buf.fill(0);
    this.back.fill(0);
    this.hStateA.fill(0);
    this.hStateB.fill(0);
  }

  resetNoise() {
    this.clear();
    const plane = this.plane;
    // Fill all 16 channels with uniform random values, giving strong alpha
    for (let c = 0; c < C; c++) {
      const off = c * plane;
      for (let i = 0; i < plane; i++) {
        if (c === 3) {
          this.buf[off + i] = Math.random() * 0.9 + 0.1; // Alpha > 0.1
        } else {
          this.buf[off + i] = Math.random();
        }
      }
    }
  }

  resetSeeds() {
    this.clear();
    const W = this.W, H = this.H;
    // Seed A in left quadrant
    const ax = Math.floor(W * 0.30), ay = Math.floor(H * 0.5);
    // Seed B in right quadrant
    const bx = Math.floor(W * 0.70), by = Math.floor(H * 0.5);
    
    this._placeSeedBlob(ax, ay, 6);
    this._placeSeedBlob(bx, by, 6);
  }

  _placeSeedBlob(cx, cy, radius = 3) {
    const W = this.W, H = this.H, plane = this.plane;
    for (let dy = -radius; dy <= radius; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= H) continue;
      for (let dx = -radius; dx <= radius; dx++) {
        const x = cx + dx;
        if (x < 0 || x >= W) continue;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= radius) {
          const i = y * W + x;
          const strength = 1.0 - dist / (radius + 0.5);
          for (let c = 0; c < C; c++) {
            this.buf[c * plane + i] = (c >= 3 ? strength : 0.8 * strength);
          }
        }
      }
    }
  }

  // --- Control Field Presets ---

  setRamp() {
    const W = this.W, H = this.H;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        this.B[y * W + x] = x / (W - 1);
      }
    }
  }

  setSplit() {
    const W = this.W, H = this.H;
    const mid = Math.floor(W / 2);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        this.B[y * W + x] = (x < mid) ? 0.0 : 1.0;
      }
    }
  }

  setUniform(val) {
    this.B.fill(Math.max(0, Math.min(1, val)));
  }

  setCircle(cx = Math.floor(this.W / 2), cy = Math.floor(this.H / 2), radius = Math.floor(this.W * 0.28)) {
    const W = this.W, H = this.H;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        this.B[y * W + x] = (d <= radius) ? 1.0 : 0.0;
      }
    }
  }

  paintB(cx, cy, radius, value) {
    const W = this.W, H = this.H;
    const r2 = radius * radius;
    for (let dy = -radius; dy <= radius; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= H) continue;
      for (let dx = -radius; dx <= radius; dx++) {
        const x = cx + dx;
        if (x < 0 || x >= W) continue;
        const d2 = dx * dx + dy * dy;
        if (d2 <= r2) {
          const falloff = 1.0 - Math.sqrt(d2) / radius;
          const i = y * W + x;
          this.B[i] = this.B[i] + (value - this.B[i]) * Math.min(1.0, falloff * 1.5);
          if (this.B[i] < 0) this.B[i] = 0;
          if (this.B[i] > 1) this.B[i] = 1;
        }
      }
    }
  }

  damage(cx, cy, radius) {
    const W = this.W, H = this.H, plane = this.plane;
    const r2 = radius * radius;
    for (let dy = -radius; dy <= radius; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= H) continue;
      for (let dx = -radius; dx <= radius; dx++) {
        const x = cx + dx;
        if (x < 0 || x >= W) continue;
        if (dx * dx + dy * dy <= r2) {
          const i = y * W + x;
          for (let c = 0; c < C; c++) this.buf[c * plane + i] = 0;
        }
      }
    }
  }

  // --- Dynamic Kernel Computation at a Cell ---

  getEffectiveKernel(model, alphaLayer, gx, gy) {
    if (!model.isDynamic) {
      return { kernels: model.kernels, alphas: [[1, 0, 0, 0], [1, 0, 0, 0]] };
    }
    const W = this.W, H = this.H, plane = this.plane;
    const cur = this.buf;
    const Nk = model.Nk, M = model.num_basis;
    const basis = model.basis_kernels;
    const ws = model.w_surround, bs = model.b_surround;
    const temp = model.temperature;
    const ks = model.ks;

    const effKernels = [];
    const all_alphas = [];

    for (let k = 0; k < Nk; k++) {
      const logits = [];
      for (let m = 0; m < M; m++) {
        const idx = k * M + m;
        let a = (bs[idx] !== undefined ? bs[idx] : 0.0) + (alphaLayer ? alphaLayer[idx] : 0.0);
        const w_km = ws[idx];
        for (let dy = -1; dy <= 1; dy++) {
          const py = gy + dy;
          if (py < 0 || py >= H) continue;
          const rowOff = py * W;
          for (let dx = -1; dx <= 1; dx++) {
            const px = gx + dx;
            if (px < 0 || px >= W) continue;
            const pix = rowOff + px;
            for (let c = 0; c < C; c++) {
              a += w_km[c][dy + 1][dx + 1] * cur[c * plane + pix];
            }
          }
        }
        logits.push(a);
      }

      let maxL = -Infinity;
      for (let m = 0; m < M; m++) if (logits[m] > maxL) maxL = logits[m];
      let sumExp = 0;
      const alpha_k = [];
      for (let m = 0; m < M; m++) {
        const ex = Math.exp((logits[m] - maxL) / temp);
        alpha_k.push(ex);
        sumExp += ex;
      }
      for (let m = 0; m < M; m++) alpha_k[m] /= sumExp;
      all_alphas.push(alpha_k);

      // Synthesize 5x5 kernel
      const K = [];
      for (let r = 0; r < ks; r++) {
        const row = [];
        for (let c = 0; c < ks; c++) {
          let v = 0;
          for (let m = 0; m < M; m++) {
            v += alpha_k[m] * basis[k][m][r][c];
          }
          row.push(v);
        }
        K.push(row);
      }
      effKernels.push(K);
    }
    return { kernels: effKernels, alphas: all_alphas };
  }

  // --- Step Simulation ---

  _updateAlphaLayer(model, x_bar, h_bar, alphaLayer, hState) {
    if (!model.isDynamic || !model.w_x_to_k) return;
    const plane = this.plane;
    const cur = this.buf;
    // Mean pooling
    for (let c = 0; c < C; c++) {
      let sum = 0;
      const off = c * plane;
      for (let i = 0; i < plane; i++) sum += cur[off + i];
      x_bar[c] = sum / plane;
    }
    for (let k = 0; k < HN; k++) {
      let sum = 0;
      const off = k * plane;
      for (let i = 0; i < plane; i++) sum += hState[off + i];
      h_bar[k] = sum / plane;
    }
    const Nk = model.Nk, M = model.num_basis;
    const wx = model.w_x_to_k, bx = model.b_x_to_k;
    const wh = model.w_h_to_k, bh = model.b_h_to_k;
    for (let idx = 0; idx < Nk * M; idx++) {
      let a = (bx ? bx[idx] : 0) + (bh ? bh[idx] : 0);
      const wx_row = wx[idx], wh_row = wh[idx];
      for (let c = 0; c < C; c++) a += wx_row[c] * x_bar[c];
      for (let k = 0; k < HN; k++) a += wh_row[k] * h_bar[k];
      alphaLayer[idx] = a;
    }
  }

  _synthesizeKernelFlat(model, alphaLayer, gx, gy, kOut0, kOut1) {
    if (!model.isDynamic) {
      kOut0.set(model.flatKernels[0]);
      kOut1.set(model.flatKernels[1]);
      return;
    }
    const W = this.W, H = this.H, plane = this.plane;
    const cur = this.buf;
    const Nk = model.Nk, M = model.num_basis;
    const flatBasis = model.flatBasis;
    const ws = model.w_surround, bs = model.b_surround;
    const temp = model.temperature;
    const ks = model.ks;
    const ks2 = ks * ks;

    for (let k = 0; k < Nk; k++) {
      let maxL = -Infinity;
      for (let m = 0; m < M; m++) {
        const idx = k * M + m;
        let a = (bs[idx] !== undefined ? bs[idx] : 0.0) + (alphaLayer ? alphaLayer[idx] : 0.0);
        const w_km = ws[idx];
        for (let dy = -1; dy <= 1; dy++) {
          const py = gy + dy;
          if (py < 0 || py >= H) continue;
          const rowOff = py * W;
          const wy = dy + 1;
          for (let dx = -1; dx <= 1; dx++) {
            const px = gx + dx;
            if (px < 0 || px >= W) continue;
            const pix = rowOff + px;
            const wx_val = dx + 1;
            for (let c = 0; c < C; c++) {
              a += w_km[c][wy][wx_val] * cur[c * plane + pix];
            }
          }
        }
        this.scratchLogits[m] = a;
        if (a > maxL) maxL = a;
      }

      let sumExp = 0;
      for (let m = 0; m < M; m++) {
        const ex = Math.exp((this.scratchLogits[m] - maxL) / temp);
        this.scratchAlpha[m] = ex;
        sumExp += ex;
      }
      const invSum = 1.0 / (sumExp > 1e-8 ? sumExp : 1e-8);
      for (let m = 0; m < M; m++) {
        this.scratchAlpha[m] *= invSum;
      }

      const kTarget = (k === 0) ? kOut0 : kOut1;
      kTarget.fill(0);
      const basisK = flatBasis[k];
      for (let m = 0; m < M; m++) {
        const weight = this.scratchAlpha[m];
        const bMat = basisK[m];
        for (let j = 0; j < ks2; j++) {
          kTarget[j] += weight * bMat[j];
        }
      }
    }
  }

  _computeProposal(model, k0, k1, hState, hNext, x, y, pOut, hOut, dxOut) {
    const W = this.W, H = this.H, plane = this.plane;
    const cur = this.buf;
    const i = y * W + x;
    const ks = model.ks, pad = model.pad;
    const inC = model.inC;
    const fc0_w = model.fc0_w, fc0_b = model.fc0_b, fc1_w = model.fc1_w;

    // 1. Perception convolution
    const isInterior = (x >= pad && x < W - pad && y >= pad && y < H - pad);

    if (isInterior && ks === 5) {
      // Branchless unrolled 5x5 convolution for interior cells
      for (let c = 0; c < C; c++) {
        const cOff = c * plane;
        pOut[c] = cur[cOff + i];
        
        let sum0 = 0, sum1 = 0;
        let kidx = 0;
        for (let dy = -2; dy <= 2; dy++) {
          const rowStart = cOff + (y + dy) * W + (x - 2);
          for (let dx = 0; dx < 5; dx++) {
            const v = cur[rowStart + dx];
            sum0 += k0[kidx] * v;
            sum1 += k1[kidx] * v;
            kidx++;
          }
        }
        pOut[16 + c] = sum0;
        pOut[32 + c] = sum1;
      }
    } else if (isInterior && ks === 3) {
      // Branchless unrolled 3x3 convolution
      for (let c = 0; c < C; c++) {
        const cOff = c * plane;
        pOut[c] = cur[cOff + i];
        
        let sum0 = 0, sum1 = 0;
        let kidx = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const rowStart = cOff + (y + dy) * W + (x - 1);
          for (let dx = 0; dx < 3; dx++) {
            const v = cur[rowStart + dx];
            sum0 += k0[kidx] * v;
            sum1 += k1[kidx] * v;
            kidx++;
          }
        }
        pOut[16 + c] = sum0;
        pOut[32 + c] = sum1;
      }
    } else {
      // Safe border convolution with boundary clipping
      for (let c = 0; c < C; c++) {
        const cOff = c * plane;
        pOut[c] = cur[cOff + i];

        let sum0 = 0, sum1 = 0;
        let kidx = 0;
        for (let dy = -pad; dy <= pad; dy++) {
          const py = y + dy;
          if (py < 0 || py >= H) {
            kidx += ks;
            continue;
          }
          const rowOff = cOff + py * W;
          for (let dx = -pad; dx <= pad; dx++) {
            const px = x + dx;
            if (px >= 0 && px < W) {
              const v = cur[rowOff + px];
              sum0 += k0[kidx] * v;
              sum1 += k1[kidx] * v;
            }
            kidx++;
          }
        }
        pOut[16 + c] = sum0;
        pOut[32 + c] = sum1;
      }
    }

    // 2. Hidden layer: h = ReLU(fc0_w * p + fc0_b)
    for (let k = 0; k < HN; k++) {
      let sum = fc0_b[k];
      const off = k * inC;
      for (let j = 0; j < inC; j++) {
        sum += fc0_w[off + j] * pOut[j];
      }
      const act = sum > 0 ? sum : 0;
      hOut[k] = act;
      if (hNext) hNext[k * plane + i] = act;
    }

    // 3. Output layer: dx = fc1_w * h
    for (let c = 0; c < C; c++) {
      let sum = 0;
      const off = c * HN;
      for (let k = 0; k < HN; k++) {
        sum += fc1_w[off + k] * hOut[k];
      }
      dxOut[c] = sum;
    }
  }

  _aliveMask(state, mask) {
    const W = this.W, H = this.H, plane = this.plane;
    const aOff = 3 * plane; // Channel 3 = Alpha
    for (let y = 0; y < H; y++) {
      const yu = y > 0, yd = y < H - 1;
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        let maxA = state[aOff + i];
        if (x > 0 && state[aOff + i - 1] > maxA) maxA = state[aOff + i - 1];
        if (x < W - 1 && state[aOff + i + 1] > maxA) maxA = state[aOff + i + 1];
        if (yu) {
          const u = i - W;
          if (state[aOff + u] > maxA) maxA = state[aOff + u];
          if (x > 0 && state[aOff + u - 1] > maxA) maxA = state[aOff + u - 1];
          if (x < W - 1 && state[aOff + u + 1] > maxA) maxA = state[aOff + u + 1];
        }
        if (yd) {
          const d = i + W;
          if (state[aOff + d] > maxA) maxA = state[aOff + d];
          if (x > 0 && state[aOff + d - 1] > maxA) maxA = state[aOff + d - 1];
          if (x < W - 1 && state[aOff + d + 1] > maxA) maxA = state[aOff + d + 1];
        }
        mask[i] = maxA > 0.1 ? 1 : 0;
      }
    }
  }

  step() {
    const W = this.W, H = this.H, plane = this.plane;
    const cur = this.buf, nxt = this.back;
    const Bf = this.B;
    const fireRate = this.fire_rate;
    const useAlive = this.aliveMaskEnabled;

    this._updateAlphaLayer(this.modelA, this.x_barA, this.h_barA, this.alpha_layerA, this.hStateA);
    this._updateAlphaLayer(this.modelB, this.x_barB, this.h_barB, this.alpha_layerB, this.hStateB);

    if (useAlive) {
      this._aliveMask(cur, this.pre);
    }

    const pA = this.pA, pB = this.pB;
    const hA = this.hA, hB = this.hB;
    const dxA = this.dxA, dxB = this.dxB;
    const k0A = this.k0A, k1A = this.k1A;
    const k0B = this.k0B, k1B = this.k1B;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;

        // When alive mask is enabled, cells with pre[i] == 0 are strictly dead
        if (useAlive && !this.pre[i]) {
          for (let c = 0; c < C; c++) nxt[c * plane + i] = 0;
          continue;
        }

        const fired = Math.random() <= fireRate;
        if (!fired) {
          for (let c = 0; c < C; c++) nxt[c * plane + i] = cur[c * plane + i];
          continue;
        }

        const bVal = Bf[i];
        const aVal = 1.0 - bVal;

        if (bVal <= 0.001) {
          this._synthesizeKernelFlat(this.modelA, this.alpha_layerA, x, y, k0A, k1A);
          this._computeProposal(this.modelA, k0A, k1A, this.hStateA, this.hNextA, x, y, pA, hA, dxA);
          for (let c = 0; c < C; c++) {
            nxt[c * plane + i] = cur[c * plane + i] + dxA[c];
          }
        } else if (bVal >= 0.999) {
          this._synthesizeKernelFlat(this.modelB, this.alpha_layerB, x, y, k0B, k1B);
          this._computeProposal(this.modelB, k0B, k1B, this.hStateB, this.hNextB, x, y, pB, hB, dxB);
          for (let c = 0; c < C; c++) {
            nxt[c * plane + i] = cur[c * plane + i] + dxB[c];
          }
        } else {
          this._synthesizeKernelFlat(this.modelA, this.alpha_layerA, x, y, k0A, k1A);
          this._computeProposal(this.modelA, k0A, k1A, this.hStateA, this.hNextA, x, y, pA, hA, dxA);
          this._synthesizeKernelFlat(this.modelB, this.alpha_layerB, x, y, k0B, k1B);
          this._computeProposal(this.modelB, k0B, k1B, this.hStateB, this.hNextB, x, y, pB, hB, dxB);
          for (let c = 0; c < C; c++) {
            nxt[c * plane + i] = cur[c * plane + i] + (aVal * dxA[c] + bVal * dxB[c]);
          }
        }
      }
    }

    // Post alive mask check
    if (useAlive) {
      this._aliveMask(nxt, this.post);
      for (let i = 0; i < plane; i++) {
        if (!(this.pre[i] && this.post[i])) {
          for (let c = 0; c < C; c++) nxt[c * plane + i] = 0;
          if (this.hNextA) for (let k = 0; k < HN; k++) this.hNextA[k * plane + i] = 0;
          if (this.hNextB) for (let k = 0; k < HN; k++) this.hNextB[k * plane + i] = 0;
        }
      }
    }

    // Double buffer swap
    this.buf = nxt;
    this.back = cur;
    const swapHA = this.hStateA; this.hStateA = this.hNextA; this.hNextA = swapHA;
    const swapHB = this.hStateB; this.hStateB = this.hNextB; this.hNextB = swapHB;
  }

  // --- RGBA Rendering ---

  readRGBA(out, showOverlay = false, overlayOpacity = 0.25) {
    const plane = this.plane;
    const cur = this.buf;
    const rOff = 0, gOff = plane, bOff = 2 * plane, aOff = 3 * plane;
    const Bf = this.B;

    for (let i = 0; i < plane; i++) {
      const p = i * 4;
      const alpha = Math.max(0, Math.min(1, cur[aOff + i]));
      
      // Standard RGBA clamp
      let r = Math.round(Math.max(0, Math.min(1, cur[rOff + i])) * 255);
      let g = Math.round(Math.max(0, Math.min(1, cur[gOff + i])) * 255);
      let b = Math.round(Math.max(0, Math.min(1, cur[bOff + i])) * 255);

      // Ink-on-white composition for visualization
      const bgR = 255, bgG = 255, bgB = 255;
      r = Math.round(r * alpha + bgR * (1 - alpha));
      g = Math.round(g * alpha + bgG * (1 - alpha));
      b = Math.round(b * alpha + bgB * (1 - alpha));

      // Control field overlay: Model A is Cool Blue (#1a8cff), Model B is Warm Orange (#ff5500)
      if (showOverlay) {
        const bv = Math.max(0, Math.min(1, Bf[i]));
        const cr = Math.round(26 + bv * (255 - 26));
        const cg = Math.round(140 + bv * (85 - 140));
        const cb = Math.round(255 + bv * (0 - 255));
        const k = overlayOpacity;
        r = Math.round(r * (1 - k) + cr * k);
        g = Math.round(g * (1 - k) + cg * k);
        b = Math.round(b * (1 - k) + cb * k);
      }

      out[p] = r;
      out[p + 1] = g;
      out[p + 2] = b;
      out[p + 3] = 255;
    }
  }
}

// ---------------------------------------------------------------------------
// UI & Interaction Controller
// ---------------------------------------------------------------------------

let engine = null;
let animTimer = null;
let isPaused = false;
let stepsPerTick = 1;
let brushMode = "modelB"; // "modelA", "modelB", "damage"
let brushRadius = 16;
let isMouseDown = false;
let hoveredX = Math.floor(SIM_SIZE / 2), hoveredY = Math.floor(SIM_SIZE / 2);

const canvas = document.getElementById("demo-canvas");
canvas.width = SIM_SIZE;
canvas.height = SIM_SIZE;
const ctx = canvas.getContext("2d");
const canvasImgData = ctx.createImageData(SIM_SIZE, SIM_SIZE);
const rgbaBuffer = new Uint8ClampedArray(SIM_PLANE * 4);

const modelASelect = document.getElementById("select-model-a");
const modelBSelect = document.getElementById("select-model-b");
const statusLabel = document.getElementById("demo-status");
const overlayToggle = document.getElementById("overlay-toggle");
const overlayOpacityInput = document.getElementById("overlay-opacity");
const uniformBlendSlider = document.getElementById("uniform-blend-slider");
const blendValueLabel = document.getElementById("blend-val");
const speedSlider = document.getElementById("speed-slider");
const speedValLabel = document.getElementById("speed-val");
const pauseBtn = document.getElementById("pause-btn");
const aliveToggle = document.getElementById("alive-mask-toggle");

// Inspector elements
const hudReadout = document.getElementById("hud-readout");
const kernelCanvasA = document.getElementById("kernel-canvas-a");
const kernelCanvasBlend = document.getElementById("kernel-canvas-blend");
const kernelCanvasB = document.getElementById("kernel-canvas-b");
const ctxKA = kernelCanvasA.getContext("2d");
const ctxKBlend = kernelCanvasBlend.getContext("2d");
const ctxKB = kernelCanvasB.getContext("2d");

const basisBarsA = [0, 1, 2, 3].map(i => document.getElementById(`basis-a-${i}`));
const basisBarsB = [0, 1, 2, 3].map(i => document.getElementById(`basis-b-${i}`));

function populateDropdowns() {
  modelASelect.innerHTML = "";
  modelBSelect.innerHTML = "";
  MODEL_REGISTRY.forEach((m, idx) => {
    const optA = document.createElement("option");
    optA.value = m.path;
    optA.innerText = m.name;
    modelASelect.appendChild(optA);

    const optB = document.createElement("option");
    optB.value = m.path;
    optB.innerText = m.name;
    modelBSelect.appendChild(optB);
  });
  // Defaults: Fire (A) vs Rocket (B)
  modelASelect.selectedIndex = 0;
  modelBSelect.selectedIndex = 1;
}

async function loadSelectedModels() {
  statusLabel.innerText = "Loading model weights…";
  try {
    const pathA = modelASelect.value;
    const pathB = modelBSelect.value;

    const [resA, resB] = await Promise.all([
      fetch(`${pathA}?t=${Date.now()}`),
      fetch(`${pathB}?t=${Date.now()}`)
    ]);
    if (!resA.ok) throw new Error(`Could not fetch ${pathA}`);
    if (!resB.ok) throw new Error(`Could not fetch ${pathB}`);

    const [wA, wB] = await Promise.all([resA.json(), resB.json()]);

    const prevB = engine ? new Float32Array(engine.B) : null;
    engine = new BlendedDynKernelCA(wA, wB, SIM_SIZE);
    
    if (prevB && prevB.length === engine.B.length) {
      engine.B.set(prevB);
    } else {
      engine.setRamp();
    }
    
    engine.aliveMaskEnabled = aliveToggle.checked;
    engine.resetSeeds();
    statusLabel.innerText = `${wA.name || "Model A"} + ${wB.name || "Model B"} active`;
    render();
    updateKernelInspector();
  } catch (err) {
    statusLabel.innerText = `Error: ${err.message}`;
    console.error("Failed to load models:", err);
  }
}

// ---------------------------------------------------------------------------
// Inspector Kernel Rendering
// ---------------------------------------------------------------------------

function renderKernelToCanvas(ctxK, K, ks = 5) {
  const w = ctxK.canvas.width, h = ctxK.canvas.height;
  ctxK.clearRect(0, 0, w, h);
  const cellW = w / ks, cellH = h / ks;

  // Find max magnitude for diverging colormap
  let maxAbs = 0.001;
  for (let r = 0; r < ks; r++) {
    for (let c = 0; c < ks; c++) {
      const v = Math.abs(K[r][c]);
      if (v > maxAbs) maxAbs = v;
    }
  }

  for (let r = 0; r < ks; r++) {
    for (let c = 0; c < ks; c++) {
      const val = K[r][c];
      const norm = Math.max(-1, Math.min(1, val / maxAbs));
      
      // Diverging colormap: negative = deep blue, zero = dark gray #222, positive = bright orange/red
      let red, green, blue;
      if (norm < 0) {
        const t = -norm;
        red = Math.round(34 * (1 - t) + 40 * t);
        green = Math.round(34 * (1 - t) + 120 * t);
        blue = Math.round(34 * (1 - t) + 255 * t);
      } else {
        const t = norm;
        red = Math.round(34 * (1 - t) + 255 * t);
        green = Math.round(34 * (1 - t) + 90 * t);
        blue = Math.round(34 * (1 - t) + 20 * t);
      }

      ctxK.fillStyle = `rgb(${red},${green},${blue})`;
      ctxK.fillRect(c * cellW, r * cellH, cellW, cellH);

      // Grid borders
      ctxK.strokeStyle = "#111";
      ctxK.lineWidth = 1;
      ctxK.strokeRect(c * cellW, r * cellH, cellW, cellH);
    }
  }

  // Highlight center pixel with a subtle white dot
  const centerPad = Math.floor(ks / 2);
  const cx = centerPad * cellW + cellW / 2;
  const cy = centerPad * cellH + cellH / 2;
  ctxK.fillStyle = "#ffffff";
  ctxK.beginPath();
  ctxK.arc(cx, cy, 3, 0, Math.PI * 2);
  ctxK.fill();
}

function updateKernelInspector() {
  if (!engine) return;
  const x = Math.max(0, Math.min(SIM_SIZE - 1, hoveredX));
  const y = Math.max(0, Math.min(SIM_SIZE - 1, hoveredY));
  const i = y * SIM_SIZE + x;
  const bVal = engine.B[i];
  const aVal = 1.0 - bVal;

  const resA = engine.getEffectiveKernel(engine.modelA, engine.alpha_layerA, x, y);
  const resB = engine.getEffectiveKernel(engine.modelB, engine.alpha_layerB, x, y);

  const KA0 = resA.kernels[0];
  const KB0 = resB.kernels[0];
  const ks = KA0.length;

  // Blended Kernel: (1 - B) * KA + B * KB
  const KBlend0 = [];
  for (let r = 0; r < ks; r++) {
    const row = [];
    for (let c = 0; c < ks; c++) {
      const vA = (r < KA0.length && c < KA0[0].length) ? KA0[r][c] : 0;
      const vB = (r < KB0.length && c < KB0[0].length) ? KB0[r][c] : 0;
      row.push(aVal * vA + bVal * vB);
    }
    KBlend0.push(row);
  }

  renderKernelToCanvas(ctxKA, KA0, ks);
  renderKernelToCanvas(ctxKBlend, KBlend0, ks);
  renderKernelToCanvas(ctxKB, KB0, ks);

  // Update basis activation bars
  const alphasA = resA.alphas[0] || [1, 0, 0, 0];
  const alphasB = resB.alphas[0] || [1, 0, 0, 0];
  for (let m = 0; m < 4; m++) {
    if (basisBarsA[m]) basisBarsA[m].style.width = `${Math.round((alphasA[m] || 0) * 100)}%`;
    if (basisBarsB[m]) basisBarsB[m].style.width = `${Math.round((alphasB[m] || 0) * 100)}%`;
  }

  const nameA = engine.modelA.name.split(" ")[0];
  const nameB = engine.modelB.name.split(" ")[0];
  hudReadout.innerText = `Cell (${x}, ${y}) · Blend: ${(aVal * 100).toFixed(0)}% ${nameA} · ${(bVal * 100).toFixed(0)}% ${nameB}`;
}

// ---------------------------------------------------------------------------
// Render & Animation Loop
// ---------------------------------------------------------------------------

function render() {
  if (!engine) return;
  const showOverlay = overlayToggle.checked;
  const opacity = parseFloat(overlayOpacityInput.value);
  engine.readRGBA(canvasImgData.data, showOverlay, opacity);
  ctx.putImageData(canvasImgData, 0, 0);
}

function loop() {
  if (engine && !isPaused) {
    for (let s = 0; s < stepsPerTick; s++) {
      engine.step();
    }
    render();
  }
  animTimer = requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// Event Listeners & Interaction
// ---------------------------------------------------------------------------

function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = SIM_SIZE / rect.width;
  const scaleY = SIM_SIZE / rect.height;
  const x = Math.floor((e.clientX - rect.left) * scaleX);
  const y = Math.floor((e.clientY - rect.top) * scaleY);
  return {
    x: Math.max(0, Math.min(SIM_SIZE - 1, x)),
    y: Math.max(0, Math.min(SIM_SIZE - 1, y))
  };
}

function applyBrushAt(e) {
  if (!engine) return;
  const { x, y } = getCanvasCoords(e);
  hoveredX = x; hoveredY = y;

  if (brushMode === "damage") {
    engine.damage(x, y, brushRadius);
  } else if (brushMode === "modelB") {
    engine.paintB(x, y, brushRadius, 1.0);
  } else if (brushMode === "modelA") {
    engine.paintB(x, y, brushRadius, 0.0);
  }
  render();
  updateKernelInspector();
}

canvas.addEventListener("mousedown", (e) => {
  isMouseDown = true;
  if (e.button === 2) {
    // Right click paints Model A
    const old = brushMode;
    brushMode = "modelA";
    applyBrushAt(e);
    brushMode = old;
  } else {
    applyBrushAt(e);
  }
});

canvas.addEventListener("mousemove", (e) => {
  const { x, y } = getCanvasCoords(e);
  hoveredX = x; hoveredY = y;
  if (isMouseDown) applyBrushAt(e);
  else updateKernelInspector();
});

window.addEventListener("mouseup", () => { isMouseDown = false; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

// Touch support
canvas.addEventListener("touchstart", (e) => {
  if (e.touches.length > 0) {
    isMouseDown = true;
    applyBrushAt(e.touches[0]);
  }
  e.preventDefault();
}, { passive: false });

canvas.addEventListener("touchmove", (e) => {
  if (e.touches.length > 0) {
    applyBrushAt(e.touches[0]);
  }
  e.preventDefault();
}, { passive: false });

canvas.addEventListener("touchend", () => { isMouseDown = false; });

// Controls
modelASelect.onchange = () => loadSelectedModels();
modelBSelect.onchange = () => loadSelectedModels();

document.getElementById("btn-ramp").onclick = () => {
  if (engine) { engine.setRamp(); render(); updateKernelInspector(); }
};
document.getElementById("btn-split").onclick = () => {
  if (engine) { engine.setSplit(); render(); updateKernelInspector(); }
};
document.getElementById("btn-circle").onclick = () => {
  if (engine) { engine.setCircle(); render(); updateKernelInspector(); }
};
document.getElementById("btn-clear-b").onclick = () => {
  if (engine) { engine.setUniform(0.0); render(); updateKernelInspector(); }
};

uniformBlendSlider.oninput = (e) => {
  const v = parseFloat(e.target.value);
  blendValueLabel.innerText = `${Math.round(v * 100)}%`;
  if (engine) {
    engine.setUniform(v);
    render();
    updateKernelInspector();
  }
};

document.getElementById("btn-noise").onclick = () => {
  if (engine) { engine.resetNoise(); render(); updateKernelInspector(); }
};
document.getElementById("btn-seeds").onclick = () => {
  if (engine) { engine.resetSeeds(); render(); updateKernelInspector(); }
};
document.getElementById("btn-clear-state").onclick = () => {
  if (engine) { engine.clear(); render(); updateKernelInspector(); }
};

pauseBtn.onclick = () => {
  isPaused = !isPaused;
  pauseBtn.innerText = isPaused ? "▶ Resume" : "⏸ Pause";
  pauseBtn.style.background = isPaused ? "#2e7d4f" : "#444";
};

document.getElementById("btn-step").onclick = () => {
  if (engine) {
    engine.step();
    render();
    updateKernelInspector();
  }
};

speedSlider.oninput = (e) => {
  stepsPerTick = parseInt(e.target.value, 10);
  speedValLabel.innerText = stepsPerTick;
};

aliveToggle.onchange = (e) => {
  if (engine) engine.aliveMaskEnabled = e.target.checked;
};

overlayToggle.onchange = () => render();
overlayOpacityInput.oninput = () => render();

// Brush tool selectors
const brushButtons = {
  modelB: document.getElementById("brush-model-b"),
  modelA: document.getElementById("brush-model-a"),
  damage: document.getElementById("brush-damage")
};

function selectBrush(mode) {
  brushMode = mode;
  Object.keys(brushButtons).forEach(k => {
    if (brushButtons[k]) {
      brushButtons[k].classList.toggle("active-brush", k === mode);
    }
  });
}

if (brushButtons.modelB) brushButtons.modelB.onclick = () => selectBrush("modelB");
if (brushButtons.modelA) brushButtons.modelA.onclick = () => selectBrush("modelA");
if (brushButtons.damage) brushButtons.damage.onclick = () => selectBrush("damage");

const brushRadiusSlider = document.getElementById("brush-radius");
const brushRadiusLabel = document.getElementById("brush-radius-val");
if (brushRadiusSlider) {
  brushRadiusSlider.oninput = (e) => {
    brushRadius = parseInt(e.target.value, 10);
    brushRadiusLabel.innerText = `${brushRadius}px`;
  };
}

const btnToggleView = document.getElementById("btn-toggle-view");
let isFitView = false;
if (btnToggleView) {
  btnToggleView.onclick = () => {
    isFitView = !isFitView;
    canvas.classList.toggle("fit-mode", isFitView);
    btnToggleView.innerText = isFitView ? "🔍 View: Fit Screen" : "🔍 View: 100% (5px/cell)";
    updateKernelInspector();
  };
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

async function init() {
  populateDropdowns();
  selectBrush("modelB");
  await loadSelectedModels();
  loop();
}

init().catch(err => console.error("Init failed:", err));

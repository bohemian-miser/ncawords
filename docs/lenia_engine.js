// Lenia web engine: steps a TRAINED Lenia physics (exported weights.json)
// live in the browser. Mirrors nca/train_lenia.py Lenia.step exactly:
// toroidal conv -> growth bump -> gains/coupling -> leaky clamp.
// CPU implementation; kernels are small (15x15) and grids 64x64, so even
// the 108-kernel 'full' variant runs at interactive rates.

export class LeniaCA {
  constructor(w, size = 64) {
    this.w = w;
    this.C = w.C; this.K = w.K; this.ks = w.ks; this.dt = w.dt;
    this.leak = w.leak ?? 0.05;
    this.width = size; this.height = size;
    this.channel_n = this.C;
    const plane = size * size;
    this.state = new Float32Array(this.C * plane);
    this._tmp = new Float32Array(this.C * plane);
    this._u = new Float32Array(plane);
    if (w.kernels) this._kern = w.kernels.map(k => Float32Array.from(k.flat()));
    if (w.basis) this._basis = w.basis.map(k => Float32Array.from(k.flat()));
    this.reset(true);
  }

  reset(noise = true) {
    for (let i = 0; i < this.state.length; i++)
      this.state[i] = noise ? Math.random() * 0.6 : 0;
    if (!noise) { // a soft blob in the middle as the non-noise seed
      const s = this.width, c = s >> 1;
      for (let y = c - 6; y < c + 6; y++)
        for (let x = c - 6; x < c + 6; x++)
          for (let ch = 0; ch < this.C; ch++)
            this.state[ch * s * s + y * s + x] = Math.random();
    }
  }

  // toroidal correlation of channel plane `src` with kernel `k` into out
  _conv(srcOff, k, out) {
    const s = this.width, ks = this.ks, r = ks >> 1, st = this.state;
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        let acc = 0;
        for (let ky = 0; ky < ks; ky++) {
          const yy = (y + ky - r + s) % s;
          const rowK = ky * ks, rowS = srcOff + yy * s;
          for (let kx = 0; kx < ks; kx++) {
            const xx = (x + kx - r + s) % s;
            acc += k[rowK + kx] * st[rowS + xx];
          }
        }
        out[y * s + x] = acc;
      }
    }
  }

  _growth(u, mu, sg) {
    const d = (u - mu) / sg;
    return 2 * Math.exp(-d * d / 2) - 1;
  }

  step() {
    const w = this.w, s = this.width, plane = s * s, v = w.variant;
    const dx = this._tmp; dx.fill(0);
    const u = this._u;

    if (v === "static1" || v === "multik" || v === "aniso" || v === "wave") {
      const n = this._kern.length;
      for (let i = 0; i < n; i++) {
        this._conv(0, this._kern[i], u);
        for (let p = 0; p < plane; p++)
          dx[p] += w.h[i] * this._growth(u[p], w.mu[i], w.sg[i]);
      }
    } else if (v === "dyn1" || v === "dynwave") {
      const B = this._basis.length;
      const ub = this._ub || (this._ub = Array.from({length: B},
        () => new Float32Array(plane)));
      for (let b = 0; b < B; b++) this._conv(0, this._basis[b], ub[b]);
      const {w0, b0, w2, b2} = w.mix;
      const H = w0.length;
      const hid = new Float32Array(H);
      for (let p = 0; p < plane; p++) {
        for (let hI = 0; hI < H; hI++) {
          let a = b0[hI] + w0[hI][0] * this.state[p];
          for (let b = 0; b < B; b++) a += w0[hI][1 + b] * ub[b][p];
          hid[hI] = a > 0 ? a : 0;
        }
        let uu = 0;
        for (let b = 0; b < B; b++) {
          let c = b2[b];
          for (let hI = 0; hI < H; hI++) c += w2[b][hI] * hid[hI];
          uu += Math.tanh(c) * ub[b][p];
        }
        dx[p] = w.h[0] * this._growth(uu, w.mu[0], w.sg[0]);
      }
    } else if (v === "sharedk") {
      const g = this._g || (this._g = new Float32Array(this.C * plane));
      for (let c = 0; c < this.C; c++) {
        this._conv(c * plane, this._kern[0], u);
        for (let p = 0; p < plane; p++)
          g[c * plane + p] = this._growth(u[p], w.mu[0], w.sg[0]);
      }
      for (let t = 0; t < this.C; t++)
        for (let src = 0; src < this.C; src++) {
          const hij = w.H[src][t];
          for (let p = 0; p < plane; p++)
            dx[t * plane + p] += hij * g[src * plane + p];
        }
    } else if (v === "full") {
      // kernel idx = (src*C + tgt)*K + k ; mu/sg/h share the same flat order
      for (let src = 0; src < this.C; src++)
        for (let t = 0; t < this.C; t++)
          for (let k = 0; k < this.K; k++) {
            const idx = (src * this.C + t) * this.K + k;
            this._conv(src * plane, this._kern[idx], u);
            const mu = w.mu[idx], sg = w.sg[idx], h = w.h[idx];
            for (let p = 0; p < plane; p++)
              dx[t * plane + p] += h * this._growth(u[p], mu, sg);
          }
    }

    const st = this.state, dt = this.dt, leak = this.leak;
    for (let i = 0; i < st.length; i++) {
      const xn = st[i] + dt * dx[i];
      const c = xn < 0 ? 0 : xn > 1 ? 1 : xn;
      st[i] = c + leak * (xn - c);
    }
  }

  damage(cx, cy, rad = 8) {
    const s = this.width, plane = s * s;
    for (let y = 0; y < s; y++)
      for (let x = 0; x < s; x++) {
        const dx2 = (x - cx) ** 2 + (y - cy) ** 2;
        if (dx2 < rad * rad)
          for (let c = 0; c < this.C; c++) this.state[c * plane + y * s + x] = 0;
      }
  }

  readChannel(c) {
    const plane = this.width * this.height;
    return this.state.slice(c * plane, (c + 1) * plane);
  }

  // channel 0 as ink-on-white; extra channels (if any) tint via 1 and 2
  readRGBA(out) {
    const s = this.width, plane = s * s;
    if (!out) out = new Uint8ClampedArray(plane * 4);
    for (let p = 0; p < plane; p++) {
      const a = Math.min(1, Math.max(0, this.state[p]));
      const g = this.C > 1 ? Math.min(1, Math.max(0, this.state[plane + p])) : 0;
      const b = this.C > 2 ? Math.min(1, Math.max(0, this.state[2 * plane + p])) : 0;
      out[p * 4 + 0] = (1 - a) * 255;
      out[p * 4 + 1] = (1 - Math.max(a, g * 0.6)) * 255;
      out[p * 4 + 2] = (1 - Math.max(a, b * 0.6)) * 255;
      out[p * 4 + 3] = 255;
    }
    return out;
  }
}

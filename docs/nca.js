// nca.js — Neural Cellular Automata inference engine (ES module, dependency-free).
//
// Implements the update rule from nca/model.py exactly (see site/API.md):
//   1. preAlive  = maxpool3x3(alpha) > 0.1        (computed on the OLD state)
//   2. p[3C]     = [state, sobelX(state), sobelY(state)]   (blocked layout)
//   3. h = relu(fc0_w @ p + fc0_b);  dx = fc1_w @ h
//   4. per-cell gate: rand() <= fireRate  ->  new = old + dx, else new = old
//   5. postAlive = maxpool3x3(newAlpha) > 0.1; if !(pre && post) zero the cell
// Out-of-bounds neighbors read as 0. State is Float32Array in [C, H, W] order.

const SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]; // /8, [ky][kx]

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function flattenWeights(weights) {
  const C = weights.channel_n;
  const HN = weights.hidden_n;
  const C3 = 3 * C;
  // Internally the perception vector is blocked: [id(0..C), sx(0..C), sy(0..C)].
  // Per API.md, trust the file's `layout` field: "blocked" fc0_w columns are
  // already [state | sobel_x | sobel_y] — use as-is. "interleaved" (or absent)
  // columns follow model.py's grouped-conv output order
  // [id0, sx0, sy0, id1, ...] — reorder to blocked at load time.
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

// Grid dimensions: word models (kind: "word") use grid_w x grid_h (grid is
// null there); single-glyph models use grid x grid.
function gridDims(weights) {
  if (weights.kind === "word" || weights.grid == null) {
    return { W: weights.grid_w, H: weights.grid_h };
  }
  return { W: weights.grid, H: weights.grid };
}

// dx for a cell whose whole 3x3 neighborhood is zero (perception vector = 0):
// dx = fc1 @ relu(fc0_b). Precomputed with the exact same arithmetic as the
// main loop so the fast path is bit-identical.
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

function rgbaFromState(state, C, plane, out) {
  // v = clamp01(1 - a + rgb) * 255 with a clamped to [0,1] (model.py to_rgb).
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

// ---------------------------------------------------------------------------
// CPUCA
// ---------------------------------------------------------------------------

export class CPUCA {
  constructor(weights) {
    this.grid = weights.grid;
    this.channel_n = weights.channel_n;
    this.hidden_n = weights.hidden_n;
    this.fire_rate = weights.fire_rate;
    const dims = gridDims(weights);
    this._W = dims.W;
    this._H = dims.H;
    this._seeds = weights.seeds || null;
    this._codeCh0 = weights.code_ch0;
    this._codeBits = weights.code_bits;
    const f = flattenWeights(weights);
    this._C = f.C;
    this._HN = f.HN;
    this._C3 = f.C3;
    this._w0 = f.w0;
    this._b0 = f.b0;
    this._w1 = f.w1;
    this._dxZero = computeDxZero(f);

    const plane = this._W * this._H;
    this._plane = plane;
    this._buf = new Float32Array(f.C * plane);
    this._back = new Float32Array(f.C * plane);
    this._pre = new Uint8Array(plane);
    this._post = new Uint8Array(plane);
    this._nz = new Uint8Array(plane);   // per-cell "any channel nonzero"
    this._nbnz = new Uint8Array(plane); // 3x3 dilation of _nz
    this._p = new Float64Array(f.C3);
    this._h = new Float64Array(f.HN);
    this.reset();
  }

  get state() { return this._buf; }
  get width() { return this._W; }
  get height() { return this._H; }

  clear() { this._buf.fill(0); }

  reset() {
    this.clear();
    if (this._seeds && this._seeds.length) {
      for (const s of this._seeds) this._placeSeed(s.x, s.y, s.code);
    } else {
      this.seed(this._W >> 1, this._H >> 1);
    }
  }

  seed(x, y) {
    const plane = this._plane, i = y * this._W + x;
    for (let c = 3; c < this._C; c++) this._buf[c * plane + i] = 1.0;
  }

  // Word-model seed: channels 3..C-1 = 1, then channels
  // code_ch0..code_ch0+code_bits-1 overwritten with the code bits (0/1).
  _placeSeed(x, y, code) {
    this.seed(x, y);
    if (!code) return;
    const plane = this._plane, i = y * this._W + x;
    const c0 = this._codeCh0, nb = this._codeBits != null ? this._codeBits : code.length;
    for (let b = 0; b < nb; b++) this._buf[(c0 + b) * plane + i] = code[b] ? 1.0 : 0.0;
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

  // maxpool3x3(alpha) > 0.1 -> out (0/1). Zero-padding is equivalent to
  // PyTorch's -inf padding here because the threshold 0.1 > 0.
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
    const C = this._C, HN = this._HN, C3 = this._C3;
    const cur = this._buf, nxt = this._back;
    const pre = this._pre, post = this._post;
    const p = this._p, hbuf = this._h;
    const w0 = this._w0, b0 = this._b0, w1 = this._w1, dxZero = this._dxZero;

    this._aliveMask(cur, pre);
    const nbnz = this._nonzeroDilated(cur);

    for (let y = 0; y < H; y++) {
      const yu = y > 0, yd = y < H - 1;
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const fired = rand() <= fireRate;
        if (!fired) {
          for (let c = 0; c < C; c++) nxt[c * plane + i] = cur[c * plane + i];
          continue;
        }
        if (!nbnz[i]) {
          // Entire 3x3 neighborhood is zero: perception is exactly 0.
          for (let c = 0; c < C; c++) nxt[c * plane + i] = cur[c * plane + i] + dxZero[c];
          continue;
        }
        const xl = x > 0, xr = x < W - 1;
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
        for (let k = 0; k < HN; k++) {
          let s = b0[k];
          const off = k * C3;
          for (let j = 0; j < C3; j++) s += w0[off + j] * p[j];
          hbuf[k] = s > 0 ? s : 0;
        }
        for (let c = 0; c < C; c++) {
          let s = 0;
          const off = c * HN;
          for (let k = 0; k < HN; k++) s += w1[off + k] * hbuf[k];
          const b = c * plane + i;
          nxt[b] = cur[b] + s;
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

  readRGBA(out) {
    return rgbaFromState(this._buf, this._C, this._plane, out);
  }
}

// ---------------------------------------------------------------------------
// GLCA — WebGL2 fragment-shader implementation.
//
// Channels are packed 4-per-texture into T = ceil(C/4) RGBA32F textures
// (texture t, lane l  <->  channel 4t+l), rendered with MRT. Each step is two
// fullscreen passes: (1) perceive + MLP + fire gate -> tmp set;
// (2) alive-mask using old alpha (pre) and tmp alpha (post) -> next set.
// The fire mask is a deterministic integer hash of (cellX, cellY, frame).
// ---------------------------------------------------------------------------

const VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

function buildUpdateFS(T, HN) {
  let s = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform ivec2 uSize;
uniform float uFireRate;
uniform uint uFrame;
uniform sampler2D uW0; // (3T+1) x HN; texel(3T,k).x = bias
uniform sampler2D uW1; // HN x T; texel(k,t) = fc1 rows 4t..4t+3, column k
`;
  for (let t = 0; t < T; t++) {
    s += `uniform sampler2D uS${t};\n`;
    s += `layout(location=${t}) out vec4 o${t};\n`;
    s += `vec4 f${t}(ivec2 q){ return (q.x<0||q.y<0||q.x>=uSize.x||q.y>=uSize.y) ? vec4(0.0) : texelFetch(uS${t}, q, 0); }\n`;
  }
  s += `
float hashRand(ivec2 q, uint f) {
  uint h = uint(q.x) * 374761393u + uint(q.y) * 668265263u + f * 2246822519u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return float(h) * 2.3283064365386963e-10; // [0,1)
}
void main() {
  ivec2 xy = ivec2(gl_FragCoord.xy);
`;
  for (let t = 0; t < T; t++) {
    s += `  vec4 nUL${t} = f${t}(xy + ivec2(-1,-1));
  vec4 nU${t}  = f${t}(xy + ivec2( 0,-1));
  vec4 nUR${t} = f${t}(xy + ivec2( 1,-1));
  vec4 nL${t}  = f${t}(xy + ivec2(-1, 0));
  vec4 nC${t}  = texelFetch(uS${t}, xy, 0);
  vec4 nR${t}  = f${t}(xy + ivec2( 1, 0));
  vec4 nDL${t} = f${t}(xy + ivec2(-1, 1));
  vec4 nD${t}  = f${t}(xy + ivec2( 0, 1));
  vec4 nDR${t} = f${t}(xy + ivec2( 1, 1));
  vec4 sx${t} = (nUR${t} + 2.0*nR${t} + nDR${t} - nUL${t} - 2.0*nL${t} - nDL${t}) * 0.125;
  vec4 sy${t} = (nDL${t} + 2.0*nD${t} + nDR${t} - nUL${t} - 2.0*nU${t} - nUR${t}) * 0.125;
`;
  }
  s += `  float hb[${HN}];
  for (int k = 0; k < ${HN}; k++) {
    float acc = texelFetch(uW0, ivec2(${3 * T}, k), 0).x;
`;
  for (let t = 0; t < T; t++) {
    s += `    acc += dot(texelFetch(uW0, ivec2(${t}, k), 0), nC${t});
    acc += dot(texelFetch(uW0, ivec2(${T + t}, k), 0), sx${t});
    acc += dot(texelFetch(uW0, ivec2(${2 * T + t}, k), 0), sy${t});
`;
  }
  s += `    hb[k] = max(acc, 0.0);
  }
`;
  for (let t = 0; t < T; t++) s += `  vec4 d${t} = vec4(0.0);\n`;
  s += `  for (int k = 0; k < ${HN}; k++) {
    float hv = hb[k];
`;
  for (let t = 0; t < T; t++) s += `    d${t} += hv * texelFetch(uW1, ivec2(k, ${t}), 0);\n`;
  s += `  }
  float m = (hashRand(xy, uFrame) <= uFireRate) ? 1.0 : 0.0;
`;
  for (let t = 0; t < T; t++) s += `  o${t} = nC${t} + m * d${t};\n`;
  s += `}\n`;
  return s;
}

function buildMaskFS(T) {
  let s = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform ivec2 uSize;
uniform sampler2D uOldA; // old texture 0 (alpha in .w)
`;
  for (let t = 0; t < T; t++) {
    s += `uniform sampler2D uN${t};\n`;
    s += `layout(location=${t}) out vec4 o${t};\n`;
  }
  s += `
float aOld(ivec2 q){ return (q.x<0||q.y<0||q.x>=uSize.x||q.y>=uSize.y) ? 0.0 : texelFetch(uOldA, q, 0).w; }
float aNew(ivec2 q){ return (q.x<0||q.y<0||q.x>=uSize.x||q.y>=uSize.y) ? 0.0 : texelFetch(uN0, q, 0).w; }
void main() {
  ivec2 xy = ivec2(gl_FragCoord.xy);
  float pre = 0.0, post = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      ivec2 q = xy + ivec2(i, j);
      pre = max(pre, aOld(q));
      post = max(post, aNew(q));
    }
  }
  bool alive = (pre > 0.1) && (post > 0.1);
`;
  for (let t = 0; t < T; t++) {
    s += `  o${t} = alive ? texelFetch(uN${t}, xy, 0) : vec4(0.0);\n`;
  }
  s += `}\n`;
  return s;
}

function compileProgram(gl, vsSrc, fsSrc) {
  const compile = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("shader compile failed: " + gl.getShaderInfoLog(sh));
    }
    return sh;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("program link failed: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

export class GLCA {
  constructor(gl, weights) {
    if (!gl || typeof gl.texStorage2D !== "function") {
      throw new Error("GLCA requires a WebGL2 context");
    }
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new Error("EXT_color_buffer_float not supported");
    }
    this.gl = gl;
    this.grid = weights.grid;
    this.channel_n = weights.channel_n;
    this.hidden_n = weights.hidden_n;
    this.fire_rate = weights.fire_rate;
    const dims = gridDims(weights);
    this._W = dims.W;
    this._H = dims.H;
    this._seeds = weights.seeds || null;
    this._codeCh0 = weights.code_ch0;
    this._codeBits = weights.code_bits;
    const f = flattenWeights(weights);
    this._C = f.C;
    this._HN = f.HN;
    this._T = Math.ceil(f.C / 4);
    this._plane = this._W * this._H;
    this._frame = 0;

    const W = this._W, H = this._H, T = this._T;

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);
    gl.disable(gl.SCISSOR_TEST);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);

    this._vao = gl.createVertexArray();

    const makeStateTex = () => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, W, H);
      return tex;
    };
    const makeSet = () => {
      const texs = [];
      for (let t = 0; t < T; t++) texs.push(makeStateTex());
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      const bufs = [];
      for (let t = 0; t < T; t++) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + t, gl.TEXTURE_2D, texs[t], 0);
        bufs.push(gl.COLOR_ATTACHMENT0 + t);
      }
      gl.drawBuffers(bufs);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error("float framebuffer incomplete");
      }
      return { texs, fbo };
    };
    this._sets = [makeSet(), makeSet()]; // ping-pong state
    this._tmp = makeSet();               // pass-1 target
    this._cur = 0;

    // --- Weight textures ---
    const HN = f.HN, C = f.C;
    const WD = 3 * T + 1;
    const w0data = new Float32Array(WD * HN * 4);
    for (let k = 0; k < HN; k++) {
      for (let sec = 0; sec < 3; sec++) {
        for (let t = 0; t < T; t++) {
          for (let lane = 0; lane < 4; lane++) {
            const c = 4 * t + lane;
            if (c < C) w0data[(k * WD + sec * T + t) * 4 + lane] = f.w0[k * f.C3 + sec * C + c];
          }
        }
      }
      w0data[(k * WD + 3 * T) * 4] = f.b0[k];
    }
    const w1data = new Float32Array(HN * T * 4);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < HN; k++) {
        for (let lane = 0; lane < 4; lane++) {
          const c = 4 * t + lane;
          if (c < C) w1data[(t * HN + k) * 4 + lane] = f.w1[c * HN + k];
        }
      }
    }
    const makeWeightTex = (w, h, data) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, w, h);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, data);
      return tex;
    };
    this._w0Tex = makeWeightTex(WD, HN, w0data);
    this._w1Tex = makeWeightTex(HN, T, w1data);

    // --- Programs ---
    this._progU = compileProgram(gl, VS, buildUpdateFS(T, HN));
    this._progM = compileProgram(gl, VS, buildMaskFS(T));
    gl.useProgram(this._progU);
    for (let t = 0; t < T; t++) gl.uniform1i(gl.getUniformLocation(this._progU, `uS${t}`), t);
    gl.uniform1i(gl.getUniformLocation(this._progU, "uW0"), T);
    gl.uniform1i(gl.getUniformLocation(this._progU, "uW1"), T + 1);
    gl.uniform2i(gl.getUniformLocation(this._progU, "uSize"), W, H);
    this._uFireRate = gl.getUniformLocation(this._progU, "uFireRate");
    this._uFrame = gl.getUniformLocation(this._progU, "uFrame");
    gl.useProgram(this._progM);
    gl.uniform1i(gl.getUniformLocation(this._progM, "uOldA"), 0);
    for (let t = 0; t < T; t++) gl.uniform1i(gl.getUniformLocation(this._progM, `uN${t}`), 1 + t);
    gl.uniform2i(gl.getUniformLocation(this._progM, "uSize"), W, H);

    this._readBuf = new Float32Array(this._plane * 4);
    this._texelBuf = new Float32Array(4);
    this.reset();
  }

  get width() { return this._W; }
  get height() { return this._H; }

  _bindFbo(set) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, set.fbo);
    gl.viewport(0, 0, this._W, this._H);
  }

  clear() {
    const gl = this.gl;
    this._bindFbo(this._sets[this._cur]);
    gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this._frame = 0;
  }

  reset() {
    this.clear();
    if (this._seeds && this._seeds.length) {
      for (const s of this._seeds) this._placeSeed(s.x, s.y, s.code);
    } else {
      this.seed(this._W >> 1, this._H >> 1);
    }
  }

  // Write one cell: channels 3..C-1 = 1, optional code bits overwrite
  // channels code_ch0.., channels 0..2 (RGB, in texture 0) are preserved.
  _writeSeedCell(x, y, code) {
    const gl = this.gl, texs = this._sets[this._cur].texs;
    const chan = new Float32Array(this._C);
    for (let c = 3; c < this._C; c++) chan[c] = 1;
    if (code) {
      const c0 = this._codeCh0, nb = this._codeBits != null ? this._codeBits : code.length;
      for (let b = 0; b < nb; b++) chan[c0 + b] = code[b] ? 1 : 0;
    }
    // Preserve RGB (channels 0..2): read the current texel of texture 0.
    const v = this._texelBuf;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._sets[this._cur].fbo);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.FLOAT, v);
    chan[0] = v[0]; chan[1] = v[1]; chan[2] = v[2];
    for (let t = 0; t < this._T; t++) {
      for (let lane = 0; lane < 4; lane++) {
        const c = 4 * t + lane;
        v[lane] = c < this._C ? chan[c] : 0;
      }
      gl.bindTexture(gl.TEXTURE_2D, texs[t]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, 1, 1, gl.RGBA, gl.FLOAT, v);
    }
  }

  seed(x, y) {
    this._writeSeedCell(x, y, null);
  }

  _placeSeed(x, y, code) {
    this._writeSeedCell(x, y, code);
  }

  damage(x, y, r) {
    const gl = this.gl, W = this._W, H = this._H, r2 = r * r;
    this._bindFbo(this._sets[this._cur]);
    gl.enable(gl.SCISSOR_TEST);
    gl.clearColor(0, 0, 0, 0);
    const y0 = Math.max(0, Math.ceil(y - r)), y1 = Math.min(H - 1, Math.floor(y + r));
    for (let yy = y0; yy <= y1; yy++) {
      const d = Math.floor(Math.sqrt(r2 - (yy - y) * (yy - y)));
      const x0 = Math.max(0, x - d), x1 = Math.min(W - 1, x + d);
      if (x1 < x0) continue;
      gl.scissor(x0, yy, x1 - x0 + 1, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.disable(gl.SCISSOR_TEST);
  }

  step(fireRate = null) {
    if (fireRate === null || fireRate === undefined) fireRate = this.fire_rate;
    const gl = this.gl, T = this._T;
    const cur = this._sets[this._cur];
    const nxt = this._sets[1 - this._cur];
    gl.disable(gl.SCISSOR_TEST);
    gl.bindVertexArray(this._vao);

    // Pass 1: update -> tmp
    gl.useProgram(this._progU);
    for (let t = 0; t < T; t++) {
      gl.activeTexture(gl.TEXTURE0 + t);
      gl.bindTexture(gl.TEXTURE_2D, cur.texs[t]);
    }
    gl.activeTexture(gl.TEXTURE0 + T);
    gl.bindTexture(gl.TEXTURE_2D, this._w0Tex);
    gl.activeTexture(gl.TEXTURE0 + T + 1);
    gl.bindTexture(gl.TEXTURE_2D, this._w1Tex);
    gl.uniform1f(this._uFireRate, fireRate);
    gl.uniform1ui(this._uFrame, this._frame >>> 0);
    this._bindFbo(this._tmp);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass 2: alive mask (pre from cur, post from tmp) -> nxt
    gl.useProgram(this._progM);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cur.texs[0]);
    for (let t = 0; t < T; t++) {
      gl.activeTexture(gl.TEXTURE0 + 1 + t);
      gl.bindTexture(gl.TEXTURE_2D, this._tmp.texs[t]);
    }
    this._bindFbo(nxt);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    this._cur = 1 - this._cur;
    this._frame = (this._frame + 1) >>> 0;
  }

  get state() {
    const gl = this.gl, plane = this._plane, C = this._C;
    const out = new Float32Array(C * plane);
    const buf = this._readBuf;
    const set = this._sets[this._cur];
    gl.bindFramebuffer(gl.FRAMEBUFFER, set.fbo);
    for (let t = 0; t < this._T; t++) {
      gl.readBuffer(gl.COLOR_ATTACHMENT0 + t);
      gl.readPixels(0, 0, this._W, this._H, gl.RGBA, gl.FLOAT, buf);
      for (let lane = 0; lane < 4; lane++) {
        const c = 4 * t + lane;
        if (c >= C) break;
        const off = c * plane;
        for (let i = 0; i < plane; i++) out[off + i] = buf[i * 4 + lane];
      }
    }
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    return out;
  }

  readRGBA(out) {
    const gl = this.gl, plane = this._plane;
    const buf = this._readBuf;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._sets[this._cur].fbo);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, this._W, this._H, gl.RGBA, gl.FLOAT, buf);
    if (!out) out = new Uint8ClampedArray(plane * 4);
    for (let i = 0; i < plane; i++) {
      let a = buf[i * 4 + 3];
      if (a < 0) a = 0; else if (a > 1) a = 1;
      const w = 1 - a;
      out[i * 4 + 0] = (w + buf[i * 4 + 0]) * 255;
      out[i * 4 + 1] = (w + buf[i * 4 + 1]) * 255;
      out[i * 4 + 2] = (w + buf[i * 4 + 2]) * 255;
      out[i * 4 + 3] = 255;
    }
    return out;
  }

  // Texture 0 of the current state set: RGBA = (r, g, b, alpha) channels.
  // Bound to the active texture unit and returned, for compositing shaders.
  bindOutputTexture() {
    const tex = this._sets[this._cur].texs[0];
    this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
    return tex;
  }

  drawTo(ctx2d) {
    const rgba = this.readRGBA();
    ctx2d.putImageData(new ImageData(rgba, this._W, this._H), 0, 0);
  }
}

// ---------------------------------------------------------------------------
// Factory + GL-vs-CPU comparison
// ---------------------------------------------------------------------------

export function createCA(weights, opts = {}) {
  if (!opts.forceCPU) {
    try {
      let gl = opts.gl || null;
      if (!gl) {
        const dims = gridDims(weights);
        let canvas = opts.canvas || null;
        if (!canvas) {
          if (typeof document !== "undefined") {
            canvas = document.createElement("canvas");
          } else if (typeof OffscreenCanvas !== "undefined") {
            canvas = new OffscreenCanvas(dims.W, dims.H);
          }
        }
        if (canvas) {
          canvas.width = dims.W;
          canvas.height = dims.H;
          gl = canvas.getContext("webgl2", { antialias: false, depth: false, stencil: false });
        }
      }
      if (gl) return { ca: new GLCA(gl, weights), mode: "gl" };
    } catch (e) {
      /* fall through to CPU */
    }
  }
  return { ca: new CPUCA(weights), mode: "cpu" };
}

export function compareGLvsCPU(weights, steps = 20) {
  const { ca: glca, mode } = createCA(weights);
  if (mode !== "gl") throw new Error("WebGL2 with float render targets is unavailable");
  const cpu = new CPUCA(weights);
  cpu.reset();
  glca.reset();
  for (let i = 0; i < steps; i++) {
    cpu.step(1.0);
    glca.step(1.0);
  }
  const a = cpu.state, b = glca.state;
  let maxAbsDiff = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbsDiff) maxAbsDiff = d;
  }
  return { maxAbsDiff };
}

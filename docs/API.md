# Contract between `nca.js` (engine) and the page (`main.js`)

## Weight file format (`site/weights/<hex4-codepoint>.json`)

```json
{
  "char": "A", "grid": 40, "glyph": 28,
  "channel_n": 12, "hidden_n": 64, "fire_rate": 0.5,
  "layout": "blocked",
  "fc0_w": [[...48 floats...] x 64],   // [hidden_n][3*channel_n]
  "fc0_b": [...64 floats...],
  "fc1_w": [[...64 floats...] x 12]    // [channel_n][hidden_n], NO bias
}
```

`site/weights/index.json` = `{"chars": ["A","B",...], "words": ["GO","GROW"]}`
(available characters / whole-word models at `weights/word_<TEXT>.json`).

`layout` semantics: `"blocked"` means fc0_w columns are ordered
`[state(C) | sobel_x(C) | sobel_y(C)]` — use as-is. `"interleaved"` (or
absent) means columns are `[id_c0, sx_c0, sy_c0, id_c1, ...]` — reorder to
blocked at load time. Trust the field.

### Word models (`weights/word_<TEXT>.json`)

Same fields plus: `kind: "word"`, `text`, `grid_w`, `grid_h` (`grid` is
null — use grid_w/grid_h), and `seeds: [{x, y, code: [5 ints 0/1], char}]`,
`code_ch0: 4`, `code_bits: 5`. One model grows the whole string on one
`grid_w x grid_h` grid. `reset()` must place ALL seeds: at each seed cell set
channels 3..C-1 = 1.0, then overwrite channels `code_ch0..code_ch0+code_bits-1`
with the seed's code bits (0.0/1.0). Everything else (step, damage, render)
is unchanged.

## The CA update rule (must match Python exactly)

State: `channel_n` floats per cell on a `grid x grid` torus-free grid
(out-of-bounds neighbors read as 0). Channels 0-2 premultiplied RGB,
3 = alpha, rest hidden. Stored as Float32Array in **[C, H, W]** order
(channel-major, row y, column x): `state[c*H*W + y*W + x]`.

Per step, computed for ALL cells from the OLD state (double buffer):

1. `preAlive(x,y)` = max of alpha over 3x3 neighborhood > 0.1.
2. Perception vector p[3C], **blocked layout**:
   - `p[c]      = state[c]` (identity)
   - `p[C + c]  = sobelX(state[c]) ` with kernel [[-1,0,1],[-2,0,2],[-1,0,1]]/8
   - `p[2C + c] = sobelY(state[c])` (transpose of sobelX)
3. `h = relu(fc0_w @ p + fc0_b)`; `dx = fc1_w @ h`.
4. Per-CELL update gate: if `rand() <= fireRate` the whole cell gets
   `new = old + dx`, else `new = old`.
5. `postAlive` computed on the NEW alpha (same maxpool). If
   `!(preAlive && postAlive)`: zero ALL channels of the cell.

Render to RGBA bytes: `v = clamp01(1 - a + rgb_c) * 255`, alpha byte 255.

Seed: all zeros, then channels `3..channel_n-1` = 1.0 at one cell.

## Engine API (`site/nca.js`, ES module)

```js
export class CPUCA {
  constructor(weights)          // parsed weight JSON
  reset()                       // clear + seed at center
  clear()                       // all zeros
  seed(x, y)                    // set channels 3.. to 1 at cell
  damage(x, y, r)               // zero all channels within radius r
  step(fireRate = null, rand = Math.random)  // one CA step; null -> weights.fire_rate
  get state()                   // Float32Array [C,H,W]
  readRGBA(out?)                // Uint8ClampedArray grid*grid*4
}

export class GLCA {             // WebGL2, float textures; same methods
  constructor(gl, weights)      // gl from a throwaway/shared canvas
  // step() uses a deterministic per-(cell,frame) hash for the fire mask
  readRGBA(out?)                // readback (slow; for tests)
  bindOutputTexture()           // texture handle for compositing, plus:
  drawTo(ctx2d)                 // blit current RGBA into a 2D canvas
}

export function createCA(weights, opts = {})
  // -> { ca, mode: "gl" | "cpu" }; tries WebGL2 w/ EXT_color_buffer_float,
  //    falls back to CPUCA. opts.forceCPU for tests.

export function compareGLvsCPU(weights, steps = 20)
  // runs both with fireRate = 1.0 (deterministic) from reset();
  // -> { maxAbsDiff } — must be < 1e-2 for shipping.
```

## Test fixtures (already generated)

- `site/test/dummy_0058.json` — real exported weights (short training run).
- `site/test/golden_0058.json` — `{grid, channel_n, steps: 30, fire_rate: 1.0,
  state: [C*H*W floats]}` — the exact state after 30 steps with
  fireRate 1.0 from `reset()`. CPUCA must match with max abs diff < 1e-3.

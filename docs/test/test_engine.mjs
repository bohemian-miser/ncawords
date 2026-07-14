// Node test for site/nca.js (CPU path). Run from the project root:
//   node site/test/test_engine.mjs
// Plain node v20, no dependencies. Exits nonzero on any failure.

import { readFileSync } from "node:fs";
import { CPUCA, createCA } from "../nca.js";

const here = new URL(".", import.meta.url);
const weights = JSON.parse(readFileSync(new URL("dummy_0058.json", here), "utf8"));
const golden = JSON.parse(readFileSync(new URL("golden_0058.json", here), "utf8"));

let failures = 0;
function check(name, cond, detail = "") {
  const ok = !!cond;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`);
  if (!ok) failures++;
}

const G = weights.grid;
const C = weights.channel_n;
const plane = G * G;
const cx = G >> 1;

// ---------------------------------------------------------------------------
// 1. Golden test: reset + 30 steps at fireRate 1.0 must match PyTorch output.
// ---------------------------------------------------------------------------
{
  check("golden fixture shape", golden.grid === G && golden.channel_n === C &&
    golden.state.length === C * plane && golden.fire_rate === 1.0);

  const ca = new CPUCA(weights);
  ca.reset();
  for (let i = 0; i < golden.steps; i++) ca.step(1.0);
  const st = ca.state;
  let maxDiff = 0;
  for (let i = 0; i < st.length; i++) {
    const d = Math.abs(st[i] - golden.state[i]);
    if (d > maxDiff) maxDiff = d;
  }
  console.log(`golden max abs diff after ${golden.steps} steps: ${maxDiff}`);
  check("golden state match", maxDiff < 1e-3, `maxAbsDiff=${maxDiff}, tol=1e-3`);
}

// ---------------------------------------------------------------------------
// 1b. Layout field is trusted: converting dummy's interleaved fc0_w columns
//     to blocked order (and labeling it so) must yield the identical result.
// ---------------------------------------------------------------------------
{
  const blocked = { ...weights, layout: "blocked" };
  blocked.fc0_w = weights.fc0_w.map((row) => {
    const out = new Array(3 * C);
    for (let c = 0; c < C; c++) {
      out[c] = row[3 * c];             // identity
      out[C + c] = row[3 * c + 1];     // sobel x
      out[2 * C + c] = row[3 * c + 2]; // sobel y
    }
    return out;
  });
  const ca = new CPUCA(blocked);
  ca.reset();
  for (let i = 0; i < golden.steps; i++) ca.step(1.0);
  let maxDiff = 0;
  for (let i = 0; i < ca.state.length; i++) {
    const d = Math.abs(ca.state[i] - golden.state[i]);
    if (d > maxDiff) maxDiff = d;
  }
  check("blocked-layout weights match golden too", maxDiff < 1e-3, `maxAbsDiff=${maxDiff}`);
}

// ---------------------------------------------------------------------------
// 2. Seed placement: reset() puts channels 3.. = 1 at the center, else zeros.
// ---------------------------------------------------------------------------
{
  const ca = new CPUCA(weights);
  ca.reset();
  const st = ca.state;
  const i0 = cx * G + cx;
  let ok = true;
  for (let c = 0; c < C; c++) {
    const want = c >= 3 ? 1 : 0;
    if (st[c * plane + i0] !== want) ok = false;
  }
  let sum = 0;
  for (let i = 0; i < st.length; i++) sum += st[i];
  check("seed placement at center", ok && sum === C - 3,
    `center=(${cx},${cx}), total sum=${sum}, expected ${C - 3}`);

  ca.clear();
  ca.seed(5, 7);
  const st2 = ca.state;
  check("seed(5,7) placement", st2[3 * plane + 7 * G + 5] === 1 &&
    st2[(C - 1) * plane + 7 * G + 5] === 1 && st2[0 * plane + 7 * G + 5] === 0);
}

// ---------------------------------------------------------------------------
// 3. damage() zeroes all channels within radius r.
// ---------------------------------------------------------------------------
{
  const ca = new CPUCA(weights);
  ca.reset();
  for (let i = 0; i < 20; i++) ca.step(1.0);
  const st = ca.state;
  let preSum = 0;
  for (let i = 0; i < st.length; i++) preSum += Math.abs(st[i]);
  check("pattern grew before damage", preSum > 0);

  const r = 5;
  ca.damage(cx, cx, r);
  let inside = 0, outside = 0;
  for (let y = 0; y < G; y++) {
    for (let x = 0; x < G; x++) {
      const d2 = (x - cx) * (x - cx) + (y - cx) * (y - cx);
      for (let c = 0; c < C; c++) {
        const v = Math.abs(ca.state[c * plane + y * G + x]);
        if (d2 <= r * r) inside += v; else outside += v;
      }
    }
  }
  check("damage zeroes cells within radius", inside === 0, `insideSum=${inside}`);
  check("damage leaves cells outside radius", outside > 0, `outsideSum=${outside}`);
}

// ---------------------------------------------------------------------------
// 4. Alive masking: isolated cell with alpha 0.05 (<= 0.1) dies in one step.
// ---------------------------------------------------------------------------
{
  const ca = new CPUCA(weights);
  ca.clear();
  const i0 = 10 * G + 10;
  ca.state[0 * plane + i0] = 0.5;  // some RGB
  ca.state[3 * plane + i0] = 0.05; // alpha below the 0.1 threshold
  ca.state[5 * plane + i0] = 0.7;  // hidden channel
  ca.step(1.0);
  let sum = 0;
  for (let i = 0; i < ca.state.length; i++) sum += Math.abs(ca.state[i]);
  check("isolated alpha=0.05 cell dies (grid all zero)", sum === 0, `sum=${sum}`);
}

// ---------------------------------------------------------------------------
// 5. readRGBA: shape, range, alpha byte, white background, out-reuse.
// ---------------------------------------------------------------------------
{
  const ca = new CPUCA(weights);
  ca.clear();
  let rgba = ca.readRGBA();
  check("readRGBA type/length", rgba instanceof Uint8ClampedArray && rgba.length === plane * 4);
  let bgOk = true;
  for (let i = 0; i < rgba.length; i++) if (rgba[i] !== 255) bgOk = false;
  check("readRGBA empty grid is white, alpha=255", bgOk);

  ca.reset();
  for (let i = 0; i < 10; i++) ca.step(1.0);
  const out = new Uint8ClampedArray(plane * 4);
  rgba = ca.readRGBA(out);
  let rangeOk = rgba === out, alphaOk = true, nonWhite = false;
  for (let i = 0; i < plane; i++) {
    if (rgba[i * 4 + 3] !== 255) alphaOk = false;
    for (let k = 0; k < 3; k++) {
      const v = rgba[i * 4 + k];
      if (!(v >= 0 && v <= 255)) rangeOk = false;
      if (v !== 255) nonWhite = true;
    }
  }
  check("readRGBA range 0..255, alpha bytes 255, reuses out", rangeOk && alphaOk);
  check("readRGBA shows the grown pattern", nonWhite);
}

// ---------------------------------------------------------------------------
// 6. fireRate 0 is a no-op (alive-masking of an already-masked state changes
//    nothing, since post-mask alpha equals the current alpha).
// ---------------------------------------------------------------------------
{
  const ca = new CPUCA(weights);
  ca.reset();
  for (let i = 0; i < 5; i++) ca.step(1.0);
  const before = ca.state.slice();
  ca.step(0, () => 0.5); // deterministic rand; 0.5 <= 0 is false -> no cell fires
  let maxDiff = 0;
  for (let i = 0; i < before.length; i++) {
    const d = Math.abs(ca.state[i] - before[i]);
    if (d > maxDiff) maxDiff = d;
  }
  check("fireRate 0 step is a no-op", maxDiff === 0, `maxDiff=${maxDiff}`);
}

// ---------------------------------------------------------------------------
// 7. Custom rand is honored (rand always 0 -> fires even at tiny fireRate,
//    since the gate is rand() <= fireRate).
// ---------------------------------------------------------------------------
{
  const a = new CPUCA(weights);
  const b = new CPUCA(weights);
  a.reset(); b.reset();
  a.step(1.0);
  b.step(1e-9, () => 0); // 0 <= 1e-9 -> every cell fires, same as fireRate 1
  let maxDiff = 0;
  for (let i = 0; i < a.state.length; i++) {
    const d = Math.abs(a.state[i] - b.state[i]);
    if (d > maxDiff) maxDiff = d;
  }
  check("step(fr, rand=()=>0) fires all cells", maxDiff === 0, `maxDiff=${maxDiff}`);
}

// ---------------------------------------------------------------------------
// 8. createCA fallback: forceCPU, and CPU fallback under Node (no WebGL2).
// ---------------------------------------------------------------------------
{
  const forced = createCA(weights, { forceCPU: true });
  check("createCA forceCPU -> cpu", forced.mode === "cpu" && forced.ca instanceof CPUCA);
  const auto = createCA(weights);
  check("createCA in Node falls back to cpu", auto.mode === "cpu" && auto.ca instanceof CPUCA);
}

// ---------------------------------------------------------------------------
// 9. Word models: rectangular grid, multi-seed reset with code channels.
//    Synthetic word model reusing dummy_0058's layers (no new fixture).
// ---------------------------------------------------------------------------
{
  const GW = 72, GH = 32, CODE_CH0 = 4, CODE_BITS = 5;
  const seeds = [
    { x: 18, y: 16, code: [0, 0, 0, 0, 0], char: "X" },
    { x: 54, y: 16, code: [1, 0, 0, 1, 0], char: "X" },
  ];
  const wordWeights = {
    kind: "word", text: "XX", grid: null, grid_w: GW, grid_h: GH,
    channel_n: weights.channel_n, hidden_n: weights.hidden_n,
    fire_rate: weights.fire_rate, layout: weights.layout,
    fc0_w: weights.fc0_w, fc0_b: weights.fc0_b, fc1_w: weights.fc1_w,
    seeds, code_ch0: CODE_CH0, code_bits: CODE_BITS,
  };
  const wplane = GW * GH;
  const ca = new CPUCA(wordWeights);

  check("word width/height getters", ca.width === GW && ca.height === GH);
  {
    const sq = new CPUCA(weights);
    check("square width/height getters", sq.width === G && sq.height === G);
  }
  check("word state length", ca.state.length === C * wplane);

  // reset() placement: channels 3..C-1 = 1, then code bits overwrite 4..8.
  ca.reset();
  let seedOk = true;
  for (const s of seeds) {
    const i = s.y * GW + s.x;
    for (let c = 0; c < C; c++) {
      let want;
      if (c < 3) want = 0;
      else if (c >= CODE_CH0 && c < CODE_CH0 + CODE_BITS) want = s.code[c - CODE_CH0];
      else want = 1;
      if (ca.state[c * wplane + i] !== want) {
        seedOk = false;
        console.log(`  mismatch seed(${s.x},${s.y}) ch${c}: got ${ca.state[c * wplane + i]}, want ${want}`);
      }
    }
  }
  let total = 0;
  for (const v of ca.state) total += v;
  // seed1: ch3,9,10,11 = 1 (code all 0) -> 4; seed2: + code bits 4,7 -> 6.
  check("word reset places both seeds with code channels", seedOk && total === 10,
    `total=${total}, expected 10`);

  // Steps run on the rectangular grid; state stays finite; far cells stay 0.
  for (let i = 0; i < 20; i++) ca.step(1.0);
  let finite = true, grew = 0;
  for (const v of ca.state) { if (!Number.isFinite(v)) finite = false; grew += Math.abs(v); }
  check("word step runs on rectangular grid, state finite", finite && grew > 0,
    `sumAbs=${grew.toFixed(3)}`);
  let cornersZero = true;
  for (const [cx2, cy2] of [[0, 0], [GW - 1, 0], [0, GH - 1], [GW - 1, GH - 1]]) {
    for (let c = 0; c < C; c++) {
      if (ca.state[c * wplane + cy2 * GW + cx2] !== 0) cornersZero = false;
    }
  }
  check("word borders stay zero (OOB reads 0, growth contained)", cornersZero);

  // damage at both ends of the rectangle (and clipped at a border).
  // (after 20 steps the pattern extends ~5 cells from each seed, so r=3
  // leaves an outer ring alive)
  const r = 3;
  ca.damage(18, 16, r);
  ca.damage(54, 16, r);
  ca.damage(0, 0, 3); // clipped at the border, must not throw
  let insideL = 0, insideR = 0, outside = 0;
  for (let y = 0; y < GH; y++) {
    for (let x = 0; x < GW; x++) {
      const dL = (x - 18) * (x - 18) + (y - 16) * (y - 16);
      const dR = (x - 54) * (x - 54) + (y - 16) * (y - 16);
      for (let c = 0; c < C; c++) {
        const v = Math.abs(ca.state[c * wplane + y * GW + x]);
        if (dL <= r * r) insideL += v;
        else if (dR <= r * r) insideR += v;
        else outside += v;
      }
    }
  }
  check("word damage zeroes both seed regions", insideL === 0 && insideR === 0,
    `L=${insideL} R=${insideR}`);
  check("word damage leaves the rest intact", outside > 0, `outside=${outside.toFixed(3)}`);
}

// ---------------------------------------------------------------------------
// Informational: CPU step throughput.
// ---------------------------------------------------------------------------
{
  const ca = new CPUCA(weights);
  ca.reset();
  for (let i = 0; i < 10; i++) ca.step(1.0); // warm up JIT + grow pattern
  const N = 120;
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) ca.step();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.log(`info: CPU throughput ${(N / (ms / 1000)).toFixed(0)} steps/sec (${(ms / N).toFixed(2)} ms/step)`);
}

console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);

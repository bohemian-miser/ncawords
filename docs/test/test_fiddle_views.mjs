// Node test for the Fiddle modal's semantic weight views and the engines'
// live weight hot-swap. Run from the project root:
//   node docs/test/test_fiddle_views.mjs
// Plain node v20, no dependencies. Exits nonzero on any failure.
//
// The two things worth proving here are the ones that fail SILENTLY:
//   1. perceptionColumn() must agree with nca.js's own reading of fc0_w, or
//      the tree confidently mislabels every weight it shows. It is checked
//      behaviourally — zero the columns the view calls "sobel x/y" and the CA
//      must stop seeing its neighbours entirely.
//   2. setWeights() must swap the physics WITHOUT disturbing the grid, or
//      live-apply quietly becomes the reset it exists to avoid.

import { readFileSync } from "node:fs";
import { CPUCA } from "../nca.js";
import { LeniaCA } from "../lenia_engine.js";
import { perceptionColumn, channelLabel } from "../fiddle.js";

const here = new URL(".", import.meta.url);
const readJson = (p) => JSON.parse(readFileSync(new URL(p, here), "utf8"));

let failures = 0;
function check(name, cond, detail = "") {
  const ok = !!cond;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`);
  if (!ok) failures++;
}

const clone = (v) => JSON.parse(JSON.stringify(v));
const ALL_FIRE = () => 0;   // rand() <= fireRate for every cell

// ---------------------------------------------------------------------------
// 1. perceptionColumn — the mapping, stated two ways.
// ---------------------------------------------------------------------------
{
  const blocked = { layout: "blocked" };
  const inter = { layout: "interleaved" };
  const C = 4;
  // blocked: [state(0..C) | sobelX(C..2C) | sobelY(2C..3C)]
  check("blocked state column", perceptionColumn(blocked, 0, 2, C) === 2);
  check("blocked sobel-x column", perceptionColumn(blocked, 1, 2, C) === C + 2);
  check("blocked sobel-y column", perceptionColumn(blocked, 2, 2, C) === 2 * C + 2);
  // interleaved: [id0, sx0, sy0, id1, sx1, sy1, …]
  check("interleaved state column", perceptionColumn(inter, 0, 2, C) === 6);
  check("interleaved sobel-x column", perceptionColumn(inter, 1, 2, C) === 7);
  check("interleaved sobel-y column", perceptionColumn(inter, 2, 2, C) === 8);
  // Whatever the layout, the three blocks must tile the row exactly once.
  for (const w of [blocked, inter]) {
    const cols = new Set();
    for (let b = 0; b < 3; b++) for (let c = 0; c < C; c++) cols.add(perceptionColumn(w, b, c, C));
    check(`${w.layout}: the 3 blocks tile all ${3 * C} columns exactly once`,
      cols.size === 3 * C && Math.max(...cols) === 3 * C - 1 && Math.min(...cols) === 0);
  }
}

// ---------------------------------------------------------------------------
// 2. perceptionColumn, behaviourally: zeroing the columns the view calls
//    "sobel x" and "sobel y" must make a cell blind to its neighbours. If the
//    mapping were wrong we would have zeroed the identity columns instead and
//    neighbour sensitivity would survive.
// ---------------------------------------------------------------------------
function zeroSobelColumns(weights) {
  const w = clone(weights);
  const C = w.channel_n;
  for (let k = 0; k < w.hidden_n; k++) {
    for (let block = 1; block <= 2; block++) {
      for (let c = 0; c < C; c++) w.fc0_w[k][perceptionColumn(w, block, c, C)] = 0;
    }
  }
  return w;
}

// Alive everywhere (alpha well over the 0.1 mask threshold) so the alive-mask
// gate can't be what makes the two runs agree.
function fillAlive(ca, C, plane, seed) {
  const st = ca.state;
  for (let c = 0; c < C; c++) {
    for (let i = 0; i < plane; i++) st[c * plane + i] = c === 3 ? 0.9 : ((i * 7 + c * 13 + seed) % 10) / 20;
  }
}

function neighbourSensitivity(weights, label) {
  const C = weights.channel_n;
  const ca = new CPUCA(weights);
  const W = ca.width, H = ca.height, plane = W * H;
  const x = 5, y = 4, i = y * W + x;
  const nb = y * W + (x + 1);   // the cell immediately to the right

  const run = (perturb) => {
    const c = new CPUCA(weights);
    fillAlive(c, C, plane, 0);
    if (perturb) c.state[0 * plane + nb] += 0.5;   // channel 0 of one neighbour
    c.step(1.0, ALL_FIRE);
    const out = [];
    for (let ch = 0; ch < C; ch++) out.push(c.state[ch * plane + i]);
    return out;
  };

  const base = run(false), moved = run(true);
  let maxDiff = 0;
  for (let ch = 0; ch < C; ch++) maxDiff = Math.max(maxDiff, Math.abs(base[ch] - moved[ch]));
  check(`${label}: baseline engine constructs`, ca.channel_n === C);
  return maxDiff;
}

for (const [file, name] of [["dummy_0058.json", "interleaved"], ["../weights/word_COMP.json", "blocked"]]) {
  const w = readJson(file);
  check(`${name}: fixture layout is ${name}`, (w.layout || "interleaved") === name);
  const before = neighbourSensitivity(w, `${name} trained`);
  const after = neighbourSensitivity(zeroSobelColumns(w), `${name} sobel-zeroed`);
  check(`${name}: trained net DOES react to a neighbour`, before > 1e-9,
    `maxDiff=${before.toExponential(3)}`);
  check(`${name}: zeroing the view's sobel columns makes it blind to neighbours`,
    after === 0, `maxDiff=${after}`);
}

// A control: zeroing the columns the view calls "state" must NOT blind it,
// which is what distinguishes a correct mapping from a transposed one.
{
  const w = readJson("dummy_0058.json");
  const z = clone(w);
  const C = z.channel_n;
  for (let k = 0; k < z.hidden_n; k++) {
    for (let c = 0; c < C; c++) z.fc0_w[k][perceptionColumn(z, 0, c, C)] = 0;
  }
  const diff = neighbourSensitivity(z, "state-zeroed");
  check("control: zeroing the STATE columns leaves neighbour sensing intact", diff > 1e-9,
    `maxDiff=${diff.toExponential(3)}`);
}

// ---------------------------------------------------------------------------
// 3. channelLabel — RGBA, code bits, hidden.
// ---------------------------------------------------------------------------
{
  const w = { channel_n: 16, code_ch0: 4, code_bits: 6 };
  check("channel 0 is R", channelLabel(w, 0) === "ch 0 · R");
  check("channel 3 is alpha", channelLabel(w, 3) === "ch 3 · alpha");
  check("channel 4 is code bit 0", channelLabel(w, 4) === "ch 4 · code bit 0");
  check("channel 9 is code bit 5", channelLabel(w, 9) === "ch 9 · code bit 5");
  check("channel 10 is past the code, so hidden", channelLabel(w, 10) === "ch 10 · hidden");
  check("no code metadata: everything past alpha is hidden",
    channelLabel({ channel_n: 12 }, 7) === "ch 7 · hidden");
  check("code_bits 0 does not claim a channel",
    channelLabel({ code_ch0: 4, code_bits: 0 }, 4) === "ch 4 · hidden");
}

// ---------------------------------------------------------------------------
// 4. NCA hot-swap: physics changes, grid does not.
// ---------------------------------------------------------------------------
{
  const w = readJson("dummy_0058.json");
  const C = w.channel_n;
  const ca = new CPUCA(w);
  const plane = ca.width * ca.height;
  fillAlive(ca, C, plane, 3);
  for (let i = 0; i < 3; i++) ca.step(1.0, ALL_FIRE);
  const before = Float32Array.from(ca.state);

  const edited = clone(w);
  edited.fc1_w[0][0] += 1.5;            // a real change to the rule
  const ok = ca.setWeights(edited);
  check("CPU setWeights accepts same-shape weights", ok === true);
  const after = Float32Array.from(ca.state);
  let identical = before.length === after.length;
  for (let i = 0; identical && i < before.length; i++) if (before[i] !== after[i]) identical = false;
  check("CPU setWeights leaves the grid state untouched", identical);

  // Same state + swapped weights must now evolve differently from a control
  // engine still carrying the trained weights.
  const control = new CPUCA(w);
  control.state.set(before);
  ca.step(1.0, ALL_FIRE);
  control.step(1.0, ALL_FIRE);
  let maxDiff = 0;
  for (let i = 0; i < ca.state.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(ca.state[i] - control.state[i]));
  }
  check("CPU setWeights actually changes the physics", maxDiff > 1e-9,
    `maxDiff=${maxDiff.toExponential(3)}`);

  // A control engine built fresh from the edited weights must agree with the
  // swapped one given the same state — the swap is not an approximation.
  const fresh = new CPUCA(edited);
  fresh.state.set(before);
  fresh.step(1.0, ALL_FIRE);
  let swapDiff = 0;
  for (let i = 0; i < fresh.state.length; i++) {
    swapDiff = Math.max(swapDiff, Math.abs(ca.state[i] - fresh.state[i]));
  }
  check("swapped engine matches one rebuilt from the same weights", swapDiff === 0,
    `maxDiff=${swapDiff}`);
}

// hidden_n may change in place (the CPU scratch is resized); channel_n and the
// grid may not — those are what the buffers are allocated around.
{
  const w = readJson("dummy_0058.json");
  const ca = new CPUCA(w);

  const wider = clone(w);
  wider.hidden_n = w.hidden_n + 2;
  for (let i = 0; i < 2; i++) {
    wider.fc0_w.push(w.fc0_w[0].slice());
    wider.fc0_b.push(0.1);
    for (const row of wider.fc1_w) row.push(0.01);
  }
  check("CPU setWeights accepts a changed hidden_n", ca.setWeights(wider) === true);
  ca.step(1.0, ALL_FIRE);
  check("engine still steps after a hidden_n swap", Number.isFinite(ca.state[0]));

  const moreCh = clone(w);
  moreCh.channel_n = w.channel_n + 1;
  check("CPU setWeights refuses a changed channel_n", ca.setWeights(moreCh) === false);

  const bigger = clone(w);
  bigger.grid = w.grid + 8;
  check("CPU setWeights refuses a changed grid", ca.setWeights(bigger) === false);
}

// ---------------------------------------------------------------------------
// 5. Lenia hot-swap: same contract.
// ---------------------------------------------------------------------------
{
  const lenia = {
    kind: "lenia", variant: "static1", C: 1, K: 2, ks: 3, dt: 0.1, leak: 0.05,
    size: 16, init: "seedblob", seed_x: 8, seed_y: 8,
    h: [1.0, 0.5], mu: [0.15, 0.3], sg: [0.02, 0.05],
    kernels: [
      [[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]],
      [[0.0, 0.1, 0.0], [0.1, 0.3, 0.1], [0.0, 0.1, 0.0]]
    ]
  };
  const ca = new LeniaCA(lenia, 16);
  for (let i = 0; i < 5; i++) ca.step();
  const before = Float32Array.from(ca.state);

  const edited = clone(lenia);
  edited.mu[0] = 0.4;
  check("Lenia setWeights accepts same C and size", ca.setWeights(edited) === true);
  let identical = true;
  for (let i = 0; i < before.length; i++) if (before[i] !== ca.state[i]) identical = false;
  check("Lenia setWeights leaves the grid state untouched", identical);

  const control = new LeniaCA(lenia, 16);
  control.state.set(before);
  ca.step();
  control.step();
  let maxDiff = 0;
  for (let i = 0; i < ca.state.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(ca.state[i] - control.state[i]));
  }
  check("Lenia setWeights actually changes the physics", maxDiff > 1e-9,
    `maxDiff=${maxDiff.toExponential(3)}`);

  const fresh = new LeniaCA(edited, 16);
  fresh.state.set(before);
  fresh.step();
  let swapDiff = 0;
  for (let i = 0; i < fresh.state.length; i++) {
    swapDiff = Math.max(swapDiff, Math.abs(ca.state[i] - fresh.state[i]));
  }
  check("swapped Lenia matches one rebuilt from the same weights", swapDiff === 0,
    `maxDiff=${swapDiff}`);

  const wideC = clone(lenia); wideC.C = 2;
  check("Lenia setWeights refuses a changed C", ca.setWeights(wideC) === false);
  const bigger = clone(lenia); bigger.size = 32;
  check("Lenia setWeights refuses a changed size", ca.setWeights(bigger) === false);

  // The dyn variants keep per-basis scratch sized by the basis count; swapping
  // between kernel banks of different sizes must not reuse a stale buffer.
  const dyn = {
    kind: "lenia", variant: "dyn1", C: 1, K: 1, ks: 3, dt: 0.1, leak: 0.05, size: 16,
    init: "noise", h: [1.0], mu: [0.15], sg: [0.02],
    basis: [[[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]]],
    mix: { w0: [[0.1, 0.2]], b0: [0.0], w2: [[0.3]], b2: [0.0] }
  };
  const dca = new LeniaCA(dyn, 16);
  dca.step();
  const dyn2 = clone(dyn);
  dyn2.basis.push([[0.0, 0.1, 0.0], [0.1, 0.2, 0.1], [0.0, 0.1, 0.0]]);
  dyn2.mix = { w0: [[0.1, 0.2, 0.15]], b0: [0.0], w2: [[0.3], [0.2]], b2: [0.0, 0.0] };
  check("Lenia setWeights accepts a resized dyn basis", dca.setWeights(dyn2) === true);
  dca.step();
  check("dyn engine still steps after a basis resize", Number.isFinite(dca.state[0]));
}

console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
process.exitCode = failures === 0 ? 0 : 1;

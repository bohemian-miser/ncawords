// Parity test for docs/demo_coexist.js's BlendedCA against nca.js's CPUCA.
//
// With THE SAME weights loaded as both model A and model B and the blend
// field at 0.5 everywhere, dx = 0.5*dx_m + 0.5*dx_m == dx_m, so the blended
// engine must reproduce a single CPUCA exactly (fireRate forced to 1 so the
// shared stochastic fire mask is deterministic). This proves the blended
// step math (perception, fc0/fc1, zero padding, alive mask) matches nca.js.
//
// Run: node docs/test/test_coexist.mjs
// Uses any exported NCA weights.json; override with WEIGHTS_URL env var.

import { CPUCA } from "../nca.js";
import { BlendedCA } from "../demo_coexist.js";

const URL_DEFAULT =
  "https://storage.googleapis.com/recipe-lanes-nca-jobs/alphaword-noise3/weights.json";
const url = process.env.WEIGHTS_URL || URL_DEFAULT;

const res = await fetch(url);
if (!res.ok) {
  console.error(`FAIL  fetch ${url}: HTTP ${res.status}`);
  process.exit(1);
}
const weights = await res.json();

let failures = 0;
function check(name, cond, detail = "") {
  const ok = !!cond;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`);
  if (!ok) failures++;
}

check("weights have channel_n/hidden_n",
  Number.isInteger(weights.channel_n) && Number.isInteger(weights.hidden_n),
  `channel_n=${weights.channel_n}, hidden_n=${weights.hidden_n}`);

const STEPS = 5;

const cpu = new CPUCA(weights);
const blend = new BlendedCA(weights, weights);

check("grid dims match CPUCA",
  blend.width === cpu.width && blend.height === cpu.height,
  `${blend.width}x${blend.height}`);

// Identical single-seed start for both engines.
const sx = blend.width >> 1, sy = blend.height >> 1;
cpu.clear(); cpu.seed(sx, sy);
blend.clear(); blend.seed(sx, sy);
blend.B.fill(0.5); // 0.5*dx_m + 0.5*dx_m == dx_m for identical models

for (let i = 0; i < STEPS; i++) {
  cpu.step(1.0);   // fire_rate forced to 1: every cell fires in both engines
  blend.step(1.0);
}

const a = cpu.state, b = blend.state;
check("state lengths match", a.length === b.length, `${a.length}`);

let maxDiff = 0, sumAbs = 0;
for (let i = 0; i < a.length; i++) {
  const d = Math.abs(a[i] - b[i]);
  if (d > maxDiff) maxDiff = d;
  sumAbs += Math.abs(a[i]);
}
check("state is nontrivial after 5 steps", sumAbs > 0, `sumAbs=${sumAbs.toFixed(3)}`);
check("blended engine matches CPUCA", maxDiff < 1e-6, `maxAbsDiff=${maxDiff}, tol=1e-6`);

console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);

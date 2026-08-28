// Node test for docs/fiddle.js — the pure, DOM-free half of the Fiddle
// weights editor. Run from the project root:
//   node docs/test/test_fiddle.mjs
// Plain node v20+, no dependencies. Exits nonzero on any failure.
//
// The whole point of these tests is the contract fiddle.js is written
// against: it must import under bare Node (no document/window/location at
// import time), diff/patch must be lossless and non-mutating, patches must
// not be a prototype-pollution vector, and a token must survive a URL
// fragment intact.

import { readFileSync } from "node:fs";

const here = new URL(".", import.meta.url);

let failures = 0;
function check(name, cond, detail = "") {
  const ok = !!cond;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`);
  if (!ok) failures++;
}

function deepEqual(a, b) {
  if (Object.is(a, b)) return true;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false;
    return true;
  }
  const oa = a !== null && typeof a === "object";
  const ob = b !== null && typeof b === "object";
  if (oa !== ob || !oa) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) {
    if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
    if (!deepEqual(a[k], b[k])) return false;
  }
  return true;
}

const clone = (v) => JSON.parse(JSON.stringify(v));
const snap = (v) => JSON.stringify(v);

// ---------------------------------------------------------------------------
// 0. Import: fiddle.js must load under plain Node (no top-level DOM access).
// ---------------------------------------------------------------------------
let F = null;
try {
  F = await import("../fiddle.js");
  check("import ../fiddle.js under Node throws nothing", true);
} catch (e) {
  check("import ../fiddle.js under Node throws nothing", false, e.message);
  console.log("\n1 TEST(S) FAILED — cannot continue without the module");
  process.exit(1);
}
const { diffWeights, applyPatch, encodeFiddle, decodeFiddle } = F;

check("exports the four pure helpers + three entry points",
  typeof diffWeights === "function" && typeof applyPatch === "function" &&
  typeof encodeFiddle === "function" && typeof decodeFiddle === "function" &&
  typeof F.initFiddle === "function" && typeof F.openFiddleModal === "function" &&
  typeof F.handleFiddleHash === "function");

// Guard against a lazily-created global leaking in at import time.
check("import did not create a window/document global",
  typeof globalThis.window === "undefined" && typeof globalThis.document === "undefined");

// Fixtures --------------------------------------------------------------
const nca = JSON.parse(readFileSync(new URL("dummy_0058.json", here), "utf8"));

function makeLenia() {
  return {
    kind: "lenia", variant: "static1",
    C: 1, K: 2, ks: 3, dt: 0.1, leak: 0.05, size: 16,
    init: "seedblob", seed_x: 8, seed_y: 8,
    h: [0.6, -0.3],
    mu: [0.15, 0.30],
    sg: [0.02, 0.05],
    kernels: [
      [[0.0, 0.1, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.0]],
      [[0.2, 0.0, 0.2], [0.0, 0.4, 0.0], [0.2, 0.0, 0.2]]
    ]
  };
}

// ---------------------------------------------------------------------------
// 1. diffWeights: the empty diff.
// ---------------------------------------------------------------------------
{
  const a = makeLenia();
  const b = makeLenia();
  const p = diffWeights(a, b);
  check("diffWeights(identical) -> empty patch", Array.isArray(p) && p.length === 0,
    `len=${Array.isArray(p) ? p.length : "not an array"}`);
  check("diffWeights(same reference) -> empty patch", diffWeights(a, a).length === 0);
  check("applyPatch(base, []) deep-equals base", deepEqual(applyPatch(a, []), a));
}

// ---------------------------------------------------------------------------
// 2. diffWeights: one scalar edit -> exactly one entry with a correct path.
// ---------------------------------------------------------------------------
{
  const base = makeLenia();
  const edited = makeLenia();
  edited.dt = 0.25;
  const p = diffWeights(base, edited);
  check("scalar edit -> exactly one patch entry", p.length === 1,
    `got ${p.length}: ${JSON.stringify(p).slice(0, 120)}`);
  check("scalar edit path is 'dt' and value is the new scalar",
    p.length === 1 && p[0][0] === "dt" && p[0][1] === 0.25,
    p.length ? JSON.stringify(p[0]) : "");
  check("scalar edit round-trips through applyPatch",
    deepEqual(applyPatch(base, p), edited));

  // A nested-in-an-object scalar (mix-style path).
  const b2 = { a: 1, m: { w0: 2, b0: 3 } };
  const e2 = { a: 1, m: { w0: 2, b0: 9 } };
  const p2 = diffWeights(b2, e2);
  check("nested object scalar path is dot-joined ('m.b0')",
    p2.length === 1 && p2[0][0] === "m.b0" && p2[0][1] === 9,
    JSON.stringify(p2));
}

// ---------------------------------------------------------------------------
// 3. diffWeights: a nested kernel cell -> 'kernels.i.r.c', round-trips.
// ---------------------------------------------------------------------------
{
  const base = makeLenia();
  const edited = makeLenia();
  edited.kernels[1][2][0] = -0.75;
  const p = diffWeights(base, edited);
  check("kernel-cell edit -> one entry at 'kernels.1.2.0'",
    p.length === 1 && p[0][0] === "kernels.1.2.0" && p[0][1] === -0.75,
    JSON.stringify(p));
  const out = applyPatch(base, p);
  check("kernel-cell patch round-trips deep-equal to the edited object",
    deepEqual(out, edited));
  check("kernel-cell patch left the untouched cells alone",
    out.kernels[0][1][1] === 0.6 && out.kernels[1][0][0] === 0.2);

  // Two independent edits in one diff.
  const e2 = makeLenia();
  e2.kernels[0][0][1] = 5;
  e2.mu[1] = 0.9;
  const p2 = diffWeights(base, e2);
  check("two edits -> two entries, both round-trip",
    p2.length === 2 && deepEqual(applyPatch(base, p2), e2),
    JSON.stringify(p2.map(x => x[0])));
}

// ---------------------------------------------------------------------------
// 4. Shape mismatch: pasting 15x15 kernels over 3x3 replaces whole nodes.
// ---------------------------------------------------------------------------
{
  const base = makeLenia();
  const edited = makeLenia();
  const big = () => Array.from({ length: 15 }, (_, r) =>
    Array.from({ length: 15 }, (_, c) => (r * 15 + c) / 225));
  edited.kernels = [big(), big()];
  edited.ks = 15;
  const p = diffWeights(base, edited);

  // Outer length matches (2 == 2) so the diff descends one level and
  // replaces each row-of-rows wholesale; nothing may be emitted per cell.
  const paths = p.map(e => e[0]).sort();
  check("15x15-over-3x3 emits whole-node replacements, not per-cell sets",
    p.length === 3 && deepEqual(paths, ["kernels.0", "kernels.1", "ks"]),
    JSON.stringify(paths));
  check("the replacement values are the full 15x15 subtrees",
    p.filter(e => e[0].startsWith("kernels."))
      .every(e => Array.isArray(e[1]) && e[1].length === 15 && e[1][0].length === 15));
  check("shape-mismatch patch applies to deep-equal the edited object",
    deepEqual(applyPatch(base, p), edited));

  // A length change on the outer array replaces the node itself.
  const e3 = makeLenia();
  e3.kernels = [e3.kernels[0]];
  e3.K = 1; e3.h = [0.6]; e3.mu = [0.15]; e3.sg = [0.02];
  const p3 = diffWeights(base, e3);
  check("array length change -> single whole-array replacement per array",
    p3.some(e => e[0] === "kernels" && Array.isArray(e[1]) && e[1].length === 1),
    JSON.stringify(p3.map(e => e[0])));
  check("length-change patch round-trips", deepEqual(applyPatch(base, p3), e3));

  // A key-set change replaces the whole object node.
  const b4 = { m: { a: 1, b: 2 } };
  const e4 = { m: { a: 1, c: 3 } };
  const p4 = diffWeights(b4, e4);
  check("object key-set change -> whole-object replacement",
    p4.length === 1 && p4[0][0] === "m" && deepEqual(p4[0][1], { a: 1, c: 3 }),
    JSON.stringify(p4));
  check("key-set-change patch round-trips", deepEqual(applyPatch(b4, p4), e4));

  // A root-level replacement uses the '' path.
  const p5 = diffWeights({ a: 1 }, [1, 2, 3]);
  check("incompatible roots -> single '' (replace everything) entry",
    p5.length === 1 && p5[0][0] === "" && deepEqual(p5[0][1], [1, 2, 3]),
    JSON.stringify(p5));
  check("'' patch replaces the whole object",
    deepEqual(applyPatch({ a: 1 }, p5), [1, 2, 3]));
}

// ---------------------------------------------------------------------------
// 5. Neither helper mutates its inputs.
// ---------------------------------------------------------------------------
{
  const base = makeLenia();
  const edited = makeLenia();
  edited.kernels[0][1][1] = 42;
  edited.h[0] = -1;
  const baseSnap = snap(base), editedSnap = snap(edited);

  const p = diffWeights(base, edited);
  check("diffWeights does not mutate base", snap(base) === baseSnap);
  check("diffWeights does not mutate edited", snap(edited) === editedSnap);

  const out = applyPatch(base, p);
  check("applyPatch does not mutate base", snap(base) === baseSnap);
  check("applyPatch returns a NEW object (not base)", out !== base);

  // The returned object must not share any structure with base either.
  out.kernels[0][0][0] = 999;
  out.h[1] = 999;
  check("applyPatch result shares no nested structure with base",
    snap(base) === baseSnap, "post-write to result changed base");

  // The patch itself must be detached from `edited`.
  const pSnap = snap(p);
  edited.kernels[0][1][1] = -7;
  check("patch values are detached from the edited object", snap(p) === pSnap);
}

// ---------------------------------------------------------------------------
// 6. Prototype-pollution guard.
// ---------------------------------------------------------------------------
{
  const attempts = [
    ["__proto__ direct", [["__proto__", { polluted: "yes" }]]],
    ["__proto__ nested", [["__proto__.polluted", "yes"]]],
    ["constructor.prototype", [["constructor.prototype.polluted", "yes"]]],
    ["prototype segment", [["a.prototype.polluted", "yes"]]],
    ["deep __proto__", [["a.b.__proto__.polluted", "yes"]]]
  ];
  let allSafe = true, threwCount = 0;
  for (const [label, patch] of attempts) {
    let threw = false;
    try {
      const out = applyPatch({ a: { b: {} } }, patch);
      // If it didn't throw it must at least not have polluted anything.
      if (out && out.polluted !== undefined && Object.getPrototypeOf(out) === Object.prototype) {
        // an own 'polluted' key on the result is fine; prototype reach is not
      }
    } catch (e) {
      threw = true; threwCount++;
    }
    const clean = ({}).polluted === undefined && Object.prototype.polluted === undefined;
    if (!clean) { allSafe = false; console.log(`  POLLUTED by ${label}`); }
    if (!threw) console.log(`  note: ${label} did not throw (but stayed clean)`);
  }
  check("applyPatch never pollutes Object.prototype", allSafe);
  check("({}).polluted is still undefined after every attempt", ({}).polluted === undefined);
  check("applyPatch rejects every forbidden path segment", threwCount === attempts.length,
    `${threwCount}/${attempts.length} threw`);

  // A JSON-parsed __proto__ key in the *value* must not pollute the clone.
  const evil = JSON.parse('{"a":1,"__proto__":{"polluted":"yes"}}');
  const out2 = applyPatch(evil, []);
  check("deep-cloning a JSON '__proto__' own-key does not pollute",
    ({}).polluted === undefined && Object.getPrototypeOf(out2) === Object.prototype);

  // Malformed patches are rejected loudly, not applied halfway.
  let badThrew = 0;
  for (const bad of [null, "nope", 42, [["only-a-path"]], [null]]) {
    try { applyPatch({ a: 1 }, bad); } catch (e) { badThrew++; }
  }
  check("applyPatch rejects malformed patches", badThrew === 5, `${badThrew}/5 threw`);
}

// ---------------------------------------------------------------------------
// 7. encodeFiddle / decodeFiddle round-trip.
// ---------------------------------------------------------------------------
{
  check("this Node has CompressionStream (gz path is exercised)",
    typeof globalThis.CompressionStream !== "undefined");

  const payload = { v: 1, run: "dummy_0058", w: nca };
  const token = await encodeFiddle(payload);
  check("gz token starts with 'gz:'", token.startsWith("gz:"), token.slice(0, 8));
  check("token is URL-fragment safe (no + / = #)",
    !/[+/=#]/.test(token), (token.match(/[+/=#]/g) || []).join(""));
  check("token is non-trivially shorter than the raw JSON",
    token.length < JSON.stringify(payload).length,
    `token=${token.length} json=${JSON.stringify(payload).length}`);

  const back = await decodeFiddle(token);
  check("gz round-trip deep-equals the payload", deepEqual(back, payload));
  check("gz round-trip preserves float precision exactly",
    back.w.fc0_w[0][0] === nca.fc0_w[0][0] && back.w.fire_rate === nca.fire_rate);

  // decodeFiddle tolerates a full '#fiddle=<token>' fragment.
  check("decodeFiddle accepts a '#fiddle=' prefixed fragment",
    deepEqual(await decodeFiddle("#fiddle=" + token), payload));

  // Odd byte lengths (1/2/3 mod 3) must all survive the hand-rolled base64url.
  let lenOk = true;
  for (const n of [1, 2, 3, 4, 5, 6, 7, 255, 256, 257]) {
    const p = { v: 1, run: "r", w: { s: "x".repeat(n) } };
    const t = await encodeFiddle(p);
    if (!deepEqual(await decodeFiddle(t), p)) { lenOk = false; console.log(`  failed at n=${n}`); }
  }
  check("base64url round-trips at every length residue", lenOk);
}

// ---------------------------------------------------------------------------
// 8. The 'raw:' fallback when CompressionStream is missing.
// ---------------------------------------------------------------------------
{
  const saveC = globalThis.CompressionStream;
  const saveD = globalThis.DecompressionStream;
  let rawToken = null, rawBack = null, err = null;
  try {
    delete globalThis.CompressionStream;
    delete globalThis.DecompressionStream;
    const payload = { v: 1, run: "dummy_0058", w: nca };
    rawToken = await encodeFiddle(payload);
    rawBack = await decodeFiddle(rawToken);
    check("without CompressionStream the token is 'raw:'", rawToken.startsWith("raw:"),
      rawToken.slice(0, 8));
    check("raw token is URL-fragment safe", !/[+/=#]/.test(rawToken));
    check("raw round-trip deep-equals the payload", deepEqual(rawBack, payload));
    // A gz token cannot be read without DecompressionStream — it must say so.
    let msg = "";
    try { await decodeFiddle("gz:H4sIAAAAAAAAA6tWKkstKlayUrJSSkosUqoFAJ5tE0YNAAAA"); }
    catch (e) { msg = e.message; }
    check("a gz token without DecompressionStream throws a clear error",
      /DecompressionStream/.test(msg), msg);
  } catch (e) {
    err = e;
  } finally {
    globalThis.CompressionStream = saveC;
    globalThis.DecompressionStream = saveD;
  }
  check("raw-path test restored the globals",
    typeof globalThis.CompressionStream !== "undefined" &&
    typeof globalThis.DecompressionStream !== "undefined");
  if (err) check("raw-path test ran without an unexpected throw", false, err.message);
}

// ---------------------------------------------------------------------------
// 9. decodeFiddle rejects garbage with a real Error.
// ---------------------------------------------------------------------------
{
  const bad = ["garbage", "gz:!!!", "raw:!!!", "", "   ", "gz:", "raw:", null, undefined, 42,
               "gz:" + "AAAA"];   // valid base64url, not a gzip stream
  let rejected = 0, allErrors = true;
  for (const t of bad) {
    try {
      await decodeFiddle(t);
      console.log(`  note: decodeFiddle(${JSON.stringify(t)}) did NOT reject`);
    } catch (e) {
      rejected++;
      if (!(e instanceof Error) || !e.message) allErrors = false;
    }
  }
  check("decodeFiddle rejects every garbage token", rejected === bad.length,
    `${rejected}/${bad.length}`);
  check("every rejection is an Error with a message", allErrors);

  let m1 = "", m2 = "";
  try { await decodeFiddle("garbage"); } catch (e) { m1 = e.message; }
  try { await decodeFiddle("gz:!!!"); } catch (e) { m2 = e.message; }
  check("decodeFiddle('garbage') names the missing prefix", /prefix/.test(m1), m1);
  check("decodeFiddle('gz:!!!') names the bad character", /base64url|character/.test(m2), m2);

  // A 'raw:' token holding non-JSON is a distinct, named failure.
  let m3 = "";
  try { await decodeFiddle("raw:bm90IGpzb24"); } catch (e) { m3 = e.message; }
  check("a raw token that isn't JSON reports a JSON error", /JSON/.test(m3), m3);
}

// ---------------------------------------------------------------------------
// 10. Payload-size choice: a one-scalar edit must ship as a patch, not the
//     whole weights object. (saveToUrl's rule, replicated against the same
//     helpers it uses: JSON.stringify(asPatch).length <= asFull.length.)
// ---------------------------------------------------------------------------
function choosePayload(run, pristine, fiddled) {
  const patch = diffWeights(pristine, fiddled);
  const asPatch = { v: 1, run, p: patch };
  const asFull = { v: 1, run, w: fiddled };
  return JSON.stringify(asPatch).length <= JSON.stringify(asFull).length ? asPatch : asFull;
}
{
  const pristine = nca;
  const fiddled = clone(nca);
  fiddled.fire_rate = 0.9123;
  const payload = choosePayload("dummy_0058", pristine, fiddled);
  check("one-scalar edit on dummy_0058 ships as a patch (p, not w)",
    payload.p !== undefined && payload.w === undefined,
    `keys=${Object.keys(payload).join(",")}`);
  check("that patch is the single fire_rate entry",
    payload.p.length === 1 && payload.p[0][0] === "fire_rate" && payload.p[0][1] === 0.9123,
    JSON.stringify(payload.p));
  check("the patch payload is dramatically smaller than full weights",
    JSON.stringify(payload).length * 20 < JSON.stringify({ v: 1, run: "dummy_0058", w: fiddled }).length,
    `patch=${JSON.stringify(payload).length} full=${JSON.stringify({ v: 1, run: "x", w: fiddled }).length}`);

  // And the patch actually reconstructs the fiddled weights at the far end.
  check("the shipped patch reconstructs the fiddled weights",
    deepEqual(applyPatch(pristine, payload.p), fiddled));

  // A token built from it stays fragment-safe and round-trips.
  const tok = await encodeFiddle(payload);
  check("the patch payload tokenizes and round-trips",
    !/[+/=#]/.test(tok) && deepEqual(await decodeFiddle(tok), payload));

  // The other side of the rule: replacing everything must ship as `w`.
  const wholesale = clone(nca);
  wholesale.fc0_w = wholesale.fc0_w.map(r => r.map(v => v * 2));
  wholesale.fc1_w = wholesale.fc1_w.map(r => r.map(v => v * 2));
  const p2 = choosePayload("dummy_0058", pristine, wholesale);
  check("a wholesale re-scale ships as full weights (w, not p)",
    p2.w !== undefined && p2.p === undefined,
    `keys=${Object.keys(p2).join(",")}`);
}

// ---------------------------------------------------------------------------
// 11. A fiddled synthetic Lenia model still builds and steps in the real
//     engine — the diff/patch pipeline must not damage the schema.
// ---------------------------------------------------------------------------
{
  const { LeniaCA } = await import("../lenia_engine.js?v=nostencil");

  const pristine = makeLenia();
  const edited = makeLenia();
  edited.mu[0] = 0.22;               // scalar in an array
  edited.kernels[0][1][1] = 0.95;    // nested kernel cell
  edited.dt = 0.2;                   // top-level scalar

  const payload = choosePayload("synthlenia", pristine, edited);
  check("lenia one-off edits ship as a patch", payload.p !== undefined && payload.p.length === 3,
    JSON.stringify((payload.p || []).map(e => e[0])));

  const token = await encodeFiddle(payload);
  const decoded = await decodeFiddle(token);
  const fiddled = applyPatch(pristine, decoded.p);
  check("lenia edit survives diff -> encode -> decode -> apply",
    deepEqual(fiddled, edited));
  check("the shared base was not mutated anywhere in that pipeline",
    deepEqual(pristine, makeLenia()));

  let ca = null, buildErr = "";
  try {
    ca = new LeniaCA(fiddled, fiddled.size ?? 64);
  } catch (e) { buildErr = e.message; }
  check("new LeniaCA(fiddled, 16) constructs", ca !== null, buildErr);

  if (ca) {
    check("engine picked up the fiddled geometry",
      ca.width === 16 && ca.height === 16 && ca.channel_n === 1 && ca.C === 1,
      `${ca.width}x${ca.height} C=${ca.C}`);
    check("engine picked up the fiddled kernel cell",
      Math.abs(ca._kern[0][4] - 0.95) < 1e-6, `got ${ca._kern[0][4]} (Float32)`);
    check("engine picked up the fiddled mu", ca.w.mu[0] === 0.22 && ca.w.dt === 0.2);

    let stepErr = "";
    try { for (let i = 0; i < 5; i++) ca.step(); } catch (e) { stepErr = e.message; }
    check("LeniaCA.step() runs on the fiddled weights", stepErr === "", stepErr);

    let finite = true;
    for (const v of ca.state) if (!Number.isFinite(v)) { finite = false; break; }
    check("state stays finite after stepping fiddled physics", finite);

    // The same object must also drive the shared engine API the modal uses.
    let apiErr = "";
    try {
      ca.reset(true); ca.resetTrained(); ca.damage(8, 8, 3);
      const rgba = ca.readRGBA(new Uint8ClampedArray(16 * 16 * 4));
      if (!rgba || rgba.length !== 16 * 16 * 4) apiErr = "readRGBA returned the wrong shape";
    } catch (e) { apiErr = e.message; }
    check("fiddled lenia supports reset/resetTrained/damage/readRGBA", apiErr === "", apiErr);
  }

  // Engine-selection rule (mirrored from lenia.js) over both fixtures.
  const pick = (w) => w.kind === "lenia" ? "lenia"
    : (w.fc0_w || (w.channel_n !== undefined && w.hidden_n !== undefined)) ? "nca"
    : "unsupported";
  check("engine selection: fiddled lenia -> lenia, dummy_0058 -> nca, junk -> unsupported",
    pick(fiddled) === "lenia" && pick(nca) === "nca" && pick({ hello: 1 }) === "unsupported");
}

// ---------------------------------------------------------------------------
// 12. applyPatch path mechanics used by the URL loader (numeric segments,
//     out-of-order entries, later entries winning).
// ---------------------------------------------------------------------------
{
  const base = { a: [1, 2, 3], b: { c: 4 } };
  const out = applyPatch(base, [["a.1", 20], ["b.c", 40], ["a.1", 21]]);
  check("later patch entries win over earlier ones", out.a[1] === 21 && out.b.c === 40,
    JSON.stringify(out));
  check("numeric segments index arrays, keeping them arrays",
    Array.isArray(out.a) && out.a.length === 3);
  check("applyPatch of a diff of two unrelated weights reproduces the target",
    deepEqual(applyPatch(makeLenia(), diffWeights(makeLenia(), nca)), nca));
}

console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);

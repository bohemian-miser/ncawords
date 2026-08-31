// Fiddle — per-run weights editor for the run gallery.
//
// A card's "Fiddle" button opens one modal that puts three things side by
// side for a single training run:
//   1. the same metadata the detail modal shows (args/tags/loss/target),
//   2. the ENTIRE weights.json as an editable collapsible tree, and
//   3. a live preview stepping the real physics from the *edited* weights.
//
// The point is hand-perturbing a trained model — scale a kernel, retype a
// growth mu, copy one run's kernel bank onto another's — and immediately
// watching what the physics does. Edits never touch the bucket: a fiddled
// model is shared by URL instead (`#fiddle=<token>`), carrying either a
// diff against the trained weights or, when that is bigger, the whole
// object, gzip+base64url packed.
//
// Constraints this module is written against:
//   * ZERO top-level DOM/window access — docs/test/*.mjs imports it under
//     plain Node to unit-test the pure helpers, so every document/location
//     touch happens lazily inside a function.
//   * Engine selection must match lenia.js's live widget exactly, since a
//     fiddled export is still the same weights.json schema.
//   * Trees get big (a 'full' Lenia variant is 108 x 15x15 kernels, an NCA
//     fc0_w is 80 x 48), so children render only when a node is expanded.

import { LeniaCA } from './lenia_engine.js?v=nostencil';
import { createCA } from './nca.js';

// ---------------------------------------------------------------------
// Pure helpers (unit-tested in Node; no DOM anywhere below this line
// until the "DOM" banner).
// ---------------------------------------------------------------------

// Prototype-pollution guard. JSON.parse happily produces an *own* property
// literally named "__proto__", and assigning that key onto a plain object
// re-points its prototype — so both the cloner and the patch applier drop
// these segments rather than write them.
const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

// Path strings are dot-joined keys/indices ('kernels.3.2.5'). Weights
// schemas have no dots in their keys, so the join is unambiguous.
const PATH_SEP = '.';

// How far past the end of an existing array a patch may write. A path
// segment addressed at an array must be a canonical index inside this
// window: without it a token can set 'kernels.length' outright, or make
// applyPatch auto-vivify a billion-slot sparse array from 'kernels.999999999',
// and every later walker over that structure hangs the tab.
const MAX_PATCH_GROWTH = 1024;

function isPlainObject(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function deepClone(v) {
    if (Array.isArray(v)) return v.map(deepClone);
    if (isPlainObject(v)) {
        const out = {};
        for (const k of Object.keys(v)) {
            if (FORBIDDEN_KEYS.has(k)) continue;
            out[k] = deepClone(v[k]);
        }
        return out;
    }
    return v;
}

function diffInto(base, edited, segs, out) {
    if (Array.isArray(base) && Array.isArray(edited)) {
        // A length change can't be expressed as per-index sets (the extra
        // indices would linger), so the whole array is replaced.
        if (base.length !== edited.length) {
            out.push([segs.join(PATH_SEP), deepClone(edited)]);
            return;
        }
        for (let i = 0; i < base.length; i++) {
            diffInto(base[i], edited[i], segs.concat(String(i)), out);
        }
        return;
    }
    if (isPlainObject(base) && isPlainObject(edited)) {
        const bk = Object.keys(base), ek = Object.keys(edited);
        // Same reasoning as arrays: a removed key can't be patched away, so
        // any key-set difference replaces the object wholesale.
        const sameKeys = bk.length === ek.length
            && ek.every(k => Object.prototype.hasOwnProperty.call(base, k));
        if (!sameKeys) {
            out.push([segs.join(PATH_SEP), deepClone(edited)]);
            return;
        }
        for (const k of ek) diffInto(base[k], edited[k], segs.concat(k), out);
        return;
    }
    if (!Object.is(base, edited)) out.push([segs.join(PATH_SEP), deepClone(edited)]);
}

// Minimal patch taking `base` to `edited`: an array of [pathString, value].
// A path of '' means "replace the whole object". Neither input is mutated.
export function diffWeights(base, edited) {
    const out = [];
    diffInto(base, edited, [], out);
    return out;
}

// A write landing on an array has to be a real index near the array's own
// end — 'length', a non-canonical index ('01', '1e3') and a wildly
// out-of-range one all build a structure no weights.json could hold.
function checkArraySeg(cur, seg) {
    if (!Array.isArray(cur)) return;
    if (!/^(0|[1-9]\d*)$/.test(seg)) {
        throw new Error(`refusing fiddle patch segment '${seg}' on an array (not an index)`);
    }
    if (Number(seg) > cur.length + MAX_PATCH_GROWTH) {
        throw new Error(`fiddle patch index ${seg} is past the end of a `
            + `${cur.length}-entry array`);
    }
}

// Applies a diffWeights patch to a deep clone of `base` and returns the
// clone; `base` is never mutated. Throws on malformed entries and on any
// path segment that could reach an object's prototype.
export function applyPatch(base, patch) {
    if (!Array.isArray(patch)) {
        throw new Error('fiddle patch must be an array of [path, value] pairs');
    }
    let out = deepClone(base);
    for (const entry of patch) {
        if (!Array.isArray(entry) || entry.length < 2) {
            throw new Error('bad fiddle patch entry: ' + JSON.stringify(entry));
        }
        const raw = String(entry[0]);
        const value = entry[1];
        const segs = raw.length ? raw.split(PATH_SEP) : [];
        for (const s of segs) {
            if (FORBIDDEN_KEYS.has(s)) {
                throw new Error(`refusing fiddle patch path segment '${s}'`);
            }
        }
        if (!segs.length) { out = deepClone(value); continue; }
        // A deep path onto a scalar root has nowhere to land; start a fresh
        // container rather than silently dropping the write.
        if (out === null || typeof out !== 'object') out = /^\d+$/.test(segs[0]) ? [] : {};
        let cur = out;
        for (let i = 0; i < segs.length - 1; i++) {
            checkArraySeg(cur, segs[i]);
            let next = cur[segs[i]];
            if (next === null || typeof next !== 'object') {
                next = /^\d+$/.test(segs[i + 1]) ? [] : {};
                cur[segs[i]] = next;
            }
            cur = next;
        }
        checkArraySeg(cur, segs[segs.length - 1]);
        cur[segs[segs.length - 1]] = deepClone(value);
    }
    return out;
}

// base64url (RFC 4648 §5): the -_ alphabet, padding stripped, so a token is
// safe inside a URL fragment. Hand-rolled rather than btoa/Buffer so the
// same code runs in the browser and under bare Node.
const B64URL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
const B64URL_REV = (() => {
    const m = Object.create(null);
    for (let i = 0; i < B64URL.length; i++) m[B64URL[i]] = i;
    m['+'] = 62; m['/'] = 63;   // tolerate a standard-alphabet token
    return m;
})();

function b64urlEncode(bytes) {
    let out = '';
    for (let i = 0; i < bytes.length; i += 3) {
        const b0 = bytes[i], b1 = bytes[i + 1], b2 = bytes[i + 2];
        out += B64URL[b0 >> 2];
        out += B64URL[((b0 & 3) << 4) | ((b1 === undefined ? 0 : b1) >> 4)];
        if (b1 === undefined) break;
        out += B64URL[((b1 & 15) << 2) | ((b2 === undefined ? 0 : b2) >> 6)];
        if (b2 === undefined) break;
        out += B64URL[b2 & 63];
    }
    return out;
}

function b64urlDecode(str) {
    const s = String(str).replace(/=+$/, '');
    const n = s.length;
    const out = new Uint8Array(Math.floor(n * 3 / 4));
    let o = 0, acc = 0, bits = 0;
    for (let i = 0; i < n; i++) {
        const v = B64URL_REV[s[i]];
        if (v === undefined) throw new Error(`bad base64url character '${s[i]}' at ${i}`);
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[o++] = (acc >> bits) & 0xff;
        }
    }
    return out.subarray(0, o);
}

// The writer promises must be caught, not merely fired: when the transform
// errors (a truncated or non-gzip token), write()/close() reject too, and an
// unawaited rejection is an unhandled-rejection crash. The one failure the
// caller should see comes off the readable side below.
async function gzipBytes(bytes) {
    const cs = new CompressionStream('gzip');
    const writer = cs.writable.getWriter();
    writer.write(bytes).catch(() => {});
    writer.close().catch(() => {});
    return new Uint8Array(await new Response(cs.readable).arrayBuffer());
}

// Inflated-token ceiling. Gzip packs weights JSON ~800:1, so buffering the
// whole stream would let a ~300 KB link expand to hundreds of MB (and peak
// near a GB once the decoded string and the parsed object exist beside it).
// The largest honest payload is a full-weights token of a few hundred KB.
const MAX_TOKEN_BYTES = 32 * 1024 * 1024;

function tooBigError() {
    const e = new Error(`fiddle token expands to more than `
        + `${Math.round(MAX_TOKEN_BYTES / (1024 * 1024))} MB`);
    e.tokenTooBig = true;   // decodeFiddle reports this as-is, not as bad gzip
    return e;
}

async function gunzipBytes(bytes) {
    const ds = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    writer.write(bytes).catch(() => {});
    writer.close().catch(() => {});
    // Read chunk by chunk instead of `new Response(...).arrayBuffer()` so the
    // running total can be checked before the bomb is fully materialised.
    const reader = ds.readable.getReader();
    const chunks = [];
    let total = 0;
    try {
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            total += value.length;
            if (total > MAX_TOKEN_BYTES) {
                reader.cancel().catch(() => {});
                throw tooBigError();
            }
            chunks.push(value);
        }
    } finally {
        try { reader.releaseLock(); } catch (e) { /* already released */ }
    }
    const out = new Uint8Array(total);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
}

// Packs a payload object into a URL-fragment token. Gzip when the platform
// has CompressionStream (every current browser and Node >= 18); the 'raw:'
// form is the honest fallback rather than a silent size blow-up.
export async function encodeFiddle(obj) {
    const bytes = new TextEncoder().encode(JSON.stringify(obj));
    if (typeof CompressionStream !== 'undefined') {
        try {
            return 'gz:' + b64urlEncode(await gzipBytes(bytes));
        } catch (e) {
            console.error('fiddle: gzip failed, falling back to raw token', e);
        }
    }
    return 'raw:' + b64urlEncode(bytes);
}

// Inverse of encodeFiddle; accepts either prefix and throws a message that
// names what was wrong (tokens arrive from user-pasted URLs).
export async function decodeFiddle(token) {
    if (typeof token !== 'string' || !token.length) {
        throw new Error('fiddle token is empty');
    }
    let t = token.trim();
    if (t.startsWith('#')) t = t.slice(1);
    if (t.startsWith('fiddle=')) t = t.slice('fiddle='.length);

    let bytes;
    if (t.startsWith('gz:')) {
        if (typeof DecompressionStream === 'undefined') {
            throw new Error('this browser cannot read gzipped fiddle tokens (no DecompressionStream)');
        }
        const packed = b64urlDecode(t.slice(3));
        try {
            bytes = await gunzipBytes(packed);
        } catch (e) {
            if (e && e.tokenTooBig) throw e;
            // The stream layer reports a truncated/garbage member as a bare
            // TypeError with no message; say what actually went wrong.
            throw new Error('fiddle token is not a readable gzip stream'
                + (e && e.message ? ': ' + e.message : ''));
        }
    } else if (t.startsWith('raw:')) {
        // Unamplified, but still worth capping before allocating the buffer.
        const body = t.slice(4);
        if (body.length > MAX_TOKEN_BYTES / 3 * 4) throw tooBigError();
        bytes = b64urlDecode(body);
    } else {
        throw new Error("unrecognized fiddle token (expected a 'gz:' or 'raw:' prefix)");
    }
    const text = new TextDecoder().decode(bytes);
    try {
        return JSON.parse(text);
    } catch (e) {
        throw new Error('fiddle token does not contain valid JSON: ' + e.message);
    }
}

// ---- shape / traversal utilities used by both the tree and the UI ----

// Dimensions of a (possibly nested) array, plus whether the nesting is
// ragged — a ragged array can't honestly be summarised as "A x B", so the
// caller falls back to an item count.
function arrayDims(a) {
    const dims = [];
    let cur = a;
    while (Array.isArray(cur)) {
        dims.push(cur.length);
        if (cur.length && Array.isArray(cur[0])) {
            const n0 = cur[0].length;
            for (let i = 1; i < cur.length; i++) {
                if (!Array.isArray(cur[i]) || cur[i].length !== n0) {
                    return { dims, ragged: true, leaf: undefined };
                }
            }
        }
        cur = cur[0];
    }
    return { dims, ragged: false, leaf: cur };
}

function leafWord(leaf, plural) {
    if (typeof leaf === 'number') return plural ? 'floats' : 'float';
    if (typeof leaf === 'string') return plural ? 'strings' : 'string';
    if (typeof leaf === 'boolean') return plural ? 'bools' : 'bool';
    if (leaf === undefined || leaf === null) return 'items';
    return plural ? 'objects' : 'object';
}

// One-line shape label for a collapsed node, e.g. "12 x 15x15 floats",
// "80 x 48", "9 keys".
function shapeSummary(v) {
    if (Array.isArray(v)) {
        const { dims, ragged, leaf } = arrayDims(v);
        if (ragged) return `${v.length} items (ragged)`;
        if (dims.length === 1) return `${dims[0]} ${leafWord(leaf, true)}`;
        if (dims.length === 2) return `${dims[0]} x ${dims[1]}`;
        if (dims.length === 3) return `${dims[0]} x ${dims[1]}x${dims[2]} ${leafWord(leaf, true)}`;
        return dims.join(' x ') + ' ' + leafWord(leaf, true);
    }
    if (isPlainObject(v)) {
        const keys = Object.keys(v);
        return `${keys.length} key${keys.length === 1 ? '' : 's'}`;
    }
    return typeof v;
}

// Compact shape used in the paste mismatch confirm(), where the two shapes
// have to be comparable at a glance.
function shapeOf(v) {
    if (Array.isArray(v)) {
        const { dims, ragged, leaf } = arrayDims(v);
        if (ragged) return `ragged array[${v.length}]`;
        return dims.join('x') + ' ' + leafWord(leaf, true);
    }
    if (v === null) return 'null';
    if (typeof v === 'object') return `object{${Object.keys(v).join(',')}}`;
    return typeof v;
}

// Leaf count, capped so counting a 108x15x15 kernel bank costs ~26 visits.
function countLeaves(v, cap = 26) {
    let n = 0;
    const walk = (x) => {
        if (n >= cap) return;
        if (Array.isArray(x)) {
            for (const e of x) { walk(e); if (n >= cap) return; }
        } else if (isPlainObject(x)) {
            for (const k of Object.keys(x)) { walk(x[k]); if (n >= cap) return; }
        } else n++;
    };
    walk(v);
    return n;
}

// Same visit budget as countLeaves, for the same reason but harder: this one
// runs on every container row, and `every` over an array walks 0..length-1
// even when the array is sparse — a hostile token that inflates a `length`
// would otherwise freeze the tab for minutes. Over budget means "no [scale]
// button", which is the right answer for a node that large anyway.
const NUMERIC_SUBTREE_BUDGET = 200000;

function isNumericSubtree(v, budget = { n: NUMERIC_SUBTREE_BUDGET }) {
    if (budget.n <= 0) return false;
    if (typeof v === 'number') { budget.n--; return true; }
    if (Array.isArray(v)) {
        if (v.length === 0 || v.length > budget.n) return false;
        for (let i = 0; i < v.length; i++) {
            if (!isNumericSubtree(v[i], budget)) return false;
        }
        return true;
    }
    if (isPlainObject(v)) {
        const keys = Object.keys(v);
        if (keys.length === 0 || keys.length > budget.n) return false;
        for (const k of keys) {
            if (!isNumericSubtree(v[k], budget)) return false;
        }
        return true;
    }
    return false;
}

function scaleNumbers(v, factor) {
    if (typeof v === 'number') return v * factor;
    if (Array.isArray(v)) return v.map(e => scaleNumbers(e, factor));
    if (isPlainObject(v)) {
        const out = {};
        for (const k of Object.keys(v)) out[k] = scaleNumbers(v[k], factor);
        return out;
    }
    return v;
}

function getAt(root, segs) {
    let cur = root;
    for (const s of segs) {
        if (cur === null || cur === undefined) return undefined;
        cur = cur[s];
    }
    return cur;
}

// ---------------------------------------------------------------------
// DOM — everything below runs only from an exported entry point, never at
// import time.
// ---------------------------------------------------------------------

const DEFAULT_BUCKET = 'recipe-lanes-nca-jobs';
const MAX_CHILD_ROWS = 500;     // rows built per expanded container
const AUTO_OPEN_LEAVES = 25;    // collapse anything bigger by default
const URL_WARN_LEN = 30000;     // share-channel-hostile URL length

// Preview build limits. The engine constructors happily allocate whatever an
// edited (or URL-supplied) weights.json asks for — `new LeniaCA` takes
// C*size*size floats up front and one step is O(convs * size^2 * ks^2) — so
// a size of 20000 or a ks of 2001 wedges the tab long before any later check
// runs. Real runs sit far inside these: grids top out around 216, ks is 15,
// and the biggest 'full' bank is 108 kernels (~1e8 per step at size 64).
const MAX_GRID = 512;
const MAX_KS = 63;
const MAX_C = 32;
const MAX_K = 512;
const MAX_HIDDEN = 4096;
const MAX_STEP_WORK = 4e8;      // multiply-adds per Lenia step

// ctx is injected by lenia.js (initFiddle) — kept behind accessors so this
// module never reaches into the gallery's module state directly.
let ctx = { getMethod: null, getTracker: null, bucketBase: '', pauseLive: null, resumeLive: null };

// The one thing that deliberately outlives a modal: values are moved
// BETWEEN runs by copying in one and pasting in another, so the clipboard
// survives close/reopen.
let clipboard = null;   // { sourceRun, path, value }

let ui = null;          // built-once modal DOM references
let session = null;     // per-open state (run, weights, engine, timers)
// Whether the tree names the axes of the learned matrices (see the semantic
// views below). Read lazily so importing this module touches no browser API.
let grouping = (() => {
    try { return localStorage.getItem('fiddle_group') !== '0'; } catch (e) { return true; }
})();
let hashListenerInstalled = false;
let selfSetToken = null;   // hash we wrote ourselves; ignore its hashchange

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function bucketBase() {
    if (ctx.bucketBase) return ctx.bucketBase;
    const cfg = (typeof window !== 'undefined' && window.NCA_CONFIG) || null;
    return `https://storage.googleapis.com/${(cfg && cfg.bucket) || DEFAULT_BUCKET}/`;
}

function runNameOf(id) {
    return String(id).startsWith('lenia_') ? String(id).slice('lenia_'.length) : String(id);
}

// Called once by lenia.js near the top of the module. Only stores the
// accessors — no DOM is built until the first open.
export function initFiddle(context) {
    ctx = Object.assign({
        getMethod: null, getTracker: null, bucketBase: '',
        pauseLive: null, resumeLive: null
    }, context || {});
}

// ---- styles ----------------------------------------------------------

const FIDDLE_CSS = `
#fiddle-modal .modal-content { max-width:920px; width:94%; background:#1e1e1e; border-radius:8px;
  padding:30px 30px 40px; align-items:stretch; margin:0 auto; }
#fiddle-modal h2 { color:#4db8ff; margin:0 0 4px; }
#fiddle-modal h4 { color:#4db8ff; margin:14px 0 6px; font-size:0.9em; text-transform:uppercase; letter-spacing:0.04em; }
#fd-sub { color:#888; font-size:0.82em; margin-bottom:10px; }
#fd-badge { display:none; background:#2a3f52; color:#8fd0ff; border:1px solid #3a6a8a;
  border-radius:4px; padding:3px 8px; font-size:0.72em; font-weight:bold; letter-spacing:0.03em; }
#fd-dirty { display:none; color:#ffb454; font-size:0.75em; font-weight:bold; letter-spacing:0.03em; }
#fd-flags { display:flex; gap:8px; align-items:center; min-height:20px; margin-bottom:8px; }
.fd-toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:10px 0;
  border-top:1px solid #333; border-bottom:1px solid #333; }
.fd-btn { padding:4px 10px; background:#444; color:#fff; border:none; border-radius:4px;
  cursor:pointer; font-size:0.8em; font-weight:bold; }
.fd-btn:hover { background:#555; }
.fd-btn.primary { background:#2d6a4f; }
.fd-btn.primary:hover { background:#37835f; }
#fd-savebox { display:none; margin-top:10px; padding:10px; background:#161616; border:1px solid #333;
  border-radius:4px; font-size:0.78em; word-break:break-all; }
#fd-savebox a { color:#4db8ff; }
#fd-save-warn { color:#ffb454; margin-top:6px; }
#fd-top { display:flex; gap:20px; flex-wrap:wrap; margin-top:14px; }
#fd-info { flex:1 1 380px; min-width:300px; }
#fd-preview { flex:0 0 300px; }
#fd-args { width:100%; border-collapse:collapse; font-size:0.8em; }
#fd-args td { padding:2px 8px 2px 0; border-bottom:1px solid #333; vertical-align:top; }
#fd-args td:first-child { color:#888; white-space:nowrap; }
.fd-meta { font-size:0.82em; color:#ccc; }
.fd-meta a { color:#4db8ff; }
.fd-target-wrap { width:110px; }
.fd-target-wrap .img-container { height:110px; }
.fd-canvas-wrap { display:inline-block; background:#000; border:2px solid #444; border-radius:4px; }
#fd-canvas { display:block; image-rendering:pixelated; cursor:crosshair; width:256px; height:256px; }
.fd-ctl { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-top:8px; font-size:0.78em; color:#ccc; }
#fd-live-status { font-size:0.76em; color:#888; margin-top:6px; min-height:1.1em; }
.fd-tree { margin-top:6px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:0.78em; background:#161616; border:1px solid #333; border-radius:4px;
  padding:8px; max-height:420px; overflow:auto; }
.fd-node { padding-left:0; }
.fd-children { margin-left:14px; border-left:1px dotted #3a3a3a; padding-left:8px; }
.fd-row { display:flex; align-items:center; gap:6px; padding:1px 0; white-space:nowrap; }
.fd-row:hover { background:#202020; }
.fd-tog { width:14px; flex:0 0 14px; background:none; border:none; color:#888; cursor:pointer;
  padding:0; font-size:0.9em; line-height:1; }
.fd-tog:hover { color:#4db8ff; }
.fd-key { color:#8fd0ff; }
.fd-shape { color:#888; }
.fd-null { color:#666; font-style:italic; }
.fd-input { background:#222; color:#eee; border:1px solid #444; border-radius:3px;
  padding:1px 4px; font-family:inherit; font-size:1em; width:9em; }
.fd-input.text { width:14em; }
.fd-input:focus { outline:none; border-color:#4db8ff; }
.fd-input.bad { border-color:#c55; background:#301c1c; }
.fd-tools { display:none; gap:3px; margin-left:6px; }
.fd-row:hover .fd-tools { display:inline-flex; }
.fd-tool { background:#2f2f2f; color:#aaa; border:1px solid #444; border-radius:3px;
  font-size:0.85em; padding:0 5px; cursor:pointer; font-family:inherit; }
.fd-tool:hover { color:#4db8ff; border-color:#4db8ff; }
.fd-more { color:#ffb454; padding:2px 0; }
.fd-view-key { color:#c9a0ff; }
.fd-role { color:#7fbf7f; }
#fd-tree-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  font-size:0.78em; color:#888; margin-top:6px; }
#fd-tree-head label { cursor:pointer; display:inline-flex; align-items:center; gap:4px; }
.fd-dialog { position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:1200;
  display:flex; align-items:center; justify-content:center; }
.fd-dialog-box { background:#1e1e1e; border:1px solid #444; border-radius:8px; padding:18px;
  width:min(720px,92vw); }
.fd-dialog-box h4 { margin-top:0 !important; }
.fd-dialog-box textarea { width:100%; height:320px; background:#111; color:#ddd; border:1px solid #444;
  border-radius:4px; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:0.8em; padding:8px; }
.fd-dialog-err { color:#ff8a8a; font-size:0.8em; min-height:1.2em; margin:6px 0; }
.fd-dialog-btns { display:flex; gap:8px; justify-content:flex-end; }
#fd-tree-msg { color:#888; font-size:0.85em; }
`;

// ---- modal construction ---------------------------------------------

function buildModal() {
    if (ui) return ui;

    const style = document.createElement('style');
    style.id = 'fiddle-styles';
    style.textContent = FIDDLE_CSS;
    document.head.appendChild(style);

    const overlay = document.createElement('div');
    overlay.id = 'fiddle-modal';
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `
      <div class="modal-content" id="fd-content">
        <span class="modal-close" id="fd-close" title="Close">&times;</span>
        <h2 id="fd-title"></h2>
        <div id="fd-sub"></div>
        <div id="fd-flags">
          <span id="fd-badge">fiddled — loaded from URL</span>
          <span id="fd-dirty">edited — applying…</span>
        </div>
        <div class="fd-toolbar">
          <button class="fd-btn" id="fd-copy-all" title="Copy the whole weights object to the fiddle clipboard">Copy all</button>
          <button class="fd-btn" id="fd-paste-all" title="Replace the whole weights object">Paste all</button>
          <button class="fd-btn" id="fd-reset-all" title="Discard every edit">Reset to trained</button>
          <button class="fd-btn" id="fd-download" title="Save the fiddled weights.json locally">Download weights.json</button>
          <button class="fd-btn primary" id="fd-save" title="Pack the edits into a shareable #fiddle= URL">Save &rarr; URL</button>
        </div>
        <div id="fd-savebox"></div>

        <div id="fd-top">
          <div id="fd-info"></div>
          <div id="fd-preview">
            <h4>Live preview</h4>
            <div class="fd-canvas-wrap"><canvas id="fd-canvas" width="64" height="64"></canvas></div>
            <div class="fd-ctl">
              <button class="fd-btn" id="fd-play" title="Play/pause">&#9654;</button>
              <button class="fd-btn" id="fd-seed" title="Reset the way training started">Seed</button>
              <button class="fd-btn" id="fd-noise" title="Reset to random noise">Noise</button>
              <button class="fd-btn" id="fd-clear" title="Zero every channel">Clear</button>
              <label>speed <input type="range" id="fd-speed" min="0.1" max="10" step="0.1" value="1" style="width:70px;vertical-align:middle;"></label>
            </div>
            <div class="fd-ctl">
              <button class="fd-btn" id="fd-apply" title="Rebuild the engine from scratch — edits already apply live, this also resets the grid">Rebuild</button>
              <button class="fd-btn" id="fd-revert" title="Restore the trained weights">RESET</button>
            </div>
            <div id="fd-live-status"></div>
          </div>
        </div>

        <h4>Structure — weights.json</h4>
        <div id="fd-tree-head">
          <label title="Name the axes of the learned matrices: an NCA's fc0_w columns split into state / sobel x / sobel y per channel, fc1_w rows into output channels, a Lenia kernel bank into source→target pairs">
            <input type="checkbox" id="fd-group" checked> group weights by role
          </label>
        </div>
        <div id="fd-tree-msg"></div>
        <div class="fd-tree" id="fd-tree"></div>
      </div>
    `;
    document.body.appendChild(overlay);

    const $ = (id) => overlay.querySelector('#' + id);
    ui = {
        overlay,
        content: $('fd-content'),
        title: $('fd-title'),
        sub: $('fd-sub'),
        badge: $('fd-badge'),
        dirty: $('fd-dirty'),
        info: $('fd-info'),
        saveBox: $('fd-savebox'),
        canvas: $('fd-canvas'),
        playBtn: $('fd-play'),
        speed: $('fd-speed'),
        applyBtn: $('fd-apply'),
        liveStatus: $('fd-live-status'),
        seedBtn: $('fd-seed'),
        noiseBtn: $('fd-noise'),
        clearBtn: $('fd-clear'),
        treeMsg: $('fd-tree-msg'),
        groupChk: $('fd-group'),
        tree: $('fd-tree')
    };
    ui.groupChk.checked = grouping;
    ui.groupChk.onchange = () => {
        grouping = ui.groupChk.checked;
        try { localStorage.setItem('fiddle_group', grouping ? '1' : '0'); } catch (e) { /* private mode */ }
        renderTree();
    };
    ui.canvasCtx = ui.canvas.getContext('2d');

    $('fd-close').onclick = closeFiddleModal;
    // Backdrop click closes; a click that started inside the panel must not.
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) closeFiddleModal();
    });
    ui.content.addEventListener('click', (e) => e.stopPropagation());

    $('fd-copy-all').onclick = () => copyNode([]);
    $('fd-paste-all').onclick = () => pasteNode([], null);
    $('fd-reset-all').onclick = resetToTrained;
    $('fd-download').onclick = downloadWeights;
    $('fd-save').onclick = () => saveToUrl().catch(e => {
        console.error('fiddle: save failed', e);
        showSaveBox(null, 'could not build a share URL: ' + e.message);
    });

    ui.playBtn.onclick = togglePlay;
    ui.seedBtn.onclick = () => previewReset('seed');
    ui.noiseBtn.onclick = () => previewReset('noise');
    ui.clearBtn.onclick = () => previewReset('clear');
    ui.applyBtn.onclick = () => rebuildEngine({ keepPaused: false });
    $('fd-revert').onclick = resetToTrained;

    ui.canvas.addEventListener('mousedown', (e) => { ui.damaging = true; damageAt(e); });
    ui.canvas.addEventListener('mousemove', (e) => { if (ui.damaging) damageAt(e); });
    ui.canvas.addEventListener('mouseup', () => { ui.damaging = false; });
    ui.canvas.addEventListener('mouseleave', () => { ui.damaging = false; });

    // The page already has an Escape handler for #lenia-modal; this one only
    // fires while the fiddle overlay is the visible modal.
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape' || !ui || ui.overlay.style.display !== 'block') return;
        if (ui.overlay.querySelector('.fd-dialog')) return;   // sub-dialog owns Escape
        closeFiddleModal();
    });

    return ui;
}

// ---- session lifecycle ----------------------------------------------

function markDirty() {
    if (!session) return;
    session.dirty = true;
    if (ui) {
        // Not '' — the stylesheet's own `#fd-dirty { display:none }` is what
        // an empty inline value falls back to, which would hide it forever.
        ui.dirty.style.display = 'inline-block';
        ui.dirty.innerText = 'edited — applying…';
        ui.saveBox.style.display = 'none';   // any shown URL is now stale
    }
    scheduleLiveApply();
}

function clearDirty() {
    if (session) session.dirty = false;
    if (ui) ui.dirty.style.display = 'none';
}

// ---- live apply ------------------------------------------------------
//
// Edits reach the running simulation on their own: the point of fiddling is
// watching the SAME grown pattern react to a changed rule, so a weight edit
// swaps the engine's parameters in place and never resets the grid. Only an
// edit the engine is allocated around (channel count, grid size, a Lenia
// grid) forces the rebuild that does reset it.

const LIVE_APPLY_DELAY = 160;   // ms of quiet before the swap — inputs fire per keystroke

function scheduleLiveApply() {
    if (!session) return;
    const s = session;
    if (s.applyTimer) clearTimeout(s.applyTimer);
    s.applyTimer = setTimeout(() => {
        s.applyTimer = null;
        if (session === s) liveApply();
    }, LIVE_APPLY_DELAY);
}

function liveApply() {
    if (!ui || !session || !session.fiddled) return;
    const w = session.fiddled;
    if (!session.ca) { rebuildEngine({ keepPaused: false }); return; }

    // Same bound the rebuild path enforces: a hand-typed ks or grid can make
    // one step() effectively never return, and setWeights would install it.
    const rangeErr = weightsOutOfRange(w);
    if (rangeErr) {
        ui.dirty.innerText = 'edited — not applied';
        liveStatus('not applied — ' + rangeErr);
        return;
    }

    let swapped = false;
    try {
        swapped = !!(session.ca.setWeights && session.ca.setWeights(w));
    } catch (e) {
        // A throw part-way through a swap leaves the engine inconsistent
        // (half the new matrices, half the old), so rebuild rather than step.
        console.error('fiddle: live weight swap failed', e);
        rebuildEngine({ keepPaused: false });
        return;
    }
    if (!swapped) {
        // Structural change — the engine is sized around what changed. Leave
        // a failed rebuild's own diagnosis in place rather than papering over
        // it with the cheerier "rebuilt" line.
        rebuildEngine({ keepPaused: false });
        if (session && session.ca) liveStatus('structure changed — engine rebuilt, grid reset');
        return;
    }
    clearDirty();
    if (session.paused) drawPreview();
    liveStatus(`applied live (${session.engine}) — grid kept, physics updated`);
}

function liveStatus(text) {
    if (ui) ui.liveStatus.innerText = text;
}

function stopLoop() {
    if (session && session.timer) {
        clearTimeout(session.timer);
        session.timer = null;
    }
    if (ui) ui.playBtn.innerHTML = '&#9654;';
}

// Dropping the JS reference does NOT release a WebGL2 context: the browser's
// live-context cap is enforced immediately and GC is not prompt, so an APPLY
// loop would silently evict the gallery cards' own contexts. Same cleanup
// nca_viewer.js does before every re-create.
function disposeEngine() {
    const gl = session && session.ca && session.ca.gl;
    if (!gl) return;
    try {
        const ext = gl.getExtension('WEBGL_lose_context');
        if (ext) ext.loseContext();
    } catch (e) { /* already lost, or a CPU engine with a non-GL 'gl' */ }
}

// The [json] sub-dialog writes into whatever session is current when OK is
// pressed, so it must never outlive the session it was opened for.
function removeSubDialog() {
    if (!ui) return;
    const dlg = ui.overlay.querySelector('.fd-dialog');
    if (dlg) dlg.remove();
}

// Transport controls do nothing without an engine (togglePlay/previewReset
// both bail on `!session.ca`), so a failed build must show them as dead
// rather than leave them looking merely paused.
function setPreviewControls(enabled) {
    if (!ui) return;
    for (const b of [ui.playBtn, ui.seedBtn, ui.noiseBtn, ui.clearBtn]) b.disabled = !enabled;
}

function closeFiddleModal() {
    stopLoop();
    if (session) {
        if (session.applyTimer) { clearTimeout(session.applyTimer); session.applyTimer = null; }
        // Drop the engine: GLCA holds GPU textures, LeniaCA holds big typed
        // arrays, and neither should survive a closed modal.
        disposeEngine();
        session.ca = null;
        session = null;
    }
    if (ui) {
        ui.overlay.style.display = 'none';
        removeSubDialog();
    }
    // The card behind the overlay was paused on open; let it tick again.
    if (ctx.resumeLive) ctx.resumeLive();
}

// Opens the modal on nothing but a message — used by the URL loader when a
// token, its run, or its weights can't be resolved. Kills any running
// preview first so a stale engine isn't left ticking behind the message.
function showFatal(title, message) {
    buildModal();
    stopLoop();
    disposeEngine();
    removeSubDialog();   // its OK handler would write into a null session
    session = null;
    if (ctx.pauseLive) ctx.pauseLive();
    ui.overlay.style.display = 'block';
    ui.title.innerText = title;
    ui.sub.innerText = '';
    ui.badge.style.display = 'none';
    ui.dirty.style.display = 'none';
    ui.saveBox.style.display = 'none';
    ui.info.innerHTML = '';
    ui.tree.innerHTML = '';
    ui.treeMsg.innerText = message;
    liveStatus('');
}

// Opens the modal for one run and loads its trained weights from the
// bucket. `preload` (used by the URL path) supplies weights that are
// already in hand plus the pristine copy to diff against.
async function openSession(id, preload) {
    buildModal();
    const run = runNameOf(id);
    const method = (ctx.getMethod && ctx.getMethod(id)) || null;
    const tracker = (ctx.getTracker && ctx.getTracker(id)) || null;
    const dir = (method && method.dir) || (bucketBase() + run + '/');

    stopLoop();
    disposeEngine();
    removeSubDialog();   // a dialog from the previous run must not survive
    // Only one simulation should be stepping at a time, and the card behind
    // this overlay is completely hidden by it.
    if (ctx.pauseLive) ctx.pauseLive();
    const s = session = {
        id, run, dir, method, tracker,
        runJson: (tracker && tracker.runJson) || null,
        pristine: null, fiddled: null,
        // Whether `pristine` really is the run's trained weights. False when
        // a URL-loaded session had to stand in its own weights, in which case
        // nothing may be diffed against it.
        pristineIsReal: true,
        ca: null, engine: null, imgData: null,
        paused: false, accum: 0, timer: null, applyTimer: null,
        dirty: false, fromUrl: !!(preload && preload.fromUrl)
    };

    ui.overlay.style.display = 'block';
    ui.title.innerText = run;
    // 'inline-block', not '': #fd-badge's own rule is display:none, so an
    // empty inline value would leave the badge hidden on URL-loaded models.
    ui.badge.style.display = session.fromUrl ? 'inline-block' : 'none';
    clearDirty();
    ui.saveBox.style.display = 'none';
    ui.tree.innerHTML = '';
    ui.treeMsg.innerText = 'loading weights.json…';
    liveStatus('loading weights…');
    renderInfo();
    if (!session.runJson) fetchRunJsonBestEffort();

    if (preload) {
        s.pristine = preload.pristine;
        s.fiddled = preload.fiddled;
        s.pristineIsReal = preload.pristineIsReal !== false;
    } else {
        try {
            const res = await fetch(dir + 'weights.json?t=' + Date.now());
            // A second open (another card, or a hashchange) while this fetch
            // was in flight owns the modal now — drop this result on the floor.
            if (session !== s) return;
            if (!res.ok) {
                ui.treeMsg.innerText = 'weights not exported yet';
                liveStatus('weights not exported yet');
                return;
            }
            const w = await res.json();
            if (session !== s) return;
            s.pristine = w;
            s.fiddled = deepClone(w);
        } catch (e) {
            console.error('fiddle: weights fetch failed', e);
            if (session !== s) return;
            ui.treeMsg.innerText = 'weights not exported yet';
            liveStatus('weights not exported yet');
            return;
        }
    }

    ui.treeMsg.innerText = '';
    renderTree();
    rebuildEngine({ keepPaused: false });
    clearDirty();
}

// The gallery may not have fetched run.json yet (and a URL-loaded modal has
// no card at all), so pull it ourselves for the info panel. Best effort:
// a miss simply leaves the panel in its "not loaded" state.
async function fetchRunJsonBestEffort() {
    const s = session;
    try {
        const res = await fetch(s.dir + 'run.json?t=' + Date.now());
        if (!res.ok) return;
        const rj = await res.json();
        if (session !== s) return;   // modal was closed/reopened meanwhile
        session.runJson = rj;
        renderInfo();
    } catch (e) { /* offline or no run.json — the panel degrades on its own */ }
}

export function openFiddleModal(id) {
    openSession(id, null).catch(e => {
        console.error('fiddle: open failed', e);
        if (ui) ui.treeMsg.innerText = 'could not open this run: ' + e.message;
    });
}

// ---- info panel ------------------------------------------------------

function collectMeta() {
    const m = session.method;
    const rj = session.runJson;
    const args = (m && m.args) || (rj && rj.args) || null;
    const tags = (m && m.tags && m.tags.length ? m.tags : (rj && rj.tags)) || [];
    const desc = (m && m.desc) || (rj && rj.text) || '';
    let finalLoss = m ? m.finalLoss : null;
    let minLoss = m ? m.minLoss : null;
    if ((finalLoss === null || finalLoss === undefined) && rj && Array.isArray(rj.losses)) {
        const vals = rj.losses
            .map(p => Number(Array.isArray(p) ? p[1] : p))
            .filter(Number.isFinite);
        if (vals.length) { finalLoss = vals[vals.length - 1]; minLoss = Math.min(...vals); }
    }
    return { args, tags, desc, finalLoss, minLoss };
}

function fmtLoss(v) { return Number.isFinite(v) ? Number(v).toFixed(4) : '–'; }

function renderInfo() {
    if (!ui || !session) return;
    const { args, tags, desc, finalLoss, minLoss } = collectMeta();
    const dir = session.dir;
    const known = !!session.method;

    ui.sub.innerHTML = known
        ? escapeHtml(desc || '(no description in run.json)')
        : `<em>${escapeHtml(session.run)}</em> — no card for this run in the gallery; `
          + 'showing weights only.';

    let argsHtml;
    if (args && Object.keys(args).length) {
        argsHtml = '<table id="fd-args">' + Object.keys(args).map(k =>
            `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(args[k]))}</td></tr>`
        ).join('') + '</table>';
    } else {
        argsHtml = '<div class="fd-meta" style="color:#666;">(run.json not loaded yet)</div>';
    }

    ui.info.innerHTML = `
      <h4>Args</h4>
      ${argsHtml}
      <h4>Tags</h4>
      <div class="fd-meta">${tags.length ? escapeHtml(tags.join(', ')) : '(none)'}</div>
      <h4>Loss</h4>
      <div class="fd-meta">final: ${escapeHtml(fmtLoss(finalLoss))} &nbsp;·&nbsp; min: ${escapeHtml(fmtLoss(minLoss))}</div>
      <h4>Trained on</h4>
      <div class="fd-target-wrap">
        <div class="img-container"><img id="fd-target" src="${escapeHtml(dir + 'target.png')}" alt="training target"></div>
      </div>
      <h4>Bucket links</h4>
      <div class="fd-meta">
        <a href="${escapeHtml(dir + 'run.json')}" target="_blank" rel="noopener">run.json</a>
        &nbsp;·&nbsp;
        <a href="${escapeHtml(dir + 'weights.json')}" target="_blank" rel="noopener">weights.json</a>
      </div>
    `;
    const target = ui.info.querySelector('#fd-target');
    if (target) target.onerror = function () { this.style.display = 'none'; };
}

// ---- semantic weight views -------------------------------------------
//
// weights.json stores the learned parameters as flat matrices whose axes only
// mean something once you know the update rule. These views name those axes.
// Every editable leaf still writes straight through to its real path, so the
// grouping is presentation only — never a second copy of the data:
//
//   NCA (nca.js): fc0_w is [hidden_n][3*channel_n], its columns being the
//   perception vector [state | sobel x | sobel y] over every channel, in an
//   order the 'layout' field pins down; fc1_w is [channel_n][hidden_n], one
//   row per output channel.
//   Lenia (lenia_engine.js): the 'full' variant indexes its kernel bank
//   (src*C + tgt)*K + k, with mu/sg/h sharing that flat order, and 'sharedk'
//   couples channels through H[src][tgt].

const PERCEPTION_BLOCKS = ['state (identity)', 'sobel x', 'sobel y'];
const FULL_BANKS = ['kernels', 'mu', 'sg', 'h'];

// Column of fc0_w holding `block` of channel `c`. model.py's grouped conv
// emits [id0, sx0, sy0, id1, …]; a 'blocked' export has already reordered
// those into three contiguous per-channel runs.
export function perceptionColumn(w, block, c, C) {
    return w.layout === 'blocked' ? block * C + c : 3 * c + block;
}

// Channel semantics from model.py: 0..2 are RGB, 3 is the alpha (aliveness)
// channel, code_ch0..+code_bits carry a word model's per-letter code, and
// everything above that is free hidden state.
export function channelLabel(w, c) {
    const named = ['R', 'G', 'B', 'alpha'];
    if (c < named.length) return `ch ${c} · ${named[c]}`;
    const c0 = w.code_ch0, nb = w.code_bits;
    if (Number.isInteger(c0) && Number.isInteger(nb) && nb > 0 && c >= c0 && c < c0 + nb) {
        return `ch ${c} · code bit ${c - c0}`;
    }
    return `ch ${c} · hidden`;
}

function isNcaWeights(w) {
    return !!w && w.kind !== 'lenia' && Array.isArray(w.fc0_w)
        && Number.isInteger(w.channel_n) && Number.isInteger(w.hidden_n);
}

// Every fc0_w cell of one perception block, optionally narrowed to one hidden
// unit — what [scale] multiplies when you scale a whole block.
function fc0Paths(w, block, C, HN, onlyK) {
    const out = [];
    for (let k = 0; k < HN; k++) {
        if (onlyK !== undefined && k !== onlyK) continue;
        for (let c = 0; c < C; c++) {
            out.push(['fc0_w', String(k), String(perceptionColumn(w, block, c, C))]);
        }
    }
    return out;
}

// Flat 'full'-variant bank entries for one source channel, optionally
// narrowed to one target channel.
function fullBankPaths(segs, C, K, src, onlyTgt) {
    const out = [];
    for (let tgt = 0; tgt < C; tgt++) {
        if (onlyTgt !== undefined && tgt !== onlyTgt) continue;
        for (let k = 0; k < K; k++) out.push(segs.concat(String((src * C + tgt) * K + k)));
    }
    return out;
}

// The view rows for one node, or null to fall back to raw keys. A spec with
// `children` is a view node: it has no value of its own, so it offers only
// [scale] (over `leafPaths`) rather than the copy/paste/json tools that need
// a real path.
function viewChildren(segs) {
    if (!grouping || !session) return null;
    const w = session.fiddled;
    if (!w || typeof w !== 'object') return null;
    const key = segs[0];

    if (segs.length === 1 && isNcaWeights(w)) {
        const C = w.channel_n, HN = w.hidden_n;
        if (key === 'fc0_w' && w.fc0_w.length === HN) {
            return PERCEPTION_BLOCKS.map((blockLabel, block) => ({
                label: blockLabel, role: true, shape: `${HN} hidden × ${C} ch`,
                leafPaths: () => fc0Paths(w, block, C, HN),
                children: () => Array.from({ length: HN }, (_, k) => ({
                    label: `h ${k}`, role: true, shape: `${C} ch`,
                    leafPaths: () => fc0Paths(w, block, C, HN, k),
                    children: () => Array.from({ length: C }, (_, c) => ({
                        segs: ['fc0_w', String(k), String(perceptionColumn(w, block, c, C))],
                        label: channelLabel(w, c)
                    }))
                }))
            }));
        }
        if (key === 'fc1_w' && Array.isArray(w.fc1_w) && w.fc1_w.length === C) {
            return w.fc1_w.map((_, c) => ({
                segs: ['fc1_w', String(c)], label: `${channelLabel(w, c)} ← hidden`
            }));
        }
        if (key === 'fc0_b' && Array.isArray(w.fc0_b) && w.fc0_b.length === HN) {
            return w.fc0_b.map((_, k) => ({ segs: ['fc0_b', String(k)], label: `h ${k}` }));
        }
    }

    if (w.kind === 'lenia') {
        const C = w.C, K = w.K, arr = w[key];
        if (segs.length === 1 && w.variant === 'full' && FULL_BANKS.includes(key)
            && Array.isArray(arr) && Number.isInteger(C) && Number.isInteger(K)
            && arr.length === C * C * K) {
            return Array.from({ length: C }, (_, src) => ({
                label: `from ch ${src}`, role: true, shape: `${C} targets × ${K}`,
                leafPaths: () => fullBankPaths(segs, C, K, src),
                children: () => Array.from({ length: C }, (_, tgt) => ({
                    label: `→ to ch ${tgt}`, role: true,
                    shape: `${K} kernel${K === 1 ? '' : 's'}`,
                    leafPaths: () => fullBankPaths(segs, C, K, src, tgt),
                    children: () => Array.from({ length: K }, (_, k) => ({
                        segs: segs.concat(String((src * C + tgt) * K + k)), label: `k ${k}`
                    }))
                }))
            }));
        }
        if (segs.length === 1 && key === 'H' && Array.isArray(arr) && arr.length === C) {
            return arr.map((_, src) => ({ segs: ['H', String(src)], label: `from ch ${src}` }));
        }
        if (segs.length === 2 && segs[0] === 'H' && Array.isArray(w.H)) {
            const row = w.H[Number(segs[1])];
            if (Array.isArray(row)) {
                return row.map((_, tgt) => ({ segs: segs.concat(String(tgt)), label: `→ to ch ${tgt}` }));
            }
        }
    }
    return null;
}

// ---- structure tree --------------------------------------------------

function renderTree() {
    if (!ui || !session) return;
    ui.tree.innerHTML = '';
    if (session.fiddled === null || session.fiddled === undefined) return;
    ui.tree.appendChild(makeNode({ segs: [], label: 'weights.json', forceOpen: true }));
}

function nodePath(segs) { return segs.join(PATH_SEP); }

// One tree row (+ a lazily filled children box for containers). `spec` is
// either a real node ({segs, label}) or a view node (one that also carries
// `children`), whose rows name an axis rather than a stored key.
function makeNode(spec) {
    const view = typeof spec.children === 'function';
    const segs = spec.segs || [];
    const value = view ? null : (segs.length ? getAt(session.fiddled, segs) : session.fiddled);
    const isContainer = view || (value !== null && typeof value === 'object');

    const node = document.createElement('div');
    node.className = 'fd-node';
    node._spec = spec;

    const row = document.createElement('div');
    row.className = 'fd-row';
    node.appendChild(row);

    const tog = document.createElement('button');
    tog.className = 'fd-tog';
    tog.innerHTML = isContainer ? '&#9656;' : '&nbsp;';
    tog.disabled = !isContainer;
    row.appendChild(tog);

    const key = document.createElement('span');
    key.className = spec.role ? 'fd-key fd-role' : 'fd-key';
    key.innerText = spec.label;
    row.appendChild(key);

    if (isContainer) {
        const shape = document.createElement('span');
        shape.className = 'fd-shape';
        shape.innerText = '— ' + (view ? (spec.shape || '') : shapeSummary(value));
        row.appendChild(shape);
    } else {
        row.appendChild(makeLeafEditor(segs, value));
    }

    const tools = document.createElement('span');
    tools.className = 'fd-tools';
    if (view) {
        // No real path to copy from or paste onto — but scaling every number
        // under one named axis ("turn sobel x down") is the point of the view.
        if (typeof spec.leafPaths === 'function') {
            tools.appendChild(toolBtn('scale', 'Multiply every number under this axis by a factor',
                () => scaleView(spec, node)));
        }
    } else {
        tools.appendChild(toolBtn('copy', 'Copy this subtree to the fiddle clipboard',
            () => copyNode(segs)));
        tools.appendChild(toolBtn('paste', 'Paste the fiddle clipboard onto this node',
            () => pasteNode(segs, node)));
        if (isContainer) {
            tools.appendChild(toolBtn('json', 'Edit this subtree as raw JSON',
                () => jsonEditNode(segs, node)));
            if (isNumericSubtree(value)) {
                tools.appendChild(toolBtn('scale', 'Multiply every number below by a factor',
                    () => scaleNode(segs, node)));
            }
        }
    }
    row.appendChild(tools);

    if (isContainer) {
        const kids = document.createElement('div');
        kids.className = 'fd-children';
        kids.style.display = 'none';
        node.appendChild(kids);
        tog.onclick = () => toggleNode(node);
        // Small subtrees open on sight; big ones (fc0_w, kernel banks) and
        // every view node stay shut so opening the modal never renders
        // thousands of rows.
        if (spec.forceOpen || (!view
            && countLeaves(value, AUTO_OPEN_LEAVES + 1) <= AUTO_OPEN_LEAVES)) {
            toggleNode(node, true);
        }
    }
    return node;
}

function toolBtn(text, title, onclick) {
    const b = document.createElement('button');
    b.className = 'fd-tool';
    b.innerText = text;
    b.title = title;
    b.onclick = (e) => { e.stopPropagation(); onclick(); };
    return b;
}

function toggleNode(node, forceOpen) {
    const kids = node.querySelector(':scope > .fd-children');
    const tog = node.querySelector(':scope > .fd-row > .fd-tog');
    if (!kids) return;
    const open = forceOpen === true ? true : kids.style.display === 'none';
    kids.style.display = open ? '' : 'none';
    tog.innerHTML = open ? '&#9662;' : '&#9656;';
    if (open && !node._built) buildChildren(node);
}

function buildChildren(node) {
    const spec = node._spec;
    const kids = node.querySelector(':scope > .fd-children');
    kids.innerHTML = '';

    let specs;
    if (typeof spec.children === 'function') {
        specs = spec.children();
    } else {
        const segs = spec.segs;
        specs = viewChildren(segs);
        if (!specs) {
            const value = segs.length ? getAt(session.fiddled, segs) : session.fiddled;
            const keys = Array.isArray(value)
                ? value.map((_, i) => String(i))
                : Object.keys(value);
            specs = keys.map(k => ({ segs: segs.concat(k), label: k }));
        }
    }

    const shown = specs.slice(0, MAX_CHILD_ROWS);
    for (const s of shown) kids.appendChild(makeNode(s));
    if (specs.length > shown.length) {
        const more = document.createElement('div');
        more.className = 'fd-more';
        more.innerText = `… ${specs.length - shown.length} more entries not shown — use [json] to view or edit them`;
        kids.appendChild(more);
    }
    node._built = true;
}

// Rebuilds one node in place after a structural edit (paste / json / scale),
// preserving its position in the tree but not its children's open state.
function refreshNode(node) {
    if (!node) { renderTree(); return; }
    const spec = node._spec;
    const replacement = makeNode(
        Object.assign({}, spec, { forceOpen: spec.forceOpen || (spec.segs && !spec.segs.length) }));
    node.replaceWith(replacement);
}

function writeAt(segs, value) {
    if (!segs.length) { session.fiddled = value; return; }
    const parent = getAt(session.fiddled, segs.slice(0, -1));
    if (parent === null || parent === undefined || typeof parent !== 'object') return;
    parent[segs[segs.length - 1]] = value;
}

// Scalar editors. The invariant that matters: a half-typed or empty number
// must never corrupt the weights object — the field turns red and the last
// valid value stays in place.
function makeLeafEditor(segs, value) {
    if (typeof value === 'boolean') {
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = value;
        cb.onchange = () => { writeAt(segs, cb.checked); markDirty(); };
        return cb;
    }
    if (typeof value === 'number') {
        const inp = document.createElement('input');
        inp.className = 'fd-input';
        inp.type = 'number';
        inp.step = 'any';
        inp.value = String(value);
        inp.oninput = () => {
            const v = parseFloat(inp.value);
            if (inp.value.trim() === '' || !Number.isFinite(v)) {
                inp.classList.add('bad');
                return;   // keep the last valid value
            }
            inp.classList.remove('bad');
            writeAt(segs, v);
            markDirty();
        };
        return inp;
    }
    if (value === null || value === undefined) {
        // Rendered as an empty text box: leaving it empty keeps null, typing
        // turns it into a string (the only sane in-place promotion).
        const inp = document.createElement('input');
        inp.className = 'fd-input text';
        inp.type = 'text';
        inp.value = '';
        inp.placeholder = value === null ? 'null' : 'undefined';
        inp.oninput = () => {
            writeAt(segs, inp.value === '' ? null : inp.value);
            markDirty();
        };
        return inp;
    }
    const inp = document.createElement('input');
    inp.className = 'fd-input text';
    inp.type = 'text';
    inp.value = String(value);
    inp.oninput = () => { writeAt(segs, inp.value); markDirty(); };
    return inp;
}

// ---- node toolbar actions -------------------------------------------

function copyNode(segs) {
    if (!session) return;
    const value = deepClone(segs.length ? getAt(session.fiddled, segs) : session.fiddled);
    clipboard = { sourceRun: session.run, path: nodePath(segs), value };
    const json = JSON.stringify(value);
    // Best effort only: the async clipboard rejects without a user gesture
    // or permission, and that must not break the in-module copy.
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
        navigator.clipboard.writeText(json).catch(() => {});
    }
    liveStatus(`copied ${nodePath(segs) || '(whole weights)'} — ${shapeOf(value)}`);
}

function pasteNode(segs, node) {
    if (!session) return;
    if (clipboard === null) {
        // Nothing copied this session — let the user hand-paste JSON.
        openJsonDialog({
            title: `Paste JSON into ${nodePath(segs) || '(whole weights)'}`,
            initial: '',
            onOk: (parsed) => { applyPasteValue(segs, node, parsed, 'pasted JSON'); }
        });
        return;
    }
    applyPasteValue(segs, node, deepClone(clipboard.value),
        `${clipboard.sourceRun}:${clipboard.path || '(whole weights)'}`);
}

function applyPasteValue(segs, node, value, sourceLabel) {
    const current = segs.length ? getAt(session.fiddled, segs) : session.fiddled;
    const a = shapeOf(current), b = shapeOf(value);
    if (a !== b) {
        const ok = confirm(
            `Shape mismatch at ${nodePath(segs) || '(whole weights)'}:\n\n`
            + `  current: ${a}\n  pasted:  ${b}\n\n`
            + 'Replace anyway? The preview may fail to build.');
        if (!ok) return;
    }
    writeAt(segs, value);
    markDirty();
    if (!segs.length) renderTree(); else refreshNode(node);
    liveStatus(`pasted ${b} from ${sourceLabel}`);
}

function jsonEditNode(segs, node) {
    const value = segs.length ? getAt(session.fiddled, segs) : session.fiddled;
    const flat = JSON.stringify(value);
    // Pretty-print only while it stays readable; a 108x15x15 kernel bank is
    // far past the point where indentation helps.
    const initial = flat.length > 200000 ? flat : JSON.stringify(value, null, 1);
    openJsonDialog({
        title: `Edit ${nodePath(segs) || '(whole weights)'} as JSON`,
        initial,
        onOk: (parsed) => {
            writeAt(segs, parsed);
            markDirty();
            if (!segs.length) renderTree(); else refreshNode(node);
            liveStatus(`replaced ${nodePath(segs) || '(whole weights)'} — ${shapeOf(parsed)}`);
        }
    });
}

function scaleNode(segs, node) {
    const value = segs.length ? getAt(session.fiddled, segs) : session.fiddled;
    const raw = prompt(`Multiply every number under ${nodePath(segs) || '(whole weights)'} by:`, '1.0');
    if (raw === null) return;
    const f = parseFloat(raw);
    if (!Number.isFinite(f)) { liveStatus(`'${raw}' is not a number — nothing scaled`); return; }
    writeAt(segs, scaleNumbers(value, f));
    markDirty();
    if (!segs.length) renderTree(); else refreshNode(node);
    liveStatus(`scaled ${nodePath(segs) || '(whole weights)'} by ${f}`);
}

// [scale] on a view node: the axis it names has no single stored path, so it
// walks the real leaves it stands for (a whole perception block, one hidden
// unit's slice of it, one src→tgt kernel group).
function scaleView(spec, node) {
    const raw = prompt(`Multiply every number under ${spec.label} by:`, '1.0');
    if (raw === null) return;
    const f = parseFloat(raw);
    if (!Number.isFinite(f)) { liveStatus(`'${raw}' is not a number — nothing scaled`); return; }
    let n = 0;
    for (const path of spec.leafPaths()) {
        const v = getAt(session.fiddled, path);
        if (typeof v === 'number') { writeAt(path, v * f); n++; }
        else if (v !== null && typeof v === 'object') { writeAt(path, scaleNumbers(v, f)); n++; }
    }
    markDirty();
    refreshNode(node);
    liveStatus(`scaled ${n} entries under ${spec.label} by ${f}`);
}

// Modal-in-modal JSON textarea. Resolves only on a successful parse; the
// parse error stays visible so the user can fix the text in place.
function openJsonDialog({ title, initial, onOk }) {
    // The dialog covers the modal, so a hashchange can swap the run behind it
    // unseen; OK must refuse rather than write these edits into another run's
    // weights (or throw on a session showFatal() has already nulled).
    const owner = session;
    const dlg = document.createElement('div');
    dlg.className = 'fd-dialog';
    dlg.innerHTML = `
      <div class="fd-dialog-box">
        <h4 style="color:#4db8ff;text-transform:uppercase;letter-spacing:0.04em;font-size:0.9em;">${escapeHtml(title)}</h4>
        <textarea spellcheck="false"></textarea>
        <div class="fd-dialog-err"></div>
        <div class="fd-dialog-btns">
          <button class="fd-btn" data-act="cancel">Cancel</button>
          <button class="fd-btn primary" data-act="ok">OK</button>
        </div>
      </div>
    `;
    const ta = dlg.querySelector('textarea');
    ta.value = initial;
    const err = dlg.querySelector('.fd-dialog-err');
    dlg.querySelector('[data-act="cancel"]').onclick = () => dlg.remove();
    dlg.querySelector('[data-act="ok"]').onclick = () => {
        if (session !== owner) {
            err.innerText = 'this editor belongs to a run that is no longer open — '
                + 'copy the text out and Cancel';
            return;
        }
        let parsed;
        try {
            parsed = JSON.parse(ta.value);
        } catch (e) {
            err.innerText = 'JSON parse error: ' + e.message;
            return;
        }
        dlg.remove();
        onOk(parsed);
    };
    dlg.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') { e.stopPropagation(); dlg.remove(); }
    });
    ui.overlay.appendChild(dlg);
    ta.focus();
}

// ---- live preview ----------------------------------------------------

// Everything the engine constructors read, bounded BEFORE either of them is
// called: they allocate (and resetTrained/reset over) whatever the numbers
// ask for, so a hand-typed or URL-supplied `size: 20000` is 1.6 GB and 14 s
// of synchronous work before any later check could object, and a `ks: 2001`
// builds instantly and then never returns from step(). Returns a reason
// string, or null when the shape is safe to build.
function weightsOutOfRange(w) {
    const bad = (name, v, hi) => (Number.isInteger(v) && v >= 1 && v <= hi)
        ? null : `${name} ${v} out of range (1..${hi})`;
    if (w.kind === 'lenia') {
        const S = w.size ?? 64;
        const r = bad('grid', S, MAX_GRID) || bad('ks', w.ks, MAX_KS)
            || bad('C', w.C, MAX_C) || bad('K', w.K, MAX_K);
        if (r) return r;
        // step() convolves the whole kernel bank (C*C*K entries on 'full'),
        // each O(size^2 * ks^2) — cheap for real runs, ruinous once edited.
        const banks = Math.max(
            Array.isArray(w.kernels) ? w.kernels.length : 0,
            Array.isArray(w.basis) ? w.basis.length : 0, 1);
        const work = S * S * w.ks * w.ks * banks;
        if (work > MAX_STEP_WORK) {
            return `${banks} kernels at ${S}x${S} with ks=${w.ks} is too much work per step`;
        }
        return null;
    }
    // NCA: gridDims' own rule (nca.js) — word models carry grid_w/grid_h.
    const wordly = w.kind === 'word' || w.grid == null;
    return bad('grid width', wordly ? w.grid_w : w.grid, MAX_GRID)
        || bad('grid height', wordly ? w.grid_h : w.grid, MAX_GRID)
        || (w.channel_n === undefined ? null : bad('channel_n', w.channel_n, MAX_C * 8))
        || (w.hidden_n === undefined ? null : bad('hidden_n', w.hidden_n, MAX_HIDDEN));
}

// Every failure exit of rebuildEngine. Without the clear, the previous
// engine's last frame stays on the canvas and reads as a merely paused
// simulation whose transport buttons have silently stopped responding.
function engineBuildFailed(message) {
    if (session) {
        session.ca = null;
        session.engine = null;
        session.paused = true;
    }
    if (ui) {
        ui.canvasCtx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
        setPreviewControls(false);
    }
    liveStatus(message);
}

// Engine selection mirrors lenia.js's live widget exactly: `kind === 'lenia'`
// is a trainable-Lenia export, anything carrying the NCA schema goes through
// createCA (WebGL2 with CPU fallback). Both share step/reset/damage/readRGBA.
function rebuildEngine({ keepPaused }) {
    if (!ui || !session || !session.fiddled) return;
    stopLoop();
    disposeEngine();
    session.ca = null;
    session.engine = null;

    const w = session.fiddled;
    let W, H;
    const recognized = w.kind === 'lenia'
        || w.fc0_w || (w.channel_n !== undefined && w.hidden_n !== undefined);
    if (!recognized) { engineBuildFailed('unrecognized weights.json format'); return; }
    const rangeErr = weightsOutOfRange(w);
    if (rangeErr) { engineBuildFailed('engine build failed: ' + rangeErr); return; }
    try {
        if (w.kind === 'lenia') {
            const S = w.size ?? 64;
            session.ca = new LeniaCA(w, S);
            session.engine = 'lenia';
            W = S; H = S;
        } else if (w.fc0_w ||
                   (w.channel_n !== undefined && w.hidden_n !== undefined)) {
            const { ca } = createCA(w);
            session.ca = ca;
            session.engine = 'nca';
            W = ca.width; H = ca.height;
            ca.reset(w.seedType === 'noise');
        }
    } catch (e) {
        // Hand-edited weights can be structurally invalid (a resized kernel,
        // a retyped channel_n) — say so instead of throwing into the void.
        console.error('fiddle: engine build failed', e);
        engineBuildFailed('engine build failed: ' + e.message);
        return;
    }

    // The engine reports its own geometry (an NCA one derives it from the
    // weights), and a canvas can't take a zero, fractional or oversized one.
    if (!Number.isInteger(W) || !Number.isInteger(H)
        || W < 1 || H < 1 || W > MAX_GRID || H > MAX_GRID) {
        engineBuildFailed(`engine build failed: bad grid size ${W}x${H}`);
        return;
    }

    // Guarded too: the canvas resize and the ImageData allocation can fail on
    // their own, and unwinding out of here would leave the modal half-built
    // (title set, tree rendered, 'loading weights…' still up) instead of
    // reporting through liveStatus() like every other failure mode.
    try {
        ui.canvas.width = W;
        ui.canvas.height = H;
        session.imgData = ui.canvasCtx.createImageData(W, H);
    } catch (e) {
        console.error('fiddle: preview canvas allocation failed', e);
        engineBuildFailed('engine build failed: ' + e.message);
        return;
    }
    ui.canvas.style.imageRendering = 'pixelated';
    ui.canvas.style.width = '256px';
    ui.canvas.style.height = `${Math.round(256 * H / W)}px`;
    // Clear only makes sense on the Lenia engine; an NCA reset always
    // re-places its trained seed(s).
    ui.clearBtn.style.display = session.engine === 'lenia' ? '' : 'none';
    setPreviewControls(true);

    session.accum = 0;
    if (!keepPaused) session.paused = false;
    drawPreview();
    clearDirty();
    liveStatus(`live fiddled physics (${session.engine}, ${W}x${H}) — click/drag to damage`);
    if (!session.paused) {
        ui.playBtn.innerHTML = '&#9208;';   // stopLoop left the play glyph up
        runLoop();
    }
}

function drawPreview() {
    if (!session || !session.ca || !session.imgData) return;
    const out = session.ca.readRGBA(session.imgData.data);
    if (out && out !== session.imgData.data) session.imgData.data.set(out);
    ui.canvasCtx.putImageData(session.imgData, 0, 0);
}

// Fractional speeds (< 1 step/tick) accumulate across ticks, same pattern as
// lenia.js's runLiveLoop.
function runLoop() {
    if (!session || !session.ca || session.paused) return;
    session.accum += parseFloat(ui.speed.value || '1');
    const steps = Math.floor(session.accum);
    session.accum -= steps;
    try {
        for (let i = 0; i < steps; i++) session.ca.step();
    } catch (e) {
        console.error('fiddle: step failed', e);
        stopLoop();
        session.paused = true;
        liveStatus('simulation stopped — ' + e.message);
        return;
    }
    if (steps > 0) drawPreview();
    session.timer = setTimeout(runLoop, 30);
}

function togglePlay() {
    if (!session || !session.ca) return;
    session.paused = !session.paused;
    ui.playBtn.innerHTML = session.paused ? '&#9654;' : '&#9208;';
    if (session.paused) {
        if (session.timer) { clearTimeout(session.timer); session.timer = null; }
    } else {
        runLoop();
    }
}

function previewReset(mode) {
    if (!session || !session.ca) return;
    if (mode === 'noise') session.ca.reset(true);
    else if (mode === 'clear') session.ca.reset(false);
    else if (session.engine === 'lenia') session.ca.resetTrained();
    else session.ca.reset(false);
    drawPreview();
}

function damageAt(e) {
    if (!session || !session.ca) return;
    const rect = ui.canvas.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    session.ca.damage(nx * session.ca.width, ny * session.ca.height, 6);
    drawPreview();
}

// ---- toolbar actions -------------------------------------------------

function resetToTrained() {
    if (!session || !session.pristine) return;
    if (!session.pristineIsReal) {
        // `pristine` is this session's own weights standing in for a
        // weights.json that could not be fetched — restoring it is a no-op
        // that would look like the button is broken.
        liveStatus("this run's trained weights.json could not be fetched, "
            + 'so there is nothing to reset to');
        return;
    }
    session.fiddled = deepClone(session.pristine);
    session.fromUrl = false;
    if (ui) {
        ui.badge.style.display = 'none';
        ui.saveBox.style.display = 'none';
    }
    renderTree();
    rebuildEngine({ keepPaused: false });
    clearDirty();
    liveStatus('restored the trained weights');
}

function downloadWeights() {
    if (!session || !session.fiddled) return;
    const blob = new Blob([JSON.stringify(session.fiddled)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${session.run}_fiddled.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function showSaveBox(url, warning) {
    if (!ui) return;
    ui.saveBox.style.display = 'block';
    if (!url) {
        ui.saveBox.innerHTML = `<div id="fd-save-warn">${escapeHtml(warning || '')}</div>`;
        return;
    }
    const shown = url.length > 120 ? url.slice(0, 117) + '…' : url;
    ui.saveBox.innerHTML = `
      <div><a href="${escapeHtml(url)}" title="${escapeHtml(url)}">${escapeHtml(shown)}</a>
      <button class="fd-btn" id="fd-save-copy" style="margin-left:8px;">Copy</button></div>
      <div>${escapeHtml(String(url.length))} characters</div>
      ${warning ? `<div id="fd-save-warn">${escapeHtml(warning)}</div>` : ''}
    `;
    const btn = ui.saveBox.querySelector('#fd-save-copy');
    btn.onclick = () => {
        if (typeof navigator !== 'undefined' && navigator.clipboard) {
            navigator.clipboard.writeText(url)
                .then(() => { btn.innerText = 'Copied'; })
                .catch(() => { btn.innerText = 'Copy failed'; });
        }
    };
}

// Packs the current edits into '#fiddle=<token>'. A diff is usually far
// smaller than the weights, but a wholesale paste makes the patch bigger
// than the object it patches — so both are built and the smaller wins.
async function saveToUrl() {
    if (!session || !session.fiddled) return;
    let payload;
    // Only diff against weights we actually fetched: a stand-in pristine is a
    // clone of the fiddled object, so every patch against it is empty and the
    // link would tell the recipient "apply nothing to the trained weights".
    if (session.pristine && session.pristineIsReal) {
        const patch = diffWeights(session.pristine, session.fiddled);
        const asPatch = { v: 1, run: session.run, p: patch };
        const asFull = { v: 1, run: session.run, w: session.fiddled };
        payload = JSON.stringify(asPatch).length <= JSON.stringify(asFull).length ? asPatch : asFull;
    } else {
        payload = { v: 1, run: session.run, w: session.fiddled };
    }
    const token = await encodeFiddle(payload);
    // Remember what we wrote: setting location.hash fires hashchange, and
    // reloading the modal from our own token would throw away the session.
    selfSetToken = token;
    location.hash = 'fiddle=' + token;
    const url = location.href;
    showSaveBox(url, url.length > URL_WARN_LEN
        ? `this URL is ${url.length} characters — some chat/mail clients truncate links this long, `
          + 'so prefer Download weights.json for sharing it'
        : '');
    liveStatus(payload.p ? `saved as a ${payload.p.length}-entry patch` : 'saved as full weights');
}

// ---- URL loading -----------------------------------------------------

function readHashToken() {
    if (typeof location === 'undefined') return null;
    const h = String(location.hash || '');
    const i = h.indexOf('#fiddle=');
    if (i !== 0) return null;
    const t = h.slice('#fiddle='.length);
    return t.length ? t : null;
}

function installHashListener() {
    if (hashListenerInstalled || typeof window === 'undefined') return;
    hashListenerInstalled = true;
    window.addEventListener('hashchange', () => {
        const token = readHashToken();
        // One-shot: only the hashchange our own Save just caused is skipped.
        // Leaving it set would make that hash permanently dead, so navigating
        // Forward onto it again would leave the modal showing another model.
        const mine = selfSetToken;
        selfSetToken = null;
        if (!token || token === mine) return;   // our own Save
        loadFromToken(token).catch(e => console.error('fiddle: hash load failed', e));
    });
}

// Rebuilds a fiddled model from a share token: fetch the run's trained
// weights (the diff base), apply the patch (or take the full payload), and
// open the modal on the result with dirty=false.
async function loadFromToken(token) {
    buildModal();
    let payload;
    try {
        payload = await decodeFiddle(token);
    } catch (e) {
        // No alert(): a bad link shows inline in the modal it would have
        // opened, with the reason on the console.
        console.error('fiddle: bad token in URL', e);
        showFatal('Fiddle', 'this #fiddle= link is not readable: ' + e.message);
        return;
    }
    if (!payload || payload.v !== 1 || typeof payload.run !== 'string') {
        console.error('fiddle: unsupported payload', payload);
        showFatal('Fiddle', 'this #fiddle= link uses an unsupported payload version');
        return;
    }

    const run = payload.run;
    const id = 'lenia_' + run;
    const method = (ctx.getMethod && ctx.getMethod(id)) || null;
    const dir = (method && method.dir) || (bucketBase() + run + '/');

    let pristine = null;
    try {
        const res = await fetch(dir + 'weights.json?t=' + Date.now());
        if (res.ok) pristine = await res.json();
    } catch (e) {
        console.error('fiddle: pristine weights fetch failed', e);
    }

    if (!pristine && payload.w === undefined) {
        // A patch is meaningless without the base it was diffed against.
        showFatal(run, 'weights not exported yet — this link patches a run whose '
            + 'weights.json is not in the bucket');
        return;
    }

    let fiddled;
    try {
        fiddled = (payload.w !== undefined)
            ? deepClone(payload.w)
            : applyPatch(pristine, payload.p || []);
    } catch (e) {
        console.error('fiddle: patch apply failed', e);
        showFatal(run, 'this #fiddle= link could not be applied: ' + e.message);
        return;
    }

    await openSession(id, {
        pristine: pristine || deepClone(fiddled),
        fiddled,
        // Without the real weights.json the stand-in above is the fiddled
        // object itself: usable as a tree/preview base, never as a diff base.
        pristineIsReal: !!pristine,
        fromUrl: true
    });
    if (!pristine && session && ui) {
        ui.treeMsg.innerText = "this run's trained weights.json could not be fetched, "
            + 'so these weights cannot be diffed against it — Save → URL will ship the '
            + 'whole object and Reset to trained is unavailable';
    }
}

// Parses '#fiddle=<token>' out of the current URL and opens the fiddled
// model. Installs the hashchange listener exactly once, whether or not a
// token is present. Returns true when a token was handled.
export async function handleFiddleHash() {
    installHashListener();
    const token = readHashToken();
    if (!token) return false;
    if (token === selfSetToken) return false;   // already open from our Save
    await loadFromToken(token);
    return true;
}

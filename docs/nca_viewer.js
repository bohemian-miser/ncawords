import { createCA } from './nca.js';

// INTERACTIVE WIDGET LOGIC (Client-Side WebGL)
const canvas = document.getElementById("nca-canvas");
const ctx = canvas.getContext("2d");

// Default config matching original
window.interactiveLoop = null;
window.isDamaging = false;
let lastDamageTime = 0;
let unit = null; // Holds the instantiated CA

// Event listener for the speed slider
document.getElementById("speed-slider").oninput = function() {
    const v = parseFloat(this.value);
    document.getElementById("speed-val").innerText =
        v >= 1 ? String(Math.round(v)) : `1/${Math.round(1 / v)}`;
};

window.loadInteractiveModel = async function() {
    if(window.interactiveLoop) clearTimeout(window.interactiveLoop);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#888";
    ctx.font = "16px sans-serif";
    ctx.fillText("Loading weights...", canvas.width/2 - 50, canvas.height/2);
    
    let dir = document.getElementById("interactive-model-select").value;
    let isNoise = !dir.startsWith("http") && dir.endsWith("_noise");

    // Map python folder names to JSON exports
    let baseJson = dir.replace('snaps_web_', 'word_').replace('snaps_', 'word_');
    if(baseJson === 'word_cloud') baseJson = 'word_cloud';

    let jsonName = baseJson + '.json';
    let fallbackJsonName = baseJson.replace('_noise', '') + '.json';

    try {
        // Absolute URLs (cloud-run weights in the public bucket) load as-is;
        // bust GCS's 1-hour object cache since training updates them live.
        let res = await fetch(dir.startsWith("http") ? `${dir}?t=${Date.now()}` : `docs/weights/${jsonName}`);

        // If the specific _noise model wasn't exported, fallback to the base model weights
        if(!res.ok && isNoise) {
            res = await fetch(`docs/weights/${fallbackJsonName}`);
            if(!res.ok) throw new Error(`Could not load docs/weights/${fallbackJsonName}`);
        } else if (!res.ok) {
            throw new Error(`Could not load ${dir.startsWith("http") ? dir : 'docs/weights/' + jsonName}`);
        }
        
        const weights = await res.json();
        // Denoising models declare seedType 'noise': reset fills the grid
        // with noise instead of placing a single seed.
        if (weights.seedType === 'noise') isNoise = true;
        
        // Clean up old webgl context if present
        if(unit && unit.ca && unit.ca.gl) {
            const ext = unit.ca.gl.getExtension('WEBGL_lose_context');
            if(ext) ext.loseContext();
        }
        
        const { ca, mode } = createCA(weights);
        unit = { ca, mode, isNoise };
        
        let cw = typeof ca.width === 'number' ? ca.width : (weights.grid_w ?? weights.grid);
        let ch = typeof ca.height === 'number' ? ca.height : (weights.grid_h ?? weights.grid);
        unit.w = cw;
        unit.h = ch;
        unit.imgData = ctx.createImageData(cw, ch);
        populateChannelSelectors(ca.channel_n);

        // Set canvas visual size. Actual resolution is cw, ch.
        canvas.width = cw;
        canvas.height = ch;
        canvas.style.width = (cw * 4) + 'px'; // upscale for UI
        canvas.style.imageRendering = 'pixelated';
        
        ca.reset(isNoise);
        drawCA();
        window.runInteractiveLoop();
    } catch(e) {
        console.error(e);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#ff5555";
        ctx.fillText("Model load failed. Please ensure JSON weights exist.", 10, 20);
    }
};

// R/G/B channel-select dropdowns: populated 0..channel_n-1, default 0/1/2.
function populateChannelSelectors(n) {
    const ids = ['rgb-r', 'rgb-g', 'rgb-b'];
    const defaults = [0, 1, 2];
    ids.forEach((id, idx) => {
        const sel = document.getElementById(id);
        if (!sel) return;
        sel.innerHTML = '';
        for (let c = 0; c < n; c++) {
            const opt = document.createElement('option');
            opt.value = String(c);
            opt.textContent = String(c);
            sel.appendChild(opt);
        }
        sel.value = String(Math.min(defaults[idx], n - 1));
        sel.onchange = drawCA;
    });
}

function selectedChannels() {
    const rSel = document.getElementById('rgb-r');
    const gSel = document.getElementById('rgb-g');
    const bSel = document.getElementById('rgb-b');
    return {
        r: rSel && rSel.value !== '' ? parseInt(rSel.value, 10) : 0,
        g: gSel && gSel.value !== '' ? parseInt(gSel.value, 10) : 1,
        b: bSel && bSel.value !== '' ? parseInt(bSel.value, 10) : 2,
    };
}

function drawCA() {
    if(!unit || !unit.ca) return;
    const { r: rC, g: gC, b: bC } = selectedChannels();
    const isDefault = rC === 0 && gC === 1 && bC === 2;

    if (!isDefault && typeof unit.ca.readChannelsRGB === 'function') {
        const out = unit.ca.readChannelsRGB(rC, gC, bC, unit.imgData.data);
        if (out && out !== unit.imgData.data) unit.imgData.data.set(out);
        ctx.putImageData(unit.imgData, 0, 0);
    } else if (unit.mode === 'gl' && typeof unit.ca.drawTo === 'function') {
        unit.ca.drawTo(ctx);
    } else {
        const out = unit.ca.readRGBA(unit.imgData.data);
        if (out && out !== unit.imgData.data) unit.imgData.data.set(out);
        ctx.putImageData(unit.imgData, 0, 0);
    }

    updateChannelGridIfVisible();
}

// Per-channel grayscale heatmap grid (all channels 0..channel_n-1).
function isChannelGridVisible() {
    const grid = document.getElementById('channel-grid');
    return !!grid && grid.style.display !== 'none' && grid.style.display !== '';
}

function updateChannelGridIfVisible() {
    if (isChannelGridVisible()) updateChannelGrid();
}

function updateChannelGrid() {
    if (!unit || !unit.ca || typeof unit.ca.readChannel !== 'function') return;
    const grid = document.getElementById('channel-grid');
    if (!grid) return;
    const n = unit.ca.channel_n;
    const w = unit.w, h = unit.h;

    if (grid.children.length !== n) {
        grid.innerHTML = '';
        for (let c = 0; c < n; c++) {
            const wrap = document.createElement('div');
            wrap.style.textAlign = 'center';
            const cv = document.createElement('canvas');
            cv.width = w;
            cv.height = h;
            cv.style.width = (w * 2) + 'px';
            cv.style.height = (h * 2) + 'px';
            cv.style.imageRendering = 'pixelated';
            cv.style.border = '1px solid #333';
            const label = document.createElement('div');
            label.textContent = 'ch ' + c;
            label.style.color = '#aaa';
            label.style.fontSize = '11px';
            wrap.appendChild(cv);
            wrap.appendChild(label);
            grid.appendChild(wrap);
        }
    }

    for (let c = 0; c < n; c++) {
        const cv = grid.children[c].firstChild;
        const cctx = cv.getContext('2d');
        const vals = unit.ca.readChannel(c);
        const img = cctx.createImageData(w, h);
        for (let i = 0; i < vals.length; i++) {
            let v = vals[i];
            if (v < 0) v = 0; else if (v > 1) v = 1;
            const g = v * 255;
            img.data[i * 4 + 0] = g;
            img.data[i * 4 + 1] = g;
            img.data[i * 4 + 2] = g;
            img.data[i * 4 + 3] = 255;
        }
        cctx.putImageData(img, 0, 0);
    }
}

window.toggleChannelGrid = function() {
    const grid = document.getElementById('channel-grid');
    if (!grid) return;
    if (isChannelGridVisible()) {
        grid.style.display = 'none';
    } else {
        grid.style.display = 'flex';
        updateChannelGrid();
    }
};

window.resetInteractive = function() {
    if(unit && unit.ca) {
        unit.ca.reset(unit.isNoise);
        drawCA();
    }
};

window.noiseInteractive = function() {
    // Immediate splash straight onto the 2D canvas — guaranteed visible
    // feedback even before (or without) the engine.
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < img.data.length; i += 4) {
        img.data[i] = Math.random() * 255;
        img.data[i + 1] = Math.random() * 255;
        img.data[i + 2] = Math.random() * 255;
        img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);

    if(!unit || !unit.ca) {
        // Nothing loaded yet: load the selected model, then fill its state.
        window.loadInteractiveModel().then(() => {
            if(unit && unit.ca) unit.ca.reset(true);
        });
        return;
    }
    unit.ca.reset(true);
    // Hold the animation so the noise state is visible — growth models can
    // annihilate uniform noise within a few CA steps.
    if(window.interactiveLoop) clearTimeout(window.interactiveLoop);
    window.interactiveLoop = setTimeout(window.runInteractiveLoop, 1000);
};

window.paused = false;
window.togglePause = function() {
    window.paused = !window.paused;
    const btn = document.getElementById('pause-btn');
    if (btn) btn.innerText = window.paused ? '▶ Play' : '⏸ Pause';
    if (!window.paused) window.runInteractiveLoop();
    else if (window.interactiveLoop) clearTimeout(window.interactiveLoop);
};

let stepAccum = 0;
window.runInteractiveLoop = function() {
    if(!unit || !unit.ca || window.paused) return;
    // Fractional speeds (< 1 step/tick) accumulate across ticks — slow
    // motion for watching fast dynamics like noise collapse.
    stepAccum += parseFloat(document.getElementById("speed-slider").value);
    const steps = Math.floor(stepAccum);
    stepAccum -= steps;
    for(let i=0; i<steps; i++) unit.ca.step();
    if (steps > 0) drawCA();
    window.interactiveLoop = setTimeout(window.runInteractiveLoop, 30);
};

window.sendDamage = function(e) {
    if(!unit || !unit.ca) return;
    const now = Date.now();
    if (now - lastDamageTime < 10) return; // Client-side can be fast
    lastDamageTime = now;
    
    const rect = canvas.getBoundingClientRect();
    // Calculate normalized coordinate across the upscaled CSS display
    const normX = (e.clientX - rect.left) / rect.width;
    const normY = (e.clientY - rect.top) / rect.height;
    
    // Damage CA using absolute grid coordinates
    unit.ca.damage(normX * unit.w, normY * unit.h, 6); // R=6
    drawCA();
};

window.startDamage = function(e) { window.isDamaging = true; window.sendDamage(e); };
window.stopDamage = function(e) { window.isDamaging = false; };
window.doDamage = function(e) { if(window.isDamaging) window.sendDamage(e); };

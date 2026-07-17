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
    document.getElementById("speed-val").innerText = this.value;
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

function drawCA() {
    if(!unit || !unit.ca) return;
    if (unit.mode === 'gl' && typeof unit.ca.drawTo === 'function') {
        unit.ca.drawTo(ctx);
    } else {
        const out = unit.ca.readRGBA(unit.imgData.data);
        if (out && out !== unit.imgData.data) unit.imgData.data.set(out);
        ctx.putImageData(unit.imgData, 0, 0);
    }
}

window.resetInteractive = function() {
    if(unit && unit.ca) {
        unit.ca.reset(unit.isNoise);
        drawCA();
    }
};

window.noiseInteractive = function() {
    if(!unit || !unit.ca) {
        // Nothing loaded yet: load the selected model first, then fill.
        window.loadInteractiveModel().then(() => {
            if(unit && unit.ca) window.noiseInteractive();
        });
        return;
    }
    unit.ca.reset(true);
    drawCA();
    // Hold the animation briefly so the filled noise is actually visible —
    // growth models can annihilate uniform noise within a few CA steps.
    if(window.interactiveLoop) clearTimeout(window.interactiveLoop);
    window.interactiveLoop = setTimeout(window.runInteractiveLoop, 800);
};

window.runInteractiveLoop = function() {
    if(!unit || !unit.ca) return;
    const steps = parseInt(document.getElementById("speed-slider").value);
    for(let i=0; i<steps; i++) unit.ca.step();
    drawCA();
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

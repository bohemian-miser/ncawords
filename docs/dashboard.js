// Interactive-playground support ONLY. The run-card gallery that used to
// live here moved to lenia.html (lenia.js) — one unified card format for
// every run in the bucket. This file just keeps the playground's model
// dropdown populated with every run that has exported weights.json, plus
// the prev/next helpers the playground buttons call (the CA loop itself
// lives in nca_viewer.js).

// Public bucket the training jobs write to; readable (and listable)
// anonymously. Overridable via an optional config.js (gitignored) that
// sets window.NCA_CONFIG.bucket.
const BUCKET = (window.NCA_CONFIG && window.NCA_CONFIG.bucket) || 'recipe-lanes-nca-jobs';
const BUCKET_BASE = (window.NCA_CONFIG && window.NCA_CONFIG.bucketBase) || `https://storage.googleapis.com/${BUCKET}/`;
const API_BASE = (window.NCA_CONFIG && window.NCA_CONFIG.apiBase) || `https://storage.googleapis.com/storage/v1/b/${BUCKET}`;
const BUCKET_LIST = `${API_BASE}/o?fields=items(name,updated),nextPageToken&maxResults=1000`;

// run name -> { url, updated } for every run with an exported weights.json.
let cloudModels = {};

async function listCloudWeights() {
    const found = {};

    // 1. Check static models.json bundled with the site
    try {
        const localRes = await fetch('models.json?t=' + Date.now());
        if (localRes.ok) {
            const staticModels = await localRes.json();
            Object.assign(found, staticModels);
        }
    } catch (e) {
        /* static models.json unavailable */
    }

    // 2. Query cloud GCS bucket if accessible
    try {
        let pageToken = null;
        do {
            const res = await fetch(BUCKET_LIST + (pageToken ? `&pageToken=${pageToken}` : ''));
            if (!res.ok) break;
            const d = await res.json();
            (d.items || []).forEach(({name, updated}) => {
                const i = name.indexOf('/');
                if (i < 0) return;
                const run = name.slice(0, i), fname = name.slice(i + 1);
                if (!found[run]) found[run] = { url: null, updated: '' };
                if (fname === 'weights.json') found[run].url = BUCKET_BASE + run + '/weights.json';
                if (updated && updated > found[run].updated) found[run].updated = updated;
            });
            pageToken = d.nextPageToken;
        } while (pageToken);
    } catch (e) {
        /* cloud list unavailable */
    }

    Object.entries(found).forEach(([run, v]) => {
        if (v.url) cloudModels[run] = v;
    });
}

function initializeDropdown() {
    const selectBox = document.getElementById('interactive-model-select');
    if (!selectBox) return;
    const prev = selectBox.value;   // preserve selection across refreshes
    selectBox.innerHTML = '';
    Object.keys(cloudModels)
        .sort((a, b) => (cloudModels[b].updated || '').localeCompare(cloudModels[a].updated || '')
                        || a.localeCompare(b))
        .forEach(run => {
            const opt = document.createElement('option');
            opt.value = cloudModels[run].url;
            const isLocal = cloudModels[run].url.startsWith('runs/');
            opt.innerText = (isLocal ? '★ ' : '☁ ') + run;
            selectBox.appendChild(opt);
        });
    if (prev && [...selectBox.options].some(o => o.value === prev)) selectBox.value = prev;
}

window.prevInteractiveModel = function() {
    const select = document.getElementById('interactive-model-select');
    if (select.options.length > 0) {
        select.selectedIndex = (select.selectedIndex - 1 + select.options.length) % select.options.length;
        if (window.loadInteractiveModel) window.loadInteractiveModel();
    }
};

window.nextInteractiveModel = function() {
    const select = document.getElementById('interactive-model-select');
    if (select.options.length > 0) {
        select.selectedIndex = (select.selectedIndex + 1) % select.options.length;
        if (window.loadInteractiveModel) window.loadInteractiveModel();
    }
};

async function bootstrap() {
    try {
        await listCloudWeights();
        initializeDropdown();
    } catch (e) { console.error('playground model listing failed', e); }
    // New exports appear without a reload.
    setInterval(async () => {
        try {
            await listCloudWeights();
            initializeDropdown();
        } catch (e) { /* transient */ }
    }, 60000);
}

bootstrap();

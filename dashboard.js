let methods = [];
let cardTrackers = [];
const container = document.getElementById('cards-container');

let serverState = {};
let activeModalDir = null;
let activeModalMaxStepRendered = -100;

// Fetch methods via API
fetch('methods.json?t=' + Date.now())
    .then(res => res.json())
    .then(data => {
        methods = data;
        
        // Build cards without innerHTML concatenations
        methods.forEach(m => {
            const card = document.createElement('div');
            card.className = 'card';
            card.id = `card_${m.id}`;
            card.onclick = () => openModal(m.title, m.dir, m.desc);
            
            card.innerHTML = `
                <h3>${m.title}</h3>
                <div class="side-by-side">
                    <div>
                        <div class="sub-desc">Live Target</div>
                        <div class="img-container"><img id="live_tgt_${m.id}" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" onerror="this.src='${m.dir}target.png'"></div>
                    </div>
                    <div>
                        <div class="sub-desc">Latest Checkpoint</div>
                        <div class="img-container"><img id="live_${m.id}" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="></div>
                    </div>
                </div>
                <div class="status" id="live_status_${m.id}">Loading Server State...</div>
            `;
            container.appendChild(card);
        });
        
        initializeDropdown();
        
        cardTrackers = methods.map(m => ({
            dir: m.dir, 
            cardObj: document.getElementById(`card_${m.id}`),
            imgObj: document.getElementById(`live_${m.id}`), 
            tgtObj: document.getElementById(`live_tgt_${m.id}`), 
            statusObj: document.getElementById(`live_status_${m.id}`), 
            lastKnownStep: -100 
        }));

        // Connect to SSE instead of polling!
        const evtSource = new EventSource("/api/status_stream");
        evtSource.onmessage = function(event) {
            serverState = JSON.parse(event.data);
            updateOverviewUI();
            updateActiveModalUI();
        };
        evtSource.onerror = function() {
            console.error("SSE Connection Error");
        };
    })
    .catch(err => console.error("Could not load methods.json", err));
    
function applySeedFilter() {
    const filterVal = document.getElementById('seed-filter').value;
    methods.forEach(m => {
        const el = document.getElementById(`card_${m.id}`);
        if (!el) return;
        if (filterVal === 'all') {
            el.style.display = 'block';
        } else if (filterVal === m.seedType) {
            el.style.display = 'block';
        } else {
            el.style.display = 'none';
        }
    });
}

function initializeDropdown() {
    const selectBox = document.getElementById("interactive-model-select");
    selectBox.innerHTML = '';
    
    fetch('docs/weights/index.json?t=' + Date.now())
        .then(r => r.json())
        .then(idx => {
            methods.forEach(m => {
                if (m.id === 'diffusion') return;
                
                let dirName = m.dir.replace('/', '');
                let lookupName = dirName.replace('snaps_web_', '').replace('snaps_', '');
                
                if (idx.words.includes(lookupName) || m.id === 'guided' || m.id === 'cloud') {
                    let opt = document.createElement('option');
                    opt.value = dirName;
                    opt.innerText = m.title;
                    selectBox.appendChild(opt);
                }
            });
        })
        .catch(err => {
            methods.forEach(m => {
                if (m.id === 'diffusion') return;
                let opt = document.createElement('option');
                opt.value = m.dir.replace('/', '');
                opt.innerText = m.title;
                selectBox.appendChild(opt);
            });
        });
}

function updateOverviewUI() {
    cardTrackers.forEach(ct => {
        const currentHighest = serverState[ct.dir];
        if (currentHighest !== undefined) {
            if (currentHighest > ct.lastKnownStep) {
                ct.lastKnownStep = currentHighest;
                if (currentHighest === -1) {
                    ct.lastStatusText = "No runs logged yet (Waiting on step 0)";
                } else {
                    ct.lastStatusText = `Latest: Step ${currentHighest} / 16000`;
                    let stepStr = String(currentHighest).padStart(5, '0');
                    if(ct.imgObj) {
                        ct.imgObj.onerror = function() { this.src = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs='; };
                        ct.imgObj.src = `${ct.dir}COMP_${stepStr}.png?t=` + Date.now();
                    }
                    if(ct.tgtObj) ct.tgtObj.src = `${ct.dir}TARGET_${stepStr}.png?t=` + Date.now();
                }
            }
            
            if (!ct.cardObj || !ct.statusObj) return;
            
            if (ct.dir.includes('proposed_targets')) {
                ct.cardObj.style.borderColor = '#4db8ff';
                ct.statusObj.style.color = '#4db8ff';
                ct.statusObj.innerText = '(Ready for training)';
                if (ct.imgObj) ct.imgObj.style.display = 'none';
            } else if (currentHighest >= 15900) {
                ct.cardObj.style.borderColor = '#00ff00';
                ct.statusObj.style.color = '#00ff00';
                ct.statusObj.innerText = (ct.lastStatusText || '') + ' (DONE)';
            } else if (currentHighest >= 0) {
                ct.cardObj.style.borderColor = '#ffaa00';
                ct.statusObj.style.color = '#ffaa00';
                ct.statusObj.innerText = (ct.lastStatusText || '') + ' (RUNNING)';
            } else {
                ct.statusObj.innerText = (ct.lastStatusText || '');
            }
        }
    });
}

window.openModal = function(title, dir, desc) {
    document.getElementById('modal-title').innerText = title;
    document.getElementById('modal-desc').innerText = desc || "No description available.";
    
    let btnBegin = document.getElementById('btn-begin-training');
    if (dir.includes('proposed_targets')) {
        btnBegin.style.display = 'inline-block';
    } else {
        btnBegin.style.display = 'none';
    }
    
    const gallery = document.getElementById('modal-gallery');
    gallery.innerHTML = '';
    activeModalDir = dir;
    activeModalMaxStepRendered = -100;
    document.getElementById('modal').style.display = 'block';
    updateActiveModalUI();
    fetchNotes(dir);
}

function fetchNotes(dir) {
    fetch('/api/notes')
        .then(r => r.json())
        .then(notes_db => {
            const list = document.getElementById('notes-list');
            list.innerHTML = '';
            if (notes_db[dir]) {
                notes_db[dir].forEach(n => {
                    const li = document.createElement('li');
                    li.innerText = new Date(n.timestamp * 1000).toLocaleString() + ": " + n.note;
                    list.appendChild(li);
                });
            } else {
                list.innerHTML = '<li><i>No notes yet.</i></li>';
            }
        });
}

window.submitNote = function(event) {
    event.stopPropagation();
    const input = document.getElementById('note-input');
    const note = input.value.trim();
    if(!note || !activeModalDir) return;
    
    fetch('/api/notes', {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({dir: activeModalDir, note: note})
    }).then(() => {
        input.value = '';
        fetchNotes(activeModalDir);
    });
}

window.beginTraining = function(event) {
    event.stopPropagation();
    if(!activeModalDir) return;
    fetch('/api/notes', {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({dir: activeModalDir, note: "[SYSTEM: BEGIN TRAINING]"})
    }).then(() => {
        fetchNotes(activeModalDir);
        let tgtStatus = document.getElementById('live_status_proposed_targets');
        if (tgtStatus) {
            tgtStatus.innerText = "Training Started!";
            tgtStatus.style.animation = "none";
            tgtStatus.style.color = "#55ff55";
        }
    });
}

window.closeModal = function() {
    document.getElementById('modal').style.display = 'none';
    activeModalDir = null;
}

window.handleOverlayClick = function(event) {
    closeModal();
}

window.handleContentClick = function(event) {
    if (event.target.closest('#modal-notes') || event.target.closest('.gallery-item')) {
        event.stopPropagation();
    } else {
        closeModal();
        event.stopPropagation();
    }
}

function updateActiveModalUI() {
    if (!activeModalDir) return;
    const maxStep = serverState[activeModalDir];
    if (maxStep !== undefined && maxStep > activeModalMaxStepRendered) {
        const gallery = document.getElementById('modal-gallery');
        let start = Math.max(0, activeModalMaxStepRendered + 100);
        if (activeModalMaxStepRendered === -100) start = 0;
        
        for (let s = start; s <= maxStep; s += 100) {
            let stepStr = String(s).padStart(5, '0');
            
            const div = document.createElement('div');
            div.className = 'gallery-item';
            div.innerHTML = `
                <img src="${activeModalDir}TARGET_${stepStr}.png" onerror="this.src='${activeModalDir}target.png'; this.onerror=null;" alt="Target Step ${s}">
                <img src="${activeModalDir}COMP_${stepStr}.png" onerror="this.style.display='none'" alt="Step ${s}">
                <span>Step ${s}</span>
            `;
            gallery.appendChild(div);
        }
        activeModalMaxStepRendered = maxStep;
    }
}

window.prevInteractiveModel = function() {
    const select = document.getElementById("interactive-model-select");
    if (select.options.length > 0) {
        select.selectedIndex = (select.selectedIndex - 1 + select.options.length) % select.options.length;
        if(window.loadInteractiveModel) window.loadInteractiveModel();
    }
}

window.nextInteractiveModel = function() {
    const select = document.getElementById("interactive-model-select");
    if (select.options.length > 0) {
        select.selectedIndex = (select.selectedIndex + 1) % select.options.length;
        if(window.loadInteractiveModel) window.loadInteractiveModel();
    }
}

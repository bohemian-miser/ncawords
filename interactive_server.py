import os
import io
import json
import time
import base64
import torch
import numpy as np
import glob
import asyncio
from PIL import Image
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Optional, Any

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from nca.model import NCA, to_rgba, to_rgb
from nca.train_web_method1 import render_word_method1 as render_word, FONT_PATH
import uuid

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

class SessionState:
    def __init__(self, model_dir, active_channel_n, hidden_n):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = model_dir
        self.active_channel_n = active_channel_n
        self.hidden_n = hidden_n
        self.current_model = NCA(self.active_channel_n, hidden_n=self.hidden_n).to(self.device)
        self.grid_h, self.grid_w = 0, 0
        self.current_x = None

        pth_path = Path(model_dir) / "latest.pth"
        if pth_path.exists():
            self.current_model.load_state_dict(torch.load(pth_path, map_location=self.device, weights_only=True))
        self.current_model.eval()
        self.reset_grid()

    def reset_grid(self):
        text = "COMP"
        tgt = render_word(text, 12, FONT_PATH)
        self.grid_h, self.grid_w = tgt.shape[1], tgt.shape[2]
        
        self.current_x = torch.zeros(1, self.active_channel_n, self.grid_h, self.grid_w, device=self.device)
        
        if "cloud" in self.model_dir:
            from nca.train_cloud import make_cloud_seed
            self.current_x = make_cloud_seed(tgt, self.active_channel_n, 1, self.device)
        elif "_noise" in self.model_dir:
            self.current_x = torch.rand_like(self.current_x)
        else:
            self.current_x[:, 3:, self.grid_h // 2, self.grid_w // 2] = 1.0

sessions: Dict[str, SessionState] = {}

class LoadRequest(BaseModel):
    dir: str = "snaps_web_method1"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    steps: int = 1
    session_id: str

class DamageRequest(BaseModel):
    x: float = 0.5
    y: float = 0.5
    r: float = 0.1
    session_id: str

@app.get("/")
def index():
    return FileResponse("dashboard.html")

@app.post("/api/load")
def load_model(req: LoadRequest):
    sess_id = req.session_id or str(uuid.uuid4())
    if "method2" in req.dir or "9_line" in req.dir:
        ac = 16
        hn = 80
    else:
        ac = 32
        hn = 128
    
    sessions[sess_id] = SessionState(req.dir, ac, hn)
    return {"status": "loaded", "session_id": sess_id}

@app.post("/api/reset")
def reset_grid(req: LoadRequest):
    if req.session_id not in sessions:
        return {"error": "Invalid session"}
    sessions[req.session_id].reset_grid()
    return {"status": "reset"}

@app.post("/api/step")
def step(req: StepRequest):
    if req.session_id not in sessions:
        return {"error": "Invalid session"}
    
    sess = sessions[req.session_id]
    with torch.no_grad():
        sess.current_x = sess.current_model(sess.current_x, steps=req.steps)
        
    img_tensor = to_rgba(sess.current_x)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    a = sess.current_x[0, 3:4].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_rgb = (1.0 - a + img_tensor * a).clip(0, 1)

    im = Image.fromarray((img_rgb * 255).astype(np.uint8))
    buffered = io.BytesIO()
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image": "data:image/png;base64," + img_str}

@app.post("/api/damage")
def damage(req: DamageRequest):
    if req.session_id not in sessions:
        return {"error": "Invalid session"}
    sess = sessions[req.session_id]
    
    h, w = sess.grid_h, sess.grid_w
    cx = int(req.x * w)
    cy = int(req.y * h)
    r = int(req.r * w)
    
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    mask = x**2 + y**2 <= r**2
    
    sess.current_x[0, :, mask] = 0.0
    return {"status": "damaged"}

@app.get("/api/notes")
def get_notes():
    if not os.path.exists("notes.json"):
        return {}
    with open("notes.json", "r") as f:
        return json.load(f)

class NoteRequest(BaseModel):
    dir: str = "unknown"
    note: str = ""

@app.post("/api/notes")
def post_notes(req: NoteRequest):
    notes_db = {}
    if os.path.exists("notes.json"):
        with open("notes.json", "r") as f:
            notes_db = json.load(f)
            
    if req.dir not in notes_db:
        notes_db[req.dir] = []
        
    notes_db[req.dir].append({
        "timestamp": time.time(),
        "note": req.note
    })
    with open("notes.json", "w") as f:
        json.dump(notes_db, f)
    return {"status": "saved", "notes": notes_db[req.dir]}

async def status_generator():
    while True:
        status = {}
        methods = [d for d in glob.glob("snaps_*") if os.path.isdir(d)]
        for d in methods:
            files = os.listdir(d)
            comps = []
            for f in files:
                if f.startswith('COMP_') or (f.startswith('TARGET_') and '_' in f):
                    try: comps.append(int(f.split('_')[1].split('.')[0]))
                    except ValueError: pass
            max_step = max(comps) if comps else -1
            status[d + '/'] = max_step
        yield f"data: {json.dumps(status)}\n\n"
        await asyncio.sleep(1.5)

@app.get("/api/status_stream")
async def status_stream():
    return StreamingResponse(status_generator(), media_type="text/event-stream")

# Mount everything else to static (CSS, JS, outputs)
app.mount("/", StaticFiles(directory=".", html=False), name="static")

if __name__ == "__main__":
    from nca.manager import update_methods
    try:
        update_methods("methods.json")
    except Exception as e:
        print(f"Warning: Failed to update methods.json: {e}")
        
    print(f"Starting Interactive NCA Server on http://localhost:8002/ (Device: {device})")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

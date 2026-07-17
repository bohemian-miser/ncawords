import os
import json
import time
import glob
import asyncio
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Optional, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return FileResponse("dashboard.html")

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

def fetch_status_sync():
    status = {}
    methods = [d for d in glob.glob("snaps_*") if os.path.isdir(d)]
    for d in methods:
        files = os.listdir(d)
        comps = []
        for f in files:
            if f.startswith('COMP_'):
                try: comps.append(int(f.split('_')[1].split('.')[0]))
                except ValueError: pass
        max_step = max(comps) if comps else -1
        status[d + '/'] = max_step
    return status

async def status_generator():
    while True:
        status = await asyncio.to_thread(fetch_status_sync)
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
        
    print(f"Starting Interactive NCA Orchestration Server on http://localhost:8002/")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

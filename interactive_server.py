import os
import json
import time
import glob
import asyncio
import threading
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

# ---------------------------------------------------------------------------
# Cloud state: Vertex job statuses + training runs discovered in the GCS
# bucket. The bucket is public-read, so the browser loads images directly
# from PUBLIC_BASE and the server only supplies listings/statuses.
# ---------------------------------------------------------------------------
PROJECT_ID = "recipe-lanes-staging"
VERTEX_LOCATION = "us-central1"
BUCKET_NAME = "recipe-lanes-nca-jobs"
PUBLIC_BASE = f"https://storage.googleapis.com/{BUCKET_NAME}/"

_cloud_cache = {"t": 0.0, "runs": {}, "jobs": {}, "weights": []}
_cloud_refreshing = threading.Lock()


def fetch_cloud_state(ttl=60, blocking=True):
    """Cloud listings take seconds; non-blocking callers (the SSE tick) get
    the stale cache back immediately while a thread refreshes it."""
    now = time.time()
    if now - _cloud_cache["t"] < ttl:
        return _cloud_cache
    if not blocking:
        if _cloud_refreshing.acquire(blocking=False):
            def _refresh():
                try:
                    _fetch_cloud_state_now()
                finally:
                    _cloud_refreshing.release()
            threading.Thread(target=_refresh, daemon=True).start()
        return _cloud_cache
    with _cloud_refreshing:
        if time.time() - _cloud_cache["t"] < ttl:
            return _cloud_cache
        return _fetch_cloud_state_now()


def _fetch_cloud_state_now():
    jobs = {}
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=PROJECT_ID, location=VERTEX_LOCATION)
        for j in aiplatform.CustomJob.list():
            name = j.display_name.removesuffix("-custom-job")
            jobs[name] = j.state.name.replace("JOB_STATE_", "")
    except Exception as e:
        print(f"Warning: could not list Vertex jobs: {e}")

    runs = {}
    weights = []
    try:
        from google.cloud import storage
        client = storage.Client(project=PROJECT_ID)
        for blob in client.list_blobs(BUCKET_NAME):
            if "/" not in blob.name:
                continue
            run, fname = blob.name.split("/", 1)
            if fname.startswith("COMP_"):
                try:
                    step = int(fname[5:10])
                except ValueError:
                    continue
                runs[run] = max(runs.get(run, -1), step)
            elif fname == "weights.json":
                weights.append(run)
    except Exception as e:
        print(f"Warning: could not list bucket runs: {e}")

    _cloud_cache.update({"t": time.time(), "runs": runs, "jobs": jobs,
                         "weights": weights})
    return _cloud_cache


@app.get("/api/jobs")
def get_jobs():
    return fetch_cloud_state()["jobs"]


@app.get("/api/methods")
def get_methods():
    methods = []
    if os.path.exists("methods.json"):
        with open("methods.json") as f:
            methods = json.load(f)

    cloud = fetch_cloud_state()
    local_ids = {m["id"] for m in methods}
    for run, max_step in sorted(cloud["runs"].items()):
        run_id = f"cloud_{run}"
        if run_id in local_ids:
            continue
        state = cloud["jobs"].get(run, "")
        entry = {
            "id": run_id,
            "title": f"☁ {run}",
            "dir": f"{PUBLIC_BASE}{run}/",
            "desc": f"Vertex AI run ({state or 'no active job'})",
            "seedType": "cloud",
            "cloud": True,
            "vertex_state": state,
        }
        if run in cloud["weights"]:
            entry["weights_url"] = f"{PUBLIC_BASE}{run}/weights.json"
        methods.append(entry)
    return methods

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

    # Cloud runs keyed by their public URL prefix, matching the 'dir' the
    # /api/methods endpoint hands to the frontend.
    cloud = fetch_cloud_state(blocking=False)
    for run, max_step in cloud["runs"].items():
        status[f"{PUBLIC_BASE}{run}/"] = max_step
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

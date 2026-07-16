import os
import io
import json
import time
import base64
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from nca.model import NCA, to_rgba, to_rgb
from nca.train_web_method1 import render_word_method1 as render_word, FONT_PATH

app = Flask(__name__, static_folder=".")
CORS(app)

# Global State
device = "cuda" if torch.cuda.is_available() else "cpu"
current_model = None
current_x = None
grid_h, grid_w = 0, 0
active_channel_n = 32

@app.route("/")
def index():
    return send_file("compare_webs.html")

@app.route("/api/load", methods=["POST"])
def load_model():
    global current_model, current_x, grid_h, grid_w, active_channel_n
    data = request.json
    model_dir = data.get("dir", "snaps_web_method1")
    text = "COMP"
    
    # Reload model logic
    if "method2" in model_dir or "9_line" in model_dir:
        active_channel_n = 16
        hidden_n = 80
    else:
        active_channel_n = 32
        hidden_n = 128
        
    current_model = NCA(active_channel_n, hidden_n=hidden_n).to(device)
    pth_path = Path(model_dir) / "latest.pth"
    if pth_path.exists():
        current_model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    current_model.eval()
    
    # Initialize grid
    tgt = render_word(text, 12, FONT_PATH)
    grid_h, grid_w = tgt.shape[1], tgt.shape[2]
    
    current_x = torch.zeros(1, active_channel_n, grid_h, grid_w, device=device)
    
    if "cloud" in model_dir:
        from nca.train_cloud import make_cloud_seed
        current_x = make_cloud_seed(tgt, active_channel_n, 1, device)
    elif "_noise" in model_dir:
        current_x = torch.rand_like(current_x)
    else:
        # Standard seed
        current_x[:, 3:, grid_h // 2, grid_w // 2] = 1.0
        
    return jsonify({"status": "loaded", "msg": f"Loaded model from {pth_path} (exists={pth_path.exists()})"})

@app.route("/api/reset", methods=["POST"])
def reset_grid():
    global current_x, grid_h, grid_w
    if current_model is None:
        return jsonify({"error": "No model loaded"})
        
    data = request.json
    model_dir = data.get("dir", "")
    
    tgt = render_word("COMP", 12, FONT_PATH)
    current_x = torch.zeros(1, active_channel_n, grid_h, grid_w, device=device)
    
    if "cloud" in model_dir:
        from nca.train_cloud import make_cloud_seed
        current_x = make_cloud_seed(tgt, active_channel_n, 1, device)
    elif "_noise" in model_dir:
        current_x = torch.rand_like(current_x)
    else:
        current_x[:, 3:, grid_h // 2, grid_w // 2] = 1.0
        
    return jsonify({"status": "reset"})

@app.route("/api/step", methods=["POST"])
def step():
    global current_x
    if current_model is None:
        return jsonify({"error": "No model loaded"})
        
    data = request.json
    steps = int(data.get("steps", 1))
    
    with torch.no_grad():
        current_x = current_model(current_x, steps=steps)
        
    img_tensor = to_rgba(current_x)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    # NCA backgrounds are alpha based, so handle proper visual conversion back to white
    a = current_x[0, 3:4].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_rgb = (1.0 - a + img_tensor * a).clip(0, 1)

    im = Image.fromarray((img_rgb * 255).astype(np.uint8))
    # Scale up for easy viewing
    im = im.resize((im.width * 8, im.height * 8), resample=Image.NEAREST)
    
    buffered = io.BytesIO()
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({"image": "data:image/png;base64," + img_str})

@app.route("/api/damage", methods=["POST"])
def damage():
    global current_x
    if current_model is None:
        return jsonify({"error": "No model loaded"})
        
    data = request.json
    cx_norm = data.get("x", 0.5) 
    cy_norm = data.get("y", 0.5)
    r_norm = data.get("r", 0.1) # normalized radius
    
    h, w = grid_h, grid_w
    cx = int(cx_norm * w)
    cy = int(cy_norm * h)
    r = int(r_norm * w)
    
    # Create circular mask
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    mask = x**2 + y**2 <= r**2
    
    current_x[0, :, mask] = 0.0
    return jsonify({"status": "damaged"})

@app.route("/api/notes", methods=["GET", "POST"])
def notes_api():
    notes_file = "notes.json"
    if request.method == "GET":
        if not os.path.exists(notes_file):
            return jsonify({})
        with open(notes_file, "r") as f:
            return jsonify(json.load(f))
    else:
        # POST
        data = request.json
        model_dir = data.get("dir", "unknown")
        note_text = data.get("note", "")
        
        notes_db = {}
        if os.path.exists(notes_file):
            with open(notes_file, "r") as f:
                notes_db = json.load(f)
                
        if model_dir not in notes_db:
            notes_db[model_dir] = []
            
        notes_db[model_dir].append({
            "timestamp": time.time(),
            "note": note_text
        })
        
        with open(notes_file, "w") as f:
            json.dump(notes_db, f)
            
        return jsonify({"status": "saved", "notes": notes_db[model_dir]})

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)

if __name__ == "__main__":
    print("Starting Interactive NCA Server on http://localhost:8002/")
    app.run(host="0.0.0.0", port=8002, threaded=True)

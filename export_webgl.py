import json
import os
from pathlib import Path
import torch
from nca.model import NCA
from nca.train import export_weights
import glob

def export_all():
    docs_weights = Path("docs/weights")
    docs_weights.mkdir(exist_ok=True, parents=True)
    
    # We will export our newest "word" topologies
    # Read from methods.json
    with open('methods.json', 'r') as f:
        methods_db = json.load(f)
        
    new_words = []
    
    for m in methods_db:
        if m["id"] == "proposed_targets":
            continue
            
        model_dir = m["dir"]
        pth_val = f"{model_dir}latest.pth"
        
        # Remove snaps_ and snaps_web_ to get base name
        name = model_dir.replace('snaps_web_', '').replace('snaps_', '').replace('/', '')
        
        # Infer properties
        if "method1" in model_dir or "method4" in model_dir or "method5" in model_dir or m["id"] == "cloud" or m["id"] == "guided_run":
            c_n, h_n = 32, 128
        else:
            c_n, h_n = 16, 80

        if not os.path.exists(pth_val):
            print(f"Skipping {name}: {pth_val} not found")
            continue
            
        print(f"Loading {name}...")
        device = "cpu"
        model = NCA(c_n, hidden_n=h_n).to(device)
        ckpt = torch.load(pth_val, map_location=device, weights_only=True)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        
        out_path = docs_weights / f"word_{name}.json"
        
        # Grid sizes varied, but generally they used word_geometry(4 letters)
        # In 9_line it was 68x20. In method4 it was 68x20 too or something similar
        # Since these are single-seed, the exact edge bounds matter less as long as it's big enough.
        # Let's give them 120x40 to be safe (plenty of room to grow), 
        # or we can look up their exact size if it breaks. nca.js handles torii / borders implicitly.
        export_weights(model, name, 0, 12, out_path)
        
        d = json.loads(out_path.read_text())
        
        # Omit seeds array to force center seeding
        d.update({
            "kind": "word", 
            "text": name, 
            "grid_w": 100, 
            "grid_h": 40,
            "grid": None
        })
        
        out_path.write_text(json.dumps(d))
        new_words.append(name)
        print(f"Exported {name} -> {out_path}")
        
    idxs = json.loads((docs_weights / "index.json").read_text())
    
    for w in new_words:
        if w not in idxs.get("words", []):
            idxs["words"].append(w)
            
    (docs_weights / "index.json").write_text(json.dumps(idxs))
    print("Updated docs/weights/index.json")

if __name__ == "__main__":
    export_all()

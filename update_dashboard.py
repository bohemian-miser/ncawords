import json
import os
import glob
from pathlib import Path

def get_latest_png(d):
    pngs = glob.glob(f"{d}/*.png")
    pngs = [p for p in pngs if "in.png" not in p and "tgt.png" not in p]
    if not pngs: return None
    return sorted(pngs)[-1]

def update():
    dirs = [d for d in glob.glob("snaps_*") if os.path.isdir(d)]
    
    notes = {}
    if os.path.exists("notes.json"):
        with open("notes.json") as f:
            notes = json.load(f)
            
    md = "# NCA Experiments Dashboard\n\n"
    md += "This dashboard is automatically updated by the background jobs.\n\n"
    
    for d in sorted(dirs):
        md += f"## Experiment: `{d}`\n"
        
        loss_file = os.path.join(d, "loss.txt")
        loss_val = "N/A"
        if os.path.exists(loss_file):
            with open(loss_file) as f:
                loss_val = f.read().strip()
        md += f"**Latest Loss**: {loss_val}\n\n"
                
        latest_png = get_latest_png(d)
        if latest_png:
            md += f"**Latest Output**:\n<img src=\"{latest_png}\" width=\"400\" />\n\n"
            
        if d in notes:
            md += "**Notes**:\n"
            for note in notes[d]:
                md += f"- {note}\n"
            md += "\n"
            
        md += "---\n"
        
    with open("EXPERIMENTS.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    update()

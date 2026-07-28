import json
import os
from pathlib import Path
import subprocess

def commit_weights():
    os.system("venv/bin/python export_webgl.py")
    subprocess.run(["git", "add", "docs/weights/"])
    # Do not use --no-verify, allow the 5 minute hook to run
    subprocess.run(["git", "commit", "-m", "chore: auto-commit checkpoint weights"])

if __name__ == "__main__":
    commit_weights()

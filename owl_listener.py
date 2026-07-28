import os
import json
import time
import subprocess

NOTES_FILE = "notes.json"
KEY = "snaps_proposed_targets/"
MY_AGENT_ID = "31ee1a4b-136e-4a22-861d-efb9370990eb"

def read_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    with open(NOTES_FILE, "r") as f:
        try:
            data = json.load(f)
            return data.get(KEY, [])
        except:
            return []

def main():
    print(f"Starting owl_listener for {KEY}")
    seen = set()
    initial_notes = read_notes()
    for n in initial_notes:
        if type(n) == dict:
            seen.add(n.get("note", ""))
        
    while True:
        notes = read_notes()
        for idx, n in enumerate(notes):
            if type(n) != dict:
                continue
            text = n.get("note", "")
            if text not in seen:
                seen.add(text)
                
                # Check for system trigger
                if "[SYSTEM: BEGIN TRAINING]" in text:
                    print("Received begin training signal. Notifying agent and exiting.")
                    subprocess.run(["agentapi", "send-message", MY_AGENT_ID, "USER FEEDBACK: [SYSTEM: BEGIN TRAINING]"])
                    return # Exit the listener naturally
                
                # If not from Owl, send it
                if not str(text).startswith("[Owl]"):
                    print(f"New user feedback detected: {text}")
                    subprocess.run(["agentapi", "send-message", MY_AGENT_ID, f"USER FEEDBACK: {text}"])
                    
        time.sleep(2)

if __name__ == "__main__":
    main()

import os
import json
import time

import glob

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
            
    with open('status.json', 'w') as f:
        json.dump(status, f)
        
    time.sleep(1.5)

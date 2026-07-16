import os
import json
import time

methods = ['snaps_web_method1', 'snaps_web_method1_noise',
           'snaps_web_method2', 'snaps_web_method2_noise',
           'snaps_web_method4', 'snaps_web_method4_noise',
           'snaps_web_method5', 'snaps_web_method5_noise',
           'snaps_9_line',      'snaps_9_line_noise',
           'snaps_cloud',       'snaps_web_evaporate',
           'snaps_web_hidden']

while True:
    status = {}
    for d in methods:
        if os.path.exists(d):
            files = os.listdir(d)
            comps = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith('COMP_')]
            max_step = max(comps) if comps else -1
            status[d + '/'] = max_step
        else:
            status[d + '/'] = -1
            
    with open('status.json', 'w') as f:
        json.dump(status, f)
        
    time.sleep(1.5)

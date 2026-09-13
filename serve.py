#!/usr/bin/env python3
"""Local Server for NCA and Lenia Interactive Viewer

Serves:
  - Runs Gallery (Lenia & NCA): http://localhost:8000/docs/lenia.html
  - Interactive Playground: http://localhost:8000/docs/dashboard.html
  - Interactive Article: http://localhost:8000/docs/index.html
  - Emulated GCS storage API: /api/storage/o (delivers local nca_runs)
  - Auto-sync: periodically pulls fresh snapshots and weights from aisb in the background
"""

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / 'nca_runs'
DOCS_DIR = BASE_DIR / 'docs'
PORT = int(os.environ.get('PORT', 8000))


class LocalViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self):
        # Enable CORS and disable aggressive caching for local dev
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Root landing redirect
        if path in ('/', '/index.html'):
            self.send_response(302)
            self.send_header('Location', '/docs/lenia.html')
            self.end_headers()
            return

        # Storage API emulation for local runs
        if path == '/api/storage/o':
            self.handle_storage_api(parsed.query)
            return

        # Fallback to normal file server
        super().do_GET()

    def handle_storage_api(self, query_str):
        qs = parse_qs(query_str)
        prefix = qs.get('prefix', [''])[0]
        delimiter = qs.get('delimiter', [''])[0]

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        if delimiter == '/' and not prefix:
            # Top-level directory listing: prefixes of all runs
            prefixes = []
            if RUNS_DIR.exists():
                for d in sorted(RUNS_DIR.iterdir()):
                    if d.is_dir() and not d.name.startswith('.'):
                        prefixes.append(f'{d.name}/')
            payload = {'prefixes': prefixes}
            self.wfile.write(json.dumps(payload).encode('utf-8'))
            return

        if prefix:
            # Per-run file listing
            run_name = prefix.strip('/')
            run_path = RUNS_DIR / run_name
            items = []
            if run_path.exists() and run_path.is_dir():
                for f in sorted(run_path.iterdir()):
                    if f.is_file() and not f.name.startswith('.'):
                        mtime = datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat()
                        items.append({'name': f'{run_name}/{f.name}', 'updated': mtime})
            payload = {'items': items}
            self.wfile.write(json.dumps(payload).encode('utf-8'))
            return

        # Listing for dashboard (finds all runs with weights.json)
        items = []
        if RUNS_DIR.exists():
            for d in sorted(RUNS_DIR.iterdir()):
                if d.is_dir():
                    w = d / 'weights.json'
                    if w.exists():
                        mtime = datetime.fromtimestamp(w.stat().st_mtime, timezone.utc).isoformat()
                        items.append({'name': f'{d.name}/weights.json', 'updated': mtime})
        payload = {'items': items}
        self.wfile.write(json.dumps(payload).encode('utf-8'))


def background_syncer():
    """Periodically pull latest snapshots and weights from aisb."""
    print('[AutoSync] Background syncer active: syncing from aisb every 30s...', flush=True)
    time.sleep(5)
    while True:
        try:
            cmd = [
                'rsync', '-rlptDz',
                '-e', 'ssh -o IdentitiesOnly=yes -o IdentityAgent=none',
                '--include=snaps_lenia*',
                '--include=snaps_grid_1f600*',
                '--include=snaps_lenia*/**',
                '--include=snaps_grid_1f600*/**',
                '--exclude=*',
                'aisb:/home/ubuntu/user4/aisb/nca/nca_runs/',
                str(RUNS_DIR) + '/'
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            
            # Also sync docs/weights
            cmd_w = [
                'rsync', '-rlptDz',
                '-e', 'ssh -o IdentitiesOnly=yes -o IdentityAgent=none',
                'aisb:/home/ubuntu/user4/aisb/nca/docs/weights/',
                str(DOCS_DIR / 'weights') + '/'
            ]
            subprocess.run(cmd_w, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        except Exception:
            pass
        time.sleep(30)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    sync_thread = threading.Thread(target=background_syncer, daemon=True)
    sync_thread.start()

    # Find open port
    port = PORT
    for p in range(PORT, PORT + 20):
        try:
            httpd = ThreadedHTTPServer(('127.0.0.1', p), LocalViewerHandler)
            port = p
            break
        except OSError:
            continue

    print('=' * 65)
    print(f' Local Viewer Server running at http://localhost:{port}')
    print('=' * 65)
    print(f'   Runs Gallery (Lenia + NCA): http://localhost:{port}/docs/lenia.html')
    print(f'   Interactive Playground:     http://localhost:{port}/docs/dashboard.html')
    print(f'   Interactive Article:        http://localhost:{port}/docs/index.html')
    print('=' * 65)
    print('Press Ctrl+C to stop.')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')


if __name__ == '__main__':
    main()

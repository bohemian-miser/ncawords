#!/bin/bash
# Serve the static site (gallery, article, dashboard) from docs/ .
set -euo pipefail
cd "$(dirname "$0")"
echo "Gallery:    http://localhost:8000/lenia.html"
echo "Article:    http://localhost:8000/index.html"
echo "Dashboard:  http://localhost:8000/dashboard.html"
exec python3 -m http.server 8000 --directory docs

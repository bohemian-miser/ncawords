#!/bin/bash
# Copy the dashboard into docs/ (served as the GitHub Pages site root),
# rewriting paths that differ between repo-root serving and docs-root serving.
set -e
cd "$(dirname "$0")/.."
cp dashboard.html docs/dashboard.html
cp dashboard.css docs/dashboard.css
sed "s|docs/weights/|weights/|g" dashboard.js > docs/dashboard.js
sed "s|'./docs/nca.js'|'./nca.js'|" nca_viewer.js > docs/nca_viewer.js
echo "Dashboard deployed into docs/. Publish by copying docs/ onto the gh-pages branch."

#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNTIME="$HERE/.runtime/sprite32"
PYTHON=python3
mkdir -p "$HERE/.runtime"
command -v git >/dev/null 2>&1 || { echo "Git is required."; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python 3 is required."; exit 1; }
if [ ! -d "$RUNTIME/.git" ]; then git clone --depth 1 https://github.com/danemarguglio/sprite32.git "$RUNTIME"; fi
if [ ! -d "$RUNTIME/.venv" ]; then "$PYTHON" -m venv "$RUNTIME/.venv"; fi
source "$RUNTIME/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$RUNTIME/requirements.txt"
python -m pip install "huggingface_hub[cli]"
mkdir -p "$RUNTIME/weights"
if [ ! -f "$RUNTIME/weights/sprite32-1.0-fast/run_config.json" ]; then
  hf download danemarguglio/sprite32 --include "sprite32-1.0-fast/*" --local-dir "$RUNTIME/weights"
fi
cp "$HERE/index.html" "$RUNTIME/scripts/sprite_web.html"
cd "$RUNTIME"
echo "Pixel Foundry: http://127.0.0.1:8077"
if command -v open >/dev/null 2>&1; then open "http://127.0.0.1:8077" >/dev/null 2>&1 || true; fi
python scripts/sprite_web.py "sprite32 1.0 fast=weights/sprite32-1.0-fast" --host 127.0.0.1 --port 8077 --no_compare

#!/bin/bash
# =============================================
#  Beatrice V2 Trainer - Linux / WSL / Git Bash start script
#  (on Windows itself, double-click start_windows.bat instead)
# =============================================
#  1. finds Python 3.10-3.13 and creates ./venv
#  2. installs requirements.txt (web server + PyTorch + trainer dependencies)
#  3. downloads the trainer's pretrained weights and audio assets from
#     Hugging Face (fierce-cats/beatrice-trainer) if they are missing
#  4. starts the server on http://127.0.0.1:8000 and opens the browser
#     (set BEATRICE_NO_BROWSER=1 to skip opening the browser)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================="
echo "  Beatrice V2 Trainer"
echo "============================================="
echo ""

# --- 1. Python ---
PYTHON=""
# The trainer needs Python >= 3.10 (its UTMOS quality model uses 3.10 syntax);
# PyTorch < 2.9 (required by torchaudio < 2.9) has no wheels for 3.14.
PY_OK='import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)'
for candidate in python3.12 python3.11 python3.13 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$PY_OK" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python 3.10 - 3.13 is required but was not found."
    echo "        Install Python 3.12 from https://www.python.org/downloads/"
    exit 1
fi
echo "Using Python: $("$PYTHON" --version 2>&1) ($(command -v "$PYTHON"))"

# --- 2. Virtual environment ---
VENV_PY=venv/bin/python
[ -x venv/Scripts/python.exe ] && VENV_PY=venv/Scripts/python.exe
if [ -d "venv" ] && ! "$VENV_PY" -c "$PY_OK" >/dev/null 2>&1; then
    echo "The existing venv is broken or uses an unsupported Python - recreating it..."
    rm -rf venv
fi
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

echo "Installing dependencies (the first run downloads PyTorch, several hundred MB)..."
python -m pip install -q --upgrade pip
if ! python -m pip install -q -r requirements.txt; then
    echo ""
    echo "[ERROR] Installing requirements.txt failed (see the messages above)."
    echo "        pyworld may need a C/C++ compiler (e.g. build-essential) to install."
    exit 1
fi

# --- 3. Trainer weights / assets ---
mkdir -p beatrice-trainer/data beatrice-trainer/output
if ! python server.py --setup; then
    echo ""
    echo "[WARNING] The trainer is not ready (see the error above): its weights could not be"
    echo "          downloaded or a dependency failed to import. Training stays unavailable"
    echo "          until this is fixed. Re-run ./start.sh, or use 'Download or update trainer'"
    echo "          in Settings for download problems."
fi

# --- 4. Server ---
echo ""
echo "============================================="
echo "  Starting server on http://127.0.0.1:8000"
echo "  Press Ctrl+C to stop."
echo "============================================="
echo ""

if [ -n "$BEATRICE_NO_BROWSER" ]; then
    exec python server.py
else
    exec python server.py --open-browser
fi

#!/bin/bash
# =============================================
#  Beatrice V2 Trainer — Cross-Platform Startup Script
# =============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================="
echo "  🎙️  Beatrice V2 Trainer"
echo "============================================="
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python 3 is required but not found."
    echo "   Install it from https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Using Python: $($PYTHON --version)"

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Download Beatrice trainer if needed
if [ ! -d "beatrice-trainer" ] || [ ! -f "beatrice-trainer/beatrice_trainer/__main__.py" ]; then
    echo ""
    echo "📥 Downloading Beatrice Trainer from HuggingFace..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fierce-cats/beatrice-trainer', local_dir='beatrice-trainer', allow_patterns=['.gitattributes', '.gitignore', 'LICENSE', 'README.md', 'assets/README.md', 'assets/default_config.json', 'assets/images/*', 'assets/pretrained/*', 'assets/test/*', 'assets/ir/000*.flac', 'assets/noise/-*.flac', 'beatrice_trainer/*', 'pyproject.toml'])"
    echo "✅ Beatrice Trainer downloaded."
fi

# Ensure data directories exist
mkdir -p beatrice-trainer/data
mkdir -p beatrice-trainer/output

echo ""
echo "============================================="
echo "  🚀 Starting server on http://localhost:8000"
echo "============================================="
echo ""

# Open browser after a short delay
(sleep 2 && open "http://localhost:8000" 2>/dev/null || xdg-open "http://localhost:8000" 2>/dev/null || start "http://localhost:8000" 2>/dev/null) &

# Start the server
$PYTHON server.py

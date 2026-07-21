@echo off
setlocal enabledelayedexpansion

echo =============================================
echo   🎙️  Beatrice V2 Trainer for Windows
echo =============================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python is required but not found on PATH.
    echo    Please install Python 3.10+ from https://www.python.org/downloads/
    echo    (Be sure to check "Add python.exe to PATH" during installation)
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version') do set PYTHON_VERSION=%%v
echo ✅ Using Python: !PYTHON_VERSION!

:: 2. Create virtual environment if needed
if not exist "venv" (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Activate venv
call venv\Scripts\activate.bat

:: 4. Install PyTorch with CUDA & requirements
echo.
echo 📦 Checking & installing backend dependencies...
python -c "import torch; print('PyTorch CUDA status:', torch.cuda.is_available())" >nul 2>nul
if %errorlevel% neq 0 (
    echo ⚡ Installing PyTorch with CUDA 12.1 acceleration...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
)

pip install -q -r requirements.txt

:: 5. Download Beatrice trainer from HuggingFace if needed
if not exist "beatrice-trainer\beatrice_trainer\__main__.py" (
    echo.
    echo 📥 Downloading Beatrice Trainer package from HuggingFace...
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fierce-cats/beatrice-trainer', local_dir='beatrice-trainer', allow_patterns=['.gitattributes', '.gitignore', 'LICENSE', 'README.md', 'assets/README.md', 'assets/default_config.json', 'assets/images/*', 'assets/pretrained/*', 'assets/test/*', 'assets/ir/000*.flac', 'assets/noise/-*.flac', 'beatrice_trainer/*', 'pyproject.toml'])"
    echo ✅ Beatrice Trainer downloaded successfully.
)

:: 6. Ensure data directories exist
if not exist "beatrice-trainer\data" mkdir "beatrice-trainer\data"
if not exist "beatrice-trainer\output" mkdir "beatrice-trainer\output"

echo.
echo =============================================
echo   🚀 Starting server on http://localhost:8000
echo =============================================
echo.

:: Open default browser after 3 seconds
start "" "http://localhost:8000"

:: 7. Start FastAPI server
python server.py

pause

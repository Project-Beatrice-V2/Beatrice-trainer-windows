@echo off
rem =============================================
rem  Beatrice V2 Trainer - Windows start script
rem =============================================
rem  1. finds Python 3.10 - 3.13 and creates .\venv
rem  2. installs PyTorch [CUDA build when an NVIDIA GPU is present]
rem     and then requirements.txt
rem  3. downloads the trainer's pretrained weights and audio assets from
rem     Hugging Face [fierce-cats/beatrice-trainer] if they are missing
rem  4. starts the server on http://127.0.0.1:8000 and opens the browser
rem
rem  Optional environment variables:
rem    BEATRICE_TORCH_INDEX=cu126 / cu128 / cpu   force a PyTorch wheel index
rem    BEATRICE_NO_BROWSER=1                      do not open the browser
rem
rem  Note: this file deliberately avoids parentheses, ampersands and other
rem  cmd.exe special characters inside echo lines and IF blocks.

setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Beatrice V2 Trainer

echo =============================================
echo   Beatrice V2 Trainer for Windows
echo =============================================
echo.

rem ---- 1. Find Python 3.10 - 3.13 ----
rem The trainer needs Python 3.10+; PyTorch below 2.9 has no wheels for 3.14.
set "PY="
set "PY_CHECK=import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)"
python -c "%PY_CHECK%" >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY call :try_py_launcher 3.12
if not defined PY call :try_py_launcher 3.11
if not defined PY call :try_py_launcher 3.13
if not defined PY call :try_py_launcher 3.10
if not defined PY (
    echo [ERROR] Python 3.10 - 3.13 was not found.
    echo         Install Python 3.12 from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" in the installer.
    echo.
    pause
    exit /b 1
)
echo Using Python:
%PY% --version

rem ---- 2. Virtual environment ----
set "VPY=%CD%\venv\Scripts\python.exe"
if exist "%VPY%" "%VPY%" -c "%PY_CHECK%" >nul 2>nul
if exist "%VPY%" if errorlevel 1 (
    echo The existing venv is broken or uses an unsupported Python - recreating it...
    rmdir /s /q "venv"
)
if not exist "%VPY%" (
    echo.
    echo Creating virtual environment...
    %PY% -m venv venv
)
if not exist "%VPY%" (
    echo [ERROR] Failed to create the virtual environment in "%CD%\venv".
    echo.
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"

echo.
echo Installing backend dependencies - the first run downloads about 3 GB, mostly PyTorch...
"%VPY%" -m pip install -q --upgrade pip

rem ---- 3. PyTorch: CUDA build when an NVIDIA GPU is present ----
rem RTX 50xx [compute capability 12.x] needs the cu128 wheels; older cards use cu126.
set "TORCH_INDEX="
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "TORCH_INDEX=cu126"
if defined TORCH_INDEX (
    for /f "tokens=1 delims=." %%c in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul ^| findstr /r "^[0-9]"') do (
        if %%c GEQ 12 set "TORCH_INDEX=cu128"
    )
)
if defined BEATRICE_TORCH_INDEX set "TORCH_INDEX=%BEATRICE_TORCH_INDEX%"
if not defined TORCH_INDEX goto :torch_from_requirements

rem Reinstall only when torch is missing, CPU-only, or torchaudio is 2.9+.
"%VPY%" -c "import sys, torch, torchaudio; v = tuple(int(x) for x in torchaudio.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if (torch.version.cuda or '%TORCH_INDEX%' == 'cpu') and v < (2, 9) else 1)" >nul 2>nul
if not errorlevel 1 goto :torch_ready
echo Installing PyTorch from the %TORCH_INDEX% wheel index...
"%VPY%" -m pip uninstall -y -q torch torchaudio >nul 2>nul
"%VPY%" -m pip install "torch>=2.1,<2.9" "torchaudio>=2.1,<2.9" --index-url https://download.pytorch.org/whl/%TORCH_INDEX%
if errorlevel 1 (
    echo [WARNING] Could not install PyTorch from the %TORCH_INDEX% index.
    echo           requirements.txt will install the CPU build instead - training will be slow.
)
goto :torch_ready

:torch_from_requirements
echo No NVIDIA GPU found - PyTorch for CPU will be installed from requirements.txt.

:torch_ready
"%VPY%" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Installing requirements.txt failed - see the messages above.
    echo.
    pause
    exit /b 1
)

rem ---- 4. Trainer weights and assets ----
if not exist "beatrice-trainer\data" mkdir "beatrice-trainer\data"
if not exist "beatrice-trainer\output" mkdir "beatrice-trainer\output"
"%VPY%" server.py --setup
if errorlevel 1 (
    echo.
    echo [WARNING] The trainer is not ready - see the error above. Its weights could not be
    echo           downloaded or a dependency failed to import. Training stays unavailable
    echo           until this is fixed. Re-run start_windows.bat, or use
    echo           "Download or update trainer" in Settings for download problems.
)

rem ---- 5. Server ----
set "OPEN_FLAG=--open-browser"
if defined BEATRICE_NO_BROWSER set "OPEN_FLAG="
echo.
echo =============================================
echo   Starting server on http://127.0.0.1:8000
echo   Close this window or press Ctrl+C to stop.
echo =============================================
echo.
"%VPY%" server.py %OPEN_FLAG%
echo.
echo Server stopped.
pause
endlocal
exit /b 0

rem ---- helpers ----
:try_py_launcher
where py >nul 2>nul
if errorlevel 1 exit /b 0
py -%1 -c "%PY_CHECK%" >nul 2>nul
if not errorlevel 1 set "PY=py -%1"
exit /b 0

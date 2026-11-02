"""
Beatrice V2 Trainer - Windows backend server
============================================
FastAPI server behind the Beatrice V2 trainer web UI on Windows.

Training always runs the real ``beatrice_trainer`` package
(``beatrice-trainer/beatrice_trainer``) in a subprocess started through
``beatrice-trainer/webui_runner.py``; it uses an NVIDIA GPU via CUDA when
available, otherwise the CPU. There is no simulated / fallback training: if the trainer
cannot start, ``/api/train/status`` reports ``state: "failed"`` with an
``error`` message.

Run ``python server.py --setup`` to download the trainer's pretrained weights
and audio assets from Hugging Face (``fierce-cats/beatrice-trainer``).
"""

import gc
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import zipfile
try:
    import psutil
except ImportError:
    psutil = None
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# ---------- Configuration ----------
PLATFORM_LABEL = "Windows"
BASE_DIR = Path(__file__).parent.resolve()
BEATRICE_DIR = BASE_DIR / "beatrice-trainer"
DATA_DIR = BEATRICE_DIR / "data"            # one sub-folder of WAVs per speaker
OUTPUT_DIR = BEATRICE_DIR / "output"        # trained models, one sub-folder per speaker
STAGING_DIR = BEATRICE_DIR / "data_active"  # per-run dataset view handed to the trainer
WORK_DIR = BEATRICE_DIR / "work"            # per-run config, job and status files
TRAINER_MAIN = BEATRICE_DIR / "beatrice_trainer" / "__main__.py"
TRAINER_RUNNER = BEATRICE_DIR / "webui_runner.py"
DEFAULT_CONFIG_FILE = BEATRICE_DIR / "assets" / "default_config.json"
TRAINER_STDOUT_LOG = BASE_DIR / "trainer_stdout.log"
STATUS_FILE = WORK_DIR / "training_status.json"

HF_REPO_ID = "fierce-cats/beatrice-trainer"
HF_REPO = f"https://huggingface.co/{HF_REPO_ID}"
# Binary assets the trainer needs. They are downloaded from Hugging Face (never
# committed: Git LFS pointer stubs of these files used to be in the repo).
TRAINER_ASSET_PATTERNS = [
    "assets/pretrained/*",
    "assets/ir/*.flac",
    "assets/noise/-*.flac",
    "assets/test/*.wav",
    "assets/images/noimage.png",
]
PRETRAINED_CONFIG_KEYS = ("phone_extractor_file", "pitch_estimator_file", "pretrained_file")
DEFAULT_PRETRAINED_FILES = {
    "phone_extractor_file": "assets/pretrained/122_checkpoint_03000000.pt",
    "pitch_estimator_file": "assets/pretrained/104_3_checkpoint_00300000.pt",
    "pretrained_file": "assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz",
}
MIN_PRETRAINED_BYTES = 1_000_000
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
MIN_PYTHON = (3, 10)
REQUIRED_TRAINER_MODULES = ("numpy", "torch", "torchaudio", "soundfile", "pyworld", "tqdm", "tensorboard")
AUDIO_SUFFIXES = (".wav", ".flac")

# Bind to loopback by default: the API can delete data and start processes.
# Set BEATRICE_HOST=0.0.0.0 only if you really want it reachable from your LAN.
HOST = os.environ.get("BEATRICE_HOST", "127.0.0.1")
try:
    PORT = int(os.environ.get("BEATRICE_PORT", "8000"))
except ValueError:
    PORT = 8000
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

# Ensure dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- App ----------
app = FastAPI(
    title="Beatrice V2 Trainer API",
    description=f"{PLATFORM_LABEL} backend for the Beatrice V2 voice conversion trainer",
    version="2.1.0",
)

# The UI is served from this same server, so CORS is only needed for a UI
# opened from another local port. Never allow arbitrary websites to call it.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
if HOST in LOOPBACK_HOSTS:
    # Blocks DNS-rebinding attacks from web pages against the local API.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

# ---------- Training State ----------
training_process: Optional[subprocess.Popen] = None
training_lock = threading.Lock()


def _idle_training_state():
    return {
        "active": False,
        "state": "idle",          # idle | starting | running | completed | failed | stopped
        "completed": False,
        "error": None,
        "exit_code": None,
        "phase": None,
        "speaker": None,
        "step": 0,
        "total_steps": 0,
        "loss": None,
        "start_time": None,
        "end_time": None,
        "memory_usage": None,
        "steps_per_sec": None,
        "eta_seconds": None,
        "device": None,
        "output_dir": None,
        "pid": None,
        "notes": [],
    }


training_state = _idle_training_state()

# ---------- Hugging Face / Kaggle Import State ----------
hf_import_state = {
    "active": False,
    "status": "",
    "extracted_count": 0,
    "total_count": 0,
}


# ==================== HELPERS ====================

def clean_speaker_name(raw) -> str:
    """Speaker names become folder names: letters, digits, '-' and '_' only."""
    name = "".join(c for c in str(raw or "").strip() if c.isalnum() or c in "-_")[:64]
    if not name:
        raise HTTPException(
            status_code=400,
            detail="Invalid speaker name. Use letters, numbers, '-' and '_' only.",
        )
    return name


def safe_filename(raw, default: str) -> str:
    """Reduce an uploaded/archived file name to a safe base name (no paths)."""
    base = os.path.basename(str(raw or "").replace("\\", "/")).strip()
    base = "".join(c for c in base if ord(c) >= 32 and c not in '<>:"/\\|?*')
    if not base or base.startswith(".") or base in ("..",):
        base = default
    return base[:200]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def _is_real_file(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes and not _is_lfs_pointer(path)
    except OSError:
        return False


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _tail_lines(path: Path, n: int = 15) -> list:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        if "\r" in line:
            line = line.split("\r")[-1]
        line = line.rstrip()
        if line:
            out.append(line)
    return out


def _training_running() -> bool:
    return training_process is not None and training_process.poll() is None


def _require_not_training(what: str):
    if _training_running():
        raise HTTPException(status_code=409, detail=f"Cannot {what} while training is running. Stop training first.")


def _audio_files(folder: Path) -> list:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


# ---------- Trainer installation checks ----------

def pretrained_paths() -> list:
    cfg = _read_json(DEFAULT_CONFIG_FILE)
    return [BEATRICE_DIR / str(cfg.get(key) or DEFAULT_PRETRAINED_FILES[key]) for key in PRETRAINED_CONFIG_KEYS]


def find_lfs_pointer_stubs() -> list:
    assets = BEATRICE_DIR / "assets"
    if not assets.is_dir():
        return []
    return [p for p in assets.rglob("*") if p.is_file() and _is_lfs_pointer(p)]


def trainer_asset_problems() -> list:
    """Human-readable reasons the trainer code/assets are not usable (empty = OK)."""
    problems = []
    if not TRAINER_MAIN.is_file():
        problems.append("trainer code is missing (beatrice-trainer/beatrice_trainer/__main__.py); re-download this repository")
    if not TRAINER_RUNNER.is_file():
        problems.append("beatrice-trainer/webui_runner.py is missing; re-download this repository")
    if not DEFAULT_CONFIG_FILE.is_file():
        problems.append("beatrice-trainer/assets/default_config.json is missing; re-download this repository")
    for p in pretrained_paths():
        if not _is_real_file(p, MIN_PRETRAINED_BYTES):
            problems.append(f"pretrained weights not downloaded: {p.relative_to(BEATRICE_DIR).as_posix()}")
    for sub in ("ir", "noise", "test"):
        d = BEATRICE_DIR / "assets" / sub
        has_audio = d.is_dir() and any(
            _is_real_file(f, 64) for f in d.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES
        )
        if not has_audio:
            problems.append(f"trainer audio assets not downloaded: assets/{sub}/")
    if not _is_real_file(BEATRICE_DIR / "assets" / "images" / "noimage.png", 64):
        problems.append("trainer asset not downloaded: assets/images/noimage.png")
    stubs = find_lfs_pointer_stubs()
    if stubs:
        problems.append(f"{len(stubs)} Git LFS pointer stub file(s) in beatrice-trainer/assets instead of real files")
    return problems


def missing_trainer_modules() -> list:
    return [m for m in REQUIRED_TRAINER_MODULES if importlib.util.find_spec(m) is None]


def trainer_setup_problems() -> list:
    problems = trainer_asset_problems()
    if sys.version_info < MIN_PYTHON:
        problems.append(
            f"Python {sys.version.split()[0]} is too old: the trainer needs Python 3.10 - 3.13 "
            "(its UTMOS quality-check model uses 3.10 syntax). Delete the venv folder and re-run "
            "the start script with a newer Python"
        )
    missing = missing_trainer_modules()
    if missing:
        problems.append(
            "missing Python packages: " + ", ".join(missing)
            + " (run the start script again, or: pip install -r requirements.txt)"
        )
    return problems


def download_trainer_assets(log=print) -> dict:
    """Download pretrained weights + audio assets from Hugging Face into beatrice-trainer/."""
    if not TRAINER_MAIN.is_file():
        raise RuntimeError(
            "The trainer code (beatrice-trainer/beatrice_trainer/__main__.py) is missing from this "
            "folder. Re-download the repository; it ships a patched trainer that reports progress."
        )
    removed = 0
    for stub in find_lfs_pointer_stubs():
        stub.unlink()
        removed += 1
    for p in pretrained_paths():
        if p.exists() and not _is_real_file(p, MIN_PRETRAINED_BYTES):
            p.unlink()
            removed += 1
    if removed:
        log(f"Removed {removed} placeholder / incomplete asset file(s).")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub is not installed. Run: pip install -r requirements.txt") from e

    log(f"Downloading trainer assets from {HF_REPO} (about 190 MB the first time) ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(BEATRICE_DIR),
        allow_patterns=TRAINER_ASSET_PATTERNS,
        max_workers=8,
    )
    problems = trainer_asset_problems()
    if problems:
        raise RuntimeError("Download finished but the trainer is still incomplete: " + "; ".join(problems))
    log("Trainer assets are ready.")
    return {"removed_placeholders": removed}


# ==================== WINDOWS HARDWARE PROFILING ====================

def get_hardware_info():
    """Profile the Windows PC (NVIDIA CUDA GPU or CPU) for training config recommendations."""
    info = {
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "is_windows": platform.system() == "Windows",
        "cpu_count": os.cpu_count() or 1,
        "cpu_count_physical": None,
        "ram_total_gb": 0,
        "ram_available_gb": 0,
        "compute_backend": "cpu",
        "cuda_available": False,
        "cuda_version": None,
        "torch_version": None,
        "torch_cuda_build": False,
        "gpu_name": None,
        "vram_total_gb": 0,
        "vram_available_gb": 0,
    }

    # Physical CPU count and RAM
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False) or info["cpu_count"]
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    except ImportError:
        info["cpu_count_physical"] = info["cpu_count"]

    # PyTorch & NVIDIA CUDA detection
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["torch_cuda_build"] = torch.version.cuda is not None
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["compute_backend"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_total_gb"] = round(props.total_memory / (1024**3), 1)
            try:
                free_mem, _total_mem = torch.cuda.mem_get_info()
                info["vram_available_gb"] = round(free_mem / (1024**3), 1)
            except Exception:
                info["vram_available_gb"] = info["vram_total_gb"]
    except ImportError:
        info["torch_version"] = None

    if not info["gpu_name"]:
        info["gpu_name"] = platform.processor() or "CPU"
    return info


def get_optimal_training_config(hw_info: dict, num_audio_files: int = 100):
    """
    Recommend training parameters for this PC.

    Only settings the real beatrice_trainer understands are tuned (batch size,
    data loader workers, mixed precision, torch.compile). beatrice_trainer has
    no gradient accumulation, so gradient_accumulation_steps is always 1.
    """
    ram_gb = hw_info.get("ram_total_gb", 16)
    vram_gb = hw_info.get("vram_total_gb", 0)
    is_cuda = hw_info.get("cuda_available", False)

    config = {
        "device": "cuda" if is_cuda else "cpu",
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 2,
        # Windows starts DataLoader workers with "spawn"; they cannot import the
        # trainer's dataset classes (defined in beatrice_trainer/__main__.py).
        "num_workers": 0,
        "pin_memory": bool(is_cuda),
        "mixed_precision": bool(is_cuda),
        "learning_rate": "5.0e-05",
        "max_steps": 10000,
        "save_every": 2000,
        "log_every": 1,
        "max_memory_gb": round(vram_gb, 1) if is_cuda else round(ram_gb * 0.5, 1),
        "prefetch_factor": None,
        "persistent_workers": False,
        # torch.compile needs Triton, which PyTorch does not ship for Windows.
        "compile_model": False,
        "channels_last": False,
        "recommended_steps_for_dataset": 10000,
        "estimated_time_minutes": 0,
        "optimization_notes": [],
    }

    if is_cuda:
        gpu_name = hw_info.get("gpu_name") or "NVIDIA GPU"
        config["optimization_notes"].append(f"NVIDIA CUDA active: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb >= 20:       # 24 GB cards (RTX 3090 / 4090 / 5090)
            config["batch_size"] = 16
        elif vram_gb >= 15:     # 16 GB cards (RTX 4080 / 4060 Ti 16 GB / 5080)
            config["batch_size"] = 8
        elif vram_gb >= 11:     # 12 GB cards (RTX 3060 12 GB / 4070)
            config["batch_size"] = 4
        elif vram_gb >= 7:      # 8 GB cards
            config["batch_size"] = 2
        else:                   # < 8 GB
            config["batch_size"] = 1
        config["optimization_notes"].append("Mixed precision (FP16 autocast) on for CUDA")
    else:
        config["optimization_notes"].append(
            "No NVIDIA CUDA GPU found - CPU training is very slow; consider the Colab notebook (GPU)"
        )

    config["effective_batch_size"] = config["batch_size"]
    config["optimization_notes"].append(
        "Data loader workers: 0 (worker processes are not supported by the trainer on Windows)"
    )

    # Steps recommendation based on dataset size
    if num_audio_files < 30:
        config["recommended_steps_for_dataset"] = 5000
    else:
        config["recommended_steps_for_dataset"] = 10000
    config["max_steps"] = config["recommended_steps_for_dataset"]

    # Very rough time estimate (not measured on Windows hardware).
    steps_per_sec = 2.0 if is_cuda else 0.05
    config["estimated_time_minutes"] = round(config["max_steps"] / steps_per_sec / 60, 1)

    # Learning rate: upstream default is 5e-5 at batch size 8; scale linearly.
    scaled_lr = 5e-5 * (config["batch_size"] / 8)
    config["learning_rate"] = f"{scaled_lr:.1e}"

    return config


def trainer_env() -> dict:
    """Extra environment variables for the trainer subprocess on Windows."""
    return {}


def _get_memory_platform(mem: dict):
    """Windows part of /api/system/memory: device-wide NVIDIA GPU memory."""
    mem.update({"gpu_backend": "none", "gpu_used_gb": 0.0, "gpu_allocated_gb": 0.0, "gpu_total_gb": 0.0})
    try:
        import torch
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            mem["gpu_backend"] = "cuda"
            mem["gpu_total_gb"] = round(total_mem / (1024**3), 2)
            # Training runs in a subprocess, so report device-wide usage.
            mem["gpu_used_gb"] = round((total_mem - free_mem) / (1024**3), 2)
            mem["gpu_allocated_gb"] = mem["gpu_used_gb"]
    except Exception:
        pass


# ==================== ENDPOINTS ====================

# --- System Status ---
@app.get("/api/status")
def get_status():
    """Hardware profile, system status and trainer installation state."""
    hw = get_hardware_info()
    problems = trainer_setup_problems()
    return {
        "status": "online",
        "system": hw["system"],
        "platform": hw["platform"],
        "device": hw["gpu_name"] or hw["machine"],
        "compute": hw["compute_backend"],
        "gpu_name": hw["gpu_name"],
        "cuda_available": hw["cuda_available"],
        "cuda_version": hw["cuda_version"],
        "torch_cuda_build": hw["torch_cuda_build"],
        "is_windows": hw["is_windows"],
        "ram_total_gb": hw["ram_total_gb"],
        "ram_available_gb": hw["ram_available_gb"],
        "vram_total_gb": hw["vram_total_gb"],
        "vram_available_gb": hw["vram_available_gb"],
        "cpu_count": hw["cpu_count"],
        "cpu_count_physical": hw["cpu_count_physical"],
        "python": sys.version.split()[0],
        "torch_version": hw["torch_version"],
        "beatrice_installed": not problems,
        "beatrice_problems": problems,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
    }


# --- Auto-Config ---
@app.get("/api/train/auto-config")
def get_auto_config():
    """Recommended training configuration for this hardware."""
    hw = get_hardware_info()

    total_files = 0
    if DATA_DIR.exists():
        for speaker_dir in DATA_DIR.iterdir():
            if speaker_dir.is_dir():
                total_files += len(list(speaker_dir.glob("*.wav")))

    config = get_optimal_training_config(hw, total_files)
    config["hardware"] = {
        "chip": hw.get("gpu_name") or hw.get("processor") or "Unknown",
        "ram_gb": hw["ram_total_gb"],
        "vram_gb": hw["vram_total_gb"],
        "compute": hw["compute_backend"],
        "cuda": hw["cuda_available"],
    }
    return config


# --- Memory Monitor ---
@app.get("/api/system/memory")
def get_memory():
    """Real-time memory usage for training monitoring."""
    mem = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        mem["ram_total_gb"] = round(vm.total / (1024**3), 2)
        mem["ram_used_gb"] = round(vm.used / (1024**3), 2)
        mem["ram_available_gb"] = round(vm.available / (1024**3), 2)
        mem["ram_percent"] = vm.percent
    except ImportError:
        pass
    _get_memory_platform(mem)
    return mem


# --- Dataset Management ---
@app.get("/api/dataset/list")
def list_datasets():
    """List all speaker datasets with audio stats."""
    speakers = []
    if DATA_DIR.exists():
        for speaker_dir in sorted(DATA_DIR.iterdir()):
            if speaker_dir.is_dir() and not speaker_dir.name.startswith(("temp_hf_", "temp_kaggle_")):
                wav_files = list(speaker_dir.glob("*.wav"))
                total_size = sum(f.stat().st_size for f in wav_files)

                total_duration = None
                try:
                    import wave
                    dur = 0
                    for wf in wav_files[:500]:  # limit for speed
                        try:
                            with wave.open(str(wf), 'r') as w:
                                dur += w.getnframes() / w.getframerate()
                        except Exception:
                            # Non-PCM WAV (e.g. float) - fall back to soundfile if present
                            import soundfile as sf
                            dur += sf.info(str(wf)).duration
                    total_duration = round(dur, 1)
                except Exception:
                    pass

                speakers.append({
                    "name": speaker_dir.name,
                    "file_count": len(wav_files),
                    "total_size": total_size,
                    "duration": total_duration,
                })
    return {"speakers": speakers}


@app.post("/api/dataset/create")
def create_speaker(data: dict):
    """Create a new speaker directory."""
    speaker = clean_speaker_name(data.get("speaker", ""))
    speaker_dir = DATA_DIR / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)
    return {"message": f"Speaker folder '{speaker}' created", "speaker": speaker, "path": str(speaker_dir)}


def _extract_wavs_from_zip(zip_path: Path, speaker_dir: Path) -> int:
    """Copy every .wav inside a ZIP into speaker_dir (flattened; no paths kept)."""
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            if member.is_dir() or "__macosx" in name.lower():
                continue
            base = safe_filename(name, "")
            if not base or not base.lower().endswith(".wav"):
                continue
            with zf.open(member) as src, open(speaker_dir / base, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def _collect_audio_files(src_dir: Path, speaker_dir: Path, progress: Optional[dict] = None) -> int:
    """Copy .wav files (and .wav files inside .zip archives) from src_dir into speaker_dir."""
    all_files = [Path(root) / f for root, _, files in os.walk(str(src_dir)) for f in files]
    if progress is not None:
        progress["total_count"] = len(all_files)
    count = 0
    for idx, file_path in enumerate(all_files):
        if progress is not None:
            progress["status"] = f"Processing downloaded file {idx + 1}/{len(all_files)}..."
        suffix = file_path.suffix.lower()
        if suffix == ".wav":
            shutil.copy(file_path, speaker_dir / safe_filename(file_path.name, f"audio_{idx}.wav"))
            count += 1
        elif suffix == ".zip":
            try:
                count += _extract_wavs_from_zip(file_path, speaker_dir)
            except Exception as zip_err:
                print(f"Error reading zip {file_path.name}: {zip_err}")
        if progress is not None:
            progress["extracted_count"] = count
    return count


@app.post("/api/dataset/upload")
async def upload_file(file: UploadFile = File(...), speaker: str = Form(...)):
    """Upload a WAV or ZIP file to a speaker's dataset."""
    speaker = clean_speaker_name(speaker)
    speaker_dir = DATA_DIR / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    filename = safe_filename(file.filename, "")
    filename_lower = filename.lower()

    if filename_lower.endswith(".wav"):
        dest = speaker_dir / filename
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        return {"message": f"Uploaded {filename}", "path": str(dest)}

    elif filename_lower.endswith(".zip"):
        zip_path = speaker_dir / f"temp_upload_{int(time.time() * 1000)}.zip"
        try:
            content = await file.read()
            with open(zip_path, "wb") as f:
                f.write(content)
            extracted_count = _extract_wavs_from_zip(zip_path, speaker_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file structure")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process ZIP file: {str(e)}")
        finally:
            if zip_path.exists():
                zip_path.unlink()

        if extracted_count == 0:
            raise HTTPException(status_code=400, detail="ZIP file uploaded but no WAV files were found inside it")
        return {
            "message": f"Successfully extracted {extracted_count} WAV files from ZIP",
            "extracted_count": extracted_count,
        }

    raise HTTPException(status_code=400, detail="Only WAV and ZIP files accepted")


def _parse_hf_repo_id(value: str) -> str:
    """Accept 'owner/name' or any huggingface.co URL and return 'owner/name'."""
    value = value.strip()
    for marker in ("huggingface.co/datasets/", "huggingface.co/"):
        if marker in value:
            value = value.split(marker, 1)[1]
            break
    value = value.split("?")[0].split("#")[0].strip("/")
    parts = [p for p in value.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return value


def _import_hf_with_datasets(repo_id: str, speaker_dir: Path) -> Optional[int]:
    """
    Import an audio dataset via the `datasets` library.
    Returns the number of WAVs written, or None if this is not a loadable audio dataset.
    Audio is read as raw bytes (no torchcodec needed) and re-encoded to 16-bit WAV.
    """
    try:
        import datasets
    except ImportError:
        try:
            hf_import_state["status"] = "Installing datasets library..."
            subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
            import datasets
        except Exception as e:
            print(f"Failed to auto-install datasets library: {e}")
            return None
    import soundfile as sf

    hf_import_state["status"] = "Loading HF dataset via datasets library..."
    print(f"Loading HF dataset via datasets library: {repo_id}...")
    ds = datasets.load_dataset(repo_id)
    if isinstance(ds, datasets.Dataset):
        ds = {"train": ds}
    if not ds:
        return None
    first_split = next(iter(ds.values()))
    audio_col = next(
        (name for name, feature in first_split.features.items() if isinstance(feature, datasets.Audio)),
        None,
    )
    if audio_col is None:
        return None

    total_rows = sum(len(split) for split in ds.values())
    hf_import_state["total_count"] = total_rows
    written = 0
    failed = 0
    done = 0
    for split_name, split in ds.items():
        split = split.cast_column(audio_col, datasets.Audio(decode=False))
        for i, row in enumerate(split):
            done += 1
            hf_import_state["status"] = f"Decoding & saving audio files ({done}/{total_rows})..."
            audio = row.get(audio_col) or {}
            raw = audio.get("bytes")
            src_path = audio.get("path") or ""
            if not raw and src_path and os.path.isfile(src_path):
                with open(src_path, "rb") as f:
                    raw = f.read()
            if not raw:
                failed += 1
                continue
            stem = Path(safe_filename(src_path, "")).stem or f"audio_{split_name}_{i}"
            target = speaker_dir / f"{stem}.wav"
            if target.exists():
                target = speaker_dir / f"{stem}_{split_name}_{i}.wav"
            try:
                data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                sf.write(str(target), data, sr, subtype="PCM_16")
                written += 1
                hf_import_state["extracted_count"] = written
            except Exception as e:
                failed += 1
                print(f"[HF] Could not decode {src_path or f'row {i}'}: {e}")
    if failed:
        print(f"[HF] {failed} row(s) could not be decoded and were skipped.")
    return written


@app.post("/api/dataset/import-hf")
def import_hf_dataset(data: dict):
    """Fetch and download a dataset from Hugging Face."""
    repo_id = str(data.get("repo_id", "")).strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="Hugging Face repository ID or URL is required")
    if not str(data.get("speaker", "")).strip():
        raise HTTPException(status_code=400, detail="Speaker name is required")
    speaker = clean_speaker_name(data.get("speaker"))
    repo_id = _parse_hf_repo_id(repo_id)
    if "/" not in repo_id:
        raise HTTPException(status_code=400, detail="Use the form 'owner/dataset-name' or paste the dataset URL")
    if hf_import_state["active"]:
        raise HTTPException(status_code=409, detail="Another import is already running")

    speaker_dir = DATA_DIR / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    hf_import_state.update({
        "active": True,
        "status": "Initializing import...",
        "extracted_count": 0,
        "total_count": 0,
    })

    try:
        # 1. `datasets` library (parquet / audiofolder datasets)
        try:
            count = _import_hf_with_datasets(repo_id, speaker_dir)
            if count:
                return {
                    "message": f"Successfully imported {count} WAV files from Hugging Face dataset '{repo_id}'",
                    "extracted_count": count,
                    "speaker": speaker,
                }
        except Exception as ds_err:
            print(f"datasets library load failed: {ds_err}. Falling back to snapshot_download...")

        # 2. FALLBACK: raw repository snapshot (WAV or ZIP files)
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="huggingface_hub is not installed. Run: pip install -r requirements.txt",
            )

        temp_dir = DATA_DIR / f"temp_hf_{int(time.time())}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            hf_import_state["status"] = "Downloading repository snapshot..."
            print(f"Downloading HF dataset snapshot for {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(temp_dir),
                allow_patterns=["*.wav", "*.WAV", "*.zip", "*.ZIP"],
            )
            extracted_count = _collect_audio_files(temp_dir, speaker_dir, hf_import_state)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to import dataset: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if extracted_count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Hugging Face dataset '{repo_id}' downloaded, but no WAV files were found or extracted",
            )
        return {
            "message": f"Successfully imported {extracted_count} WAV files from Hugging Face dataset '{repo_id}'",
            "extracted_count": extracted_count,
            "speaker": speaker,
        }
    finally:
        hf_import_state["active"] = False


@app.get("/api/dataset/kaggle-config")
def get_kaggle_config():
    """Check if Kaggle credentials are configured in ~/.kaggle/kaggle.json."""
    config_file = Path.home() / ".kaggle" / "kaggle.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "username": data.get("username", ""),
                "key_configured": bool(data.get("key")),
                "configured": True,
            }
        except Exception as e:
            print(f"Error reading kaggle.json: {e}")
    return {"username": "", "key_configured": False, "configured": False}


@app.post("/api/dataset/import-kaggle")
def import_kaggle_dataset(data: dict):
    """Fetch and download a dataset from Kaggle."""
    dataset_ref = str(data.get("dataset_ref", "")).strip()
    username = str(data.get("username", "")).strip()
    key = str(data.get("key", "")).strip()

    if not dataset_ref:
        raise HTTPException(status_code=400, detail="Kaggle dataset reference or URL is required")
    if not str(data.get("speaker", "")).strip():
        raise HTTPException(status_code=400, detail="Speaker name is required")
    speaker = clean_speaker_name(data.get("speaker"))
    if hf_import_state["active"]:
        raise HTTPException(status_code=409, detail="Another import is already running")

    # Extract owner/dataset if a full Kaggle URL is pasted
    for marker in ("kaggle.com/datasets/", "kaggle.com/"):
        if marker in dataset_ref:
            dataset_ref = dataset_ref.split(marker, 1)[1]
            break
    dataset_ref = dataset_ref.split("?")[0].split("#")[0].strip("/")
    parts = [p for p in dataset_ref.split("/") if p]
    if len(parts) >= 2:
        dataset_ref = f"{parts[0]}/{parts[1]}"

    speaker_dir = DATA_DIR / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    hf_import_state.update({
        "active": True,
        "status": "Initializing Kaggle import...",
        "extracted_count": 0,
        "total_count": 0,
    })

    try:
        if username and key and key != "••••••••••••••••":
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            config_file = kaggle_dir / "kaggle.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({"username": username, "key": key}, f)
            config_file.chmod(0o600)  # required by the Kaggle package
            print(f"Saved Kaggle credentials to {config_file}")
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
        elif not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
            config_file = Path.home() / ".kaggle" / "kaggle.json"
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        creds = json.load(f)
                    os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
                    os.environ["KAGGLE_KEY"] = creds.get("key", "")
                except Exception as e:
                    print(f"Failed to load existing Kaggle config: {e}")

        if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
            raise HTTPException(
                status_code=400,
                detail="Kaggle credentials not found. Please provide Username and API Key.",
            )

        try:
            import kaggle
        except ImportError:
            hf_import_state["status"] = "Installing kaggle library..."
            subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
            import kaggle

        temp_dir = DATA_DIR / f"temp_kaggle_{int(time.time())}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            hf_import_state["status"] = f"Downloading Kaggle dataset {dataset_ref}..."
            print(f"Downloading Kaggle dataset {dataset_ref} to {temp_dir}...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset_ref, path=str(temp_dir), unzip=True)
            extracted_count = _collect_audio_files(temp_dir, speaker_dir, hf_import_state)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to import dataset: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if extracted_count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Kaggle dataset '{dataset_ref}' downloaded, but no WAV files were found or extracted",
            )
        return {
            "message": f"Successfully imported {extracted_count} WAV files from Kaggle dataset '{dataset_ref}'",
            "extracted_count": extracted_count,
            "speaker": speaker,
        }
    finally:
        hf_import_state["active"] = False


@app.get("/api/dataset/import-progress")
def get_import_progress():
    """Get status of Hugging Face or Kaggle dataset import progress."""
    return hf_import_state


@app.delete("/api/dataset/{speaker}")
def delete_speaker(speaker: str):
    """Delete a speaker's dataset."""
    name = clean_speaker_name(speaker)
    if name != speaker:
        raise HTTPException(status_code=400, detail="Invalid speaker name")
    if training_state.get("speaker") == name:
        _require_not_training("delete this speaker")
    speaker_dir = DATA_DIR / name
    if not speaker_dir.is_dir() or speaker_dir.parent != DATA_DIR:
        raise HTTPException(status_code=404, detail="Speaker not found")
    shutil.rmtree(speaker_dir)
    return {"message": f"Deleted speaker '{name}'"}


def _remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


@app.get("/api/dataset/export")
def export_dataset():
    """Export all datasets as a ZIP file."""
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise HTTPException(status_code=404, detail="No datasets to export")
    fd, zip_path = tempfile.mkstemp(prefix="beatrice_dataset_", suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                file_path = Path(root) / file
                zf.write(file_path, file_path.relative_to(DATA_DIR))
    tasks = BackgroundTasks()
    tasks.add_task(_remove_file, zip_path)
    return FileResponse(
        path=zip_path, filename="beatrice_dataset.zip",
        media_type="application/zip", background=tasks,
    )


# --- Training ---

def _int_param(data: dict, key: str, default: int, lo: int, hi: int) -> int:
    value = data.get(key, default)
    if value is None or value == "":
        value = default
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
        if not number.is_integer():
            raise ValueError
        number = int(number)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"'{key}' must be a whole number")
    if not lo <= number <= hi:
        raise HTTPException(status_code=400, detail=f"'{key}' must be between {lo} and {hi}")
    return number


def _float_param(data: dict, key: str, default: float, lo: float, hi: float) -> float:
    value = data.get(key, default)
    if value is None or value == "":
        value = default
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"'{key}' must be a number (e.g. 5e-5)")
    if not math.isfinite(number) or not lo <= number <= hi:
        raise HTTPException(status_code=400, detail=f"'{key}' must be between {lo:g} and {hi:g}")
    return number


def _bool_param(data: dict, key: str, default: bool) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    return bool(isjunction and isjunction(path))


def _unlink_link(path: Path):
    try:
        os.unlink(path)
    except OSError:
        os.rmdir(path)  # Windows directory symlink / junction


def _clear_staging_dir():
    """Remove the per-run dataset view without ever following links into DATA_DIR."""
    if _is_link(STAGING_DIR):
        _unlink_link(STAGING_DIR)
        return
    if not STAGING_DIR.exists():
        return
    for entry in STAGING_DIR.iterdir():
        if _is_link(entry):
            _unlink_link(entry)
        elif entry.is_dir():
            for inner in entry.iterdir():
                if _is_link(inner):
                    _unlink_link(inner)
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _stage_speaker(speaker: str) -> Path:
    """
    Build data_active/<speaker>/<speaker> -> data/<speaker> so the trainer only
    sees this speaker and names its checkpoints/paraphernalia after it.
    """
    _clear_staging_dir()
    stage_root = STAGING_DIR / speaker
    stage_root.mkdir(parents=True, exist_ok=True)
    src = DATA_DIR / speaker
    dst = stage_root / speaker
    try:
        os.symlink(src, dst, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows without Developer Mode cannot create symlinks: copy instead.
        shutil.copytree(src, dst)
    return stage_root


def _prepare_output_dir(speaker: str) -> Path:
    """
    Each run writes to output/<speaker>/. An earlier run there is moved aside to
    output/<speaker>-<timestamp>/ so previous models are never deleted (the
    trainer refuses to start in a folder that already holds a checkpoint).
    """
    out_dir = OUTPUT_DIR / speaker
    if out_dir.is_dir() and any(out_dir.iterdir()):
        stamp = datetime.fromtimestamp(out_dir.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
        archived = OUTPUT_DIR / f"{speaker}-{stamp}"
        n = 1
        while archived.exists():
            n += 1
            archived = OUTPUT_DIR / f"{speaker}-{stamp}-{n}"
        out_dir.rename(archived)
        print(f"[train] Kept previous run of '{speaker}' in {archived}")
    return out_dir


def _start_trainer_process(job_file: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(trainer_env())
    cpu_threads = str(os.cpu_count() or 1)
    env.setdefault("OMP_NUM_THREADS", cpu_threads)
    env.setdefault("MKL_NUM_THREADS", cpu_threads)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    log = open(TRAINER_STDOUT_LOG, "w", encoding="utf-8")
    try:
        log.write(f"[server] {datetime.now().isoformat(timespec='seconds')} starting beatrice_trainer\n")
        log.flush()
        return subprocess.Popen(
            [sys.executable, "-u", str(TRAINER_RUNNER), str(job_file)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(BEATRICE_DIR),
            env=env,
        )
    finally:
        log.close()


@app.post("/api/train/start")
def start_training(data: dict):
    """Start real Beatrice training for one speaker."""
    global training_process, training_state

    with training_lock:
        if _training_running():
            raise HTTPException(status_code=409, detail="Training already in progress")

        if not str(data.get("speaker", "")).strip():
            raise HTTPException(status_code=400, detail="Speaker name required")
        speaker = clean_speaker_name(data.get("speaker"))
        steps = _int_param(data, "steps", 10000, 1, 1_000_000)
        batch_size = _int_param(data, "batch_size", 4, 1, 256)
        learning_rate = _float_param(data, "learning_rate", 5e-5, 1e-8, 1.0)
        num_workers = _int_param(data, "num_workers", 0, 0, 64)
        use_amp = _bool_param(data, "mixed_precision", False)
        use_compile = _bool_param(data, "compile_model", False)
        # "gradient_accumulation_steps" is accepted for compatibility but
        # beatrice_trainer has no gradient accumulation, so it is ignored.

        speaker_dir = DATA_DIR / speaker
        if not _audio_files(speaker_dir):
            raise HTTPException(status_code=400, detail=f"No WAV files found for speaker '{speaker}'")

        problems = trainer_setup_problems()
        if problems:
            raise HTTPException(
                status_code=400,
                detail="The Beatrice trainer is not ready: " + "; ".join(problems)
                + ". Re-run the start script (it installs packages and downloads the weights) or use "
                "'Download or update trainer' in Settings, then try again.",
            )

        notes = []
        hw = get_hardware_info()
        compile_ok = use_compile
        if use_compile and not hw.get("cuda_available"):
            compile_ok = False
            notes.append("torch.compile disabled: it is only supported here with an NVIDIA CUDA GPU.")
        elif use_compile and importlib.util.find_spec("triton") is None:
            compile_ok = False
            notes.append("torch.compile disabled: it needs Triton, which is not installed.")
        if num_workers > 0:
            num_workers = 0
            notes.append(
                "num_workers set to 0: DataLoader worker processes cannot load the trainer's "
                "dataset classes on macOS/Windows ('spawn' start method)."
            )
        if not hw.get("cuda_available"):
            notes.append("No NVIDIA CUDA GPU detected by PyTorch - training runs on the CPU and will be very slow.")

        # Training config = upstream defaults + the user's settings
        h = _read_json(DEFAULT_CONFIG_FILE)
        h.update({
            "n_steps": steps,
            "batch_size": batch_size,
            "learning_rate_g": learning_rate,
            "learning_rate_d": learning_rate,
            "num_workers": num_workers,
            "use_amp": use_amp,
            "compile_convnext": compile_ok,
            "compile_d4c": compile_ok,
            "compile_discriminator": compile_ok,
        })

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        config_file = WORK_DIR / f"config_{speaker}.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(h, f, indent=4)

        try:
            data_dir = _stage_speaker(speaker)
            out_dir = _prepare_output_dir(speaker)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Could not prepare training folders: {e}")

        for stale in (STATUS_FILE, Path(str(STATUS_FILE) + ".tmp")):
            if stale.exists():
                stale.unlink()

        job = {
            "speaker": speaker,
            "beatrice_dir": str(BEATRICE_DIR),
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "config_file": str(config_file),
            "status_file": str(STATUS_FILE),
            "total_steps": steps,
            "notes": notes,
        }
        job_file = WORK_DIR / "job.json"
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)

        training_process = _start_trainer_process(job_file)

        training_state = _idle_training_state()
        training_state.update({
            "active": True,
            "state": "starting",
            "phase": "starting",
            "speaker": speaker,
            "total_steps": steps,
            "start_time": time.time(),
            "output_dir": str(out_dir),
            "pid": training_process.pid,
            "notes": notes,
        })

    return {
        "message": "Training started",
        "pid": training_process.pid,
        "speaker": speaker,
        "total_steps": steps,
        "output_dir": str(out_dir),
        "notes": notes,
    }


@app.get("/api/train/status")
def get_training_status():
    """Current training progress, as reported by the real trainer."""
    global training_process

    st = training_state
    data = _read_json(STATUS_FILE) if st["start_time"] else {}
    if data and st["state"] != "stopped":
        for key in ("step", "loss", "steps_per_sec", "eta_seconds", "device", "phase"):
            if key in data:
                st[key] = data[key]
        if "memory_mb" in data:
            st["memory_usage"] = data["memory_mb"]
        if data.get("error"):
            st["error"] = data["error"]

    proc = training_process
    if proc is not None:
        rc = proc.poll()
        if rc is None and not st["error"]:
            st["active"] = True
            st["state"] = "running" if st.get("phase") == "training" else "starting"
        elif rc is not None:
            training_process = None
            st["active"] = False
            st["exit_code"] = rc
            st["end_time"] = time.time()
            if st["state"] != "stopped":
                if rc == 0 and data.get("completed") and not st["error"]:
                    st["state"] = "completed"
                    st["completed"] = True
                    st["phase"] = "finished"
                else:
                    st["state"] = "failed"
                    st["completed"] = False
                    if not st["error"]:
                        tail = _tail_lines(TRAINER_STDOUT_LOG, 15)
                        st["error"] = (
                            f"The training process exited unexpectedly (exit code {rc}) without "
                            "reporting an error. Last lines of trainer_stdout.log:\n" + "\n".join(tail)
                        )
        else:
            # The runner reported an error and is about to exit.
            st["active"] = False
            st["state"] = "failed"
            st["completed"] = False
    if st["error"] and st["state"] not in ("stopped",):
        st["active"] = False
        st["state"] = "failed"
        st["completed"] = False

    elapsed = None
    if st["start_time"]:
        elapsed = round((st["end_time"] or time.time()) - st["start_time"], 1)

    return {
        "active": st["active"],
        "state": st["state"],
        "completed": st["completed"],
        "error": st["error"],
        "exit_code": st["exit_code"],
        "phase": st["phase"],
        "speaker": st["speaker"],
        "step": st["step"],
        "total_steps": st["total_steps"],
        "loss": st["loss"],
        "start_time": st["start_time"],
        "elapsed_seconds": elapsed,
        "memory_usage_mb": st.get("memory_usage"),
        "steps_per_sec": st.get("steps_per_sec"),
        "eta_seconds": st.get("eta_seconds"),
        "device": st.get("device"),
        "output_dir": st.get("output_dir"),
        "pid": st.get("pid"),
        "notes": st.get("notes", []),
    }


def _kill_process_tree(proc: subprocess.Popen):
    children = []
    try:
        import psutil
        children = psutil.Process(proc.pid).children(recursive=True)
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except OSError:
        pass
    for child in children:
        try:
            child.kill()
        except Exception:
            pass


@app.post("/api/train/stop")
def stop_training():
    """Stop the current training run."""
    global training_process
    proc = training_process
    was_running = proc is not None and proc.poll() is None
    if proc is not None:
        _kill_process_tree(proc)
        training_process = None
    if was_running:
        training_state.update({
            "active": False,
            "state": "stopped",
            "phase": "stopped",
            "completed": False,
            "error": None,
            "end_time": time.time(),
        })
    else:
        training_state["active"] = False

    gc.collect()
    return {"message": "Training stopped" if was_running else "No training was running"}


@app.get("/api/train/console-logs")
def get_console_logs(lines: int = 100):
    """Retrieve the last N lines of the training stdout/stderr logs."""
    if not TRAINER_STDOUT_LOG.exists():
        return {"logs": []}
    try:
        lines = max(1, min(int(lines), 2000))
        with open(TRAINER_STDOUT_LOG, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        cleaned_content = []
        for line in content[-lines:]:
            # tqdm overwrites its line with carriage returns: keep the last segment
            if "\r" in line:
                line = line.split("\r")[-1]
            cleaned_content.append(line.strip())
        return {"logs": cleaned_content}
    except Exception as e:
        return {"logs": [f"Error reading logs: {str(e)}"]}


# --- Models ---
@app.get("/api/models/list")
def list_models():
    """List trained models: paraphernalia_* folders (what the voice changer loads) and model files."""
    models = []
    if OUTPUT_DIR.exists():
        for path in sorted(OUTPUT_DIR.rglob('*')):
            # Skip files inside a paraphernalia directory
            if any(part.startswith("paraphernalia_") for part in path.relative_to(OUTPUT_DIR).parts[:-1]):
                continue

            if path.is_file() and path.suffix in ('.bin', '.pth', '.pt', '.onnx', '.safetensors'):
                models.append({
                    "name": path.name,
                    "path": path.relative_to(OUTPUT_DIR).as_posix(),
                    "size": path.stat().st_size,
                    "created": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                })
            elif path.is_dir() and path.name.startswith("paraphernalia_"):
                dir_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                models.append({
                    "name": path.name,
                    "path": path.relative_to(OUTPUT_DIR).as_posix(),
                    "size": dir_size,
                    "created": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                })
    return {"models": models}


def _resolve_model_path(name: str) -> Path:
    model_path = (OUTPUT_DIR / name).resolve()
    if not _is_within(model_path, OUTPUT_DIR) or model_path == OUTPUT_DIR.resolve():
        raise HTTPException(status_code=403, detail="Access denied")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return model_path


@app.get("/api/models/download/{name:path}")
def download_model(name: str):
    """Download a model file, or a paraphernalia folder as a ZIP."""
    model_path = _resolve_model_path(name)

    if model_path.is_dir():
        fd, zip_path = tempfile.mkstemp(prefix=f"{model_path.name}_", suffix=".zip")
        os.close(fd)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = Path(root) / file
                    zf.write(file_path, file_path.relative_to(model_path))
        tasks = BackgroundTasks()
        tasks.add_task(_remove_file, zip_path)
        return FileResponse(
            path=zip_path,
            filename=f"{model_path.name}.zip",
            media_type="application/zip",
            background=tasks,
        )

    return FileResponse(path=str(model_path), filename=model_path.name)


@app.delete("/api/models/{name:path}")
def delete_model(name: str):
    """Delete a trained model file or paraphernalia directory."""
    model_path = _resolve_model_path(name)
    try:
        if model_path.is_dir():
            shutil.rmtree(model_path)
        else:
            model_path.unlink()
        return {"message": f"Successfully deleted '{name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@app.post("/api/setup/beatrice")
def setup_beatrice():
    """Download (or update) the trainer's pretrained weights and audio assets from Hugging Face."""
    _require_not_training("update the trainer")
    try:
        result = download_trainer_assets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download Beatrice trainer assets: {e}")
    missing = missing_trainer_modules()
    message = "Beatrice trainer assets downloaded from Hugging Face."
    if missing:
        message += " Missing Python packages: " + ", ".join(missing) + " (re-run the start script)."
    return {
        "message": message,
        "installed": not missing,
        "beatrice_problems": trainer_setup_problems(),
        **result,
    }


# --- Reset ---
@app.post("/api/reset")
def reset_all():
    """Delete all datasets and models."""
    _require_not_training("reset")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {"message": "All data cleared"}


# --- Static Frontend ---
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the main web UI."""
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<h1>Beatrice V2 Trainer</h1><p>index.html not found</p>", status_code=200)

app.mount("/", StaticFiles(directory=str(BASE_DIR), html=False), name="static")


# ---------- Main ----------

def run_setup_cli() -> int:
    """`python server.py --setup`: used by the start scripts."""
    print("Checking Beatrice trainer files ...")
    try:
        problems = trainer_asset_problems()
        if problems:
            for p in problems:
                print(f"  - {p}")
            download_trainer_assets()
        else:
            print("Trainer code, pretrained weights and assets are present.")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    missing = missing_trainer_modules()
    if missing:
        print("ERROR: missing Python packages: " + ", ".join(missing))
        return 1
    # find_spec() is not enough: e.g. pyworld fails at import time without pkg_resources.
    for name in REQUIRED_TRAINER_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:
            print(f"ERROR: Python package '{name}' is installed but fails to import: {type(e).__name__}: {e}")
            return 1
    if sys.version_info < MIN_PYTHON:
        print(f"ERROR: Python {sys.version.split()[0]} is too old for the trainer (needs 3.10 - 3.13).")
        return 1
    print("Trainer dependencies import correctly.")
    return 0


def _open_browser_when_ready(url: str, host: str, port: int):
    def wait_and_open():
        probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        for _ in range(240):
            try:
                with socket.create_connection((probe_host, port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        webbrowser.open(url)
    threading.Thread(target=wait_and_open, daemon=True).start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Beatrice V2 Trainer web UI server")
    parser.add_argument("--setup", action="store_true",
                        help="download missing trainer weights/assets from Hugging Face, then exit")
    parser.add_argument("--open-browser", action="store_true",
                        help="open the web UI in the default browser once the server is up")
    args = parser.parse_args()

    if args.setup:
        sys.exit(run_setup_cli())

    hw = get_hardware_info()
    url_host = "127.0.0.1" if HOST in ("0.0.0.0", "::") else HOST
    url = f"http://{url_host}:{PORT}"
    problems = trainer_setup_problems()
    print("=" * 60)
    print(f"  Beatrice V2 Trainer - {PLATFORM_LABEL} backend")
    print("=" * 60)
    print(f"  Device:    {hw.get('gpu_name') or hw['machine']}")
    print(f"  RAM:       {hw['ram_total_gb']} GB")
    if hw["cuda_available"]:
        print(f"  Compute:   CUDA {hw['cuda_version']} ({hw['vram_total_gb']} GB VRAM)")
    elif hw["torch_cuda_build"]:
        print("  Compute:   CPU (CUDA build of PyTorch installed, but no usable NVIDIA GPU/driver)")
    else:
        print("  Compute:   CPU (CPU-only PyTorch build; re-run start_windows.bat on a PC with an NVIDIA GPU)")
    print(f"  PyTorch:   {hw.get('torch_version') or 'not installed'}")
    print(f"  Trainer:   {'ready' if not problems else 'NOT READY - ' + '; '.join(problems)}")
    print(f"  Data:      {DATA_DIR}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Web UI:    {url}")
    print(f"  API docs:  {url}/docs")
    if HOST not in LOOPBACK_HOSTS:
        print(f"  WARNING:   listening on {HOST} - anyone on your network can use this trainer")
    print("=" * 60, flush=True)
    if args.open_browser:
        _open_browser_when_ready(url, HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

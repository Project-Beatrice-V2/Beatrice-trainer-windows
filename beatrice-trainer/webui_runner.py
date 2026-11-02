"""
Training runner for the Beatrice V2 Trainer web UI.

This file is NOT part of upstream beatrice-trainer. ``server.py`` starts it as

    python -u webui_runner.py <job.json>

with the working directory set to this folder. It runs the real
``beatrice_trainer`` package (``beatrice_trainer/__main__.py``) and reports
progress through the JSON status file named in the job; the trainer itself
updates that file every step via the ``BEATRICE_STATUS_LOG_FILE`` env var.

There is deliberately no fallback of any kind. If the trainer or one of its
dependencies cannot be imported, or training raises, the status file gets
``active: false, completed: false, error: "<message>"``, the traceback is
printed to stdout (``trainer_stdout.log``) and the process exits non-zero.
"""

import importlib
import importlib.util
import json
import os
import runpy
import sys
import time
import traceback

# Modules beatrice_trainer imports at start-up (see beatrice-trainer/pyproject.toml).
REQUIRED_MODULES = (
    "numpy",
    "torch",
    "torchaudio",
    "soundfile",
    "pyworld",
    "tqdm",
    "tensorboard",
)

MAX_ERROR_CHARS = 1500


class StatusFile:
    def __init__(self, path, total_steps):
        self.path = path
        self.total_steps = total_steps

    def read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def write(self, **fields):
        status = {
            "active": True,
            "completed": False,
            "error": None,
            "phase": None,
            "step": 0,
            "total_steps": self.total_steps,
            "loss": None,
            "updated_at": time.time(),
        }
        status.update(fields)
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(status, f)
            os.replace(tmp, self.path)
        except OSError:
            # os.replace can fail on Windows while the server has the file open.
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(status, f)
            except OSError:
                pass


def _short(text):
    text = str(text).strip()
    if len(text) > MAX_ERROR_CHARS:
        text = text[: MAX_ERROR_CHARS - 3] + "..."
    return text


def main():
    if len(sys.argv) != 2:
        print("usage: python webui_runner.py <job.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8") as f:
        job = json.load(f)

    status = StatusFile(job["status_file"], int(job["total_steps"]))

    def fail(message, print_traceback=True):
        if print_traceback:
            traceback.print_exc()
        message = _short(message)
        print(f"[webui_runner] ERROR: {message}", flush=True)
        last = status.read()
        status.write(
            active=False,
            completed=False,
            error=message,
            phase="failed",
            step=last.get("step", 0) or 0,
            loss=last.get("loss"),
            device=last.get("device"),
        )
        return 1

    status.write(phase="starting")

    beatrice_dir = os.path.abspath(job["beatrice_dir"])
    os.chdir(beatrice_dir)
    if beatrice_dir not in sys.path:
        sys.path.insert(0, beatrice_dir)

    print(f"[webui_runner] Python {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"[webui_runner] Speaker: {job.get('speaker')}", flush=True)
    print(f"[webui_runner] Data dir: {job['data_dir']}", flush=True)
    print(f"[webui_runner] Output dir: {job['out_dir']}", flush=True)
    print(f"[webui_runner] Config: {job['config_file']}", flush=True)
    for note in job.get("notes", []):
        print(f"[webui_runner] Note: {note}", flush=True)

    # ---- 1. Can the real trainer be imported at all? ----
    try:
        for name in REQUIRED_MODULES:
            importlib.import_module(name)
        spec = importlib.util.find_spec("beatrice_trainer")
        if spec is None or spec.origin is None:
            raise ImportError(f"package 'beatrice_trainer' not found in {beatrice_dir}")
        main_py = os.path.join(beatrice_dir, "beatrice_trainer", "__main__.py")
        if not os.path.isfile(main_py):
            raise ImportError(f"{main_py} is missing")
    except Exception as e:
        return fail(
            "The Beatrice trainer could not start because it failed to import "
            f"({type(e).__name__}: {e}). Re-run the start script so the packages in "
            "requirements.txt get installed, then start training again."
        )

    import torch

    print(f"[webui_runner] PyTorch {torch.__version__}", flush=True)
    try:
        # Safety net for torch.compile: donated buffers conflict with the
        # retain_graph=True discriminator backward pass in the trainer.
        import torch._functorch.config

        torch._functorch.config.donated_buffer = False
    except Exception:
        pass

    # ---- 2. Run beatrice_trainer exactly like `python -m beatrice_trainer ...` ----
    os.environ["BEATRICE_STATUS_LOG_FILE"] = job["status_file"]
    sys.argv = [
        "beatrice_trainer",
        "-d", job["data_dir"],
        "-o", job["out_dir"],
        "-c", job["config_file"],
    ]
    status.write(phase="loading")
    print("[webui_runner] Launching beatrice_trainer ...", flush=True)
    try:
        runpy.run_module("beatrice_trainer", run_name="__main__", alter_sys=True)
    except SystemExit as e:
        if e.code not in (None, 0):
            return fail(f"The Beatrice trainer exited with status {e.code}.", print_traceback=False)
    except KeyboardInterrupt:
        last = status.read()
        status.write(active=False, completed=False, phase="stopped",
                     error="Training was interrupted.", step=last.get("step", 0) or 0)
        return 130
    except BaseException as e:  # noqa: BLE001 - report every failure to the UI
        return fail(f"Training failed: {type(e).__name__}: {e}")

    # ---- 3. Finished: make sure the status says so ----
    last = status.read()
    status.write(
        active=False,
        completed=True,
        phase="finished",
        step=last.get("total_steps") or status.total_steps,
        loss=last.get("loss"),
        device=last.get("device"),
    )
    print("[webui_runner] Training finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

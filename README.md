<div align="center">

<img src="docs/media/app-icon.png" alt="Beatrice app icon" width="128" />

# Beatrice Trainer for Windows

**Train your own voice for the Beatrice voice changer, on your Windows PC.**<br/>
Add recordings of one speaker, press Start, then import the finished model into Beatrice with **Upload Custom Model**.

[![Download ZIP](https://img.shields.io/badge/Download-ZIP-000000?style=for-the-badge&logo=github&logoColor=white&labelColor=000000)](https://github.com/Project-Beatrice-V2/Beatrice-trainer-windows/archive/refs/heads/main.zip)
[![Website](https://img.shields.io/badge/Website-project--beatrice.vercel.app-7c8cff?style=for-the-badge&labelColor=000000)](https://project-beatrice.vercel.app)
[![Stars across all Beatrice repos](https://img.shields.io/endpoint?url=https%3A%2F%2Fproject-beatrice.vercel.app%2Fapi%2Fstars%3Fformat%3Dshields&style=for-the-badge)](https://github.com/Project-Beatrice-V2)

![NVIDIA GPU](https://img.shields.io/badge/NVIDIA%20GPU-recommended-a887dc?style=flat-square&labelColor=000000)
![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011-a887dc?style=flat-square&labelColor=000000)
![Python](https://img.shields.io/badge/Python-3.10%20→%203.13-a887dc?style=flat-square&labelColor=000000)
![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA-a887dc?style=flat-square&labelColor=000000)
![Runs locally](https://img.shields.io/badge/runs-locally-3dffa8?style=flat-square&labelColor=000000)
![License](https://img.shields.io/badge/license-MIT-a1a1aa?style=flat-square&labelColor=000000)

[Website](https://project-beatrice.vercel.app) · [Beatrice for Windows](https://github.com/satiricalguru/Beatrice-voicechanger-windows) · [Beatrice for Mac](https://github.com/satiricalguru/Beatrice-voicechanger-mac) · [Trainer for Mac](https://github.com/Project-Beatrice-V2/Beatrice-trainer-macos) · [Report a bug](https://github.com/Project-Beatrice-V2/Beatrice-trainer-windows/issues/new)

<br/>

<img src="docs/media/trainer-demo.gif" alt="Beatrice Trainer: dashboard, dataset, a training run with the live loss chart, models, cloud notebooks and light mode" width="100%" />

</div>

---

## ✨ Why Beatrice Trainer

- 🎙️ **A real Beatrice model of your voice.** It runs the upstream [Beatrice trainer](https://huggingface.co/fierce-cats/beatrice-trainer) on your recordings. There is no fake or "demo" training mode: if the trainer can't run, you see why.
- ⚡ **Uses your NVIDIA GPU.** The start script installs the CUDA build of PyTorch when it finds an NVIDIA card, and auto-tuning picks the batch size from your VRAM.
- 📥 **Three ways to add audio.** Drag in WAV or ZIP files, or import a dataset from Hugging Face or Kaggle.
- 📈 **Watch it learn.** Live step count, loss chart, time left and the trainer's own console output, in a browser tab that keeps working while you switch pages.
- 📦 **One click to Beatrice.** Download the finished model as a ZIP and import it into the voice changer with **Upload Custom Model**.
- ☁️ **No fast GPU?** Ready-made [Colab and Kaggle notebooks](https://github.com/Project-Beatrice-V2/Beatrice-colab) train in the cloud for free.

## 📸 Tour

<table>
  <tr>
    <td width="50%"><img src="docs/media/trainer-training.jpg" alt="Training page with step, loss, time left, speed, a loss chart and the console" /></td>
    <td width="50%"><img src="docs/media/trainer-dataset.jpg" alt="Dataset page with the drop zone and the list of speakers" /></td>
  </tr>
  <tr>
    <td><b>Training</b><br/>Progress, loss and the trainer's console output, updated every two seconds.</td>
    <td><b>Dataset</b><br/>Name the speaker and add audio from your disk, Hugging Face or Kaggle.</td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/media/trainer-models.jpg" alt="Models page listing trained voice models and checkpoints" /></td>
    <td width="50%"><img src="docs/media/trainer-cloud.jpg" alt="Cloud training page with Colab and Kaggle notebooks" /></td>
  </tr>
  <tr>
    <td><b>Models</b><br/>Download a voice model as a ZIP for the voice changer, or keep the checkpoints.</td>
    <td><b>Cloud training</b><br/>Open the notebooks in Colab or Kaggle and package your dataset for upload.</td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/media/trainer-dashboard-light.jpg" alt="Dashboard in light mode" /></td>
    <td width="50%"><img src="docs/media/trainer-error.jpg" alt="Training page showing a clear training failed message" /></td>
  </tr>
  <tr>
    <td><b>Light mode</b><br/>Dark, light or follow the system.</td>
    <td><b>Honest errors</b><br/>If training can't start or crashes, the reason and a fix are shown right away.</td>
  </tr>
</table>

<sub>Screenshots and the demo use sample data served by a local mock of the trainer API.</sub>

## 🚀 Get started

1. **Install Python 3.12** from [python.org](https://www.python.org/downloads/) (3.10 to 3.13 work). In the installer, tick **Add python.exe to PATH**.
2. **Download the trainer:** [download the ZIP](https://github.com/Project-Beatrice-V2/Beatrice-trainer-windows/archive/refs/heads/main.zip) and extract it, or clone it:
   ```cmd
   git clone https://github.com/Project-Beatrice-V2/Beatrice-trainer-windows.git
   cd Beatrice-trainer-windows
   ```
3. **Double-click `start_windows.bat`** (or run it in Command Prompt). `start.bat` does the same.
4. Your browser opens **http://127.0.0.1:8000**. Next time, just run `start_windows.bat` again.

`start_windows.bat` creates a `venv`, installs PyTorch (the CUDA build when `nvidia-smi` finds an NVIDIA GPU, the cu128 wheels for RTX 50-series cards) and `requirements.txt`, downloads the trainer's pretrained weights from Hugging Face with `python server.py --setup`, then starts the server. The first run downloads about 3 GB, mostly PyTorch. If anything fails, the window stays open and shows the error.

Optional environment variables: `BEATRICE_TORCH_INDEX=cu126|cu128|cpu` forces a PyTorch wheel index, `BEATRICE_NO_BROWSER=1` stops it opening a browser tab.

> [!NOTE]
> **An NVIDIA GPU with CUDA is recommended.** The upstream trainer notes that its default settings use about 9 GB of VRAM and finish in about 40 minutes on an RTX 4090; auto-tune lowers the batch size on smaller cards. Without an NVIDIA GPU, training runs on the CPU: it works, but it is very slow. The [cloud notebooks](https://github.com/Project-Beatrice-V2/Beatrice-colab) are a free alternative.

## 🧩 How training works

```mermaid
flowchart LR
  rec([🎙️ Recordings]) --> ds
  subgraph trainer [Beatrice Trainer · 127.0.0.1:8000]
    ds[Dataset<br/>beatrice-trainer/data/&lt;speaker&gt;] --> train[Training<br/>beatrice_trainer on CUDA or CPU]
    train --> out[Model folder<br/>paraphernalia_*]
  end
  out -- Download ZIP --> zip([model .zip])
  zip -- Upload Custom Model --> app([Beatrice voice changer])
```

1. **Dataset.** Enter a speaker name and add WAV files (ZIPs are unpacked), or import a Hugging Face or Kaggle dataset. Files go to `beatrice-trainer/data/<speaker>/`.
2. **Train.** Pick the speaker on the Training page. Auto-tune fills in batch size, gradient accumulation, workers and mixed precision for your PC; change them under *Advanced settings*. Press **Start training**.
3. **Export.** When the run finishes, open **Models** and click **Download ZIP** next to the voice model. Checkpoints (`.pt.gz`) stay in `beatrice-trainer/output/<speaker>/`.
4. **Use it.** In the [Beatrice voice changer](https://github.com/satiricalguru/Beatrice-voicechanger-windows), open the model menu, choose **Upload Custom Model** and pick the ZIP.

## 💻 Requirements

| | |
|:--|:--|
| **Windows** | Windows 10 or 11, 64-bit |
| **GPU** | NVIDIA GPU with CUDA recommended. CPU-only training works but is very slow. |
| **VRAM** | Upstream defaults use about 9 GB; auto-tune picks a smaller batch on 8 to 12 GB cards |
| **Python** | 3.10 to 3.13 (3.12 recommended) |
| **Disk** | About 3 GB for PyTorch and packages, plus the trainer weights (about 175 MB), datasets and checkpoints |

<details>
<summary><b>Troubleshooting</b></summary>
<br/>

**`start_windows.bat` closes immediately (fixed).**
Older versions of the script used characters that `cmd.exe` can't parse inside `echo` lines and `if` blocks, so the window could vanish before showing anything. The current script avoids them and pauses on every error. Update (`git pull` or download the ZIP again). If a window still closes, open Command Prompt in the folder and run `start_windows.bat` there to read the output.

**Training finished instantly, or said "complete" straight away (fixed).**
Older versions quietly ran a stand-in "benchmark" loop when the real trainer couldn't start, so a run could look finished without training anything. That fallback is gone: the real trainer runs, and if it can't, the Training page shows **Training could not start** or **Training failed** with the reason. Update and run `start_windows.bat` again.

**"The Beatrice trainer is not ready".**
Weights or packages are missing. Run `start_windows.bat` again, or open **Settings** and click **Download or update trainer**.

**Training runs on the CPU although you have an NVIDIA GPU.**
Update the NVIDIA driver so `nvidia-smi` works, delete the `venv` folder and run `start_windows.bat` again. To force a wheel index, run `set BEATRICE_TORCH_INDEX=cu126` (or `cu128` for RTX 50-series) first.

**"Python … is too old".**
Install Python 3.12 from python.org with **Add python.exe to PATH** ticked, delete the `venv` folder and run `start_windows.bat` again.

**Out of memory during training.**
Lower **Batch size** under *Advanced settings* on the Training page, then start again.

**The browser didn't open, or port 8000 is busy.**
Open http://127.0.0.1:8000 yourself. To use another port, run `set BEATRICE_PORT=8001` before `start_windows.bat`.
</details>

<details>
<summary><b>Project layout</b></summary>
<br/>

```
server.py           FastAPI server: API and the web UI on 127.0.0.1:8000
index.html          Web UI layout
index.css           The Aurora Void design system (dark and light)
app.js              Web UI logic: datasets, training, models, settings
fonts/              Geist and Geist Mono (SIL OFL)
start_windows.bat   Sets up the venv, installs PyTorch and packages, downloads weights, starts the server
start.bat           Same as start_windows.bat
start.sh            The same steps for macOS and Linux
beatrice-trainer/   The Beatrice trainer (fierce-cats/beatrice-trainer) plus webui_runner.py
colab_repo/         The Colab and Kaggle notebooks
```
</details>

## 🌐 The Beatrice family

| | |
|:--|:--|
| 🪟 [Beatrice for Windows](https://github.com/satiricalguru/Beatrice-voicechanger-windows) | The real-time voice changer that uses your models |
| 🍎 [Beatrice for Mac](https://github.com/satiricalguru/Beatrice-voicechanger-mac) | The same app for Apple Silicon Macs |
| 🌍 [Website](https://project-beatrice.vercel.app) | Downloads, guides and voice samples |
| 🧠 [Trainer for Mac](https://github.com/Project-Beatrice-V2/Beatrice-trainer-macos) · [Colab](https://github.com/Project-Beatrice-V2/Beatrice-colab) | Train on a Mac or in the cloud |
| 📂 [Dataset Web UI](https://github.com/Project-Beatrice-V2/Beatrice-dataset-webui-windows) | Prepare recordings for training |
| 🎭 [Voice models](https://github.com/Project-Beatrice-V2/Beatrice-voice-models) | Ready-made models to import |

## 🙏 Credits

- **Beatrice trainer** by Project Beatrice: [fierce-cats/beatrice-trainer](https://huggingface.co/fierce-cats/beatrice-trainer) (MIT)
- **Beatrice engine** by [prj-beatrice](https://github.com/prj-beatrice/beatrice-vst)
- **Geist** and **Geist Mono** by Vercel (SIL Open Font License, see `fonts/OFL-Geist.txt`)

## 📜 License

The web UI and server are **MIT** licensed, © 2026 Project Beatrice V2 (see [LICENSE](LICENSE)). The Beatrice trainer in `beatrice-trainer/` is MIT licensed by Project Beatrice; the fonts are under the SIL Open Font License.

<div align="center">
<br/>

**If you trained a voice you love, ⭐ star the repo, it really helps.**

<a href="#beatrice-trainer-for-windows">Back to top ↑</a>

</div>

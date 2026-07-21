<p align="center">
  <img src="assets/logo.jpg" alt="Project Beatrice V2 Logo" width="150" />
</p>

# 🎙️ Project Beatrice V2 Trainer

> Train high-fidelity, real-time voice conversion models using **Beatrice V2** on Google Colab or Kaggle.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](BeatriceV2_Trainer_Notebook_Colab.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](BeatriceV2_Trainer_Notebook_Kaggle.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Beatrice V2](https://img.shields.io/badge/Model-Beatrice%20V2-blueviolet)](https://github.com/Project-Beatrice-V2)

**Project Beatrice V2 Trainer** is a streamlined workspace containing Jupyter notebooks optimized for training custom voice conversion models. It automates the preparation of datasets, configures training parameters using a visual UI, monitors training via TensorBoard, and packages your trained models into **VST3 Paraphernalia** format for real-time inference.

---

## ✨ Features

- **🚀 Dual-Platform Support:** Ready-to-go notebooks for both **Google Colab** (with Google Drive integration) and **Kaggle** (utilizing local persistence).
- **📂 Automated Dataset Setup:**
  - **Option A:** Download directly from Hugging Face (supports standard audio datasets, raw repositories, and zipped archives).
  - **Option B:** Upload your own WAV files (manually or via ZIP).
- **🎨 Settings UI:** Interactive Gradio/Colab-easy-ui interface to easily tune hyperparameters before launching the trainer.
- **📈 Real-Time Monitoring:** Launch TensorBoard with a single click to track loss curves and audio fidelity.
- **🔄 Smart Resuming:** Automatically detects the latest checkpoint (`checkpoint_latest.pt.gz`) to resume training if interrupted.
- **📦 VST3 Paraphernalia Exporter:** Packages the final weights and config into the exact structure required by Beatrice VST3 real-time voice conversion plugins.

---

## 🛠️ Step-by-Step Guide

### 1. Installation & Environment setup
- **Colab:** Mounts your Google Drive under `/content/drive/MyDrive/Beatrice_Trainer` to keep your datasets and checkpoints secure and persistent.
- **Kaggle:** Sets up directory structure under `/kaggle/working` to maximize disk throughput.
- Clones the official high-performance Beatrice training repository.

### 2. Prepare Your Dataset
Ensure your audio files are in **WAV** format. We recommend high-quality, dry (no reverb), noise-free voice recordings.
- **Hugging Face:** Input a repo ID (e.g., `meoconxinhxan/trump-speech-dataset-tts`) and speaker name. The notebook automatically extracts and downsamples the audio.
- **Manual Upload:** Upload a `.zip` containing your `.wav` files, or upload them individually.

### 3. Hyperparameter Configuration (Settings UI)
Run the Settings UI cell to launch an interactive panel. Here you can configure:
- Extraction path and dataset mapping.
- Audio backend configurations.
- GPU accelerator verification.

### 4. Start TensorBoard
Runs TensorBoard in the background and tunnels the port so you can visualize the training losses and generator metrics in real-time.

### 5. Launch Training
Configure your target step count, batch size, learning rate, and worker threads:
```python
training_steps = 5000
batch_size = 8
learning_rate = 0.00005
```
Resuming is enabled by default—if the connection drops, simply run the cell again to pick up where you left off.

### 6. Export & Download
When training completes, choose your download format:
- **VST3 Paraphernalia (Recommended):** Tiny ZIP containing only the essential metadata and checkpoint weights formatted for the VST3 voice changer plugin.
- **Latest Checkpoint (.pt.gz):** The raw PyTorch archive containing generator and discriminator states for further training.
- **Entire Output Folder:** Zips everything including intermediate training outputs.

---

## 📓 Kaggle-Specific Guide & Manual Actions

Unlike Google Colab, Kaggle's environment runs behind a secure firewall and has different file import/download mechanisms. If you are training on Kaggle, please note the following manual steps:

1. **Activate GPU Accelerator:**
   - In the right-hand panel of your Kaggle notebook under **Notebook Options**, set **Accelerator** to **GPU T4 x2** or **GPU P100**.

2. **Dataset Import (Option 2b):**
   - Kaggle does not support interactive file uploading dialogs.
   - To use your own voice/dataset:
     1. Click **+ Add Input** in the right-hand sidebar.
     2. Upload your ZIP or WAV files as a new Kaggle Dataset.
     3. Copy the dataset path (e.g., `/kaggle/input/your-dataset-name`) and paste it into the `kaggle_dataset_path` variable in the notebook cell.

3. **Gradio/Settings UI Network Tunneling:**
   - Because Kaggle blocks local network ports, the Settings UI cell will run locally but cannot be accessed directly via localhost links.
   - **Note:** You can skip the Settings UI cell entirely (the training script automatically uses optimal defaults), or configure an SSH/HTTPS tunnel (like `ngrok` or `localtunnel`) mapped to the printed local port.

4. **Downloading Outputs:**
   - Rather than triggering automatic browser downloads, Kaggle will generate a downloadable link via Jupyter `FileLink` in the cell output.
   - Click the link generated in **Step 6** of the notebook to download your VST3 compatibility ZIP file.
   - **Alternative (Manual Download):** You can also find your zipped outputs and checkpoints directly in the right-hand **"Data"** / **"Output"** sidebar under the `/kaggle/working` directory. Click the three dots next to the file (e.g., `beatrice_vst3_files.zip`) and choose **Download**.

---

## ⚠️ Important Requirements

- **GPU Acceleration:** Beatrice V2 requires a GPU to train efficiently. 
  - On **Google Colab**, ensure your runtime type is set to **T4 GPU** or better.
  - On **Kaggle**, enable **GPU T4 x2** or **P100** under the Settings panel.
- **Google Drive Storage:** Make sure you have at least 1-2 GB of free space on Google Drive for checkpoints if training on Colab.

## 🤝 Contributing

Contributions to improve dataset parsing, ui layouts, or training recipes are always welcome! Feel free to fork the repository and submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

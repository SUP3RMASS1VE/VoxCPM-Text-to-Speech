
# 🎙️ VoxCPM Text-to-Speech

A Gradio-powered application that uses **[VoxCPM-0.5B](https://modelscope.cn/models/openbmb/VoxCPM-0.5B)** for expressive text-to-speech generation, with optional **voice cloning** and **Whisper transcription** for reference audio.

## ✨ Features

- 🗣️ **Text-to-Speech (TTS):** Generate expressive, natural-sounding speech from text.  
- 🎤 **Voice Cloning:** Provide a short audio sample + transcript to mimic the reference voice.  
- 🔊 **Customizable Inference:** Adjust CFG scale, timesteps, normalization, denoising, and retries.  
- 📝 **Whisper Integration:** Automatically transcribes uploaded reference audio.  
- 🎨 **Beautiful UI:** Dark glassmorphic theme with purple accents.  
- 💾 **Outputs Saved:** All generated speech is stored in `outputs/` as `.wav` files.  

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/SUP3RMASS1VE/VoxCPM-Text-to-Speech.git
cd VoxCPM-Text-to-Speech
````

### 1. Install PyTorch

Choose the correct installation based on your OS:

#### **Windows**

```bash
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install triton-windows==3.3.1.post19
```

#### **Linux**

```bash
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install triton
```

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Gradio app:

```bash
python app.py
```

By default, it will:

* Cache models inside the `models/` directory.
* Save generated audio inside the `outputs/` directory.
* Launch a **local Gradio UI** in your browser.

---

## 🖥️ Interface Overview

* **Text to Synthesize:** Enter the text you want to convert into speech.
* **Voice Cloning (optional):**

  * Upload a short reference audio clip (3–10 seconds recommended).
  * Transcript will auto-fill using **Whisper Tiny** (editable).
* **Advanced Settings:** Fine-tune CFG scale, inference timesteps, retries, and randomness (seed).
* **Generated Speech:** Listen to or download synthesized audio directly in the UI.

---

## ⚡ Tips

* 📢 For best cloning, use **clear audio with no background noise**.
* 🎛️ Increase **inference timesteps** for higher quality (slower).
* 🎲 Set a **seed** for reproducible results (`-1` = random).
* 🧹 The system automatically clears KV cache between chunks to prevent memory issues.

---

## 📂 Project Structure

```
VoxCPM-Text-to-Speech/
├── app.py              # Main Gradio app
├── outputs/            # Generated audio files
├── models/             # Cached models (VoxCPM + Whisper)
└── README.md           # Documentation
```

---

## 📜 License

This project is licensed under the **MIT License**.
See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

* [OpenBMB](https://modelscope.cn/organization/openbmb) for **VoxCPM-0.5B**.
* [OpenAI](https://github.com/openai/whisper) for **Whisper**.
* [ModelScope](https://modelscope.cn/) for model hosting.
* [Gradio](https://gradio.app) for the web interface.

---


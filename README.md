
# 🎙️ VoxCPM Text-to-Speech
<img width="1899" height="875" alt="Screenshot 2025-09-22 200240" src="https://github.com/user-attachments/assets/b74690f7-be8d-4b91-ab1b-087d4a9caabc" />

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

Install via [Pinokio](https://pinokio.co)
You can use the Pinokio script here for one-click setup:
[Pinokio App Installer](https://pinokio.co/item.html?uri=https%3A%2F%2Fgithub.com%2FSUP3RMASS1VE%2FVoxCPM-Text-to-Speech-Pinokio&parent_frame=&theme=null)
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


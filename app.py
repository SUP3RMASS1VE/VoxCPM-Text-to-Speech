import warnings
import logging
import os

# -------------------------------
# Suppress ALL noisy warnings FIRST
# -------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress ModelScope logging completely
logging.getLogger("modelscope").setLevel(logging.CRITICAL)
logging.getLogger("modelscope").disabled = True

# Set up local models directory
MODELS_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Set environment variables to use local models directory
os.environ["MODELSCOPE_CACHE"] = MODELS_DIR
os.environ["MODELSCOPE_LOG_LEVEL"] = "40"  # 40 = ERROR level in logging
os.environ["HF_HOME"] = MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODELS_DIR, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(MODELS_DIR, "datasets")

import torch
import gradio as gr
import soundfile as sf
import numpy as np
from voxcpm import VoxCPM
import tempfile
import time
import whisper

# -------------------------------
# Enable faster matmul precision (if GPU supports it)
# -------------------------------
torch.set_float32_matmul_precision('high')

# -------------------------------
# Load the models once at startup
# -------------------------------
print(f"📁 Models will be stored in: {MODELS_DIR}")

# Load VoxCPM model to local directory
print("🔄 Loading VoxCPM model...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B", cache_dir=MODELS_DIR)
print("✅ VoxCPM model loaded")

# Load Whisper model to local directory
print("🔄 Loading Whisper model...")
whisper_models_dir = os.path.join(MODELS_DIR, "whisper")
os.makedirs(whisper_models_dir, exist_ok=True)
whisper_model = whisper.load_model("tiny", download_root=whisper_models_dir)
print("✅ Whisper model loaded")


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    if audio_path is None:
        return ""
    
    try:
        print(f"🎤 Transcribing audio: {audio_path}")
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"].strip()
        print(f"📝 Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return ""


def chunk_text(text, max_chars=500):
    """Split text into chunks that won't overflow the KV cache"""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_chars, save current chunk
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def generate_speech(
    text,
    prompt_audio,
    prompt_text,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
    retry_badcase,
    retry_badcase_max_times,
    retry_badcase_ratio_threshold,
    seed
):
    print("🚀 generate_speech function called!")
    print(f"📝 Text input: {text}")
    print(f"🎛️ Parameters: cfg={cfg_value}, steps={inference_timesteps}")
    
    if not text:
        print("❌ No text provided")
        gr.Warning("Please enter text to generate speech")
        return None

    # Handle prompt audio and text - both must be provided or both must be None
    prompt_wav_path = None
    final_prompt_text = None
    
    if prompt_audio is not None and prompt_text and prompt_text.strip():
        # Both audio and text provided - use voice cloning
        prompt_wav_path = prompt_audio
        final_prompt_text = prompt_text.strip()
        print(f"🎤 Using voice cloning with audio: {prompt_wav_path}")
        print(f"📝 Using prompt text: {final_prompt_text}")
    else:
        # Either missing audio or text - use default voice generation
        prompt_wav_path = None
        final_prompt_text = None
        print("🎵 Using default voice generation (no voice cloning)")

    try:
        # Split long text into chunks to avoid KV cache overflow
        text_chunks = chunk_text(text, max_chars=500)
        print(f"📄 Split text into {len(text_chunks)} chunks")
        
        # Determine the seed to use for ALL chunks
        if seed != -1:
            actual_seed = int(seed)
            print(f"🎲 Using provided seed: {actual_seed}")
        else:
            actual_seed = np.random.randint(0, 2147483647)
            print(f"🎲 Generated random seed: {actual_seed}")
        
        all_wavs = []
        
        for i, chunk in enumerate(text_chunks):
            print(f"🎵 Generating speech for chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
            
            # Set the SAME seed for every chunk
            torch.manual_seed(actual_seed)
            np.random.seed(actual_seed)
            print(f"🎲 Using seed {actual_seed} for chunk {i+1}")
            
            # Generate speech for this chunk
            wav = model.generate(
                text=chunk,
                prompt_wav_path=prompt_wav_path,
                prompt_text=final_prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=int(inference_timesteps),
                normalize=normalize,
                denoise=denoise,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=int(retry_badcase_max_times),
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold
            )
            
            # Convert to numpy if needed
            if torch.is_tensor(wav):
                wav = wav.cpu().numpy()
            
            # Ensure it's 1D array
            if wav.ndim > 1:
                wav = wav.squeeze()
            
            all_wavs.append(wav)
            
            # Clear any cached states between chunks to prevent memory buildup
            if hasattr(model, 'tts_model') and hasattr(model.tts_model, 'base_lm') and hasattr(model.tts_model.base_lm, 'kv_cache'):
                try:
                    model.tts_model.base_lm.kv_cache.clear()
                    print(f"🧹 Cleared KV cache after chunk {i+1}")
                except:
                    pass  # If clearing fails, continue anyway
        
        # Concatenate all audio chunks
        if len(all_wavs) > 1:
            # Add small silence between chunks (0.2 seconds)
            silence = np.zeros(int(16000 * 0.2))
            wav = np.concatenate([np.concatenate([chunk, silence]) for chunk in all_wavs[:-1]] + [all_wavs[-1]])
            print(f"🔗 Concatenated {len(all_wavs)} audio chunks")
        else:
            wav = all_wavs[0]

        print(f"✅ Speech generation completed!")
        print(f"Generated wav shape: {wav.shape}, dtype: {wav.dtype}")
            
        # Normalize audio to prevent clipping
        if wav.max() > 1.0 or wav.min() < -1.0:
            wav = wav / np.max(np.abs(wav))
        
        # Convert to 16-bit integer format for Gradio
        wav_int16 = (wav * 32767).astype(np.int16)

        # -------------------------------
        # Save to outputs/ folder
        # -------------------------------
        os.makedirs("outputs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"outputs/generated_{timestamp}.wav"
        
        try:
            sf.write(filename, wav, 16000)  # Save as float32 for better quality
            print(f"✅ Audio saved to: {os.path.abspath(filename)}")
        except Exception as save_error:
            print(f"⚠️ Failed to save audio: {save_error}")

        # Return audio as tuple (sample_rate, numpy_array) for Gradio
        print(f"🎵 Returning audio with shape: {wav_int16.shape}, dtype: {wav_int16.dtype}")
        return (16000, wav_int16)

    except Exception as e:
        print(f"💥 ERROR in generate_speech: {str(e)}")
        print(f"💥 Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        gr.Error(f"Error generating speech: {str(e)}")
        return None


# -------------------------------
# Gradio Interface
# -------------------------------
# Create custom dark glassmorphic theme with purple accents
purple_theme = gr.themes.Base(
    primary_hue="purple",
    secondary_hue="violet",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    # Same dark background for both light and dark mode
    body_background_fill="linear-gradient(135deg, rgba(15, 23, 42, 1) 0%, rgba(30, 41, 59, 1) 100%)",
    body_background_fill_dark="linear-gradient(135deg, rgba(15, 23, 42, 1) 0%, rgba(30, 41, 59, 1) 100%)",
    
    # Same panel styling for both modes
    panel_background_fill="rgba(15, 23, 42, 0.4)",
    panel_background_fill_dark="rgba(15, 23, 42, 0.4)",
    panel_border_width="1px",
    panel_border_color="rgba(147, 51, 234, 0.4)",
    panel_border_color_dark="rgba(147, 51, 234, 0.4)",
    
    # Button styling
    button_primary_background_fill="linear-gradient(135deg, #9333ea, #4f46e5)",
    button_primary_background_fill_hover="linear-gradient(135deg, #7c3aed, #4338ca)",
    button_primary_text_color="white",
    button_primary_border_color="rgba(147, 51, 234, 0.3)",
    
    # Same input styling for both modes
    input_background_fill="rgba(15, 23, 42, 0.3)",
    input_background_fill_dark="rgba(15, 23, 42, 0.3)",
    input_border_color="rgba(147, 51, 234, 0.4)",
    input_border_color_dark="rgba(147, 51, 234, 0.4)",
    
    # Slider styling
    slider_color="linear-gradient(135deg, #9333ea, #4f46e5)",
    
    # Same block styling for both modes
    block_background_fill="rgba(15, 23, 42, 0.3)",
    block_background_fill_dark="rgba(15, 23, 42, 0.3)",
    block_border_width="1px",
    block_border_color="rgba(147, 51, 234, 0.4)",
    block_border_color_dark="rgba(147, 51, 234, 0.4)",
    block_radius="16px",
    
    # Same shadow effects for both modes
    block_shadow="0 8px 32px rgba(0, 0, 0, 0.4)",
    block_shadow_dark="0 8px 32px rgba(0, 0, 0, 0.4)",
)

with gr.Blocks(
    title="VoxCPM Text-to-Speech", 
    theme=purple_theme,
    css="""
    /* Force dark theme for both light and dark mode */
    .gradio-container, .gradio-container.light, .gradio-container.dark {
        background: linear-gradient(135deg, rgba(15, 23, 42, 1) 0%, rgba(30, 41, 59, 1) 100%) !important;
        color: white !important;
    }
    
    /* Force dark styling on all elements */
    .block, .gr-box, .gr-form, .gr-panel {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(147, 51, 234, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        border-radius: 16px !important;
        color: white !important;
    }
    
    /* Force dark input styling */
    .gr-textbox, .gr-number, .gr-slider, .gr-dropdown, .gr-checkbox, .gr-radio {
        background: rgba(15, 23, 42, 0.3) !important;
        border: 1px solid rgba(147, 51, 234, 0.4) !important;
        color: white !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
    }
    
    /* Force dark text colors - comprehensive */
    .gr-textbox input, .gr-number input, .gr-dropdown select, 
    .gr-textbox textarea, label, .gr-markdown, p, span, div,
    .gr-button, .gr-checkbox label, .gr-radio label, .gr-slider label,
    .gr-accordion summary, .gr-accordion details, .gr-form label,
    .gr-box label, .gr-panel label, .markdown, .prose,
    .gr-info, .gr-warning, .gr-error, .gr-success,
    .settings-panel, .settings-panel *, .modal *, .modal-content *,
    .gr-modal *, .gr-dialog *, .overlay *, .popup *,
    h1, h2, h3, h4, h5, h6, strong, em, code, pre {
        color: white !important;
    }
    
    /* Purple button styling */
    .btn-primary, .gr-button-primary {
        background: linear-gradient(135deg, #9333ea, #4f46e5) !important;
        border: 1px solid rgba(147, 51, 234, 0.4) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-primary:hover, .gr-button-primary:hover {
        box-shadow: 0 6px 25px rgba(147, 51, 234, 0.4) !important;
        transform: translateY(-2px) !important;
        background: linear-gradient(135deg, #7c3aed, #4338ca) !important;
    }
    
    /* Audio component dark styling */
    .gr-audio {
        background: rgba(15, 23, 42, 0.3) !important;
        border: 1px solid rgba(147, 51, 234, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
    }
    
    /* Accordion dark styling */
    .gr-accordion {
        background: rgba(15, 23, 42, 0.3) !important;
        border: 1px solid rgba(147, 51, 234, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
    }
    
    /* Purple gradient text for headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #9333ea, #4f46e5) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    /* Force dark slider styling */
    .gr-slider input[type="range"] {
        background: rgba(147, 51, 234, 0.3) !important;
    }
    
    /* Force dark checkbox styling */
    .gr-checkbox input[type="checkbox"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(147, 51, 234, 0.4) !important;
    }
    
    /* Force dark styling on settings panel and modals */
    .settings-panel, .modal, .modal-content, .gr-modal, .gr-dialog,
    .overlay, .popup, .dropdown-menu, .context-menu {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(147, 51, 234, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        color: white !important;
    }
    
    /* Force all text elements to white */
    * {
        color: white !important;
    }
    
    /* Override any remaining light text */
    body, html, .gradio-container * {
        color: white !important;
    }
    
    /* Additional glassmorphic container effects */
    .gradio-container {
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
    }
    """
) as demo:
    gr.Markdown(
        """
        # 🎙️ VoxCPM Text-to-Speech 🎙️

        Generate highly expressive speech using VoxCPM-0.5B model. Optionally clone voices by providing reference audio.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter the text you want to convert to speech...",
                lines=3,
                value="You wake up to find your dog holding a job interview in your living room. He’s wearing a tie, sipping coffee, and asking YOU why you’re qualified to live in his house. What do you say?"
            )

            generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

            with gr.Accordion("Voice Cloning", open=True):
                prompt_audio = gr.Audio(
                    label="Reference Audio (Upload a reference audio file for voice cloning)",
                    type="filepath",
                    sources=["upload"]
                )
                prompt_text = gr.Textbox(
                    label="Reference Text (Auto-transcribed with Whisper)",
                    placeholder="Will be automatically filled when you upload audio above...",
                    lines=2,
                    interactive=True
                )
                gr.Markdown("💡 **Tip:** Upload your reference audio and the text will be automatically transcribed using Whisper Tiny!")

            with gr.Accordion("Advanced Settings", open=False):
                cfg_value = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.1,
                    label="CFG Value",
                    info="LM guidance on LocDiT, higher for better adherence to prompt"
                )

                inference_timesteps = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Inference Timesteps",
                    info="Higher for better quality, lower for faster speed"
                )

                with gr.Row():
                    normalize = gr.Checkbox(
                        value=True,
                        label="Normalize",
                        info="Enable external TN tool"
                    )
                    denoise = gr.Checkbox(
                        value=True,
                        label="Denoise",
                        info="Enable external Denoise tool"
                    )
                    retry_badcase = gr.Checkbox(
                        value=True,
                        label="Retry Bad Cases",
                        info="Enable retrying for bad cases"
                    )

                with gr.Row():
                    retry_badcase_max_times = gr.Number(
                        value=3,
                        minimum=1,
                        maximum=10,
                        step=1,
                        label="Max Retry Times"
                    )
                    retry_badcase_ratio_threshold = gr.Number(
                        value=6.0,
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        label="Retry Ratio Threshold"
                    )
                
                seed = gr.Number(
                    value=-1,
                    minimum=-1,
                    maximum=2147483647,
                    step=1,
                    label="Seed",
                    info="Random seed for reproducible generation (-1 for random)"
                )

        with gr.Column(scale=1):
            # Output section
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy",   # expects (sample_rate, numpy_array)
                autoplay=False,
                show_download_button=True
            )

            gr.Markdown(
                """
                ### Tips:
                - For voice cloning, upload a clear reference audio (3-10 seconds recommended)
                - Higher CFG values provide better prompt adherence but may affect naturalness
                - Increase inference timesteps for better quality at the cost of speed
                - The retry mechanism helps handle edge cases automatically
                """
            )

    # Auto-transcribe when audio is uploaded
    prompt_audio.change(
        fn=transcribe_audio,
        inputs=[prompt_audio],
        outputs=[prompt_text],
        show_progress="minimal"
    )
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_speech,
        inputs=[
            text_input,
            prompt_audio,
            prompt_text,
            cfg_value,
            inference_timesteps,
            normalize,
            denoise,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            seed
        ],
        outputs=audio_output,
        show_progress="full"
    )

demo.launch()

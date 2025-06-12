# Audio Transcription Web UI

A beautiful web interface for the audio transcription tool using Gradio.

## Features

🎵 **Easy to Use**: Simple web interface - no command line required  
🚀 **GPU Acceleration**: Automatic GPU detection and acceleration  
🌍 **Multi-Platform**: Supports YouTube, Instagram, TikTok, and 1000+ sites  
📝 **Multiple Formats**: Export as TXT, SRT, JSON, or VTT  
🔊 **Multiple Models**: Choose from Whisper tiny to large models  
🎯 **Language Support**: Auto-detection or manual language selection

## Quick Start

1. **Activate the virtual environment:**

   ```bash
   .\venv\Scripts\Activate.ps1
   ```

2. **Launch the web interface:**

   ```bash
   python gradio_ui.py
   ```

3. **Open your browser** and go to `http://localhost:7860`

## Using the Web Interface

### Input Section

- **URL**: Enter any video/audio URL (YouTube, Instagram, etc.)
- **Whisper Model**: Choose model size (tiny = fastest, large = most accurate)
- **Language**: Leave empty for auto-detection or specify (e.g., 'en', 'es', 'fr')
- **Audio Format**: Choose download format (MP3 recommended)
- **GPU Acceleration**: Enable for 3-5x faster processing
- **Output Formats**: Select which file formats to generate

### Model Comparison

| Model  | Speed      | Accuracy        | VRAM Usage | Best For      |
| ------ | ---------- | --------------- | ---------- | ------------- |
| tiny   | ⚡ Fastest | ⭐ Basic        | ~1GB       | Quick tests   |
| base   | 🏃 Fast    | ⭐⭐ Good       | ~1GB       | General use   |
| small  | 🚶 Medium  | ⭐⭐⭐ Better   | ~2GB       | Balanced      |
| medium | 🐌 Slow    | ⭐⭐⭐⭐ High   | ~5GB       | High quality  |
| large  | 🐢 Slowest | ⭐⭐⭐⭐⭐ Best | ~10GB      | Best accuracy |

### GPU Acceleration

**Automatic Detection:**

- ✅ **NVIDIA GPUs**: CUDA acceleration (3-5x faster)
- ✅ **Apple Silicon**: MPS acceleration (2-3x faster)
- ✅ **CPU Fallback**: Works on any system

**Performance Comparison:**

- CPU: ~12 seconds for 1 minute audio
- GPU: ~4 seconds for 1 minute audio

## Example Usage

1. **YouTube Video:**

   - URL: `https://www.youtube.com/watch?v=jNQXAC9IVRw`
   - Model: `base`
   - Language: `en` (or leave empty)
   - Formats: `txt`, `srt`

2. **Instagram Video:**
   - URL: `https://www.instagram.com/reel/xyz`
   - Model: `small`
   - Language: (auto-detect)
   - Formats: `txt`

## Troubleshooting

**Port Already in Use:**

```bash
python -c "import gradio_ui; gradio_ui.create_interface().launch(server_port=7861)"
```

**Missing Dependencies:**

```bash
pip install -r requirements.txt
```

**GPU Not Detected:**

- Check CUDA installation for NVIDIA GPUs
- Ensure PyTorch with CUDA support is installed

## Command Line Alternative

For power users, the command line interface is still available:

```bash
python audio_transcription.py "https://example.com/video" --model base --use-gpu
```

## File Structure

```
audio-transcript/
├── gradio_ui.py          # Web interface
├── audio_transcription.py # Core functionality
├── requirements.txt       # Dependencies
├── transcriptions/        # Output files
└── README_GRADIO.md      # This file
```

## Support

- **Supported Sites**: YouTube, Instagram, TikTok, Twitter, and 1000+ others
- **Supported Languages**: 99+ languages via Whisper
- **Supported Formats**: MP3, WAV, M4A, WebM, MP4, and more
- **Output Formats**: TXT, SRT, JSON, VTT

Enjoy transcribing! 🎉

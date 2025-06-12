#!/usr/bin/env python3
"""
Gradio Web UI for Audio Transcription

A user-friendly web interface for downloading and transcribing audio from URLs
using yt-dlp and OpenAI Whisper.
"""

import gradio as gr
import os
import tempfile
import time
from pathlib import Path
import torch

# Import the AudioTranscriber from our existing script
from audio_transcription import AudioTranscriber


class GradioAudioTranscriber:
    """Wrapper class for Gradio integration"""

    def __init__(self):
        self.transcriber = None

    def process_audio(
        self,
        url,
        file_input,
        model="base",
        language=None,
        audio_format="mp3",
        output_formats=None,
        use_gpu=False,
        progress=gr.Progress(),
    ):
        """
        Process audio transcription with progress updates for Gradio
        """
        # Check if we have either URL or file input
        if not url.strip() and file_input is None:
            return "❌ Please enter a URL or upload a file", "", "", "", ""

        if url.strip() and file_input is not None:
            return "❌ Please use either URL or file upload, not both", "", "", "", ""

        if output_formats is None:
            output_formats = ["txt"]

        try:
            # Initialize transcriber with current settings
            progress(0.1, "Initializing transcriber...")

            # Ensure transcriptions directory exists
            import os

            transcriptions_dir = os.path.join(os.getcwd(), "transcriptions")
            os.makedirs(transcriptions_dir, exist_ok=True)

            self.transcriber = AudioTranscriber(
                whisper_model=model, output_dir=transcriptions_dir, use_gpu=use_gpu
            )

            # Handle audio source (URL or file)
            if file_input is not None:
                # File upload
                progress(0.2, "Processing uploaded file...")
                start_time = time.time()
                audio_path = file_input.name  # Use the uploaded file path
                download_time = time.time() - start_time
                print(f"Using uploaded file: {audio_path}")
            else:
                # URL download
                progress(0.2, "Downloading audio...")
                start_time = time.time()
                audio_path = self.transcriber.download_audio(url, audio_format)
                download_time = time.time() - start_time

            # Load Whisper model
            progress(0.4, f"Loading Whisper model ({model})...")
            self.transcriber.load_whisper_model()

            # Transcribe audio
            progress(0.6, "Transcribing audio...")
            start_time = time.time()
            # Handle empty language input
            lang = language.strip() if language and language.strip() else None
            result = self.transcriber.transcribe_audio(audio_path, lang)
            transcribe_time = time.time() - start_time

            # Generate output filename
            progress(0.8, "Saving results...")
            audio_name = Path(audio_path).stem
            safe_name = "".join(
                c for c in audio_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()

            # Save transcription files
            self.transcriber.save_transcription(result, safe_name, output_formats)

            # Cleanup
            progress(0.9, "Cleaning up...")
            try:
                # Only cleanup downloaded files, not uploaded files
                if file_input is None and audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass

            progress(1.0, "Complete!")

            # Prepare results
            transcription_text = result["text"]

            # Generate summary
            detected_language = result.get("language", "Unknown")
            duration = result.get("duration", "Unknown")

            summary = f"""
### Transcription Complete! ✅

**Duration:** {duration} seconds  
**Processing Time:** {download_time:.2f} seconds  
**Transcription Time:** {transcribe_time:.2f} seconds  
**Detected Language:** {detected_language}  
**Model Used:** {model}  
**GPU Acceleration:** {'✅ Enabled' if use_gpu and self.transcriber.device != 'cpu' else '❌ Disabled'}  
"""

            # Prepare file paths for download
            base_path = Path("transcriptions") / safe_name

            file_paths = []
            file_contents = []

            if "txt" in output_formats:
                txt_path = f"{base_path}.txt"
                if os.path.exists(txt_path):
                    file_paths.append(txt_path)
                    with open(txt_path, "r", encoding="utf-8") as f:
                        file_contents.append(f.read())

            if "srt" in output_formats:
                srt_path = f"{base_path}.srt"
                if os.path.exists(srt_path):
                    file_paths.append(srt_path)
                    with open(srt_path, "r", encoding="utf-8") as f:
                        file_contents.append(f.read())

            if "json" in output_formats:
                json_path = f"{base_path}.json"
                if os.path.exists(json_path):
                    file_paths.append(json_path)

            if "vtt" in output_formats:
                vtt_path = f"{base_path}.vtt"
                if os.path.exists(vtt_path):
                    file_paths.append(vtt_path)

            # Return file paths as strings to avoid permission issues
            return (
                summary,
                transcription_text,
                str(file_paths[0]) if file_paths else None,
                str(file_paths[1]) if len(file_paths) > 1 else None,
                str(file_paths[2]) if len(file_paths) > 2 else None,
            )

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return error_msg, "", "", "", ""


def create_interface():
    """Create the Gradio interface"""

    # Initialize the wrapper
    transcriber_wrapper = GradioAudioTranscriber()

    # Check GPU availability
    gpu_available = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_info = f"🚀 NVIDIA GPU Available: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_info = "🍎 Apple Silicon GPU Available"
    else:
        gpu_info = "💻 CPU Only (No GPU acceleration available)"

    with gr.Blocks(
        title="Audio Transcription Tool",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            min-width: 1200px !important;
            width: 1200px !important;
            margin: 0 auto !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .file-upload {
            width: 100% !important;
        }
        .gradio-file {
            width: 100% !important;
        }
        """,
    ) as interface:
        # Header
        gr.Markdown(
            """
            # 🎵 Audio Transcription Tool
            
            Transform videos and audio from any URL into accurate text transcriptions using OpenAI Whisper.
            
            **Supported Platforms:** YouTube, Instagram, TikTok, Twitter, and 1000+ other sites via yt-dlp
            """,
            elem_classes=["main-header"],
        )

        # Input Options - URL or File Upload
        with gr.Tabs():
            with gr.TabItem("🌐 URL Input"):
                url_input = gr.Textbox(
                    label="Video/Audio URL",
                    placeholder="Enter video/audio URL (YouTube, Instagram, TikTok, etc.)",
                    lines=2,
                    scale=1,
                )

            with gr.TabItem("📁 File Upload"):
                file_input = gr.File(
                    label="Upload Audio/Video File",
                    file_types=["audio", "video"],
                    file_count="single",
                )

        # Advanced Options - Collapsed
        with gr.Accordion("⚙️ Advanced Options", open=False):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="base",
                    label="Whisper Model",
                    info="Larger models = better accuracy but slower processing",
                )

                language_input = gr.Textbox(
                    label="Language (Optional)",
                    placeholder="e.g., 'en', 'es', 'fr' - Leave empty for auto-detection",
                    max_lines=1,
                )

            with gr.Row():
                audio_format = gr.Dropdown(
                    choices=["mp3", "wav", "m4a"], value="mp3", label="Audio Format"
                )

                use_gpu = gr.Checkbox(
                    label="Use GPU Acceleration",
                    value=gpu_available,
                    interactive=gpu_available,
                    info="Enables faster processing if GPU is available",
                )

            output_formats = gr.CheckboxGroup(
                choices=["txt", "srt", "json", "vtt"],
                value=["txt"],
                label="Output Formats",
                info="Select which formats to generate",
            )

        # Start Button - Full width
        transcribe_btn = gr.Button(
            "🚀 Start Transcription", variant="primary", size="lg", scale=1
        )

        # Results section
        gr.Markdown("## 📊 Results")

        status_output = gr.Markdown(
            "Ready to transcribe! Enter a URL or upload a file and click 'Start Transcription'.",
            elem_id="status",
        )

        transcription_output = gr.Textbox(
            label="Transcription Text",
            lines=15,
            max_lines=20,
            show_copy_button=True,
        )

        # File downloads
        gr.Markdown("### 📁 Download Files")
        with gr.Row():
            file1_download = gr.File(label="Primary File", visible=False)
            file2_download = gr.File(label="Secondary File", visible=False)
            file3_download = gr.File(label="Additional File", visible=False)

        # Event handlers
        def update_file_visibility(*files):
            """Update file download visibility based on available files"""
            visibility = []
            for f in files:
                visibility.append(gr.File(visible=bool(f)))
            return visibility

        transcribe_btn.click(
            fn=transcriber_wrapper.process_audio,
            inputs=[
                url_input,
                file_input,
                model_dropdown,
                language_input,
                audio_format,
                output_formats,
                use_gpu,
            ],
            outputs=[
                status_output,
                transcription_output,
                file1_download,
                file2_download,
                file3_download,
            ],
            show_progress=True,
        ).then(
            fn=update_file_visibility,
            inputs=[file1_download, file2_download, file3_download],
            outputs=[file1_download, file2_download, file3_download],
        )

    return interface


def main():
    """Launch the Gradio interface"""
    interface = create_interface()

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create a public link
        debug=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()

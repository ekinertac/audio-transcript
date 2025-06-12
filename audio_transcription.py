#!/usr/bin/env python3
"""
Audio Transcription Script using yt-dlp and OpenAI Whisper

This script downloads videos/audio from URLs using yt-dlp,
extracts audio, and transcribes it to text using OpenAI Whisper.

Requirements:
- yt-dlp: pip install yt-dlp
- openai-whisper: pip install openai-whisper
- ffmpeg: Required for audio processing (install via system package manager)

Usage:
    python audio_transcription.py <URL> [options]
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    import yt_dlp
    import whisper
    import torch
except ImportError as e:
    print(f"Error: Missing required package. Please install: {e}")
    print("Run: pip install yt-dlp openai-whisper")
    sys.exit(1)


class AudioTranscriber:
    """Main class for downloading and transcribing audio"""

    def __init__(
        self,
        whisper_model="base",
        output_dir="transcriptions",
        temp_dir=None,
        use_gpu=False,
    ):
        """
        Initialize the transcriber

        Args:
            whisper_model (str): Whisper model size (tiny, base, small, medium, large)
            output_dir (str): Directory to save transcriptions
            temp_dir (str): Directory for temporary files (None for system temp)
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.whisper_model = whisper_model
        self.output_dir = Path(output_dir)
        self.temp_dir = temp_dir
        self.use_gpu = use_gpu
        self.model = None

        # Determine device to use
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"🚀 CUDA acceleration enabled: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                print("🍎 MPS acceleration enabled: Apple Silicon GPU")
            else:
                self.device = "cpu"
                print(
                    "⚠️ GPU requested but neither CUDA nor MPS available, falling back to CPU"
                )
        else:
            self.device = "cpu"
            print("🖥️ Using CPU for processing")

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Available Whisper models
        self.available_models = ["tiny", "base", "small", "medium", "large"]

    def load_whisper_model(self):
        """Load the Whisper model"""
        if self.model is None:
            print(f"Loading Whisper model: {self.whisper_model} on {self.device}")
            try:
                self.model = whisper.load_model(self.whisper_model, device=self.device)
                print(f"Model loaded successfully on {self.device}!")
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                raise

    def download_audio(self, url, audio_format="mp3"):
        """
        Download audio from URL using yt-dlp

        Args:
            url (str): URL to download from
            audio_format (str): Audio format (mp3, wav, m4a)

        Returns:
            str: Path to downloaded audio file
        """
        print(f"Downloading audio from: {url}")

        # Create temporary directory
        if self.temp_dir:
            temp_dir = Path(self.temp_dir)
            temp_dir.mkdir(exist_ok=True)
        else:
            temp_dir = Path(tempfile.gettempdir()) / "audio_transcription"
            temp_dir.mkdir(exist_ok=True)

        # Configure yt-dlp options
        ydl_opts = {
            "format": "bestaudio/best",
            "extractaudio": True,
            "audioformat": audio_format,
            "outtmpl": str(temp_dir / "%(title)s.%(ext)s"),
            "noplaylist": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info to get the filename
                info = ydl.extract_info(url, download=False)
                title = info.get("title", "audio")

                # Sanitize filename
                safe_title = "".join(
                    c for c in title if c.isalnum() or c in (" ", "-", "_")
                ).rstrip()
                audio_path = temp_dir / f"{safe_title}.{audio_format}"

                # Update template with sanitized name
                ydl_opts["outtmpl"] = str(temp_dir / f"{safe_title}.%(ext)s")

                # Download
                with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                    ydl_download.download([url])

                # Find the actual downloaded file
                for file in temp_dir.glob(f"{safe_title}.*"):
                    if file.suffix.lower() in [
                        ".mp3",
                        ".wav",
                        ".m4a",
                        ".ogg",
                        ".flac",
                        ".webm",
                        ".mp4",
                    ]:
                        audio_path = file
                        break

                print(f"Audio downloaded: {audio_path}")
                return str(audio_path)

        except Exception as e:
            print(f"Error downloading audio: {e}")
            raise

    def transcribe_audio(self, audio_path, language=None):
        """
        Transcribe audio file using Whisper

        Args:
            audio_path (str): Path to audio file
            language (str): Language code (auto-detect if None)

        Returns:
            dict: Transcription result with text and segments
        """
        print(f"Transcribing audio: {audio_path}")

        # Load model if not already loaded
        if self.model is None:
            self.load_whisper_model()

        try:
            # Transcribe
            result = self.model.transcribe(
                audio_path, language=language, task="transcribe", verbose=True
            )

            print("Transcription completed!")
            return result

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise

    def save_transcription(self, result, output_name, formats=["txt"]):
        """
        Save transcription in multiple formats

        Args:
            result (dict): Whisper transcription result
            output_name (str): Base name for output files
            formats (list): List of formats to save (txt, srt, json, vtt)
        """
        import json

        base_path = self.output_dir / output_name

        # Save as plain text
        if "txt" in formats:
            txt_path = f"{base_path}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            print(f"Saved transcription: {txt_path}")

        # Save as SRT subtitles
        if "srt" in formats:
            srt_path = f"{base_path}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"], 1):
                    start_time = self.seconds_to_srt_time(segment["start"])
                    end_time = self.seconds_to_srt_time(segment["end"])
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
            print(f"Saved SRT subtitles: {srt_path}")

        # Save as JSON (full result)
        if "json" in formats:
            json_path = f"{base_path}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved JSON result: {json_path}")

        # Save as VTT subtitles
        if "vtt" in formats:
            vtt_path = f"{base_path}.vtt"
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    start_time = self.seconds_to_vtt_time(segment["start"])
                    end_time = self.seconds_to_vtt_time(segment["end"])
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
            print(f"Saved VTT subtitles: {vtt_path}")

    @staticmethod
    def seconds_to_srt_time(seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    @staticmethod
    def seconds_to_vtt_time(seconds):
        """Convert seconds to VTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def process_url(
        self,
        url,
        language=None,
        audio_format="mp3",
        output_formats=["txt"],
        cleanup=True,
    ):
        """
        Complete workflow: download, transcribe, and save

        Args:
            url (str): URL to process
            language (str): Language code for transcription
            audio_format (str): Audio format for download
            output_formats (list): Formats to save transcription
            cleanup (bool): Whether to delete temporary audio file

        Returns:
            dict: Transcription result
        """
        audio_path = None
        try:
            # Download audio
            start_time = time.time()
            audio_path = self.download_audio(url, audio_format)
            download_time = time.time() - start_time

            # Transcribe
            start_time = time.time()
            result = self.transcribe_audio(audio_path, language)
            transcribe_time = time.time() - start_time

            # Generate output filename
            audio_name = Path(audio_path).stem
            safe_name = "".join(
                c for c in audio_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()

            # Save transcription
            self.save_transcription(result, safe_name, output_formats)

            # Print summary
            print(f"\n{'='*50}")
            print(f"TRANSCRIPTION COMPLETE")
            print(f"{'='*50}")
            print(f"Audio Duration: {result.get('duration', 'Unknown')} seconds")
            print(f"Download Time: {download_time:.2f} seconds")
            print(f"Transcription Time: {transcribe_time:.2f} seconds")
            print(f"Detected Language: {result.get('language', 'Unknown')}")
            print(f"Model Used: {self.whisper_model}")
            print(f"Output Directory: {self.output_dir}")
            print(f"{'='*50}")

            return result

        except Exception as e:
            print(f"Error processing URL: {e}")
            raise
        finally:
            # Cleanup temporary audio file
            if cleanup and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"Cleaned up temporary file: {audio_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {audio_path}: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Download and transcribe audio from URLs using yt-dlp and Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_transcription.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  python audio_transcription.py "https://example.com/audio.mp3" --model large --language en
  python audio_transcription.py "URL" --output-dir ./results --output-formats txt srt json
  python audio_transcription.py "URL" --model base --use-gpu --language tr
  
Available Whisper Models:
  tiny   - Fastest, least accurate, ~1GB VRAM
  base   - Good balance, ~1GB VRAM (default)
  small  - Better accuracy, ~2GB VRAM  
  medium - High accuracy, ~5GB VRAM
  large  - Best accuracy, ~10GB VRAM

GPU Acceleration:
  --use-gpu - Enable GPU acceleration (Auto-detects CUDA/MPS)
  
  Windows/Linux + NVIDIA: CUDA acceleration (3-5x faster)
  macOS + Apple Silicon: MPS acceleration (2-3x faster)
  Fallback: CPU processing if GPU unavailable
        """,
    )

    parser.add_argument("url", help="URL to download and transcribe")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language",
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified",
    )
    parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=["mp3", "wav", "m4a"],
        help="Audio format for download (default: mp3)",
    )
    parser.add_argument(
        "--output-dir",
        default="transcriptions",
        help="Output directory for transcriptions (default: transcriptions)",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=["txt"],
        choices=["txt", "srt", "json", "vtt"],
        help="Output formats (default: txt)",
    )
    parser.add_argument("--temp-dir", help="Temporary directory for downloads")
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Don't delete temporary audio files"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration if available (requires CUDA-enabled PyTorch)",
    )

    args = parser.parse_args()

    try:
        # Create transcriber
        transcriber = AudioTranscriber(
            whisper_model=args.model,
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            use_gpu=args.use_gpu,
        )

        # Process URL
        result = transcriber.process_url(
            url=args.url,
            language=args.language,
            audio_format=args.audio_format,
            output_formats=args.output_formats,
            cleanup=not args.no_cleanup,
        )

        print(f"\nTranscription text preview:")
        print(f"{'='*50}")
        print(result["text"][:500] + ("..." if len(result["text"]) > 500 else ""))
        print(f"{'='*50}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

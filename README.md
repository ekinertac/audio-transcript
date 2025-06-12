# Audio Transcription Tool

A Python script that downloads audio/video from URLs using **yt-dlp** and transcribes it to text using **OpenAI Whisper**.

## Features

- ✅ Download audio from any URL supported by yt-dlp (YouTube, podcasts, etc.)
- ✅ Automatic audio format conversion
- ✅ High-quality transcription using OpenAI Whisper
- ✅ Multiple output formats (TXT, SRT, JSON, VTT)
- ✅ Language detection and translation
- ✅ Subtitle generation with timestamps
- ✅ Command-line interface
- ✅ Configurable model sizes for speed vs accuracy

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install yt-dlp openai-whisper
```

### 2. Install FFmpeg (Required)

**Windows:**

```bash
# Using Chocolatey
choco install ffmpeg

# Using Scoop
scoop install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt update && sudo apt install ffmpeg
```

## Transcription Tools Comparison

| Tool                      | Pros                                             | Cons                          | Best For                   |
| ------------------------- | ------------------------------------------------ | ----------------------------- | -------------------------- |
| **OpenAI Whisper** ⭐     | Free, runs locally, high accuracy, 99+ languages | Slower than cloud APIs        | Privacy, offline use       |
| **WhisperX**              | Enhanced Whisper with speaker diarization        | More complex setup            | Multi-speaker content      |
| **Google Speech-to-Text** | Fast, cloud-powered                              | Requires API key, costs money | Real-time applications     |
| **Azure Speech Services** | Enterprise features                              | Requires API key, costs money | Business applications      |
| **Assembly AI**           | Specialized features                             | Requires API key, costs money | Professional transcription |

**For this script, we use OpenAI Whisper because it's:**

- Free and runs completely offline
- Extremely accurate (state-of-the-art)
- Supports 99+ languages
- No API keys required
- Privacy-friendly (data stays on your machine)

## Usage

### Basic Usage

```bash
# Transcribe a YouTube video
python audio_transcription.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Transcribe any audio URL
python audio_transcription.py "https://example.com/podcast.mp3"
```

### Advanced Usage

```bash
# Use larger model for better accuracy
python audio_transcription.py "URL" --model large

# Specify language (faster than auto-detection)
python audio_transcription.py "URL" --language en

# Translate to English
python audio_transcription.py "URL" --task translate --language es

# Custom output directory and formats
python audio_transcription.py "URL" --output-dir ./results --output-formats txt srt

# Keep temporary audio file
python audio_transcription.py "URL" --no-cleanup
```

### Whisper Models

| Model    | Size    | Speed         | Accuracy | VRAM   | Best For           |
| -------- | ------- | ------------- | -------- | ------ | ------------------ |
| `tiny`   | 39 MB   | ~10x faster   | Lower    | ~1 GB  | Quick transcripts  |
| `base`   | 74 MB   | ~7x faster    | Good     | ~1 GB  | **Default choice** |
| `small`  | 244 MB  | ~4x faster    | Better   | ~2 GB  | Balanced quality   |
| `medium` | 769 MB  | ~2x faster    | High     | ~5 GB  | High quality       |
| `large`  | 1550 MB | 1x (baseline) | Highest  | ~10 GB | Best accuracy      |

### Command Line Options

```
positional arguments:
  url                   URL to download and transcribe

options:
  -h, --help            show help message
  --model {tiny,base,small,medium,large}
                        Whisper model size (default: base)
  --language LANGUAGE   Language code (e.g., 'en', 'es', 'fr')
  --task {transcribe,translate}
                        Task to perform (default: transcribe)
  --audio-format {mp3,wav,m4a}
                        Audio format for download (default: mp3)
  --output-dir OUTPUT_DIR
                        Output directory (default: transcriptions)
  --output-formats {txt,srt,json,vtt} [{txt,srt,json,vtt} ...]
                        Output formats (default: txt srt json)
  --temp-dir TEMP_DIR   Temporary directory for downloads
  --no-cleanup          Don't delete temporary audio files
```

## Output Formats

The script generates multiple output formats:

### 1. Plain Text (.txt)

```
This is the transcribed text without timestamps.
Perfect for reading or further text processing.
```

### 2. SRT Subtitles (.srt)

```
1
00:00:00,000 --> 00:00:03,000
This is the transcribed text with timestamps.

2
00:00:03,000 --> 00:00:06,000
Perfect for video subtitles.
```

### 3. JSON (.json)

```json
{
  "text": "Full transcription...",
  "segments": [...],
  "language": "en"
}
```

### 4. VTT Subtitles (.vtt)

```
WEBVTT

00:00:00.000 --> 00:00:03.000
This is WebVTT format for web players.
```

## Examples

### Example 1: Transcribe YouTube Video

```bash
python audio_transcription.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --model base
```

### Example 2: Spanish to English Translation

```bash
python audio_transcription.py "https://spanish-podcast.com/episode.mp3" \
  --language es \
  --task translate \
  --model medium
```

### Example 3: High-Quality Transcription

```bash
python audio_transcription.py "https://interview.com/audio.wav" \
  --model large \
  --output-formats txt srt json \
  --output-dir ./interviews
```

## Supported URLs

Thanks to yt-dlp, this script supports 1000+ websites including:

- **Video platforms:** YouTube, Vimeo, Dailymotion, Twitch
- **Audio platforms:** SoundCloud, Spotify (where allowed), Bandcamp
- **Social media:** Twitter, Facebook, Instagram, TikTok
- **News sites:** BBC, CNN, NPR
- **Educational:** Coursera, edX, Khan Academy
- **Direct media files:** MP3, MP4, WAV, etc.

## Performance Tips

1. **Model Selection:**

   - Use `tiny` or `base` for quick transcripts
   - Use `medium` or `large` for important content

2. **Language Specification:**

   - Specify `--language` when known (faster than auto-detection)

3. **Hardware:**

   - GPU acceleration automatically used when available
   - More RAM = can use larger models

4. **Audio Quality:**
   - Higher quality audio = better transcription
   - Use `--audio-format wav` for best quality

## Troubleshooting

### Common Issues

**"FFmpeg not found"**

```bash
# Install FFmpeg (see installation section above)
# Ensure it's in your system PATH
```

**"CUDA out of memory"**

```bash
# Use smaller model
python audio_transcription.py "URL" --model tiny
```

**"URL not supported"**

```bash
# Check if yt-dlp supports the site
yt-dlp --list-extractors | grep -i "sitename"
```

**"Transcription quality poor"**

```bash
# Try larger model
python audio_transcription.py "URL" --model large

# Or specify language
python audio_transcription.py "URL" --language en
```

## Advanced: Using as Python Module

```python
from audio_transcription import AudioTranscriber

# Create transcriber
transcriber = AudioTranscriber(whisper_model="base", output_dir="./results")

# Process URL
result = transcriber.process_url(
    url="https://example.com/audio.mp3",
    language="en",
    task="transcribe"
)

# Access transcription
print(result['text'])
print(f"Detected language: {result['language']}")
```

## License

MIT License - Feel free to use and modify as needed.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Alternatives

- **WhisperX:** Enhanced Whisper with speaker diarization
- **Faster-Whisper:** Optimized Whisper implementation
- **Whisper.cpp:** C++ implementation for better performance
- **Cloud APIs:** Google, Azure, AWS for different use cases

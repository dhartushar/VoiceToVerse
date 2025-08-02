# VoiceToVerse

Transcribe audio files with speaker diarization using state-of-the-art open-source models — built with FastAPI and powered by `faster-whisper` and `pyannote.audio`.

---

## Overview

**VoiceToVerse** is an API service that:

- Transcribes spoken content using [`faster-whisper`](https://github.com/guillaumekln/faster-whisper)
- Identifies *who spoke when* using [`pyannote.audio`](https://huggingface.co/pyannote/segmentation-3.0) (speaker diarization)
- Accepts common audio formats: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.mp4`, `.avi`
- Returns structured transcripts with timestamps and speaker labels
- Built with FastAPI for high performance and easy integration

> ⚠️ Speaker diarization may take longer than transcription — allow a few extra seconds for response.

---

## Models Used

- **Transcription**: [`faster-whisper`](https://github.com/guillaumekln/faster-whisper)
- **Diarization**: [`pyannote/speaker-diarization`](https://huggingface.co/pyannote/speaker-diarization)

---

## Requirements

Before running the application, complete the following steps:

1. **Create a Hugging Face account**  
    https://huggingface.co/join

2. **Accept model licenses**  
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)  
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

3. **Generate Hugging Face Access Token**  
    https://huggingface.co/settings/tokens  
   - Create a token with **read** access.

4. **Create a `.env` file** in the root directory and add your token(HUGGINGFACE_TOKEN):  


---

## Setup Instructions

### Option 1: With `script.sh`

```bash
git clone https://github.com/your-username/VoiceToVerse.git
cd VoiceToVerse
./script.sh
```

### Option 2: With `requirements.txt`

```bash
git clone https://github.com/your-username/VoiceToVerse.git
cd VoiceToVerse
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the server
run the fasAPI server with 
```
python main.py
```

##API usage

```
curl -X POST http://example.com/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/file.mp3"
```

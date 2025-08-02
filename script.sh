# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install FastAPI first (very fast)
pip install fastapi uvicorn python-multipart

# Install CPU-only PyTorch (much faster than GPU version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core audio processing
pip install librosa soundfile numpy pydantic python-dotenv

# Install Faster Whisper (medium speed)
pip install faster-whisper

# Install pyannote last (this is the slowest)
pip install pyannote.audio
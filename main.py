import os
from dotenv import load_dotenv
import tempfile
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from logger_config import logger 
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np

load_dotenv()

# Pydantic models for request/response
class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

class TranscriptionResponse(BaseModel):
    status: str
    duration: float
    language: str
    segments: List[TranscriptionSegment]
    speakers_count: int

class TranscriptionService:
    def __init__(self):
        self.whisper_model = None
        self.diarization_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    async def initialize_models(self):
        """Initialize Whisper and diarization models"""
        try:
            # Initialize Faster Whisper model
            logger.info("Whisper Model Loading.")
            self.whisper_model = WhisperModel(
                "base",  # Good balance of speed and quality
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            logger.info("Whisper model loaded successfully!")
            
            # Initialize pyannote diarization pipeline
            logger.info("Diarization Pipeline Loading.")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
                )
                
                if self.device == "cuda":
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                
                logger.info("Diarization pipeline loaded successfully!")
                
            except Exception as e:
                logger.warning(f"Could not load diarization pipeline: {e}")
                logger.info("Continuing with transcription only...")
                self.diarization_pipeline = None
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise e
    
    def preprocess_audio(self, audio_path: str) -> str:
        """Robust audio preprocessing"""
        try:
            # Read audio file
            data, samplerate = sf.read(audio_path, always_2d=False)
            logger.info(f"Original audio: shape={getattr(data, 'shape', 'scalar')}, sr={samplerate}")
            
            # Convert to numpy array and ensure proper shape
            data = np.asarray(data, dtype=np.float32)
            
            # Handle different audio shapes
            if data.ndim > 1:
                # Multi-channel audio: convert to mono
                if data.shape[1] > 1:
                    data = np.mean(data, axis=1)
                else:
                    data = data.squeeze()
            
            # Ensure 1D array
            data = np.atleast_1d(data)
            
            # Basic audio validation
            if len(data) == 0:
                raise ValueError("Empty audio data")
            
            # Normalize audio
            if np.max(np.abs(data)) > 0:
                data = data / np.max(np.abs(data)) * 0.9
            
            # Resample if needed
            if samplerate != 16000:
                logger.info(f"Resampling from {samplerate}Hz to 16000Hz")
                # Simple linear interpolation resampling
                original_length = len(data)
                target_length = int(original_length * 16000 / samplerate)
                
                if target_length > 0:
                    indices = np.linspace(0, original_length - 1, target_length)
                    data = np.interp(indices, np.arange(original_length), data)
                    samplerate = 16000
            
            # Save processed audio
            processed_path = str(Path(audio_path).with_suffix('.processed.wav'))
            sf.write(processed_path, data, samplerate)
            logger.info(f"Processed audio saved: {processed_path}")
            
            return processed_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            logger.info("Using original audio file")
            return audio_path
    
    async def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using Faster Whisper"""
        try:
            logger.info("Starting transcription...")
            segments, info = self.whisper_model.transcribe(
                audio_path,
                beam_size=5,
                language=None,  # Auto-detect language
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            
            # Convert segments to list
            transcription_segments = []
            for segment in segments:
                transcription_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                })
            
            logger.info(f"Transcription completed: {len(transcription_segments)} segments")
            
            return {
                "segments": transcription_segments,
                "language": info.language,
                "duration": info.duration
            }
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise e
    
    async def perform_diarization(self, audio_path: str) -> Dict[str, Any]:
        """Perform speaker diarization"""
        if not self.diarization_pipeline:
            logger.warning("Diarization pipeline not available")
            return {
                "speaker_segments": [],
                "speakers": ["SPEAKER_00"],
                "speakers_count": 1
            }
        
        try:
            logger.info("Starting speaker diarization...")
            diarization = self.diarization_pipeline(audio_path)
            
            # Extract speaker segments
            speaker_segments = []
            speakers = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
                speakers.add(speaker)
            
            logger.info(f"Diarization completed: {len(speakers)} speakers found")
            
            return {
                "speaker_segments": speaker_segments,
                "speakers": list(speakers),
                "speakers_count": len(speakers)
            }
            
        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            # Return single speaker as fallback
            return {
                "speaker_segments": [],
                "speakers": ["SPEAKER_00"],
                "speakers_count": 1
            }
    
    def align_transcription_with_speakers(self, transcription_segments: List[Dict], 
                                        speaker_segments: List[Dict]) -> List[TranscriptionSegment]:
        """Align transcription segments with speaker information"""
        aligned_segments = []
        
        # If no speaker segments, assign default speaker
        if not speaker_segments:
            for trans_seg in transcription_segments:
                aligned_segments.append(TranscriptionSegment(
                    start=trans_seg["start"],
                    end=trans_seg["end"],
                    text=trans_seg["text"],
                    speaker="SPEAKER_00"
                ))
            return aligned_segments
        
        # Align with speaker segments
        for trans_seg in transcription_segments:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            
            # Find overlapping speaker segments
            overlapping_speakers = []
            for spk_seg in speaker_segments:
                spk_start = spk_seg["start"]
                spk_end = spk_seg["end"]
                
                # Check for overlap
                if (trans_start < spk_end and trans_end > spk_start):
                    overlap_duration = min(trans_end, spk_end) - max(trans_start, spk_start)
                    overlapping_speakers.append((spk_seg["speaker"], overlap_duration))
            
            # Assign speaker with maximum overlap
            if overlapping_speakers:
                best_speaker = max(overlapping_speakers, key=lambda x: x[1])[0]
            else:
                best_speaker = "SPEAKER_00"
            
            aligned_segments.append(TranscriptionSegment(
                start=trans_start,
                end=trans_end,
                text=trans_seg["text"],
                speaker=best_speaker
            ))
        
        return aligned_segments
    
    async def process_audio_file(self, audio_path: str) -> TranscriptionResponse:
        """Main processing function"""
        try:
            logger.info(f"Processing audio file: {audio_path}")
            
            # Preprocess audio
            processed_audio_path = self.preprocess_audio(audio_path)
            
            # Run transcription and diarization concurrently if diarization is available
            logger.info("self.diarization_pipeline: %s", self.diarization_pipeline)
            if self.diarization_pipeline:
                logger.info("Running transcription and diarization concurrently...")
                transcription_task = asyncio.create_task(
                    self.transcribe_audio(processed_audio_path)
                )
                diarization_task = asyncio.create_task(
                    self.perform_diarization(processed_audio_path)
                )
                
                transcription_result, diarization_result = await asyncio.gather(
                    transcription_task, diarization_task
                )
            else:
                logger.info("Running transcription only...")
                transcription_result = await self.transcribe_audio(processed_audio_path)
                diarization_result = {
                    "speaker_segments": [],
                    "speakers": ["SPEAKER_00"],
                    "speakers_count": 1
                }
            
            # Align transcription with speakers
            aligned_segments = self.align_transcription_with_speakers(
                transcription_result["segments"],
                diarization_result["speaker_segments"]
            )
            
            # Clean up processed audio file if different from original
            if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            
            logger.info("Audio processing completed successfully")
            
            return TranscriptionResponse(
                status="success",
                duration=transcription_result["duration"],
                language=transcription_result["language"],
                segments=aligned_segments,
                speakers_count=diarization_result["speakers_count"]
            )
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Audio Transcription API with Speaker Diarization",
    description="FastAPI backend for audio transcription with speaker diarization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize transcription service
transcription_service = TranscriptionService()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting Audio Transcription API...")
    await transcription_service.initialize_models()
    logger.info("API ready to accept requests!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Audio Transcription API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "whisper_model_loaded": transcription_service.whisper_model is not None,
        "diarization_model_loaded": transcription_service.diarization_pipeline is not None,
        "device": transcription_service.device
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Transcribe audio file with speaker diarization
    
    Supported formats: WAV, MP3, M4A, FLAC, OGG, MP4, AVI
    """
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".avi"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_path = temp_file.name
        
        # Save uploaded file
        content = await file.read()
        temp_file.write(content)
    
    try:
        logger.info(f"Received file: {file.filename} ({len(content)} bytes)")
        
        # Process the audio file
        result = await transcription_service.process_audio_file(temp_path)
        
        # Schedule cleanup of temporary file
        background_tasks.add_task(os.remove, temp_path)
        
        return result
    
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Request failed: {e}")
        raise e

if __name__ == "__main__":
    # Set environment variables for optimal performance
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1  # Single worker to avoid model loading issues
    )
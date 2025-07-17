try:
    import whisper
except ImportError:
    whisper = None
    logging.error("The 'whisper' package is not installed. Please install it with 'pip install whisper'.")
import torch
import numpy as np
import wave
import io
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self):
        # Initialize Whisper model (open source)
        self.whisper_model = whisper.load_model("base")
        
        # Initialize Piper TTS
        self.piper_model_path = self._download_piper_model()
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        
    def _download_piper_model(self) -> str:
        """Download Piper TTS model if not exists"""
        model_dir = Path("./models/piper")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "en_US-lessac-medium.onnx"
        config_path = model_dir / "en_US-lessac-medium.onnx.json"
        
        if not model_path.exists():
            logger.info("Downloading Piper TTS model...")
            # Download from GitHub releases
            import requests
            
            model_url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx"
            config_url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json"
            
            # Download model
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            
            # Download config
            response = requests.get(config_url)
            with open(config_path, 'wb') as f:
                f.write(response.content)
            
            logger.info("Piper TTS model downloaded successfully")
        
        return str(model_path)
    
    async def speech_to_text(self, audio_data: bytes) -> Dict[str, Any]:
        """Convert speech to text using Whisper"""
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            # Use Whisper to transcribe
            result = self.whisper_model.transcribe(tmp_file_path)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "confidence": self._calculate_confidence(result),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Speech to text error: {e}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    async def text_to_speech(self, text: str, voice_speed: float = 1.0) -> Dict[str, Any]:
        """Convert text to speech using Piper TTS"""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Run Piper TTS
            cmd = [
                "piper",
                "--model", self.piper_model_path,
                "--output_file", output_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                raise Exception(f"Piper TTS failed: {stderr}")
            
            # Read generated audio
            with open(output_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(output_path)
            
            return {
                "audio_data": audio_data,
                "format": "wav",
                "duration": self._get_audio_duration(audio_data),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Text to speech error: {e}")
            return {
                "audio_data": b"",
                "error": str(e),
                "success": False
            }
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence score from Whisper result"""
        # Whisper doesn't provide direct confidence scores
        # We estimate based on segment probabilities
        if "segments" in whisper_result:
            probs = [segment.get("avg_logprob", 0) for segment in whisper_result["segments"]]
            if probs:
                return max(0, min(1, np.mean(probs) + 5) / 5)  # Normalize to 0-1
        return 0.7  # Default confidence
    
    def _get_audio_duration(self, audio_data: bytes) -> float:
        """Get duration of audio data"""
        try:
            import wave
            with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                return frames / sample_rate
        except:
            return 0.0
    
    def validate_audio_format(self, audio_data: bytes) -> bool:
        """Validate audio format"""
        try:
            # Check if it's a valid WAV file
            with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                return wav_file.getnframes() > 0
        except:
            return False
    
    def preprocess_audio(self, audio_data: bytes) -> bytes:
        """Preprocess audio for better recognition"""
        try:
            # Load audio with librosa
            import librosa
            import soundfile as sf
            
            # Convert to numpy array
            audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=self.sample_rate)
            
            # Normalize audio
            audio_array = librosa.util.normalize(audio_array)
            
            # Remove silence
            audio_array, _ = librosa.effects.trim(audio_array)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            sf.write(output_buffer, audio_array, self.sample_rate, format='WAV')
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio_data  # Return original if preprocessing fails

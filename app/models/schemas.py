from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import uuid
from datetime import datetime

class QueryType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"
    DOCUMENT = "document"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"

class QueryRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    query: str = Field(..., description="User query text")
    query_type: QueryType = Field(default=QueryType.TEXT)
    preferred_llm: Optional[Union[LLMProvider, str]] = None  # Allow string or enum
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[Dict[str, Any]] = None
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    def model_post_init(self, __context):
        """Post-process the model after validation"""
        # Convert string preferred_llm to enum if needed
        if isinstance(self.preferred_llm, str):
            try:
                self.preferred_llm = LLMProvider(self.preferred_llm.lower())
            except ValueError:
                self.preferred_llm = None  # Invalid provider, set to None

class ImageUpload(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    image_data: str = Field(..., description="Base64 encoded image data")
    image_type: str = Field(default="png")
    query: str = Field(..., description="Query about the image")

class DocumentUpload(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    file_name: str
    file_type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class RetrievalContext(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    chunks: List[str]
    sources: List[str]
    relevance_scores: List[float]
    metadata: Optional[Dict[str, Any]] = None

class LLMResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    content: str
    model_used: str
    tokens_used: int
    response_time: float
    confidence_score: Optional[float] = None

class AgentStep(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    agent_name: str
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None

class QueryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    response: str
    model_used: str
    session_id: str
    context_used: Optional[RetrievalContext] = None
    agent_steps: List[AgentStep] = []
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthCheck(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]

# Document processing models
class DocumentChunk(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    content: str
    metadata: Dict[str, Any]
    context: Dict[str, str] = {}
    embedding: Optional[List[float]] = None

class IngestionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool
    document_hash: Optional[str] = None
    chunks: List[DocumentChunk] = []
    total_chunks: int = 0
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class QueryRefinementRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    query: str
    context: Optional[str] = None
    refinement_type: Optional[str] = None
    user_preferences: Dict[str, Any] = {}

class QueryRefinementResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    original_query: str
    refined_query: str
    sub_queries: List[str] = []
    refinement_type: str
    confidence_score: float
    reasoning: str

class RetrievalResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    chunk: DocumentChunk
    similarity_score: float
    rank: int
    search_type: str

# ✅ MISSING VOICE MODELS - ADDED BELOW

class VoiceRequest(BaseModel):
    """Request model for text-to-speech conversion"""
    model_config = ConfigDict(protected_namespaces=())
    
    text: str = Field(..., description="Text to convert to speech")
    language_code: Optional[str] = Field(default="en-US", description="Language code (e.g., en-US, es-ES)")
    voice_name: Optional[str] = Field(default=None, description="Specific voice name (e.g., en-US-Wavenet-C)")
    voice_gender: Optional[str] = Field(default="NEUTRAL", description="Voice gender (MALE, FEMALE, NEUTRAL)")
    speaking_rate: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Speaking rate (0.25 to 4.0)")
    pitch: Optional[float] = Field(default=0.0, ge=-20.0, le=20.0, description="Voice pitch (-20.0 to 20.0)")
    audio_encoding: Optional[str] = Field(default="MP3", description="Audio encoding format (MP3, WAV, OGG_OPUS)")

class VoiceResponse(BaseModel):
    """Response model for text-to-speech conversion"""
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool = Field(..., description="Whether the voice generation was successful")
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio content")
    audio_format: Optional[str] = Field(default=None, description="Audio format (mp3, wav, ogg)")
    text: str = Field(..., description="The text that was converted to speech")
    original_text: Optional[str] = Field(default=None, description="Original text before cleaning")
    voice_settings: Optional[Dict[str, Any]] = Field(default=None, description="Voice settings used")
    text_length: Optional[int] = Field(default=None, description="Length of processed text")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class SpeechToTextRequest(BaseModel):
    """Request model for speech-to-text conversion"""
    model_config = ConfigDict(protected_namespaces=())
    
    language_code: Optional[str] = Field(default="en-US", description="Language code for recognition")
    audio_encoding: Optional[str] = Field(default="MP3", description="Audio encoding format")
    sample_rate: Optional[int] = Field(default=16000, description="Audio sample rate in Hz")
    enable_word_confidence: Optional[bool] = Field(default=True, description="Enable word-level confidence scores")
    enable_automatic_punctuation: Optional[bool] = Field(default=True, description="Enable automatic punctuation")

class SpeechToTextResponse(BaseModel):
    """Response model for speech-to-text conversion"""
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool = Field(..., description="Whether the speech recognition was successful")
    text: str = Field(..., description="Transcribed text from audio")
    confidence: Optional[float] = Field(default=None, description="Overall confidence score (0.0 to 1.0)")
    words: Optional[List[Dict[str, Any]]] = Field(default=None, description="Word-level timing and confidence")
    language_code: Optional[str] = Field(default=None, description="Detected or specified language")
    audio_size: Optional[int] = Field(default=None, description="Size of processed audio in bytes")
    processing_time: Optional[float] = Field(default=None, description="Time taken to process audio")
    error: Optional[str] = Field(default=None, description="Error message if recognition failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

# Enhanced voice models (existing ones updated)
class VoiceQuery(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    audio_data: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="wav", description="Audio format (wav, mp3, etc.)")
    language: str = Field(default="en-US", description="Language code")
    transcription_method: str = Field(default="whisper", description="Speech recognition method")
    voice_settings: Dict[str, Any] = Field(default_factory=dict)
    query_settings: Dict[str, Any] = Field(default_factory=dict)

class VoiceSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    language: str = "en-US"
    voice_speed: float = 1.0
    voice_pitch: float = 1.0
    voice_gender: str = "female"
    tts_engine: str = "google"
    stt_engine: str = "google"
    voice_name: Optional[str] = None
    auto_play: bool = True
    enable_voice_responses: bool = True

# Voice-enhanced RAG models
class VoiceRAGRequest(BaseModel):
    """Request model for RAG with voice capabilities"""
    model_config = ConfigDict(protected_namespaces=())
    
    query: str = Field(..., description="User query text")
    generate_voice: bool = Field(default=False, description="Generate voice response")
    voice_settings: Optional[VoiceSettings] = Field(default=None, description="Voice generation settings")
    rag_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="RAG-specific settings")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

class VoiceRAGResponse(BaseModel):
    """Response model for RAG with voice capabilities"""
    model_config = ConfigDict(protected_namespaces=())
    
    text_response: str = Field(..., description="Text response from RAG")
    voice_response: Optional[VoiceResponse] = Field(default=None, description="Voice response if generated")
    context_used: Optional[RetrievalContext] = Field(default=None, description="Context used for RAG")
    model_used: str = Field(..., description="LLM model used")
    session_id: str = Field(..., description="Session identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

# Audio processing models
class AudioMetadata(BaseModel):
    """Metadata for audio files"""
    model_config = ConfigDict(protected_namespaces=())
    
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    sample_rate: Optional[int] = Field(default=None, description="Sample rate in Hz")
    channels: Optional[int] = Field(default=None, description="Number of audio channels")
    format: Optional[str] = Field(default=None, description="Audio format")
    size: Optional[int] = Field(default=None, description="File size in bytes")

class AudioProcessingResult(BaseModel):
    """Result of audio processing operations"""
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool = Field(..., description="Whether processing was successful")
    processed_audio: Optional[str] = Field(default=None, description="Base64 encoded processed audio")
    metadata: Optional[AudioMetadata] = Field(default=None, description="Audio metadata")
    processing_time: Optional[float] = Field(default=None, description="Time taken for processing")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")

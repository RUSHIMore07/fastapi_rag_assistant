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
    model_used: str  # This field caused the warning
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
    model_used: str  # This field caused the warning
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



# Add these to your existing schemas

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




# Add these new models to your existing schemas

class VoiceQuery(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    audio_data: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="wav", description="Audio format (wav, mp3, etc.)")
    language: str = Field(default="en-US", description="Language code")
    transcription_method: str = Field(default="whisper", description="Speech recognition method")
    voice_settings: Dict[str, Any] = Field(default_factory=dict)
    query_settings: Dict[str, Any] = Field(default_factory=dict)

class VoiceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool
    transcribed_text: str
    response_text: str
    audio_response: Optional[bytes] = None
    confidence_score: float = 0.0
    model_used: str = ""
    context_used: Optional[RetrievalContext] = None
    processing_time: float = 0.0
    voice_engine: str = ""
    error: Optional[str] = None

class VoiceSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    language: str = "en-US"
    voice_speed: float = 1.0
    voice_pitch: float = 1.0
    voice_gender: str = "female"
    tts_engine: str = "openai"
    stt_engine: str = "whisper"
    voice_name: Optional[str] = None

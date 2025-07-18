# # import sys
# # import os
# # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# # from pydantic_settings import BaseSettings
# # from typing import Optional, List

# # class Settings(BaseSettings):
# #     # API Configuration
# #     app_name: str = "Agentic RAG Assistant"
# #     debug: bool = False
# #     version: str = "1.0.0"
    
# #     # Add LOG_LEVEL attribute
# #     log_level: str = "INFO"  # Add this line
    
# #     # LLM API Keys
# #     openai_api_key: Optional[str] = None
# #     google_api_key: Optional[str] = None
# #     groq_api_key: Optional[str] = None
    
# #     # Redis
# #     redis_url: str = "redis://localhost:6379"
    
# #     # FAISS Configuration
# #     faiss_index_path: str = "./data/faiss_index"
# #     embedding_model: str = "text-embedding-ada-002"
    
# #     # Model Configuration
# #     default_llm: str = "gpt-4o-mini"
# #     available_models: List[str] = [
# #         "gpt-4o-mini", "gpt-4o-mini", "gemini-pro", 
# #         "groq-mixtral", "ollama-llama3.2:latest"
# #     ]
    
# #     # File Upload
# #     max_file_size: int = 10 * 1024 * 1024  # 10MB
# #     allowed_file_types: List[str] = [
# #         "pdf", "txt", "docx", "jpg", "jpeg", "png"
# #     ]
    
# #     class Config:
# #         env_file = ".env"
# #         env_file_encoding = "utf-8"

# # settings = Settings()


# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from pydantic_settings import BaseSettings
# from typing import Optional, List

# class Settings(BaseSettings):
#     # API Configuration
#     app_name: str = "Agentic RAG Assistant"
#     debug: bool = False
#     version: str = "1.0.0"
#     log_level: str = "INFO"
    
#     # LLM API Keys
#     openai_api_key: Optional[str] = None
#     google_api_key: Optional[str] = None
#     groq_api_key: Optional[str] = None
    
#     # Redis
#     redis_url: str = "redis://localhost:6379"
    
#     # FAISS Configuration
#     faiss_index_path: str = "./data/faiss_index"
#     embedding_model: str = "text-embedding-3-small"
    
#     # Model Configuration
#     default_llm: str = "gpt-4"
#     available_models: List[str] = [
#         "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo", 
#         "gemini-pro", "groq-mixtral", "llama3.2:latest"
#     ]
    
#     # File Upload
#     max_file_size: int = 10 * 1024 * 1024  # 10MB
#     allowed_file_types: List[str] = [
#         "pdf", "txt", "docx", "jpg", "jpeg", "png"
#     ]
    
#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"
#         case_sensitive = False

# settings = Settings()


from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # ✅ Allow extra fields
    )
    
    # API Configuration
    app_name: str = "Agentic RAG Assistant"
    debug: bool = False
    version: str = "1.0.0"
    log_level: str = "INFO"
    
    # LLM API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    
    # Google Cloud Configuration
    google_application_credentials: Optional[str] = None
    
    # Text-to-Speech Settings
    tts_language_code: str = "en-US"
    tts_voice_name: str = "en-US-Wavenet-D"
    tts_voice_gender: str = "NEUTRAL"
    tts_audio_encoding: str = "MP3"
    tts_speaking_rate: float = 1.0
    tts_pitch: float = 0.0
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # FAISS Configuration
    faiss_index_path: str = "./data/faiss_index"
    embedding_model: str = "text-embedding-3-small"
    
    # Model Configuration
    default_llm: str = "gpt-4"
    available_models: List[str] = [
        "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo", 
        "gemini-pro", "groq-mixtral", "llama3.2:latest"
    ]
    
    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [
        "pdf", "txt", "docx", "jpg", "jpeg", "png", "wav", "mp3", "ogg", "m4a"
    ]

settings = Settings()

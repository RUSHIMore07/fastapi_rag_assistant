# from typing import Dict, Any, Optional
# from app.llm_routing.model_interfaces import (
#     OpenAIProvider, GoogleProvider, GroqProvider, OllamaProvider
# )
# from app.config.settings import settings
# from app.models.schemas import LLMProvider, QueryType
# import logging

# logger = logging.getLogger(__name__)

# class LLMRouter:
#     def __init__(self):
#         self.providers = {}
#         self.initialize_providers()
    
#     def initialize_providers(self):
#         """Initialize all available LLM providers"""
#         if settings.openai_api_key:
#             self.providers[LLMProvider.OPENAI] = OpenAIProvider(settings.openai_api_key)
        
#         if settings.google_api_key:
#             self.providers[LLMProvider.GOOGLE] = GoogleProvider(settings.google_api_key)
        
#         if settings.groq_api_key:
#             self.providers[LLMProvider.GROQ] = GroqProvider(settings.groq_api_key)
        
#         # Ollama is available by default for local deployment
#         self.providers[LLMProvider.OLLAMA] = OllamaProvider()
    
#     def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
#         """Analyze query complexity for routing decisions"""
#         query_length = len(query.split())
        
#         # Simple heuristics for complexity analysis
#         complexity_indicators = {
#             "length": query_length,
#             "has_code": "```",
#             "has_math": any(op in query for op in ["∑", "∫", "√", "∞", "π"]),
#             "has_reasoning": any(word in query.lower() for word in ["analyze", "compare", "explain", "reasoning"]),
#             "has_creativity": any(word in query.lower() for word in ["creative", "story", "poem", "imagine"])
#         }
        
#         # Determine complexity score
#         complexity_score = 0
#         if query_length > 50:
#             complexity_score += 2
#         if complexity_indicators["has_code"]:
#             complexity_score += 3
#         if complexity_indicators["has_math"]:
#             complexity_score += 2
#         if complexity_indicators["has_reasoning"]:
#             complexity_score += 2
#         if complexity_indicators["has_creativity"]:
#             complexity_score += 1
        
#         return {
#             "complexity_score": complexity_score,
#             "indicators": complexity_indicators,
#             "recommended_capability": self.get_recommended_capability(complexity_score)
#         }
    
#     def get_recommended_capability(self, complexity_score: int) -> str:
#         """Get recommended model capability based on complexity"""
#         if complexity_score >= 5:
#             return "advanced"  # GPT-4, Gemini Pro
#         elif complexity_score >= 3:
#             return "intermediate"  # GPT-3.5, Groq
#         else:
#             return "basic"  # Ollama, lightweight models
    
#     def route_query(self, query: str, query_type: QueryType, 
#                    preferred_provider: Optional[LLMProvider] = None) -> LLMProvider:
#         """Route query to appropriate LLM provider"""
        
#         # If user specified a preference, use it
#         if preferred_provider and preferred_provider in self.providers:
#             return preferred_provider
        
#         # Analyze query for automatic routing
#         analysis = self.analyze_query_complexity(query)
#         recommended_capability = analysis["recommended_capability"]
        
#         # Route based on query type and complexity
#         if query_type == QueryType.IMAGE or query_type == QueryType.MULTIMODAL:
#             # For image queries, prefer vision-capable models
#             if LLMProvider.OPENAI in self.providers:
#                 return LLMProvider.OPENAI  # GPT-4V
#             elif LLMProvider.GOOGLE in self.providers:
#                 return LLMProvider.GOOGLE  # Gemini Pro Vision
        
#         # Route based on complexity
#         if recommended_capability == "advanced":
#             if LLMProvider.OPENAI in self.providers:
#                 return LLMProvider.OPENAI
#             elif LLMProvider.GOOGLE in self.providers:
#                 return LLMProvider.GOOGLE
#         elif recommended_capability == "intermediate":
#             if LLMProvider.GROQ in self.providers:
#                 return LLMProvider.GROQ
#             elif LLMProvider.OPENAI in self.providers:
#                 return LLMProvider.OPENAI
#         else:  # basic
#             if LLMProvider.OLLAMA in self.providers:
#                 return LLMProvider.OLLAMA
#             elif LLMProvider.GROQ in self.providers:
#                 return LLMProvider.GROQ
        
#         # Fallback to first available provider
#         return list(self.providers.keys())
    
#     async def generate_response(self, query: str, provider: LLMProvider, **kwargs) -> Dict[str, Any]:
#         """Generate response using specified provider"""
#         if provider not in self.providers:
#             return {
#                 "content": "",
#                 "error": f"Provider {provider} not available",
#                 "success": False
#             }
        
#         return await self.providers[provider].generate_response(query, **kwargs)



from typing import Dict, Any, Optional
from app.llm_routing.model_interfaces import (
    OpenAIProvider, GoogleProvider, GroqProvider, OllamaProvider
)
from app.config.settings import settings
from app.models.schemas import LLMProvider, QueryType
import logging

logger = logging.getLogger(__name__)

class LLMRouter:
    def __init__(self):
        self.providers = {}
        self.initialize_providers()
    
    def initialize_providers(self):
        """Initialize all available LLM providers"""
        if settings.openai_api_key:
            self.providers[LLMProvider.OPENAI] = OpenAIProvider(settings.openai_api_key)
        
        if settings.google_api_key:
            self.providers[LLMProvider.GOOGLE] = GoogleProvider(settings.google_api_key)
        
        if settings.groq_api_key:
            self.providers[LLMProvider.GROQ] = GroqProvider(settings.groq_api_key)
        
        # Ollama is available by default for local deployment
        self.providers[LLMProvider.OLLAMA] = OllamaProvider()
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity for routing decisions"""
        query_length = len(query.split())
        
        # Simple heuristics for complexity analysis
        complexity_indicators = {
            "length": query_length,
            "has_code": "```",
            "has_math": any(op in query for op in ["∑", "∫", "√", "∞", "π"]),
            "has_reasoning": any(word in query.lower() for word in ["analyze", "compare", "explain", "reasoning"]),
            "has_creativity": any(word in query.lower() for word in ["creative", "story", "poem", "imagine"])
        }
        
        # Determine complexity score
        complexity_score = 0
        if query_length > 50:
            complexity_score += 2
        if complexity_indicators["has_code"]:
            complexity_score += 3
        if complexity_indicators["has_math"]:
            complexity_score += 2
        if complexity_indicators["has_reasoning"]:
            complexity_score += 2
        if complexity_indicators["has_creativity"]:
            complexity_score += 1
        
        return {
            "complexity_score": complexity_score,
            "indicators": complexity_indicators,
            "recommended_capability": self.get_recommended_capability(complexity_score)
        }
    
    def get_recommended_capability(self, complexity_score: int) -> str:
        """Get recommended model capability based on complexity"""
        if complexity_score >= 5:
            return "advanced"  # GPT-4, Gemini Pro
        elif complexity_score >= 3:
            return "intermediate"  # GPT-3.5, Groq
        else:
            return "basic"  # Ollama, lightweight models
    
    def route_query(self, query: str, query_type: QueryType, 
                   preferred_provider: Optional[LLMProvider] = None) -> LLMProvider:
        """Route query to appropriate LLM provider"""
        
        # If user specified a preference, use it
        if preferred_provider and preferred_provider in self.providers:
            return preferred_provider
        
        # Analyze query for automatic routing
        analysis = self.analyze_query_complexity(query)
        recommended_capability = analysis["recommended_capability"]
        
        # Route based on query type and complexity
        if query_type == QueryType.IMAGE or query_type == QueryType.MULTIMODAL:
            # For image queries, prefer vision-capable models
            if LLMProvider.OPENAI in self.providers:
                return LLMProvider.OPENAI  # GPT-4V
            elif LLMProvider.GOOGLE in self.providers:
                return LLMProvider.GOOGLE  # Gemini Pro Vision
        
        # Route based on complexity
        if recommended_capability == "advanced":
            if LLMProvider.OPENAI in self.providers:
                return LLMProvider.OPENAI
            elif LLMProvider.GOOGLE in self.providers:
                return LLMProvider.GOOGLE
        elif recommended_capability == "intermediate":
            if LLMProvider.GROQ in self.providers:
                return LLMProvider.GROQ
            elif LLMProvider.OPENAI in self.providers:
                return LLMProvider.OPENAI
        else:  # basic
            if LLMProvider.OLLAMA in self.providers:
                return LLMProvider.OLLAMA
            elif LLMProvider.GROQ in self.providers:
                return LLMProvider.GROQ
        
        # Fallback to first available provider
        available_providers = list(self.providers.keys())
        return available_providers if available_providers else LLMProvider.OPENAI
    
    def _convert_string_to_provider(self, provider_string: str) -> Optional[LLMProvider]:
        """Convert string to LLMProvider enum"""
        # Model mapping from frontend strings to backend enums
        model_mapping = {
            "gpt-4": LLMProvider.OPENAI,
            "gpt-4o-mini": LLMProvider.OPENAI,
            "gpt-3.5-turbo": LLMProvider.OPENAI,
            "gemini-pro": LLMProvider.GOOGLE,
            "groq-mixtral": LLMProvider.GROQ,
            "llama3.2:latest": LLMProvider.OLLAMA,
            "llama3.2": LLMProvider.OLLAMA,
            "openai": LLMProvider.OPENAI,
            "google": LLMProvider.GOOGLE,
            "groq": LLMProvider.GROQ,
            "ollama": LLMProvider.OLLAMA
        }
        
        return model_mapping.get(provider_string.lower())
    
    async def generate_response(self, query: str, provider, **kwargs) -> Dict[str, Any]:
        """Generate response using specified provider"""
        # Convert string to enum if needed
        if isinstance(provider, str):
            provider_enum = self._convert_string_to_provider(provider)
            if not provider_enum:
                return {
                    "content": "",
                    "error": f"Unknown provider: {provider}",
                    "success": False
                }
            provider = provider_enum
        
        # Check if provider is available
        if provider not in self.providers:
            return {
                "content": "",
                "error": f"Provider {provider} not available",
                "success": False
            }
        
        try:
            return await self.providers[provider].generate_response(query, **kwargs)
        except Exception as e:
            logger.error(f"Error generating response with {provider}: {e}")
            return {
                "content": "",
                "error": f"Response generation failed: {str(e)}",
                "success": False
            }

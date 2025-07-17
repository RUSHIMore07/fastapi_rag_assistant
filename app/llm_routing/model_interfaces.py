# from abc import ABC, abstractmethod
# from typing import Dict, Any, List
# import openai
# import google.generativeai as genai
# from groq import Groq
# import logging
# import time

# logger = logging.getLogger(__name__)

# class BaseLLMProvider(ABC):
#     @abstractmethod
#     async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
#         pass

# class OpenAIProvider(BaseLLMProvider):
#     def __init__(self, api_key: str):
#         self.client = openai.OpenAI(api_key=api_key)
    
#     async def generate_response(self, prompt: str, model: str = "gpt-4o-mini", **kwargs) -> Dict[str, Any]:
#         try:
#             start_time = time.time()
#             response = self.client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=kwargs.get("max_tokens", 1000),
#                 temperature=kwargs.get("temperature", 0.7)
#             )
#             end_time = time.time()
            
#             return {
#                 "content": response.choices[0].message.content,
#                 "model": model,
#                 "tokens_used": response.usage.total_tokens,
#                 "response_time": end_time - start_time,
#                 "success": True
#             }
#         except Exception as e:
#             logger.error(f"OpenAI API error: {e}")
#             return {
#                 "content": "",
#                 "error": str(e),
#                 "success": False
#             }

# class GoogleProvider(BaseLLMProvider):
#     def __init__(self, api_key: str):
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel('gemini-pro')
    
#     async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
#         try:
#             start_time = time.time()
#             response = self.model.generate_content(prompt)
#             end_time = time.time()
            
#             return {
#                 "content": response.text,
#                 "model": "gemini-pro",
#                 "tokens_used": 0,  # Gemini doesn't provide token count
#                 "response_time": end_time - start_time,
#                 "success": True
#             }
#         except Exception as e:
#             logger.error(f"Gemini API error: {e}")
#             return {
#                 "content": "",
#                 "error": str(e),
#                 "success": False
#             }

# class GroqProvider(BaseLLMProvider):
#     def __init__(self, api_key: str):
#         self.client = Groq(api_key=api_key)
    
#     async def generate_response(self, prompt: str, model: str = "mixtral-8x7b-32768", **kwargs) -> Dict[str, Any]:
#         try:
#             start_time = time.time()
#             response = self.client.chat.completions.create(
#                 messages=[{"role": "user", "content": prompt}],
#                 model=model,
#                 max_tokens=kwargs.get("max_tokens", 1000),
#                 temperature=kwargs.get("temperature", 0.7)
#             )
#             end_time = time.time()
            
#             return {
#                 "content": response.choices[0].message.content,
#                 "model": model,
#                 "tokens_used": response.usage.total_tokens,
#                 "response_time": end_time - start_time,
#                 "success": True
#             }
#         except Exception as e:
#             logger.error(f"Groq API error: {e}")
#             return {
#                 "content": "",
#                 "error": str(e),
#                 "success": False
#             }

# class OllamaProvider(BaseLLMProvider):
#     def __init__(self, base_url: str = "http://localhost:11434"):
#         self.base_url = base_url
    
#     async def generate_response(self, prompt: str, model: str = "llama3.2:latest", **kwargs) -> Dict[str, Any]:
#         try:
#             import requests
#             start_time = time.time()
            
#             response = requests.post(
#                 f"{self.base_url}/api/generate",
#                 json={
#                     "model": model,
#                     "prompt": prompt,
#                     "stream": False
#                 }
#             )
            
#             end_time = time.time()
            
#             if response.status_code == 200:
#                 result = response.json()
#                 return {
#                     "content": result.get("response", ""),
#                     "model": model,
#                     "tokens_used": 0,  # Ollama doesn't provide token count
#                     "response_time": end_time - start_time,
#                     "success": True
#                 }
#             else:
#                 return {
#                     "content": "",
#                     "error": f"HTTP {response.status_code}",
#                     "success": False
#                 }
#         except Exception as e:
#             logger.error(f"Ollama API error: {e}")
#             return {
#                 "content": "",
#                 "error": str(e),
#                 "success": False
#             }



from abc import ABC, abstractmethod
from typing import Dict, Any, List
import openai
import google.generativeai as genai
from groq import Groq
import logging
import time

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    async def generate_response(self, prompt: str, model: str = "gpt-4", **kwargs) -> Dict[str, Any]:
        try:
            # Handle different model names
            if model in ["gpt-4o-mini", "gpt-4-mini"]:
                model = "gpt-4o-mini"
            elif model in ["gpt-4", "gpt-4o"]:
                model = "gpt-4"
            elif model in ["gpt-3.5-turbo", "gpt-3.5"]:
                model = "gpt-3.5-turbo"
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            end_time = time.time()
            
            return {
                "content": response.choices[0].message.content,
                "model": model,
                "tokens_used": response.usage.total_tokens,
                "response_time": end_time - start_time,
                "success": True
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "content": "",
                "error": str(e),
                "success": False
            }

class GoogleProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            end_time = time.time()
            
            return {
                "content": response.text,
                "model": "gemini-pro",
                "tokens_used": 0,  # Gemini doesn't provide token count
                "response_time": end_time - start_time,
                "success": True
            }
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "content": "",
                "error": str(e),
                "success": False
            }

class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    async def generate_response(self, prompt: str, model: str = "mixtral-8x7b-32768", **kwargs) -> Dict[str, Any]:
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            end_time = time.time()
            
            return {
                "content": response.choices.message.content,
                "model": model,
                "tokens_used": response.usage.total_tokens,
                "response_time": end_time - start_time,
                "success": True
            }
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {
                "content": "",
                "error": str(e),
                "success": False
            }

class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def generate_response(self, prompt: str, model: str = "llama3.2:latest", **kwargs) -> Dict[str, Any]:
        try:
            import requests
            start_time = time.time()
            
            # Handle different model names
            if model in ["llama3.2", "llama3.2:latest"]:
                model = "llama3.2:latest"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "content": result.get("response", ""),
                    "model": model,
                    "tokens_used": 0,  # Ollama doesn't provide token count
                    "response_time": end_time - start_time,
                    "success": True
                }
            else:
                return {
                    "content": "",
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "success": False
                }
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return {
                "content": "",
                "error": str(e),
                "success": False
            }

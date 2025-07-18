# import os
# import asyncio
# import logging
# from typing import Dict, Any, Optional, List
# import requests
# import base64
# import io
# import json
# from app.config.settings import settings

# logger = logging.getLogger(__name__)

# class GoogleVoiceService:
#     def __init__(self):
#         # Initialize API URLs first
#         self.tts_url = "https://texttospeech.googleapis.com/v1/text:synthesize"
#         self.stt_url = "https://speech.googleapis.com/v1/speech:recognize"
#         self.voices_url = "https://texttospeech.googleapis.com/v1/voices"
        
#         # Use API key from settings for REST API
#         self.api_key = getattr(settings, 'google_api_key', None)
        
#         if not self.api_key:
#             logger.error("Google API key not found in settings")
#         else:
#             logger.info("Google Voice Service initialized with API key")
        
#         # Set client attributes to None since we're using REST API
#         self.tts_client = None
#         self.stt_client = None
    
#     async def text_to_speech(self, 
#                            text: str, 
#                            language_code: str = None,
#                            voice_name: str = None,
#                            voice_gender: str = None,
#                            speaking_rate: float = None,
#                            pitch: float = None,
#                            audio_encoding: str = None) -> Dict[str, Any]:
#         """Convert text to speech using Google Cloud TTS REST API"""
        
#         if not self.api_key:
#             return {
#                 "success": False,
#                 "error": "Google API key not configured",
#                 "audio_data": None
#             }
        
#         try:
#             # Use settings defaults if not provided
#             language_code = language_code or getattr(settings, 'tts_language_code', 'en-US')
#             voice_name = voice_name or getattr(settings, 'tts_voice_name', 'en-US-Wavenet-D')
#             voice_gender = voice_gender or getattr(settings, 'tts_voice_gender', 'NEUTRAL')
#             speaking_rate = speaking_rate or getattr(settings, 'tts_speaking_rate', 1.0)
#             pitch = pitch or getattr(settings, 'tts_pitch', 0.0)
#             audio_encoding = audio_encoding or getattr(settings, 'tts_audio_encoding', 'MP3')
            
#             # Clean text for better speech synthesis
#             cleaned_text = self.cleanup_text_for_speech(text)
            
#             # Prepare request payload for REST API
#             payload = {
#                 "input": {"text": cleaned_text},
#                 "voice": {
#                     "languageCode": language_code,
#                     "name": voice_name,
#                     "ssmlGender": voice_gender
#                 },
#                 "audioConfig": {
#                     "audioEncoding": audio_encoding,
#                     "speakingRate": speaking_rate,
#                     "pitch": pitch
#                 }
#             }
            
#             # Make API request using REST API
#             response = requests.post(
#                 f"{self.tts_url}?key={self.api_key}",
#                 json=payload,
#                 headers={"Content-Type": "application/json"},
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 audio_base64 = result["audioContent"]
                
#                 return {
#                     "success": True,
#                     "audio_base64": audio_base64,
#                     "audio_format": audio_encoding.lower(),
#                     "text": cleaned_text,
#                     "original_text": text,
#                     "language_code": language_code,
#                     "voice_name": voice_name,
#                     "voice_gender": voice_gender,
#                     "speaking_rate": speaking_rate,
#                     "pitch": pitch,
#                     "text_length": len(cleaned_text)
#                 }
#             else:
#                 error_msg = f"API request failed: {response.status_code}"
#                 try:
#                     error_detail = response.json()
#                     error_msg += f" - {error_detail}"
#                 except:
#                     error_msg += f" - {response.text}"
                
#                 return {
#                     "success": False,
#                     "error": error_msg,
#                     "audio_data": None
#                 }
                
#         except Exception as e:
#             logger.error(f"Error synthesizing speech: {e}")
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "audio_data": None
#             }
    
#     async def speech_to_text(self, 
#                            audio_data: bytes, 
#                            language_code: str = "en-US",
#                            audio_encoding: str = "MP3",
#                            sample_rate: int = 16000) -> Dict[str, Any]:
#         """Convert speech to text using Google Cloud STT REST API"""
        
#         if not self.api_key:
#             return {
#                 "success": False,
#                 "error": "Google API key not configured",
#                 "text": ""
#             }
        
#         try:
#             # Encode audio to base64
#             audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
#             # Map common audio formats to Google Cloud encoding
#             encoding_map = {
#                 "mp3": "MP3",
#                 "wav": "LINEAR16",
#                 "ogg": "OGG_OPUS",
#                 "flac": "FLAC",
#                 "webm": "WEBM_OPUS"
#             }
            
#             google_encoding = encoding_map.get(audio_encoding.lower(), "MP3")
            
#             # Prepare request payload
#             payload = {
#                 "config": {
#                     "encoding": google_encoding,
#                     "sampleRateHertz": sample_rate,
#                     "languageCode": language_code,
#                     "enableAutomaticPunctuation": True,
#                     "enableWordConfidence": True,
#                     "enableWordTimeOffsets": True
#                 },
#                 "audio": {
#                     "content": audio_base64
#                 }
#             }
            
#             # Make API request
#             response = requests.post(
#                 f"{self.stt_url}?key={self.api_key}",
#                 json=payload,
#                 headers={"Content-Type": "application/json"},
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
                
#                 if "results" in result and len(result["results"]) > 0:
#                     alternative = result["results"][0]["alternatives"][0]
                    
#                     # Extract word-level information if available
#                     words = []
#                     if "words" in alternative:
#                         for word in alternative["words"]:
#                             words.append({
#                                 "word": word["word"],
#                                 "confidence": word.get("confidence", 0.0),
#                                 "start_time": self.parse_duration(word.get("startTime", "0s")),
#                                 "end_time": self.parse_duration(word.get("endTime", "0s"))
#                             })
                    
#                     return {
#                         "success": True,
#                         "text": alternative["transcript"],
#                         "confidence": alternative.get("confidence", 0.0),
#                         "words": words,
#                         "language_code": language_code,
#                         "audio_size": len(audio_data)
#                     }
#                 else:
#                     return {
#                         "success": True,
#                         "text": "",
#                         "confidence": 0.0,
#                         "words": [],
#                         "language_code": language_code,
#                         "audio_size": len(audio_data)
#                     }
#             else:
#                 error_msg = f"API request failed: {response.status_code}"
#                 try:
#                     error_detail = response.json()
#                     error_msg += f" - {error_detail}"
#                 except:
#                     error_msg += f" - {response.text}"
                
#                 return {
#                     "success": False,
#                     "error": error_msg,
#                     "text": ""
#                 }
                
#         except Exception as e:
#             logger.error(f"Error recognizing speech: {e}")
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "text": ""
#             }
    
#     def parse_duration(self, duration_str: str) -> float:
#         """Parse duration string (e.g., '1.5s') to float"""
#         try:
#             return float(duration_str.rstrip('s'))
#         except:
#             return 0.0
    
#     def cleanup_text_for_speech(self, text: str) -> str:
#         """Clean up text for better speech synthesis"""
#         import re
        
#         # Remove markdown formatting
#         text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
#         text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
#         text = re.sub(r'`(.*?)`', r'\1', text)        # Code
#         text = re.sub(r'#{1,6}\s+', '', text)         # Headers
#         text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
#         # Remove citations and references
#         text = re.sub(r'\[\d+\]', '', text)
#         text = re.sub(r'Source \d+:', '', text)
        
#         # Replace common abbreviations for better pronunciation
#         text = re.sub(r'\bAPI\b', 'A P I', text)
#         text = re.sub(r'\bRAG\b', 'R A G', text)
#         text = re.sub(r'\bLLM\b', 'L L M', text)
#         text = re.sub(r'\bAI\b', 'A I', text)
#         text = re.sub(r'\bML\b', 'M L', text)
#         text = re.sub(r'\bUI\b', 'U I', text)
#         text = re.sub(r'\bURL\b', 'U R L', text)
#         text = re.sub(r'\bHTTP\b', 'H T T P', text)
#         text = re.sub(r'\bJSON\b', 'J S O N', text)
        
#         # Clean up extra whitespace
#         text = re.sub(r'\s+', ' ', text).strip()
        
#         # Limit length for TTS (Google has a 5000 character limit)
#         if len(text) > 4900:
#             text = text[:4900] + "..."
        
#         return text
    
#     def get_available_voices(self, language_code: str = None) -> List[Dict[str, Any]]:
#         """Get list of available voices using REST API"""
#         if not self.api_key:
#             return []
        
#         try:
#             url = f"{self.voices_url}?key={self.api_key}"
#             if language_code:
#                 url += f"&languageCode={language_code}"
            
#             response = requests.get(url, timeout=10)
            
#             if response.status_code == 200:
#                 result = response.json()
#                 voices = result.get("voices", [])
                
#                 # Format voices for easier use
#                 formatted_voices = []
#                 for voice in voices:
#                     formatted_voices.append({
#                         "name": voice.get("name", ""),
#                         "language_codes": voice.get("languageCodes", []),
#                         "ssml_gender": voice.get("ssmlGender", "NEUTRAL"),
#                         "natural_sample_rate_hertz": voice.get("naturalSampleRateHertz", 22050)
#                     })
                
#                 return formatted_voices
#             else:
#                 logger.error(f"Error getting voices: {response.status_code} - {response.text}")
#                 return []
                
#         except Exception as e:
#             logger.error(f"Error getting available voices: {e}")
#             return []
    
#     def get_supported_languages(self) -> List[str]:
#         """Get list of supported languages"""
#         return [
#             "en-US", "en-GB", "en-AU", "en-CA", "en-IN",
#             "es-ES", "es-US", "es-MX", "fr-FR", "fr-CA",
#             "de-DE", "it-IT", "pt-BR", "pt-PT", "ru-RU",
#             "ja-JP", "ko-KR", "zh-CN", "zh-TW", "zh-HK",
#             "hi-IN", "ar-SA", "nl-NL", "pl-PL", "tr-TR",
#             "sv-SE", "da-DK", "no-NO", "fi-FI", "cs-CZ"
#         ]
    
#     def get_voice_by_language(self, language_code: str) -> str:
#         """Get a recommended voice for a given language"""
#         voice_recommendations = {
#             "en-US": "en-US-Wavenet-D",
#             "en-GB": "en-GB-Wavenet-A",
#             "es-ES": "es-ES-Wavenet-A",
#             "fr-FR": "fr-FR-Wavenet-A",
#             "de-DE": "de-DE-Wavenet-A",
#             "it-IT": "it-IT-Wavenet-A",
#             "pt-BR": "pt-BR-Wavenet-A",
#             "ja-JP": "ja-JP-Wavenet-A",
#             "ko-KR": "ko-KR-Wavenet-A",
#             "zh-CN": "zh-CN-Wavenet-A",
#             "hi-IN": "hi-IN-Wavenet-A",
#             "ar-SA": "ar-XA-Wavenet-A",
#             "ru-RU": "ru-RU-Wavenet-A"
#         }
        
#         return voice_recommendations.get(language_code, "en-US-Wavenet-D")


import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
import base64
import io
import json
from app.config.settings import settings

# Import Google Cloud client libraries
from google.cloud import texttospeech
from google.cloud import speech

logger = logging.getLogger(__name__)

class GoogleVoiceService:
    def __init__(self):
        # Set up service account authentication
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        
        # Initialize Google Cloud clients
        try:
            self.tts_client = texttospeech.TextToSpeechClient()
            self.stt_client = speech.SpeechClient()
            logger.info("Google Cloud Voice services initialized with service account")
            self.use_client_library = True
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Voice services: {e}")
            self.tts_client = None
            self.stt_client = None
            self.use_client_library = False


    async def text_to_speech(self, 
                        text: str, 
                        language_code: str = None,
                        voice_name: str = None,
                        voice_gender: str = None,
                        speaking_rate: float = None,
                        pitch: float = None,
                        audio_encoding: str = None) -> Dict[str, Any]:
        """Convert text to speech using Google Cloud TTS Client Library"""
        
        if not self.tts_client:
            return {
                "success": False,
                "error": "Text-to-Speech client not initialized",
                "audio_data": None
            }
        
        try:
            # Use settings defaults if not provided
            language_code = language_code or getattr(settings, 'tts_language_code', 'en-US')
            voice_gender = voice_gender or getattr(settings, 'tts_voice_gender', 'NEUTRAL')
            speaking_rate = speaking_rate or getattr(settings, 'tts_speaking_rate', 1.0)
            pitch = pitch or getattr(settings, 'tts_pitch', 0.0)
            audio_encoding = audio_encoding or getattr(settings, 'tts_audio_encoding', 'MP3')
            
            # 🔧 FIX: Get appropriate voice name based on gender and language
            if not voice_name:
                voice_name = self.get_voice_by_language_and_gender(language_code, voice_gender)
            
            # Clean text for better speech synthesis
            cleaned_text = self.cleanup_text_for_speech(text)
            
            # Prepare synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
            
            # Configure voice parameters
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,  # Use specific voice name
                ssml_gender=getattr(texttospeech.SsmlVoiceGender, voice_gender)
            )
            
            # Configure audio output
            audio_config = texttospeech.AudioConfig(
                audio_encoding=getattr(texttospeech.AudioEncoding, audio_encoding),
                speaking_rate=speaking_rate,
                pitch=pitch
            )
            
            # Perform synthesis
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Convert audio content to base64 for frontend
            audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "audio_format": audio_encoding.lower(),
                "text": cleaned_text,
                "original_text": text,
                "language_code": language_code,
                "voice_name": voice_name,  # Return the actual voice used
                "voice_gender": voice_gender,
                "speaking_rate": speaking_rate,
                "pitch": pitch,
                "text_length": len(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None
            }

    def get_voice_by_language_and_gender(self, language_code: str, voice_gender: str) -> str:
        """Get appropriate voice name based on language and gender"""
        
        # Define gender-specific voices for each language
        voice_mapping = {
            "en-US": {
                "FEMALE": "en-US-Wavenet-C",  # Female voice
                "MALE": "en-US-Wavenet-B",    # Male voice
                "NEUTRAL": "en-US-Wavenet-D"  # Neutral voice
            },
            "en-GB": {
                "FEMALE": "en-GB-Wavenet-A",  # Female voice
                "MALE": "en-GB-Wavenet-B",    # Male voice
                "NEUTRAL": "en-GB-Wavenet-C"  # Neutral voice
            },
            "es-ES": {
                "FEMALE": "es-ES-Wavenet-C",  # Female voice
                "MALE": "es-ES-Wavenet-B",    # Male voice
                "NEUTRAL": "es-ES-Wavenet-A"  # Neutral voice
            },
            "fr-FR": {
                "FEMALE": "fr-FR-Wavenet-A",  # Female voice
                "MALE": "fr-FR-Wavenet-B",    # Male voice
                "NEUTRAL": "fr-FR-Wavenet-C"  # Neutral voice
            },
            "de-DE": {
                "FEMALE": "de-DE-Wavenet-A",  # Female voice
                "MALE": "de-DE-Wavenet-B",    # Male voice
                "NEUTRAL": "de-DE-Wavenet-C"  # Neutral voice
            },
            "it-IT": {
                "FEMALE": "it-IT-Wavenet-A",  # Female voice
                "MALE": "it-IT-Wavenet-C",    # Male voice
                "NEUTRAL": "it-IT-Wavenet-B"  # Neutral voice
            },
            "pt-BR": {
                "FEMALE": "pt-BR-Wavenet-A",  # Female voice
                "MALE": "pt-BR-Wavenet-B",    # Male voice
                "NEUTRAL": "pt-BR-Wavenet-C"  # Neutral voice
            },
            "ja-JP": {
                "FEMALE": "ja-JP-Wavenet-A",  # Female voice
                "MALE": "ja-JP-Wavenet-C",    # Male voice
                "NEUTRAL": "ja-JP-Wavenet-B"  # Neutral voice
            }
        }
        
        # Get voice for the specified language and gender
        if language_code in voice_mapping and voice_gender in voice_mapping[language_code]:
            return voice_mapping[language_code][voice_gender]
        
        # Fallback to default voice if not found
        return "en-US-Wavenet-C"  # Default to female voice



    
    async def text_to_speech_old(self, 
                           text: str, 
                           language_code: str = None,
                           voice_name: str = None,
                           voice_gender: str = None,
                           speaking_rate: float = None,
                           pitch: float = None,
                           audio_encoding: str = None) -> Dict[str, Any]:
        """Convert text to speech using Google Cloud TTS Client Library"""
        
        if not self.tts_client:
            return {
                "success": False,
                "error": "Text-to-Speech client not initialized",
                "audio_data": None
            }
        
        try:
            # Use settings defaults if not provided
            language_code = language_code or getattr(settings, 'tts_language_code', 'en-US')
            voice_name = voice_name or getattr(settings, 'tts_voice_name', 'en-US-Wavenet-D')
            voice_gender = voice_gender or getattr(settings, 'tts_voice_gender', 'NEUTRAL')
            speaking_rate = speaking_rate or getattr(settings, 'tts_speaking_rate', 1.0)
            pitch = pitch or getattr(settings, 'tts_pitch', 0.0)
            audio_encoding = audio_encoding or getattr(settings, 'tts_audio_encoding', 'MP3')
            
            # Clean text for better speech synthesis
            cleaned_text = self.cleanup_text_for_speech(text)
            
            # Prepare synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
            
            # Configure voice parameters
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
                ssml_gender=getattr(texttospeech.SsmlVoiceGender, voice_gender)
            )
            
            # Configure audio output
            audio_config = texttospeech.AudioConfig(
                audio_encoding=getattr(texttospeech.AudioEncoding, audio_encoding),
                speaking_rate=speaking_rate,
                pitch=pitch
            )
            
            # Perform synthesis
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Convert audio content to base64 for frontend
            audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
            
            return {
                "success": True,
                "audio_base64": audio_base64,  # Only base64, no binary data
                "audio_format": audio_encoding.lower(),
                "text": cleaned_text,
                "original_text": text,
                "language_code": language_code,
                "voice_name": voice_name,
                "voice_gender": voice_gender,
                "speaking_rate": speaking_rate,
                "pitch": pitch,
                "text_length": len(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None
            }
    
    async def speech_to_text(self, 
                           audio_data: bytes, 
                           language_code: str = "en-US",
                           audio_encoding: str = "MP3",
                           sample_rate: int = 16000) -> Dict[str, Any]:
        """Convert speech to text using Google Cloud STT Client Library"""
        
        if not self.stt_client:
            return {
                "success": False,
                "error": "Speech-to-Text client not initialized",
                "text": ""
            }
        
        try:
            # Map common audio formats to Google Cloud encoding
            encoding_map = {
                "mp3": speech.RecognitionConfig.AudioEncoding.MP3,
                "wav": speech.RecognitionConfig.AudioEncoding.LINEAR16,
                "ogg": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
                "flac": speech.RecognitionConfig.AudioEncoding.FLAC,
                "webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
            }
            
            google_encoding = encoding_map.get(audio_encoding.lower(), speech.RecognitionConfig.AudioEncoding.MP3)
            
            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=google_encoding,
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                enable_word_time_offsets=True
            )
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_data)
            
            # Perform recognition
            response = self.stt_client.recognize(config=config, audio=audio)
            
            # Process results
            if response.results:
                result = response.results[0]
                alternative = result.alternatives[0]
                
                # Extract word-level information
                words = []
                for word in alternative.words:
                    words.append({
                        "word": word.word,
                        "confidence": word.confidence,
                        "start_time": word.start_time.total_seconds(),
                        "end_time": word.end_time.total_seconds()
                    })
                
                return {
                    "success": True,
                    "text": alternative.transcript,
                    "confidence": alternative.confidence,
                    "words": words,
                    "language_code": language_code,
                    "audio_size": len(audio_data)
                }
            else:
                return {
                    "success": True,
                    "text": "",
                    "confidence": 0.0,
                    "words": [],
                    "language_code": language_code,
                    "audio_size": len(audio_data)
                }
                
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def cleanup_text_for_speech(self, text: str) -> str:
        """Clean up text for better speech synthesis"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'#{1,6}\s+', '', text)         # Headers
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Remove citations and references
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'Source \d+:', '', text)
        
        # Replace common abbreviations for better pronunciation
        text = re.sub(r'\bAPI\b', 'A P I', text)
        text = re.sub(r'\bRAG\b', 'R A G', text)
        text = re.sub(r'\bLLM\b', 'L L M', text)
        text = re.sub(r'\bAI\b', 'A I', text)
        text = re.sub(r'\bML\b', 'M L', text)
        text = re.sub(r'\bUI\b', 'U I', text)
        text = re.sub(r'\bURL\b', 'U R L', text)
        text = re.sub(r'\bHTTP\b', 'H T T P', text)
        text = re.sub(r'\bJSON\b', 'J S O N', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length for TTS (Google has a 5000 character limit)
        if len(text) > 4900:
            text = text[:4900] + "..."
        
        return text
    
    def get_available_voices(self, language_code: str = None) -> List[Dict[str, Any]]:
        """Get list of available voices using client library"""
        if not self.tts_client:
            return []
        
        try:
            voices = self.tts_client.list_voices(language_code=language_code)
            
            # Format voices for easier use
            formatted_voices = []
            for voice in voices.voices:
                formatted_voices.append({
                    "name": voice.name,
                    "language_codes": list(voice.language_codes),
                    "ssml_gender": voice.ssml_gender.name,
                    "natural_sample_rate_hertz": voice.natural_sample_rate_hertz
                })
            
            return formatted_voices
            
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [
            "en-US", "en-GB", "en-AU", "en-CA", "en-IN",
            "es-ES", "es-US", "es-MX", "fr-FR", "fr-CA",
            "de-DE", "it-IT", "pt-BR", "pt-PT", "ru-RU",
            "ja-JP", "ko-KR", "zh-CN", "zh-TW", "zh-HK",
            "hi-IN", "ar-SA", "nl-NL", "pl-PL", "tr-TR",
            "sv-SE", "da-DK", "no-NO", "fi-FI", "cs-CZ"
        ]
    
    def get_voice_by_language(self, language_code: str) -> str:
        """Get a recommended voice for a given language"""
        voice_recommendations = {
            "en-US": "en-US-Wavenet-D",
            "en-GB": "en-GB-Wavenet-A",
            "es-ES": "es-ES-Wavenet-A",
            "fr-FR": "fr-FR-Wavenet-A",
            "de-DE": "de-DE-Wavenet-A",
            "it-IT": "it-IT-Wavenet-A",
            "pt-BR": "pt-BR-Wavenet-A",
            "ja-JP": "ja-JP-Wavenet-A",
            "ko-KR": "ko-KR-Wavenet-A",
            "zh-CN": "zh-CN-Wavenet-A",
            "hi-IN": "hi-IN-Wavenet-A",
            "ar-SA": "ar-XA-Wavenet-A",
            "ru-RU": "ru-RU-Wavenet-A"
        }
        
        return voice_recommendations.get(language_code, "en-US-Wavenet-D")

# # #working
# # import streamlit as st
# # import json
# # import base64
# # import requests
# # from typing import Dict, Any
# # from frontend.utils.api_client import APIClient

# # def render_enhanced_voice_chat(api_client: APIClient, config: Dict[str, Any]):
# #     """Enhanced chat interface that returns both text and voice responses"""
# #     st.header("🎤 Enhanced Voice RAG Chat")
    
# #     # Voice settings
# #     with st.expander("🔊 Voice Settings"):
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             auto_voice = st.checkbox("Auto-generate Voice Response", value=True, key="enhanced_auto_voice")
# #             auto_play = st.checkbox("Auto-play Voice Response", value=True, key="enhanced_auto_play")
            
# #             language_code = st.selectbox(
# #                 "Language",
# #                 ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP"],
# #                 index=0,
# #                 key="enhanced_language_select"
# #             )
            
# #             # 🔧 Enhanced voice selection with specific names
# #             voice_options = get_voice_options_for_language(language_code)
# #             selected_voice = st.selectbox(
# #                 "Voice",
# #                 options=list(voice_options.keys()),
# #                 key="enhanced_voice_select"
# #             )
            
# #             voice_name = voice_options[selected_voice]["name"]
# #             voice_gender = voice_options[selected_voice]["gender"]
        
# #         with col2:
# #             speaking_rate = st.slider("Speaking Rate", 0.25, 4.0, 1.0, 0.25, key="enhanced_speaking_rate")
# #             voice_pitch = st.slider("Voice Pitch", -20.0, 20.0, 0.0, 1.0, key="enhanced_voice_pitch")
# #             audio_format = st.selectbox("Audio Format", ["MP3", "WAV", "OGG_OPUS"], index=0, key="enhanced_audio_format")
            
# #             # Show selected voice info
# #             st.info(f"Selected: {selected_voice}")
# #     # Voice Test Section (collapsible)
# #     with st.expander("🧪 Voice Test"):
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             test_text = st.text_input("Test Text", value="Hello, this is a voice test.", key="voice_test_text")
        
# #         with col2:
# #             if st.button("🔊 Test Voice", key="test_voice_btn"):
# #                 with st.spinner("Testing voice generation..."):
# #                     voice_request = {
# #                         "text": test_text,
# #                         "language_code": language_code,
# #                         "voice_gender": voice_gender,
# #                         "speaking_rate": speaking_rate,
# #                         "pitch": voice_pitch,
# #                         "audio_encoding": audio_format
# #                     }
                    
# #                     tts_response = api_client.post("/api/v1/text-to-speech", data=voice_request)
                    
# #                     if tts_response and tts_response.get('success'):
# #                         if tts_response.get('audio_base64'):
# #                             audio_bytes = base64.b64decode(tts_response['audio_base64'])
# #                             st.audio(audio_bytes, format=f"audio/{audio_format.lower()}", autoplay=auto_play)
# #                             st.success("✅ Voice test successful!")
# #                         else:
# #                             st.error("❌ No audio data received")
# #                     else:
# #                         st.error(f"❌ Voice test failed: {tts_response.get('error', 'Unknown error')}")
    
# #     # Initialize chat history
# #     if "enhanced_voice_messages" not in st.session_state:
# #         st.session_state.enhanced_voice_messages = []
    
# #     # Chat controls
# #     col1, col2, col3 = st.columns([1, 2, 1])
    
# #     with col1:
# #         if st.button("🗑️ Clear Chat", key="enhanced_voice_clear_chat"):
# #             st.session_state.enhanced_voice_messages = []
# #             st.rerun()
    
# #     with col2:
# #         voice_status = "🔊 Voice ON" if auto_voice else "🔇 Voice OFF"
# #         st.info(f"Status: {voice_status} | Language: {language_code} | Gender: {voice_gender}")
    
# #     with col3:
# #         if auto_voice:
# #             st.success("🎤 Ready")
# #         else:
# #             st.warning("🔇 Silent")
    
# #     # Display chat messages
# #     for message in st.session_state.enhanced_voice_messages:
# #         with st.chat_message(message["role"]):
# #             st.markdown(message["content"])
            
# #             # Display audio player if available (for assistant messages)
# #             if message["role"] == "assistant" and message.get("audio_data"):
# #                 audio_bytes = base64.b64decode(message["audio_data"])
# #                 st.audio(
# #                     audio_bytes, 
# #                     format=f"audio/{message.get('audio_format', 'mp3').lower()}",
# #                     autoplay=auto_play
# #                 )
    
# #     # Chat input
# #     if prompt := st.chat_input("Ask me anything about your documents...", key="enhanced_voice_chat_input"):
# #         # Add user message
# #         st.session_state.enhanced_voice_messages.append({
# #             "role": "user", 
# #             "content": prompt
# #         })
        
# #         # Display user message
# #         with st.chat_message("user"):
# #             st.markdown(prompt)
        
# #         # Generate response with voice
# #         with st.chat_message("assistant"):
# #             with st.spinner("🧠 Processing query and generating voice response..."):
                
# #                 response = generate_text_and_voice_response(
# #                     prompt, 
# #                     api_client, 
# #                     auto_voice,
# #                     {
# #                         "language_code": language_code,
# #                         "voice_gender": voice_gender,
# #                         "speaking_rate": speaking_rate,
# #                         "pitch": voice_pitch,
# #                         "audio_format": audio_format
# #                     },
# #                     config,
# #                     auto_play
# #                 )
                
# #                 if response:
# #                     # Display text response
# #                     st.markdown(response["text"])
                    
# #                     # Display audio if available
# #                     if response.get("audio_data"):
# #                         audio_bytes = base64.b64decode(response["audio_data"])
# #                         st.audio(
# #                             audio_bytes, 
# #                             format=f"audio/{response.get('audio_format', 'mp3').lower()}",
# #                             autoplay=auto_play
# #                         )
                        
# #                         # Success indicator
# #                         col1, col2 = st.columns([3, 1])
# #                         with col1:
# #                             st.success("🔊 Voice response generated successfully!")
# #                         with col2:
# #                             # Manual voice toggle for this message
# #                             if st.button("🔄 Replay", key=f"replay_{len(st.session_state.enhanced_voice_messages)}"):
# #                                 st.audio(audio_bytes, format=f"audio/{response.get('audio_format', 'mp3').lower()}")
                    
# #                     elif auto_voice:
# #                         st.warning("⚠️ Voice generation was enabled but failed")
                    
# #                     # Add to chat history
# #                     st.session_state.enhanced_voice_messages.append(response)
# #                 else:
# #                     st.error("❌ Failed to generate response")

# # def generate_text_and_voice_response(query: str, api_client: APIClient, 
# #                                    generate_voice: bool, voice_settings: Dict[str, Any], 
# #                                    config: Dict[str, Any], auto_play: bool = True) -> Dict[str, Any]:
# #     """Generate both text and voice response"""
    
# #     # Prepare RAG request with voice generation
# #     rag_request = {
# #         "query": query,
# #         "use_refinement": config.get("enable_context", True),
# #         "search_type": config.get("default_search_type", "hybrid"),
# #         "preferred_llm": config.get("selected_model", "gpt-4"),
# #         "generate_voice": generate_voice,
# #         "voice_language": voice_settings["language_code"],
# #         "voice_gender": voice_settings["voice_gender"],
# #         "speaking_rate": voice_settings["speaking_rate"],
# #         "pitch": voice_settings["pitch"],
# #         "audio_encoding": voice_settings["audio_format"],
# #         "max_tokens": config.get("max_tokens", 1000),
# #         "temperature": config.get("temperature", 0.7)
# #     }
    
# #     # Call the RAG with voice API
# #     response = api_client.post("/api/v1/rag-with-voice", data=rag_request)
    
# #     if response:
# #         # Extract text response
# #         text_response = response.get("response", "Sorry, I couldn't process your request.")
        
# #         # Extract voice response if available
# #         voice_response = response.get("voice_response", {})
# #         voice_success = voice_response.get("success", False)
# #         audio_data = voice_response.get("audio_base64") if voice_success else None
        
# #         return {
# #             "role": "assistant",
# #             "content": text_response,
# #             "text": text_response,
# #             "audio_data": audio_data,
# #             "audio_format": voice_settings["audio_format"].lower(),
# #             "voice_success": voice_success,
# #             "metadata": {
# #                 "model_used": response.get("model_used"),
# #                 "voice_enabled": generate_voice,
# #                 "context_sources": len(response.get("context_used", {}).get("sources", []))
# #             }
# #         }
# #     else:
# #         return None



# # def get_voice_options_for_language(language_code: str) -> dict:
# #     """Get available voice options for a specific language"""
    
# #     voice_options = {
# #         "en-US": {
# #             "Emma (Female)": {"name": "en-US-Wavenet-C", "gender": "FEMALE"},
# #             "John (Male)": {"name": "en-US-Wavenet-B", "gender": "MALE"},
# #             "Alex (Neutral)": {"name": "en-US-Wavenet-D", "gender": "NEUTRAL"},
# #             "Joanna (Female)": {"name": "en-US-Wavenet-F", "gender": "FEMALE"},
# #             "Michael (Male)": {"name": "en-US-Wavenet-A", "gender": "MALE"}
# #         },
# #         "en-GB": {
# #             "Sophie (Female)": {"name": "en-GB-Wavenet-A", "gender": "FEMALE"},
# #             "Oliver (Male)": {"name": "en-GB-Wavenet-B", "gender": "MALE"},
# #             "Emma (Female)": {"name": "en-GB-Wavenet-C", "gender": "FEMALE"},
# #             "Thomas (Male)": {"name": "en-GB-Wavenet-D", "gender": "MALE"}
# #         },
# #         "es-ES": {
# #             "Carmen (Female)": {"name": "es-ES-Wavenet-C", "gender": "FEMALE"},
# #             "Pablo (Male)": {"name": "es-ES-Wavenet-B", "gender": "MALE"},
# #             "Lucia (Female)": {"name": "es-ES-Wavenet-A", "gender": "FEMALE"}
# #         },
# #         "fr-FR": {
# #             "Marie (Female)": {"name": "fr-FR-Wavenet-A", "gender": "FEMALE"},
# #             "Pierre (Male)": {"name": "fr-FR-Wavenet-B", "gender": "MALE"},
# #             "Celine (Female)": {"name": "fr-FR-Wavenet-C", "gender": "FEMALE"}
# #         },
# #         "de-DE": {
# #             "Anna (Female)": {"name": "de-DE-Wavenet-A", "gender": "FEMALE"},
# #             "Hans (Male)": {"name": "de-DE-Wavenet-B", "gender": "MALE"},
# #             "Petra (Female)": {"name": "de-DE-Wavenet-C", "gender": "FEMALE"}
# #         },
# #         "it-IT": {
# #             "Giulia (Female)": {"name": "it-IT-Wavenet-A", "gender": "FEMALE"},
# #             "Marco (Male)": {"name": "it-IT-Wavenet-C", "gender": "MALE"},
# #             "Chiara (Female)": {"name": "it-IT-Wavenet-B", "gender": "FEMALE"}
# #         },
# #         "pt-BR": {
# #             "Camila (Female)": {"name": "pt-BR-Wavenet-A", "gender": "FEMALE"},
# #             "Ricardo (Male)": {"name": "pt-BR-Wavenet-B", "gender": "MALE"}
# #         },
# #         "ja-JP": {
# #             "Akiko (Female)": {"name": "ja-JP-Wavenet-A", "gender": "FEMALE"},
# #             "Takeshi (Male)": {"name": "ja-JP-Wavenet-C", "gender": "MALE"}
# #         }
# #     }
    
# #     return voice_options.get(language_code, {
# #         "Default Female": {"name": "en-US-Wavenet-C", "gender": "FEMALE"}
# #     })

# import streamlit as st
# import json
# import base64
# import requests
# from typing import Dict, Any
# from frontend.utils.api_client import APIClient

# def render_enhanced_voice_chat(api_client: APIClient, config: Dict[str, Any]):
#     """Enhanced chat interface that returns both text and voice responses"""
#     st.header("🎤 Enhanced Voice RAG Chat")
    
#     # Voice settings
#     with st.expander("🔊 Voice Settings"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             auto_voice = st.checkbox("Auto-generate Voice Response", value=True, key="enhanced_auto_voice")
#             auto_play = st.checkbox("Auto-play Voice Response", value=True, key="enhanced_auto_play")
            
#             language_code = st.selectbox(
#                 "Language",
#                 ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP"],
#                 index=0,
#                 key="enhanced_language_select"
#             )
            
#             # 🔧 Enhanced voice selection with specific names
#             voice_options = get_voice_options_for_language(language_code)
#             selected_voice = st.selectbox(
#                 "Voice",
#                 options=list(voice_options.keys()),
#                 key="enhanced_voice_select"
#             )
            
#             voice_name = voice_options[selected_voice]["name"]
#             voice_gender = voice_options[selected_voice]["gender"]
        
#         with col2:
#             speaking_rate = st.slider("Speaking Rate", 0.25, 4.0, 1.0, 0.25, key="enhanced_speaking_rate")
#             voice_pitch = st.slider("Voice Pitch", -20.0, 20.0, 0.0, 1.0, key="enhanced_voice_pitch")
#             audio_format = st.selectbox("Audio Format", ["MP3", "WAV", "OGG_OPUS"], index=0, key="enhanced_audio_format")
            
#             # Show selected voice info
#             st.info(f"Selected: {selected_voice}")
    
#     # Voice Test Section (collapsible)
#     with st.expander("🧪 Voice Test"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             test_text = st.text_input("Test Text", value="Hello, this is a voice test.", key="voice_test_text")
        
#         with col2:
#             if st.button("🔊 Test Voice", key="test_voice_btn"):
#                 with st.spinner("Testing voice generation..."):
#                     # 🔧 FIX: Use voice_name instead of voice_gender for proper voice selection
#                     voice_request = {
#                         "text": test_text,
#                         "language_code": language_code,
#                         "voice_name": voice_name,      # ✅ Use specific voice name
#                         "voice_gender": voice_gender,  # Keep for compatibility
#                         "speaking_rate": speaking_rate,
#                         "pitch": voice_pitch,
#                         "audio_encoding": audio_format
#                     }
                    
#                     st.write(f"**Testing with voice:** {selected_voice}")
#                     st.write(f"**Voice name:** {voice_name}")
                    
#                     tts_response = api_client.post("/api/v1/text-to-speech", data=voice_request)
                    
#                     if tts_response and tts_response.get('success'):
#                         if tts_response.get('audio_base64'):
#                             audio_bytes = base64.b64decode(tts_response['audio_base64'])
#                             st.audio(audio_bytes, format=f"audio/{audio_format.lower()}", autoplay=auto_play)
#                             st.success("✅ Voice test successful!")
#                         else:
#                             st.error("❌ No audio data received")
#                     else:
#                         st.error(f"❌ Voice test failed: {tts_response.get('error', 'Unknown error')}")
    
#     # Initialize chat history
#     if "enhanced_voice_messages" not in st.session_state:
#         st.session_state.enhanced_voice_messages = []
    
#     # Chat controls
#     col1, col2, col3 = st.columns([1, 2, 1])
    
#     with col1:
#         if st.button("🗑️ Clear Chat", key="enhanced_voice_clear_chat"):
#             st.session_state.enhanced_voice_messages = []
#             st.rerun()
    
#     with col2:
#         voice_status = "🔊 Voice ON" if auto_voice else "🔇 Voice OFF"
#         st.info(f"Status: {voice_status} | Voice: {selected_voice}")
    
#     with col3:
#         if auto_voice:
#             st.success("🎤 Ready")
#         else:
#             st.warning("🔇 Silent")
    
#     # Display chat messages
#     for message in st.session_state.enhanced_voice_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
            
#             # Display audio player if available (for assistant messages)
#             if message["role"] == "assistant" and message.get("audio_data"):
#                 audio_bytes = base64.b64decode(message["audio_data"])
#                 st.audio(
#                     audio_bytes, 
#                     format=f"audio/{message.get('audio_format', 'mp3').lower()}",
#                     autoplay=auto_play
#                 )
    
#     # Chat input
#     if prompt := st.chat_input("Ask me anything about your documents...", key="enhanced_voice_chat_input"):
#         # Add user message
#         st.session_state.enhanced_voice_messages.append({
#             "role": "user", 
#             "content": prompt
#         })
        
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate response with voice
#         with st.chat_message("assistant"):
#             with st.spinner("🧠 Processing query and generating voice response..."):
                
#                 response = generate_text_and_voice_response(
#                     prompt, 
#                     api_client, 
#                     auto_voice,
#                     {
#                         "language_code": language_code,
#                         "voice_name": voice_name,        # ✅ Pass voice_name
#                         "voice_gender": voice_gender,    # Keep for compatibility
#                         "speaking_rate": speaking_rate,
#                         "pitch": voice_pitch,
#                         "audio_format": audio_format
#                     },
#                     config,
#                     auto_play
#                 )
                
#                 if response:
#                     # Display text response
#                     st.markdown(response["text"])
                    
#                     # Display audio if available
#                     if response.get("audio_data"):
#                         audio_bytes = base64.b64decode(response["audio_data"])
#                         st.audio(
#                             audio_bytes, 
#                             format=f"audio/{response.get('audio_format', 'mp3').lower()}",
#                             autoplay=auto_play
#                         )
                        
#                         # Success indicator
#                         col1, col2 = st.columns([3, 1])
#                         with col1:
#                             st.success("🔊 Voice response generated successfully!")
#                         with col2:
#                             # Manual voice toggle for this message
#                             if st.button("🔄 Replay", key=f"replay_{len(st.session_state.enhanced_voice_messages)}"):
#                                 st.audio(audio_bytes, format=f"audio/{response.get('audio_format', 'mp3').lower()}")
                    
#                     elif auto_voice:
#                         st.warning("⚠️ Voice generation was enabled but failed")
                    
#                     # Add to chat history
#                     st.session_state.enhanced_voice_messages.append(response)
#                 else:
#                     st.error("❌ Failed to generate response")

# def generate_text_and_voice_response(query: str, api_client: APIClient, 
#                                    generate_voice: bool, voice_settings: Dict[str, Any], 
#                                    config: Dict[str, Any], auto_play: bool = True) -> Dict[str, Any]:
#     """Generate both text and voice response"""
    
#     # Prepare RAG request with voice generation
#     rag_request = {
#         "query": query,
#         "use_refinement": config.get("enable_context", True),
#         "search_type": config.get("default_search_type", "hybrid"),
#         "preferred_llm": config.get("selected_model", "gpt-4"),
#         "generate_voice": generate_voice,
#         "voice_language": voice_settings["language_code"],
#         "voice_name": voice_settings["voice_name"],        # ✅ Pass voice_name
#         "voice_gender": voice_settings["voice_gender"],    # Keep for compatibility
#         "speaking_rate": voice_settings["speaking_rate"],
#         "pitch": voice_settings["pitch"],
#         "audio_encoding": voice_settings["audio_format"],
#         "max_tokens": config.get("max_tokens", 1000),
#         "temperature": config.get("temperature", 0.7)
#     }
    
#     # Call the RAG with voice API
#     response = api_client.post("/api/v1/rag-with-voice", data=rag_request)
    
#     if response:
#         # Extract text response
#         text_response = response.get("response", "Sorry, I couldn't process your request.")
        
#         # Extract voice response if available
#         voice_response = response.get("voice_response", {})
#         voice_success = voice_response.get("success", False)
#         audio_data = voice_response.get("audio_base64") if voice_success else None
        
#         return {
#             "role": "assistant",
#             "content": text_response,
#             "text": text_response,
#             "audio_data": audio_data,
#             "audio_format": voice_settings["audio_format"].lower(),
#             "voice_success": voice_success,
#             "metadata": {
#                 "model_used": response.get("model_used"),
#                 "voice_enabled": generate_voice,
#                 "voice_used": voice_settings["voice_name"],
#                 "context_sources": len(response.get("context_used", {}).get("sources", []))
#             }
#         }
#     else:
#         return None

# def get_voice_options_for_language(language_code: str) -> dict:
#     """Get available voice options for a specific language"""
    
#     voice_options = {
#         "en-US": {
#             "Emma (Female)": {"name": "en-US-Wavenet-C", "gender": "FEMALE"},
#             "John (Male)": {"name": "en-US-Wavenet-B", "gender": "MALE"},
#             "Alex (Neutral)": {"name": "en-US-Wavenet-D", "gender": "NEUTRAL"},
#             "Joanna (Female)": {"name": "en-US-Wavenet-F", "gender": "FEMALE"},
#             "Michael (Male)": {"name": "en-US-Wavenet-A", "gender": "MALE"},
#             "Jenny (Female)": {"name": "en-US-Wavenet-G", "gender": "FEMALE"},
#             "Matthew (Male)": {"name": "en-US-Wavenet-I", "gender": "MALE"}
#         },
#         "en-GB": {
#             "Sophie (Female)": {"name": "en-GB-Wavenet-A", "gender": "FEMALE"},
#             "Oliver (Male)": {"name": "en-GB-Wavenet-B", "gender": "MALE"},
#             "Emma (Female)": {"name": "en-GB-Wavenet-C", "gender": "FEMALE"},
#             "Thomas (Male)": {"name": "en-GB-Wavenet-D", "gender": "MALE"},
#             "Alice (Female)": {"name": "en-GB-Wavenet-F", "gender": "FEMALE"}
#         },
#         "es-ES": {
#             "Carmen (Female)": {"name": "es-ES-Wavenet-C", "gender": "FEMALE"},
#             "Pablo (Male)": {"name": "es-ES-Wavenet-B", "gender": "MALE"},
#             "Lucia (Female)": {"name": "es-ES-Wavenet-A", "gender": "FEMALE"},
#             "Diego (Male)": {"name": "es-ES-Wavenet-D", "gender": "MALE"}
#         },
#         "fr-FR": {
#             "Marie (Female)": {"name": "fr-FR-Wavenet-A", "gender": "FEMALE"},
#             "Pierre (Male)": {"name": "fr-FR-Wavenet-B", "gender": "MALE"},
#             "Celine (Female)": {"name": "fr-FR-Wavenet-C", "gender": "FEMALE"},
#             "Antoine (Male)": {"name": "fr-FR-Wavenet-D", "gender": "MALE"},
#             "Eloise (Female)": {"name": "fr-FR-Wavenet-E", "gender": "FEMALE"}
#         },
#         "de-DE": {
#             "Anna (Female)": {"name": "de-DE-Wavenet-A", "gender": "FEMALE"},
#             "Hans (Male)": {"name": "de-DE-Wavenet-B", "gender": "MALE"},
#             "Petra (Female)": {"name": "de-DE-Wavenet-C", "gender": "FEMALE"},
#             "Klaus (Male)": {"name": "de-DE-Wavenet-D", "gender": "MALE"},
#             "Marlene (Female)": {"name": "de-DE-Wavenet-F", "gender": "FEMALE"}
#         },
#         "it-IT": {
#             "Giulia (Female)": {"name": "it-IT-Wavenet-A", "gender": "FEMALE"},
#             "Chiara (Female)": {"name": "it-IT-Wavenet-B", "gender": "FEMALE"},
#             "Marco (Male)": {"name": "it-IT-Wavenet-C", "gender": "MALE"},
#             "Lorenzo (Male)": {"name": "it-IT-Wavenet-D", "gender": "MALE"}
#         },
#         "pt-BR": {
#             "Camila (Female)": {"name": "pt-BR-Wavenet-A", "gender": "FEMALE"},
#             "Ricardo (Male)": {"name": "pt-BR-Wavenet-B", "gender": "MALE"},
#             "Vitoria (Female)": {"name": "pt-BR-Wavenet-C", "gender": "FEMALE"}
#         },
#         "ja-JP": {
#             "Akiko (Female)": {"name": "ja-JP-Wavenet-A", "gender": "FEMALE"},
#             "Yuki (Female)": {"name": "ja-JP-Wavenet-B", "gender": "FEMALE"},
#             "Takeshi (Male)": {"name": "ja-JP-Wavenet-C", "gender": "MALE"},
#             "Hiroshi (Male)": {"name": "ja-JP-Wavenet-D", "gender": "MALE"}
#         }
#     }
    
#     return voice_options.get(language_code, {
#         "Default Female": {"name": "en-US-Wavenet-C", "gender": "FEMALE"}
#     })
import streamlit as st
import json
import base64
import requests
from typing import Dict, Any
from frontend.utils.api_client import APIClient

def render_enhanced_voice_chat(api_client: APIClient, config: Dict[str, Any]):
    """Enhanced chat interface that returns both text and voice responses"""
    st.header("🎤 Enhanced Voice RAG Chat")
    
    # Voice settings
    with st.expander("🔊 Voice Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            auto_voice = st.checkbox("Auto-generate Voice Response", value=True, key="enhanced_auto_voice")
            auto_play = st.checkbox("Auto-play Voice Response", value=True, key="enhanced_auto_play")
            
            language_code = st.selectbox(
                "Language",
                ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP"],
                index=0,
                key="enhanced_language_select"
            )
            
            # 🔧 Enhanced voice selection with specific names
            voice_options = get_voice_options_for_language(language_code)
            selected_voice = st.selectbox(
                "Voice",
                options=list(voice_options.keys()),
                key="enhanced_voice_select"
            )
            
            voice_name = voice_options[selected_voice]["name"]
            voice_gender = voice_options[selected_voice]["gender"]
        
        with col2:
            speaking_rate = st.slider("Speaking Rate", 0.25, 4.0, 1.0, 0.25, key="enhanced_speaking_rate")
            voice_pitch = st.slider("Voice Pitch", -20.0, 20.0, 0.0, 1.0, key="enhanced_voice_pitch")
            audio_format = st.selectbox("Audio Format", ["MP3", "WAV", "OGG_OPUS"], index=0, key="enhanced_audio_format")
            
            # Show selected voice info
            st.info(f"Selected: {selected_voice}")
    
    # Voice Test Section (collapsible)
    with st.expander("🧪 Voice Test"):
        col1, col2 = st.columns(2)
        
        with col1:
            test_text = st.text_input("Test Text", value="Hello, this is a voice test.", key="voice_test_text")
        
        with col2:
            if st.button("🔊 Test Voice", key="test_voice_btn"):
                with st.spinner("Testing voice generation..."):
                    # 🔧 FIX: Use voice_name instead of voice_gender for proper voice selection
                    voice_request = {
                        "text": test_text,
                        "language_code": language_code,
                        "voice_name": voice_name,      # ✅ Use specific voice name
                        "voice_gender": voice_gender,  # Keep for compatibility
                        "speaking_rate": speaking_rate,
                        "pitch": voice_pitch,
                        "audio_encoding": audio_format
                    }
                    
                    st.write(f"**Testing with voice:** {selected_voice}")
                    st.write(f"**Voice name:** {voice_name}")
                    
                    tts_response = api_client.post("/api/v1/text-to-speech", data=voice_request)
                    
                    if tts_response and tts_response.get('success'):
                        if tts_response.get('audio_base64'):
                            audio_bytes = base64.b64decode(tts_response['audio_base64'])
                            st.audio(audio_bytes, format=f"audio/{audio_format.lower()}", autoplay=auto_play)
                            st.success("✅ Voice test successful!")
                        else:
                            st.error("❌ No audio data received")
                    else:
                        st.error(f"❌ Voice test failed: {tts_response.get('error', 'Unknown error')}")
    
    # Initialize chat history
    if "enhanced_voice_messages" not in st.session_state:
        st.session_state.enhanced_voice_messages = []
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🗑️ Clear Chat", key="enhanced_voice_clear_chat"):
            st.session_state.enhanced_voice_messages = []
            st.rerun()
    
    with col2:
        voice_status = "🔊 Voice ON" if auto_voice else "🔇 Voice OFF"
        st.info(f"Status: {voice_status} | Voice: {selected_voice}")
    
    with col3:
        if auto_voice:
            st.success("🎤 Ready")
        else:
            st.warning("🔇 Silent")
    
    # Display chat messages
    for message in st.session_state.enhanced_voice_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display audio player if available (for assistant messages)
            if message["role"] == "assistant" and message.get("audio_data"):
                audio_bytes = base64.b64decode(message["audio_data"])
                st.audio(
                    audio_bytes, 
                    format=f"audio/{message.get('audio_format', 'mp3').lower()}",
                    autoplay=auto_play
                )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents...", key="enhanced_voice_chat_input"):
        # Add user message
        st.session_state.enhanced_voice_messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with voice
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing query and generating voice response..."):
                
                response = generate_text_and_voice_response(
                    prompt, 
                    api_client, 
                    auto_voice,
                    {
                        "language_code": language_code,
                        "voice_name": voice_name,        # ✅ Pass voice_name
                        "voice_gender": voice_gender,    # Keep for compatibility
                        "speaking_rate": speaking_rate,
                        "pitch": voice_pitch,
                        "audio_format": audio_format
                    },
                    config,
                    auto_play
                )
                
                if response:
                    # Display text response
                    st.markdown(response["text"])
                    
                    # Display audio if available
                    if response.get("audio_data"):
                        audio_bytes = base64.b64decode(response["audio_data"])
                        st.audio(
                            audio_bytes, 
                            format=f"audio/{response.get('audio_format', 'mp3').lower()}",
                            autoplay=auto_play
                        )
                        
                        # Success indicator
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.success("🔊 Voice response generated successfully!")
                        with col2:
                            # Manual voice toggle for this message
                            if st.button("🔄 Replay", key=f"replay_{len(st.session_state.enhanced_voice_messages)}"):
                                st.audio(audio_bytes, format=f"audio/{response.get('audio_format', 'mp3').lower()}")
                    
                    elif auto_voice:
                        st.warning("⚠️ Voice generation was enabled but failed")
                    
                    # Add to chat history
                    st.session_state.enhanced_voice_messages.append(response)
                else:
                    st.error("❌ Failed to generate response")

def generate_text_and_voice_response(query: str, api_client: APIClient, 
                                   generate_voice: bool, voice_settings: Dict[str, Any], 
                                   config: Dict[str, Any], auto_play: bool = True) -> Dict[str, Any]:
    """Generate both text and voice response"""
    
    # Prepare RAG request with voice generation
    rag_request = {
        "query": query,
        "use_refinement": config.get("enable_context", True),
        "search_type": config.get("default_search_type", "hybrid"),
        "preferred_llm": config.get("selected_model", "gpt-4"),
        "generate_voice": generate_voice,
        "voice_language": voice_settings["language_code"],
        "voice_name": voice_settings["voice_name"],        # ✅ Pass voice_name
        "voice_gender": voice_settings["voice_gender"],    # Keep for compatibility
        "speaking_rate": voice_settings["speaking_rate"],
        "pitch": voice_settings["pitch"],
        "audio_encoding": voice_settings["audio_format"],
        "max_tokens": config.get("max_tokens", 1000),
        "temperature": config.get("temperature", 0.7)
    }
    
    # Call the RAG with voice API
    response = api_client.post("/api/v1/rag-with-voice", data=rag_request)
    
    if response:
        # Extract text response
        text_response = response.get("response", "Sorry, I couldn't process your request.")
        
        # Extract voice response if available
        voice_response = response.get("voice_response", {})
        voice_success = voice_response.get("success", False)
        audio_data = voice_response.get("audio_base64") if voice_success else None
        
        return {
            "role": "assistant",
            "content": text_response,
            "text": text_response,
            "audio_data": audio_data,
            "audio_format": voice_settings["audio_format"].lower(),
            "voice_success": voice_success,
            "metadata": {
                "model_used": response.get("model_used"),
                "voice_enabled": generate_voice,
                "voice_used": voice_settings["voice_name"],
                "context_sources": len(response.get("context_used", {}).get("sources", []))
            }
        }
    else:
        return None

def get_voice_options_for_language(language_code: str) -> dict:
    """Get available voice options for a specific language"""
    
    voice_options = {
        "en-US": {
            "Emma (Female)": {"name": "en-US-Wavenet-C", "gender": "FEMALE"},
            "John (Male)": {"name": "en-US-Wavenet-B", "gender": "MALE"},
            "Alex (Neutral)": {"name": "en-US-Wavenet-D", "gender": "NEUTRAL"},
            "Joanna (Female)": {"name": "en-US-Wavenet-F", "gender": "FEMALE"},
            "Michael (Male)": {"name": "en-US-Wavenet-A", "gender": "MALE"},
            "Jenny (Female)": {"name": "en-US-Wavenet-G", "gender": "FEMALE"},
            "Matthew (Male)": {"name": "en-US-Wavenet-I", "gender": "MALE"}
        },
        "en-GB": {
            "Sophie (Female)": {"name": "en-GB-Wavenet-A", "gender": "FEMALE"},
            "Oliver (Male)": {"name": "en-GB-Wavenet-B", "gender": "MALE"},
            "Emma (Female)": {"name": "en-GB-Wavenet-C", "gender": "FEMALE"},
            "Thomas (Male)": {"name": "en-GB-Wavenet-D", "gender": "MALE"},
            "Alice (Female)": {"name": "en-GB-Wavenet-F", "gender": "FEMALE"}
        },
        "es-ES": {
            "Carmen (Female)": {"name": "es-ES-Wavenet-C", "gender": "FEMALE"},
            "Pablo (Male)": {"name": "es-ES-Wavenet-B", "gender": "MALE"},
            "Lucia (Female)": {"name": "es-ES-Wavenet-A", "gender": "FEMALE"},
            "Diego (Male)": {"name": "es-ES-Wavenet-D", "gender": "MALE"}
        },
        "fr-FR": {
            "Marie (Female)": {"name": "fr-FR-Wavenet-A", "gender": "FEMALE"},
            "Pierre (Male)": {"name": "fr-FR-Wavenet-B", "gender": "MALE"},
            "Celine (Female)": {"name": "fr-FR-Wavenet-C", "gender": "FEMALE"},
            "Antoine (Male)": {"name": "fr-FR-Wavenet-D", "gender": "MALE"},
            "Eloise (Female)": {"name": "fr-FR-Wavenet-E", "gender": "FEMALE"}
        },
        "de-DE": {
            "Anna (Female)": {"name": "de-DE-Wavenet-A", "gender": "FEMALE"},
            "Hans (Male)": {"name": "de-DE-Wavenet-B", "gender": "MALE"},
            "Petra (Female)": {"name": "de-DE-Wavenet-C", "gender": "FEMALE"},
            "Klaus (Male)": {"name": "de-DE-Wavenet-D", "gender": "MALE"},
            "Marlene (Female)": {"name": "de-DE-Wavenet-F", "gender": "FEMALE"}
        },
        "it-IT": {
            "Giulia (Female)": {"name": "it-IT-Wavenet-A", "gender": "FEMALE"},
            "Chiara (Female)": {"name": "it-IT-Wavenet-B", "gender": "FEMALE"},
            "Marco (Male)": {"name": "it-IT-Wavenet-C", "gender": "MALE"},
            "Lorenzo (Male)": {"name": "it-IT-Wavenet-D", "gender": "MALE"}
        },
        "pt-BR": {
            "Camila (Female)": {"name": "pt-BR-Wavenet-A", "gender": "FEMALE"},
            "Ricardo (Male)": {"name": "pt-BR-Wavenet-B", "gender": "MALE"},
            "Vitoria (Female)": {"name": "pt-BR-Wavenet-C", "gender": "FEMALE"}
        },
        "ja-JP": {
            "Akiko (Female)": {"name": "ja-JP-Wavenet-A", "gender": "FEMALE"},
            "Yuki (Female)": {"name": "ja-JP-Wavenet-B", "gender": "FEMALE"},
            "Takeshi (Male)": {"name": "ja-JP-Wavenet-C", "gender": "MALE"},
            "Hiroshi (Male)": {"name": "ja-JP-Wavenet-D", "gender": "MALE"}
        }
    }
    
    return voice_options.get(language_code, {
        "Default Female": {"name": "en-US-Wavenet-C", "gender": "FEMALE"}
    })

import streamlit as st
import json
import base64
import requests
from typing import Dict, Any
from frontend.utils.api_client import APIClient




def render_voice_chat(api_client: APIClient, config: Dict[str, Any]):
    """Voice-enabled chat interface"""
    st.header("🎤 Voice-Enabled RAG Chat")
    
    # Voice settings with unique keys
    with st.expander("🔊 Voice Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            voice_enabled = st.checkbox("Enable Voice Responses", value=False, key="voice_enabled_check")
            language_code = st.selectbox(
                "Language",
                ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP"],
                index=0,
                key="voice_language_select"
            )
        
        with col2:
            voice_gender = st.selectbox(
                "Voice Gender",
                ["NEUTRAL", "MALE", "FEMALE"],
                index=0,
                key="voice_gender_select"
            )
            speaking_rate = st.slider("Speaking Rate", 0.25, 4.0, 1.0, 0.25, key="voice_speaking_rate")
        
        voice_pitch = st.slider("Voice Pitch", -20.0, 20.0, 0.0, 1.0, key="voice_pitch_slider")
        audio_format = st.selectbox("Audio Format", ["MP3", "WAV", "OGG_OPUS"], index=0, key="voice_audio_format")
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Initialize chat history
    if "voice_messages" not in st.session_state:
        st.session_state.voice_messages = []
    
    # Display chat messages
    for message in st.session_state.voice_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display audio player if available
            if message.get("audio_data") and message["role"] == "assistant":
                audio_bytes = base64.b64decode(message["audio_data"])
                st.audio(audio_bytes, format=f"audio/{audio_format.lower()}")
    
    # File upload for speech input
    st.subheader("🎙️ Speech Input")
    uploaded_audio = st.file_uploader(
        "Upload audio file",
        type=['wav', 'mp3', 'ogg', 'm4a', 'flac'],
        help="Record audio on your device and upload it here",
        key="voice_audio_uploader"
    )
    
    if uploaded_audio is not None:
        st.audio(uploaded_audio)
        
        if st.button("🎯 Process Speech", key="voice_process_speech"):
            # Your speech processing logic here...
            pass
    
    # Text input with UNIQUE KEY
    if prompt := st.chat_input("Type your message or upload audio...", key="voice_chat_input"):
        # Add user message
        st.session_state.voice_messages.append({"role": "user", "content": prompt})
        
        # Process query
        process_voice_query(prompt, api_client, voice_enabled, {
            "language_code": language_code,
            "voice_gender": voice_gender,
            "speaking_rate": speaking_rate,
            "pitch": voice_pitch,
            "audio_encoding": audio_format
        })



def process_voice_query(query: str, api_client: APIClient, voice_enabled: bool, voice_settings: Dict[str, Any]):
    """Process query through RAG and generate voice response if enabled"""
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            # RAG request
            rag_request = {
                "query": query,
                "use_refinement": True,
                "search_type": "hybrid",
                "generate_voice": voice_enabled,
                "voice_language": voice_settings["language_code"],
                "voice_gender": voice_settings["voice_gender"],
                "speaking_rate": voice_settings["speaking_rate"],
                "pitch": voice_settings["pitch"],
                "audio_encoding": voice_settings["audio_encoding"]
            }
            
            # Process through RAG with voice
            response = api_client.post("/api/v1/rag-with-voice", data=rag_request)
            
            if response:
                assistant_response = response.get("response", "Sorry, I couldn't process your request.")
                st.markdown(assistant_response)
                
                # Prepare message data
                message_data = {
                    "role": "assistant",
                    "content": assistant_response
                }
                
                # Add voice response if available
                if voice_enabled and response.get("voice_response") and response["voice_response"]["success"]:
                    voice_data = response["voice_response"]
                    audio_bytes = base64.b64decode(voice_data["audio_base64"])
                    st.audio(audio_bytes, format=f"audio/{voice_settings['audio_encoding'].lower()}")
                    
                    # Add audio data to message
                    message_data["audio_data"] = voice_data["audio_base64"]
                
                st.session_state.voice_messages.append(message_data)
                
                # Show processing info
                if response.get("metadata"):
                    with st.expander("ℹ️ Processing Details"):
                        st.json(response["metadata"])
            else:
                st.error("❌ Failed to process query")

import streamlit as st
import requests
import base64
import io
import wave
import numpy as np
from typing import Dict, Any
from frontend.utils.api_client import APIClient
import tempfile
import os

class VoiceInterface:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.sample_rate = 16000
        self.channels = 1
    
    def render_voice_controls(self):
        """Render voice interface controls"""
        st.markdown("### 🎤 Voice Interface")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Audio recorder using streamlit-webrtc (open source)
            if st.button("🎤 Start Recording"):
                st.session_state.recording = True
                self._start_recording()
        
        with col2:
            if st.button("⏹️ Stop Recording"):
                if st.session_state.get("recording", False):
                    st.session_state.recording = False
                    self._stop_recording()
        
        with col3:
            if st.button("🔊 Play Response"):
                if st.session_state.get("last_audio_response"):
                    st.audio(st.session_state.last_audio_response, format="audio/wav")
        
        # Display recording status
        if st.session_state.get("recording", False):
            st.info("🎤 Recording... Click 'Stop Recording' when done.")
        
        # File upload as alternative
        st.markdown("**Or upload audio file:**")
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload an audio file to convert to text"
        )
        
        if uploaded_audio is not None:
            return self._process_uploaded_audio(uploaded_audio)
        
        return None
    
    def _start_recording(self):
        """Start audio recording"""
        # JavaScript code for web audio recording
        recording_js = """
        <script>
        let mediaRecorder;
        let audioChunks = [];
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    // Store audio data
                    window.recordedAudio = audioBlob;
                    
                    // Clear chunks for next recording
                    audioChunks = [];
                };
                
                mediaRecorder.start();
                console.log('Recording started');
            } catch (err) {
                console.error('Error accessing microphone:', err);
            }
        }
        
        startRecording();
        </script>
        """
        
        st.components.v1.html(recording_js, height=0)
    
    def _stop_recording(self):
        """Stop audio recording"""
        stop_js = """
        <script>
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            console.log('Recording stopped');
        }
        </script>
        """
        
        st.components.v1.html(stop_js, height=0)
    
    def _process_uploaded_audio(self, audio_file) -> Dict[str, Any]:
        """Process uploaded audio file"""
        try:
            # Display audio player
            st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")
            
            if st.button("🎤 Process Audio"):
                with st.spinner("Converting speech to text..."):
                    # Upload audio to API
                    files = {"audio_file": audio_file}
                    response = requests.post(
                        f"{self.api_client.base_url}/api/v1/voice/speech-to-text",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display transcription
                        st.success("🎯 Speech Recognition Complete!")
                        st.write(f"**Transcribed Text:** {result['text']}")
                        st.write(f"**Language:** {result['language']}")
                        st.write(f"**Confidence:** {result['confidence']:.2f}")
                        
                        # Store in session state
                        st.session_state.voice_query = result['text']
                        
                        return result
                    else:
                        st.error(f"Speech recognition failed: {response.json().get('detail', 'Unknown error')}")
                        return None
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None
    
    def process_voice_query(self, query_text: str) -> Dict[str, Any]:
        """Process voice query through RAG pipeline"""
        try:
            # Prepare RAG request
            rag_request = {
                "query": query_text,
                "use_refinement": True,
                "search_type": "hybrid",
                "preferred_llm": "gpt-4"
            }
            
            # Call RAG API
            response = self.api_client.post("/api/v1/complete-rag", data=rag_request)
            
            if response:
                # Generate speech for response
                tts_request = {
                    "text": response["response"],
                    "voice_speed": 1.0
                }
                
                tts_response = requests.post(
                    f"{self.api_client.base_url}/api/v1/voice/text-to-speech",
                    json=tts_request
                )
                
                if tts_response.status_code == 200:
                    # Store audio response
                    st.session_state.last_audio_response = tts_response.content
                    
                    return {
                        "query": query_text,
                        "response": response["response"],
                        "has_audio": True,
                        "model_used": response.get("model_used", "unknown")
                    }
                else:
                    return {
                        "query": query_text,
                        "response": response["response"],
                        "has_audio": False,
                        "model_used": response.get("model_used", "unknown")
                    }
            
            return None
            
        except Exception as e:
            st.error(f"Error processing voice query: {str(e)}")
            return None

def render_voice_interface(api_client: APIClient):
    """Main voice interface component"""
    st.header("🎤 Voice Interface")
    
    # Initialize voice interface
    voice_interface = VoiceInterface(api_client)
    
    # Render voice controls
    audio_result = voice_interface.render_voice_controls()
    
    # Process voice query if we have one
    if st.session_state.get("voice_query"):
        query_text = st.session_state.voice_query
        
        st.markdown("### 🔄 Processing Voice Query")
        st.write(f"**Query:** {query_text}")
        
        if st.button("🚀 Process with RAG"):
            with st.spinner("Processing query through RAG pipeline..."):
                result = voice_interface.process_voice_query(query_text)
                
                if result:
                    st.success("✅ Query processed successfully!")
                    
                    # Display response
                    st.markdown("### 📝 Response")
                    st.write(result["response"])
                    
                    # Audio response
                    if result["has_audio"]:
                        st.markdown("### 🔊 Audio Response")
                        st.audio(st.session_state.last_audio_response, format="audio/wav")
                    
                    # Metadata
                    st.markdown("### ℹ️ Metadata")
                    st.write(f"**Model Used:** {result['model_used']}")
                    
                    # Clear processed query
                    del st.session_state.voice_query
    
    # Voice settings
    with st.expander("🔧 Voice Settings"):
        voice_speed = st.slider("Voice Speed", 0.5, 2.0, 1.0, 0.1)
        auto_play = st.checkbox("Auto-play responses", value=True)
        st.session_state.voice_settings = {
            "speed": voice_speed,
            "auto_play": auto_play
        }
    
    # Voice tips
    with st.expander("💡 Voice Tips"):
        st.markdown("""
        **Recording Tips:**
        - Speak clearly and at a moderate pace
        - Ensure quiet environment for best results
        - Keep queries concise and specific
        - Wait for the recording to fully process
        
        **Supported Audio Formats:**
        - WAV (recommended)
        - MP3
        - OGG
        - FLAC
        
        **Features:**
        - Real-time speech recognition
        - Text-to-speech responses
        - Multi-language support
        - Noise reduction
        """)

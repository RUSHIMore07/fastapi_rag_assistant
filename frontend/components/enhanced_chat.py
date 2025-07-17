import streamlit as st
import json
import time
import requests
from typing import Dict, Any
from frontend.utils.api_client import APIClient

def render_enhanced_chat(api_client: APIClient, config: Dict[str, Any]):
    """Enhanced chat interface with complete RAG pipeline"""
    st.header("💬 RAG Chat Interface")
    
    # Chat configuration
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        use_refinement = st.checkbox("🔧 Enable Query Refinement", value=False)
    
    with col2:
        if use_refinement:
            refinement_type = st.selectbox(
                "Refinement Type",
                ["auto", "rewrite", "decompose", "clarify", "expand"],
                help="Auto will let the AI choose the best refinement strategy"
            )
        else:
            refinement_type = None
    
    with col3:
        search_type = st.selectbox(
            "Search Strategy",
            ["hybrid", "vector", "keyword"],
            index=0,
            help="Hybrid combines vector similarity and keyword matching"
        )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_results = st.slider("Number of Results", 5, 20, 10)
            show_sources = st.checkbox("Show Sources", value=True)
        
        with col2:
            show_refinement = st.checkbox("Show Refinement Details", value=False)
            show_processing = st.checkbox("Show Processing Details", value=False)
    
    # Initialize chat history
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # Chat controls
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.rag_messages = []
            st.rerun()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show refinement details if available
                if message.get("refinement_info") and show_refinement:
                    with st.expander("🔧 Query Refinement Details"):
                        refinement = message["refinement_info"]
                        st.write(f"**Original Query:** {refinement.get('original_query', 'N/A')}")
                        st.write(f"**Refined Query:** {refinement.get('refined_query', 'N/A')}")
                        if refinement.get('sub_queries'):
                            st.write("**Sub-queries:**")
                            for i, sub_query in enumerate(refinement['sub_queries'], 1):
                                st.write(f"{i}. {sub_query}")
                        st.write(f"**Refinement Type:** {refinement.get('refinement_type', 'N/A')}")
                        st.write(f"**Confidence:** {refinement.get('confidence_score', 0):.2f}")
                
                # Show sources if available
                if message.get("sources") and show_sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:** {source['filename']}")
                            st.text(source["content"][:200] + "...")
                            st.write(f"*Relevance Score: {source['similarity_score']:.3f}*")
                
                # Show processing details if available
                if message.get("processing_info") and show_processing:
                    with st.expander("⚙️ Processing Details"):
                        st.json(message["processing_info"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response using complete RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Processing your query through the RAG pipeline..."):
                
                # Prepare request data
                request_data = {
                    "query": prompt,
                    "use_refinement": use_refinement,
                    "refinement_type": refinement_type if refinement_type != "auto" else None,
                    "search_type": search_type,
                    "preferred_llm": config["selected_model"].lower(),
                    "session_id": st.session_state.session_id,
                    "k": num_results
                }
                
                # Show request data in debug mode
                if config.get("show_debug"):
                    st.write("Debug - Request Data:")
                    st.json(request_data)
                
                # Call complete RAG pipeline
                response = api_client.post("/api/v1/complete-rag", data=request_data)
            
            if response:
                assistant_response = response.get("response", "Sorry, I couldn't process your request.")
                st.markdown(assistant_response)
                
                # Prepare sources data
                sources_data = []
                if response.get("context_used"):
                    context = response["context_used"]
                    chunks = context.get("chunks", [])
                    sources = context.get("sources", [])
                    scores = context.get("relevance_scores", [])
                    
                    for chunk, source, score in zip(chunks, sources, scores):
                        sources_data.append({
                            "content": chunk,
                            "filename": source,
                            "similarity_score": score
                        })
                
                # Prepare message data
                message_data = {
                    "role": "assistant",
                    "content": assistant_response,
                    "refinement_info": response.get("metadata", {}).get("refinement_info"),
                    "sources": sources_data,
                    "processing_info": {
                        "model_used": response.get("model_used"),
                        "search_type": search_type,
                        "use_refinement": use_refinement,
                        "total_chunks": len(sources_data),
                        "processing_metadata": response.get("metadata", {})
                    }
                }
                
                st.session_state.rag_messages.append(message_data)
                
                # Show quick stats
                if sources_data:
                    st.success(f"✅ Answer generated from {len(sources_data)} document sources")
                
            else:
                st.error("❌ Failed to get response from the RAG pipeline")
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": "Sorry, I encountered an error while processing your query. Please try again.",
                    "sources": [],
                    "processing_info": {}
                })




# Add these imports at the top
from frontend.components.voice_interface import VoiceInterface

def render_enhanced_chat_with_voice(api_client: APIClient, config: Dict[str, Any]):
    """Enhanced chat interface with voice capabilities"""
    st.header("💬 Voice-Enabled RAG Chat")
    
    # Initialize voice interface
    voice_interface = VoiceInterface(api_client)
    
    # Voice controls section
    with st.expander("🎤 Voice Controls", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Voice Query"):
                st.session_state.voice_mode = True
        
        with col2:
            enable_voice_responses = st.checkbox("🔊 Enable Voice Responses", value=True)
    
    # Voice input section
    if st.session_state.get("voice_mode", False):
        st.markdown("### 🎤 Voice Input")
        
        # Audio recorder
        audio_file = st.file_uploader("Record or upload audio", type=['wav', 'mp3', 'ogg'])
        
        if audio_file:
            # Process audio
            with st.spinner("Processing voice input..."):
                files = {"audio_file": audio_file}
                response = requests.post(
                    f"{api_client.base_url}/api/v1/voice/speech-to-text",
                    files=files
                )
                
                if response.status_code == 200:
                    result = response.json()
                    voice_query = result["text"]
                    
                    st.success(f"🎯 Recognized: {voice_query}")
                    
                    # Process through RAG
                    if st.button("🚀 Process Voice Query"):
                        st.session_state.voice_mode = False
                        # Add to regular chat flow
                        st.session_state.pending_voice_query = voice_query
                        st.rerun()
    
    # Regular chat interface
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # Display chat messages
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Voice playback for assistant messages
            if message["role"] == "assistant" and enable_voice_responses:
                if message.get("audio_data"):
                    st.audio(message["audio_data"], format="audio/wav")
    
    # Handle pending voice query
    if st.session_state.get("pending_voice_query"):
        prompt = st.session_state.pending_voice_query
        del st.session_state.pending_voice_query
        
        # Process voice query
        process_chat_message(prompt, api_client, config, enable_voice_responses)
    
    # Chat input
    if prompt := st.chat_input("Type your message or use voice input..."):
        process_chat_message(prompt, api_client, config, enable_voice_responses)

def process_chat_message(prompt: str, api_client: APIClient, config: Dict[str, Any], enable_voice_responses: bool):
    """Process chat message with optional voice response"""
    
    # Add user message
    st.session_state.rag_messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # RAG pipeline request
            request_data = {
                "query": prompt,
                "use_refinement": True,
                "search_type": "hybrid",
                "preferred_llm": config["selected_model"].lower()
            }
            
            response = api_client.post("/api/v1/complete-rag", data=request_data)
            
            if response:
                assistant_response = response.get("response", "Sorry, I couldn't process your request.")
                st.markdown(assistant_response)
                
                # Generate voice response if enabled
                audio_data = None
                if enable_voice_responses:
                    try:
                        tts_request = {"text": assistant_response, "voice_speed": 1.0}
                        tts_response = requests.post(
                            f"{api_client.base_url}/api/v1/voice/text-to-speech",
                            json=tts_request
                        )
                        
                        if tts_response.status_code == 200:
                            audio_data = tts_response.content
                            st.audio(audio_data, format="audio/wav")
                    except Exception as e:
                        st.warning(f"Voice generation failed: {e}")
                
                # Add to messages
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "audio_data": audio_data
                })

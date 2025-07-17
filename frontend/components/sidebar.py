import streamlit as st
from typing import Dict, Any
from frontend.utils.api_client import APIClient

def render_sidebar(api_client: APIClient) -> Dict[str, Any]:
    """Enhanced sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # API Status
    try:
        health_response = api_client.get("/health")
        if health_response and health_response.get("status") == "healthy":
            st.sidebar.success("🟢 API Online")
        else:
            st.sidebar.error("🔴 API Offline")
    except Exception as e:
        st.sidebar.error("🔴 API Connection Error")
    
    # Model Selection
    st.sidebar.subheader("🤖 Model Configuration")
    
    try:
        models_response = api_client.get("/api/v1/models")
        available_models = models_response.get("available_models", []) if models_response else []
    except Exception as e:
        available_models = ["gpt-4", "gpt-4o-mini", "gemini-pro"]  # Fallback
        st.sidebar.warning("⚠️ Using fallback models")
    
    # Model mapping for display
    model_mapping = {
        "GPT-4": "gpt-4",
        "GPT4 Mini": "gpt-4o-mini",
        "Gemini Pro": "gemini-pro",
        "Groq Mixtral": "groq-mixtral",
        "llama3.2": "llama3.2:latest"
    }
    
    # Filter available models
    display_models = [k for k, v in model_mapping.items() if v in available_models]
    
    if display_models:
        selected_display = st.sidebar.selectbox(
            "Select LLM Model",
            options=display_models,
            index=0
        )
        selected_model = model_mapping[selected_display]
    else:
        selected_model = "gpt-4"
        st.sidebar.warning("⚠️ No models available")
    
    # Model Parameters
    st.sidebar.subheader("🎛️ Model Parameters")
    max_tokens = st.sidebar.slider("Max Tokens", 100, 4000, 1000)
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    
    # RAG Settings
    st.sidebar.subheader("🔍 RAG Settings")
    chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200)
    
    # Search Settings
    st.sidebar.subheader("🔎 Search Settings")
    default_search_type = st.sidebar.selectbox(
        "Default Search Type",
        ["hybrid", "vector", "keyword"],
        index=0
    )
    
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        0.0, 1.0, 0.0, 0.1
    )
    
    # Advanced Options
    with st.sidebar.expander("🔧 Advanced Options"):
        enable_context = st.checkbox("Use Document Context", value=True)
        show_agent_steps = st.checkbox("Show Agent Steps", value=False)
        show_refinement_details = st.checkbox("Show Refinement Details", value=True)
        show_processing_details = st.checkbox("Show Processing Details", value=False)
        show_debug = st.checkbox("Debug Mode", value=False)
        enable_caching = st.checkbox("Enable Caching", value=True)
    
    # System Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Info")
    
    try:
        if models_response:
            st.sidebar.write(f"**Default Model:** {models_response.get('default_model', 'Unknown')}")
            st.sidebar.write(f"**Available Models:** {len(available_models)}")
        
        # Document Statistics
        docs_response = api_client.get("/api/v1/documents")
        if docs_response:
            doc_count = docs_response.get("document_count", 0)
            vector_count = docs_response.get("index_info", {}).get("total_vectors", 0)
            st.sidebar.write(f"**Indexed Documents:** {doc_count}")
            st.sidebar.write(f"**Total Vectors:** {vector_count}")
        
        # Session info
        st.sidebar.write(f"**Session ID:** {st.session_state.get('session_id', 'N/A')[:8]}...")
        
    except Exception as e:
        st.sidebar.write("**Status:** Could not load system info")
    
    # Quick Actions
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Refresh", key="refresh_sidebar"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", key="clear_all_sidebar"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key not in ['session_id']:  # Keep session_id
                    del st.session_state[key]
            st.rerun()
    
    return {
        "selected_model": selected_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "default_search_type": default_search_type,
        "similarity_threshold": similarity_threshold,
        "enable_context": enable_context,
        "show_agent_steps": show_agent_steps,
        "show_refinement_details": show_refinement_details,
        "show_processing_details": show_processing_details,
        "show_debug": show_debug,
        "enable_caching": enable_caching
    }

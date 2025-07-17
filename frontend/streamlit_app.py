import streamlit as st
import requests
import json
import sys
import os
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.utils.api_client import APIClient
from frontend.components.sidebar import render_sidebar
from frontend.components.document_upload import render_document_upload
from frontend.components.enhanced_chat import render_enhanced_chat
from frontend.components.advanced_search import render_advanced_search
from frontend.components.query_refinement import render_query_refinement
from frontend.components.analytics import render_analytics

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize API client
    api_client = APIClient(API_BASE_URL)
    
    # Test API connection
    try:
        health_response = api_client.get("/health")
        if not health_response:
            st.error("❌ Cannot connect to FastAPI backend. Please ensure it's running on port 8000.")
            st.info("To start the backend, run: `python main.py`")
            st.stop()
    except Exception as e:
        st.error(f"❌ API Connection Error: {e}")
        st.stop()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .metric-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<div class="main-header">🤖 Agentic RAG-Driven Multi-Modal Assistant</div>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    config = render_sidebar(api_client)
    

 
    # # Main interface tabs - add voice tab
    # tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    #     "💬 RAG Chat", 
    #     "🎤 Voice Interface",  # New voice tab
    #     "📄 Document Upload", 
    #     "🔍 Advanced Search", 
    #     "🔧 Query Refinement", 
    #     "📊 Analytics",
    #     "⚙️ System"
    # ])
    
    # with tab1:
    #     render_enhanced_chat_with_voice(api_client, config)  # Updated with voice
    
    # with tab2:
    #     render_voice_interface(api_client)  # New voice interface
    
    # # ... rest of existing tabs ...




    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 RAG Chat", 
        "📄 Document Upload", 
        "🔍 Advanced Search", 
        "🔧 Query Refinement", 
        "📊 Analytics",
        "⚙️ System"
    ])
    
    with tab1:
        render_enhanced_chat(api_client, config)
    
    with tab2:
        render_document_upload(api_client)
    
    with tab3:
        render_advanced_search(api_client)
    
    with tab4:
        render_query_refinement(api_client)
    
    with tab5:
        render_analytics(api_client)
    
    with tab6:
        render_system_info(api_client)

def render_system_info(api_client):
    """Render system information tab"""
    st.header("⚙️ System Information")
    
    # System health
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Health")
        health_response = api_client.get("/health")
        if health_response:
            st.success("✅ API is healthy")
            st.json(health_response)
        else:
            st.error("❌ API is not responding")
    
    with col2:
        st.subheader("Available Models")
        models_response = api_client.get("/api/v1/models")
        if models_response:
            st.json(models_response)
        else:
            st.warning("⚠️ Could not load models")
    
    # Document statistics
    st.subheader("Document Statistics")
    docs_response = api_client.get("/api/v1/documents")
    if docs_response:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", docs_response.get("document_count", 0))
        
        with col2:
            st.metric("Vector Dimension", docs_response.get("index_info", {}).get("dimension", 0))
        
        with col3:
            st.metric("Index Type", docs_response.get("index_info", {}).get("index_type", "Unknown"))
    
    # Clear all data
    st.subheader("⚠️ Danger Zone")
    if st.button("🗑️ Clear All Documents", type="secondary"):
        if st.session_state.get("confirm_clear_all", False):
            response = api_client.delete("/api/v1/documents/clear")
            if response:
                st.success("✅ All documents cleared!")
                st.session_state.confirm_clear_all = False
                st.rerun()
            else:
                st.error("❌ Failed to clear documents")
        else:
            st.session_state.confirm_clear_all = True
            st.warning("⚠️ Click again to confirm deletion of ALL documents")



if __name__ == "__main__":
    main()

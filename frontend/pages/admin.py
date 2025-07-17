import streamlit as st
from frontend.utils.api_client import APIClient

def render_admin_page():
    """Render admin page"""
    st.title("🔧 Admin Dashboard")
    
    api_client = APIClient()
    
    # System status
    st.header("System Status")
    
    health_response = api_client.get("/health")
    if health_response:
        st.json(health_response)
    
    # Model management
    st.header("Model Management")
    
    models_response = api_client.get("/api/v1/models")
    if models_response:
        st.json(models_response)
    
    # Document management
    st.header("Document Management")
    
    if st.button("Clear All Documents"):
        response = api_client.delete("/api/v1/documents/clear")
        if response:
            st.success("Documents cleared successfully!")

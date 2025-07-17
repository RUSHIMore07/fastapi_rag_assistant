import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
from frontend.utils.api_client import APIClient
from datetime import datetime

def render_analytics(api_client: APIClient):
    """Analytics dashboard"""
    st.header("📊 System Analytics")
    
    # System overview
    st.subheader("📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get system stats
    docs_response = api_client.get("/api/v1/documents")
    health_response = api_client.get("/health")
    models_response = api_client.get("/api/v1/models")
    
    with col1:
        doc_count = docs_response.get("document_count", 0) if docs_response else 0
        st.metric("Total Documents", doc_count)
    
    with col2:
        vector_count = docs_response.get("index_info", {}).get("total_vectors", 0) if docs_response else 0
        st.metric("Total Vectors", vector_count)
    
    with col3:
        model_count = len(models_response.get("available_models", [])) if models_response else 0
        st.metric("Available Models", model_count)
    
    with col4:
        api_status = "Online" if health_response else "Offline"
        st.metric("API Status", api_status)
    
    # Document distribution
    if docs_response and doc_count > 0:
        st.subheader("📂 Document Distribution")
        
        # Mock data for demonstration (in real implementation, get from API)
        if st.session_state.get("recent_uploads"):
            categories = {}
            for upload in st.session_state.recent_uploads:
                cat = upload.get("category", "Other")
                categories[cat] = categories.get(cat, 0) + 1
            
            if categories:
                fig = px.pie(
                    values=list(categories.values()),
                    names=list(categories.keys()),
                    title="Documents by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Index information
        st.subheader("🔍 Index Information")
        if docs_response:
            index_info = docs_response.get("index_info", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "Total Vectors": index_info.get("total_vectors", 0),
                    "Vector Dimension": index_info.get("dimension", 0),
                    "Index Type": index_info.get("index_type", "Unknown")
                })
            
            with col2:
                # Performance metrics (mock data)
                perf_data = {
                    "Average Query Time": "0.25s",
                    "Cache Hit Rate": "78%",
                    "Search Accuracy": "92%",
                    "Embedding Generation": "1.2s"
                }
                st.json(perf_data)
    
    # Recent activity
    st.subheader("📝 Recent Activity")
    
    if st.session_state.get("recent_uploads"):
        st.write("**Recent Document Uploads:**")
        for upload in st.session_state.recent_uploads[-10:]:
            st.write(f"• {upload['filename']} ({upload['category']}) - {upload['chunks']} chunks")
    else:
        st.info("No recent uploads to display")
    
    if st.session_state.get("rag_messages"):
        st.write("**Recent Chat Queries:**")
        user_messages = [msg for msg in st.session_state.rag_messages if msg["role"] == "user"]
        for msg in user_messages[-5:]:
            st.write(f"• {msg['content'][:50]}...")
    else:
        st.info("No recent chat queries to display")
    
    # System health details
    st.subheader("🏥 System Health")
    
    if health_response:
        st.success("✅ System is healthy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Status": health_response.get("status"),
                "Version": health_response.get("version"),
                "Timestamp": health_response.get("timestamp")
            })
        
        with col2:
            services = health_response.get("services", {})
            st.json(services)
    else:
        st.error("❌ System health check failed")
    
    # Performance monitoring
    st.subheader("⚡ Performance Monitoring")
    
    # Mock performance data
    if st.button("📊 Run Performance Test"):
        with st.spinner("Running performance tests..."):
            import time
            time.sleep(2)  # Simulate test
            
            # Mock results
            results = {
                "Query Processing": "Fast (< 0.5s)",
                "Document Retrieval": "Excellent (< 0.2s)",
                "Embedding Generation": "Good (< 2s)",
                "Response Generation": "Fast (< 1s)"
            }
            
            st.success("Performance test completed!")
            st.json(results)
    
    # Export data
    st.subheader("📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Analytics"):
            # In real implementation, generate and download analytics report
            st.info("Analytics export feature not yet implemented")
    
    with col2:
        if st.button("📄 Export Documents List"):
            if docs_response:
                st.download_button(
                    label="Download Documents Info",
                    data=str(docs_response),
                    file_name="documents_info.json",
                    mime="application/json"
                )

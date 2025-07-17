import streamlit as st
import json
import requests
from typing import Dict, Any
from frontend.utils.api_client import APIClient

def render_document_upload(api_client: APIClient):
    """Document upload interface using the new ingest API"""
    st.header("📄 Document Ingestion")
    
    # Upload section
    st.subheader("📤 Upload New Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx'],
        help="Upload PDF, TXT, or DOCX files to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        # Show file preview
        st.info(f"**File:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                ["recursive", "semantic", "small"],
                index=0,
                help="Choose how to split the document into chunks"
            )
        
        with col2:
            doc_category = st.selectbox(
                "Document Category",
                ["General", "Research", "Technical", "Business", "Legal", "Medical", "Other"]
            )
        
        # Additional metadata
        col1, col2 = st.columns(2)
        
        with col1:
            doc_source = st.text_input("Source", placeholder="e.g., Wikipedia, Internal docs")
        
        with col2:
            doc_author = st.text_input("Author", placeholder="Document author")
        
        doc_description = st.text_area(
            "Description", 
            placeholder="Brief description of the document content"
        )
        
        # Tags
        doc_tags = st.text_input(
            "Tags (comma-separated)", 
            placeholder="tag1, tag2, tag3"
        )
        
        # Process document button
        if st.button("📤 Process Document", type="primary"):
            with st.spinner("Processing document through ingestion pipeline..."):
                try:
                    # Prepare metadata
                    metadata = {
                        "source": doc_source or uploaded_file.name,
                        "category": doc_category,
                        "description": doc_description,
                        "author": doc_author,
                        "tags": [tag.strip() for tag in doc_tags.split(",") if tag.strip()],
                        "filename": uploaded_file.name,
                        "size": uploaded_file.size
                    }
                    
                    # Create form data for file upload
                    files = {"file": uploaded_file.getvalue()}
                    form_data = {
                        "chunking_strategy": chunking_strategy,
                        "metadata": json.dumps(metadata)
                    }
                    
                    # Upload using the new ingest API
                    response = requests.post(
                        f"{api_client.base_url}/api/v1/ingest",
                        files={"file": uploaded_file},
                        data=form_data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                        
                        # Show processing results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Chunks", result['total_chunks'])
                        
                        with col2:
                            st.metric("Text Length", f"{result['text_length']:,}")
                        
                        with col3:
                            st.metric("Strategy", result['chunking_strategy'])
                        
                        with col4:
                            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                        
                        # Show detailed results
                        with st.expander("📊 Processing Details"):
                            st.json(result)
                        
                        # Add to recent uploads
                        if "recent_uploads" not in st.session_state:
                            st.session_state.recent_uploads = []
                        
                        st.session_state.recent_uploads.append({
                            "filename": uploaded_file.name,
                            "category": doc_category,
                            "chunks": result['total_chunks'],
                            "timestamp": result.get('metadata', {}).get('upload_time')
                        })
                        
                        # Clear the file uploader
                        st.rerun()
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Upload failed: {error_detail}")
                        
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
    
    # Document management section
    st.markdown("---")
    st.subheader("📊 Document Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 View Documents"):
            docs_response = api_client.get("/api/v1/documents")
            if docs_response:
                st.subheader("Document Statistics")
                
                # Display metrics
                info = docs_response.get("index_info", {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Documents", docs_response.get("document_count", 0))
                
                with col2:
                    st.metric("Total Vectors", info.get("total_vectors", 0))
                
                st.json(docs_response)
    
    with col2:
        if st.button("🔍 Search Test"):
            test_query = st.text_input("Test search query:", "artificial intelligence")
            if test_query:
                search_response = api_client.get(f"/api/v1/search?query={test_query}&limit=3")
                if search_response:
                    st.write(f"Found {search_response.get('result_count', 0)} results")
                    for i, result in enumerate(search_response.get('results', [])[:3]):
                        st.write(f"**Result {i+1}:** {result['content'][:100]}...")
    
    with col3:
        if st.button("🗑️ Clear All"):
            if st.session_state.get("confirm_clear_docs", False):
                response = api_client.delete("/api/v1/documents/clear")
                if response:
                    st.success("✅ All documents cleared!")
                    st.session_state.confirm_clear_docs = False
                    st.rerun()
                else:
                    st.error("❌ Failed to clear documents")
            else:
                st.session_state.confirm_clear_docs = True
                st.warning("⚠️ Click again to confirm deletion")
    
    # Recent uploads
    if st.session_state.get("recent_uploads"):
        st.subheader("📝 Recent Uploads")
        for upload in st.session_state.recent_uploads[-5:]:  # Show last 5
            with st.expander(f"📄 {upload['filename']} ({upload['category']})"):
                st.write(f"**Chunks:** {upload['chunks']}")
                st.write(f"**Uploaded:** {upload['timestamp']}")

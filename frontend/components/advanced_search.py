import streamlit as st
from typing import Dict, Any
from frontend.utils.api_client import APIClient

def render_advanced_search(api_client: APIClient):
    """Advanced search interface with multiple strategies"""
    st.header("🔍 Advanced Search")
    
    # Search configuration
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter your search query...",
            help="Search through all uploaded documents"
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["hybrid", "vector", "keyword"],
            index=0,
            help="Choose search strategy"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_results = st.slider("Number of Results", 5, 50, 10)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.0, 0.1)
        
        with col2:
            # Document filters
            st.subheader("Filters")
            filter_category = st.multiselect(
                "Document Category",
                ["General", "Research", "Technical", "Business", "Legal", "Medical", "Other"]
            )
            
            filter_file_type = st.multiselect(
                "File Type",
                ["pdf", "txt", "docx"]
            )
    
    # Search button
    if st.button("🔍 Search", type="primary") and search_query:
        with st.spinner("Searching through documents..."):
            
            # Prepare filters
            filters = {}
            if filter_category:
                filters["category"] = filter_category
            if filter_file_type:
                filters["file_type"] = filter_file_type
            
            # Prepare search request
            search_request = {
                "query": search_query,
                "search_type": search_type,
                "k": num_results,
                "filters": filters if filters else None
            }
            
            # Perform search
            response = api_client.post("/api/v1/advanced-search", data=search_request)
            
            if response and response.get("results"):
                results = response["results"]
                
                # Filter by similarity threshold
                if similarity_threshold > 0:
                    results = [r for r in results if r["similarity_score"] >= similarity_threshold]
                
                st.success(f"Found {len(results)} results for: **{search_query}**")
                
                # Display results
                for i, result in enumerate(results):
                    with st.expander(f"📄 Result {i+1} - Score: {result['similarity_score']:.3f}"):
                        
                        # Content preview
                        content = result["content"]
                        if len(content) > 500:
                            st.markdown(f"**Content Preview:**")
                            st.text(content[:500] + "...")
                            
                            if st.button(f"Show Full Content", key=f"show_full_{i}"):
                                st.markdown(f"**Full Content:**")
                                st.text(content)
                        else:
                            st.markdown(f"**Content:**")
                            st.text(content)
                        
                        # Metadata
                        metadata = result.get("metadata", {})
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Document Info:**")
                            st.text(f"📁 File: {metadata.get('filename', 'Unknown')}")
                            st.text(f"🏷️ Category: {metadata.get('category', 'Unknown')}")
                            st.text(f"📊 Rank: {result.get('rank', 'N/A')}")
                        
                        with col2:
                            st.markdown("**Search Info:**")
                            st.text(f"🔍 Search Type: {result.get('search_type', 'Unknown')}")
                            st.text(f"⭐ Score: {result['similarity_score']:.4f}")
                            st.text(f"📅 Uploaded: {metadata.get('upload_time', 'Unknown')}")
                        
                        # Action buttons
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button(f"💬 Ask About This", key=f"ask_{i}"):
                                # Set up for chat
                                chat_query = f"Based on this content, tell me more about: {search_query}"
                                st.session_state.search_to_chat = {
                                    "query": chat_query,
                                    "context": content[:1000]
                                }
                                st.info("Query prepared for chat. Go to RAG Chat tab to continue.")
                        
                        with col2:
                            if st.button(f"📋 Copy Text", key=f"copy_{i}"):
                                st.text_area(f"Content {i+1}", content, height=100, key=f"copy_area_{i}")
                        
                        with col3:
                            if st.button(f"🔗 View Source", key=f"source_{i}"):
                                st.json(metadata)
                
            elif response and response.get("total_results") == 0:
                st.info("No results found. Try different search terms or adjust filters.")
            
            else:
                st.error("Search failed. Please try again.")
    
    # Search suggestions
    st.markdown("---")
    st.subheader("💡 Search Suggestions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Technical Terms"):
            st.session_state.search_suggestion = "technical implementation methodology"
    
    with col2:
        if st.button("📊 Data Analysis"):
            st.session_state.search_suggestion = "data analysis results findings"
    
    with col3:
        if st.button("🎯 Best Practices"):
            st.session_state.search_suggestion = "best practices recommendations guidelines"
    
    # Apply search suggestion
    if st.session_state.get("search_suggestion"):
        st.info(f"Suggested search: {st.session_state.search_suggestion}")
        if st.button("Use This Search"):
            st.session_state.search_query = st.session_state.search_suggestion
            st.session_state.search_suggestion = None
            st.rerun()
    
    # Search tips
    with st.expander("💡 Search Tips"):
        st.markdown("""
        **Search Strategies:**
        - **Hybrid**: Best overall results, combines semantic and keyword matching
        - **Vector**: Great for conceptual and semantic searches
        - **Keyword**: Good for exact term matching
        
        **Tips:**
        - Use specific keywords for better results
        - Try different phrasings if you don't find what you're looking for
        - Use filters to narrow down results by category or file type
        - Adjust similarity threshold to control result quality
        """)

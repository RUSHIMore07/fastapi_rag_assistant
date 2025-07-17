import streamlit as st
from frontend.utils.api_client import APIClient

def render_search(api_client: APIClient):
    """Render search interface"""
    st.header("🔍 Search Documents")
    
    # Search input
    search_query = st.text_input(
        "Search in uploaded documents:",
        placeholder="Enter your search query..."
    )
    
    # Search parameters
    col1, col2 = st.columns(2)
    
    with col1:
        search_limit = st.slider("Number of results", 1, 20, 5)
    
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["Semantic", "Keyword", "Hybrid"]
        )
    
    # Search button
    if st.button("🔍 Search", type="primary") and search_query:
        with st.spinner("Searching..."):
            response = api_client.get(
                "/api/v1/search",
                params={"query": search_query, "limit": search_limit}
            )
            
            if response and response.get("results"):
                st.success(f"Found {response['result_count']} results for: **{search_query}**")
                
                # Display results
                for i, result in enumerate(response["results"], 1):
                    with st.expander(f"📄 Result {i} - Similarity: {result['similarity_score']:.3f}"):
                        # Content preview
                        content = result["content"]
                        if len(content) > 500:
                            st.markdown(f"**Content Preview:**\n\n{content[:500]}...")
                            
                            if st.button(f"Show Full Content {i}", key=f"show_full_{i}"):
                                st.markdown(f"**Full Content:**\n\n{content}")
                        else:
                            st.markdown(f"**Content:**\n\n{content}")
                        
                        # Metadata
                        if result.get("metadata"):
                            st.markdown("**Metadata:**")
                            metadata = result["metadata"]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if metadata.get("filename"):
                                    st.text(f"📁 File: {metadata['filename']}")
                                if metadata.get("category"):
                                    st.text(f"🏷️ Category: {metadata['category']}")
                            
                            with col2:
                                if metadata.get("source"):
                                    st.text(f"🔗 Source: {metadata['source']}")
                                if metadata.get("upload_time"):
                                    st.text(f"📅 Uploaded: {metadata['upload_time']}")
                        
                        # Action buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"💬 Ask about this", key=f"ask_{i}"):
                                # Add to chat context
                                chat_query = f"Based on this content: '{content[:200]}...', {search_query}"
                                st.session_state.search_to_chat = chat_query
                                st.info("Query added to chat context. Go to Chat tab to continue.")
                        
                        with col2:
                            if st.button(f"📋 Copy Content", key=f"copy_{i}"):
                                # In a real app, you might use a clipboard library
                                st.text_area(f"Content {i}", content, height=100)
            
            elif response and response.get("result_count") == 0:
                st.info("No results found. Try different search terms.")
            
            else:
                st.error("Search failed. Please try again.")
    
    # Search tips
    with st.expander("💡 Search Tips"):
        st.markdown("""
        - Use specific keywords for better results
        - Try different phrasings if you don't find what you're looking for
        - Semantic search works well with natural language questions
        - Use quotes for exact phrase matching
        """)
    
    # Quick search suggestions
    st.subheader("Quick Search Suggestions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Technical Terms"):
            st.session_state.search_suggestion = "technical implementation details"
    
    with col2:
        if st.button("📊 Data Analysis"):
            st.session_state.search_suggestion = "data analysis methodology"
    
    with col3:
        if st.button("🎯 Best Practices"):
            st.session_state.search_suggestion = "best practices recommendations"
    
    # Apply search suggestion
    if st.session_state.get("search_suggestion"):
        st.rerun()

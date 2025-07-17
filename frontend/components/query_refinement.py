import streamlit as st
from typing import Dict, Any
from frontend.utils.api_client import APIClient

def render_query_refinement(api_client: APIClient):
    """Query refinement interface"""
    st.header("🔧 Query Refinement")
    
    st.markdown("""
    This tool helps improve your queries using AI to get better search results.
    """)
    
    # Query input
    original_query = st.text_area(
        "Original Query",
        placeholder="Enter your query to refine...",
        height=100
    )
    
    # Refinement options
    col1, col2 = st.columns(2)
    
    with col1:
        refinement_type = st.selectbox(
            "Refinement Type",
            ["auto", "rewrite", "decompose", "clarify", "expand"],
            index=0,
            help="Choose how to refine your query"
        )
    
    with col2:
        include_context = st.checkbox(
            "Include Context",
            help="Provide additional context for better refinement"
        )
    
    # Context input
    context = None
    if include_context:
        context = st.text_area(
            "Context",
            placeholder="Provide additional context about your query...",
            height=80
        )
    
    # Refine button
    if st.button("🔧 Refine Query", type="primary") and original_query:
        with st.spinner("Refining your query..."):
            
            # Prepare refinement request
            refinement_request = {
                "query": original_query,
                "refinement_type": refinement_type if refinement_type != "auto" else None,
                "context": context,
                "user_preferences": {}
            }
            
            # Call refinement API
            response = api_client.post("/api/v1/refine-query", data=refinement_request)
            
            if response:
                st.success("✅ Query refined successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Original Query")
                    st.text_area("", value=response["original_query"], height=100, disabled=True)
                
                with col2:
                    st.subheader("✨ Refined Query")
                    refined_query = response["refined_query"]
                    st.text_area("", value=refined_query, height=100, disabled=True)
                
                # Show refinement details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Refinement Type", response["refinement_type"])
                
                with col2:
                    st.metric("Confidence Score", f"{response['confidence_score']:.2f}")
                
                with col3:
                    st.metric("Sub-queries", len(response.get("sub_queries", [])))
                
                # Show sub-queries if available
                if response.get("sub_queries"):
                    st.subheader("🔍 Sub-queries")
                    for i, sub_query in enumerate(response["sub_queries"], 1):
                        st.write(f"**{i}.** {sub_query}")
                
                # Show reasoning
                st.subheader("🧠 Reasoning")
                st.info(response.get("reasoning", "No reasoning provided"))
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔍 Search with Refined Query"):
                        st.session_state.refined_search_query = refined_query
                        st.info("Query saved for search. Go to Advanced Search tab.")
                
                with col2:
                    if st.button("💬 Use in Chat"):
                        st.session_state.refined_chat_query = refined_query
                        st.info("Query saved for chat. Go to RAG Chat tab.")
                
                with col3:
                    if st.button("📋 Copy Refined Query"):
                        st.text_area("Copy this:", value=refined_query, height=60)
                
            else:
                st.error("❌ Failed to refine query. Please try again.")
    
    # Refinement examples
    st.markdown("---")
    st.subheader("📚 Refinement Examples")
    
    examples = [
        {
            "type": "rewrite",
            "original": "AI benefits",
            "refined": "What are the specific advantages and benefits of implementing artificial intelligence in business operations?"
        },
        {
            "type": "decompose",
            "original": "Compare machine learning and deep learning performance",
            "refined": ["What is machine learning performance?", "What is deep learning performance?", "How do machine learning and deep learning performance compare?"]
        },
        {
            "type": "clarify",
            "original": "How does it work?",
            "refined": "How does the neural network architecture work in image recognition systems?"
        },
        {
            "type": "expand",
            "original": "Python programming",
            "refined": "Python programming language syntax, libraries, frameworks, development best practices, and application examples"
        }
    ]
    
    for example in examples:
        with st.expander(f"📝 {example['type'].title()} Example"):
            st.write(f"**Original:** {example['original']}")
            if isinstance(example['refined'], list):
                st.write("**Refined Sub-queries:**")
                for i, sub in enumerate(example['refined'], 1):
                    st.write(f"{i}. {sub}")
            else:
                st.write(f"**Refined:** {example['refined']}")

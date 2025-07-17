import streamlit as st
from typing import Dict, Any
from frontend.utils.api_client import APIClient
import time

def render_chat(api_client: APIClient, config: Dict[str, Any]):
    """Render chat interface"""
    st.header("💬 Chat with Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat controls
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        auto_scroll = st.checkbox("Auto-scroll", value=True)
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata if available
                if message.get("metadata") and config.get("show_agent_steps"):
                    with st.expander("Response Details"):
                        st.json(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare request data with proper field names and types
                request_data = {
                    "query": prompt,
                    "query_type": "text",  # Use string, not enum
                    "preferred_llm": config["selected_model"].lower(),  # Ensure lowercase
                    "session_id": st.session_state.get("session_id", f"session_{int(time.time())}"),
                    "max_tokens": config["max_tokens"],
                    "temperature": config["temperature"],
                    "context": None  # Add context field
                }
                
                # Debug: Show what we're sending
                if config.get("show_debug"):
                    st.write("Debug - Sending:", request_data)
                
                response = api_client.post("/api/v1/query", data=request_data)
            
            if response:
                assistant_response = response.get("response", "Sorry, I couldn't process your request.")
                st.markdown(assistant_response)
                
                # Add assistant response to chat history
                message_data = {
                    "role": "assistant", 
                    "content": assistant_response,
                    "metadata": {
                        "model_used": response.get("model_used"),
                        "processing_time": f"{response.get('metadata', {}).get('processing_time', 0):.2f}s",
                        "agent_steps": response.get("agent_steps", []),
                        "context_used": response.get("context_used")
                    }
                }
                st.session_state.messages.append(message_data)
                
                # Show response details
                if config.get("show_agent_steps") and response.get("agent_steps"):
                    with st.expander("Agent Execution Steps"):
                        for step in response.get("agent_steps", []):
                            st.json(step)
                
                # Show context if used
                if config.get("enable_context") and response.get("context_used"):
                    context = response.get("context_used")
                    if context.get("chunks"):
                        with st.expander("Context Sources"):
                            for i, (chunk, source) in enumerate(zip(context["chunks"], context["sources"])):
                                st.markdown(f"**Source {i+1}:** {source}")
                                st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            else:
                st.error("Failed to get response from the API")





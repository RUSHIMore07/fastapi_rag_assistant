@echo off
echo Starting Agentic RAG Assistant...
start "FastAPI Backend" start_backend.bat
timeout /t 3
start "Streamlit Frontend" start_frontend.bat
echo Services started! Check the opened windows.

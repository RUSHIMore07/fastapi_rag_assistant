import subprocess
import sys
import time
import os

def start_fastapi():
    """Start FastAPI backend"""
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=os.getcwd()
    )

def start_streamlit():
    """Start Streamlit frontend"""
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"],
        cwd=os.getcwd()
    )

def main():
    print("Starting Agentic RAG Assistant...")
    
    # Start FastAPI backend
    print("Starting FastAPI backend...")
    fastapi_process = start_fastapi()
    
    # Wait a bit for FastAPI to start
    time.sleep(3)
    
    # Start Streamlit frontend
    print("Starting Streamlit frontend...")
    streamlit_process = start_streamlit()
    
    print("\n" + "="*50)
    print("✅ Services started successfully!")
    print("FastAPI Backend: http://localhost:8000")
    print("Streamlit Frontend: http://localhost:8501")
    print("API Documentation: http://localhost:8000/docs")
    print("="*50)
    
    try:
        # Wait for processes to complete
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        fastapi_process.terminate()
        streamlit_process.terminate()

if __name__ == "__main__":
    main()

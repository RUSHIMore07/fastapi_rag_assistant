import os
from turtle import st
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import asyncio
import logging
from datetime import datetime
import base64

from app.models.schemas import (
    QueryRequest, QueryResponse, ImageUpload, DocumentUpload, 
    HealthCheck, LLMProvider, QueryType, RetrievalContext, VoiceRAGRequest
)
from app.agents.orchestrator import AgenticOrchestrator
from app.config.settings import settings
from app.utils.logging import setup_logging
from app.multi_modal.processors import DocumentProcessor
from app.retrieval.vector_store import FAISSVectorStore

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Agentic RAG-Driven Multi-Modal Assistant",
    version=settings.version,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
orchestrator = AgenticOrchestrator()
document_processor = DocumentProcessor()
vector_store = FAISSVectorStore()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Agentic RAG Assistant API")
    # Initialize any required services here

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Agentic RAG Assistant API")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic RAG-Driven Multi-Modal Assistant API",
        "version": settings.version,
        "status": "running"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    services = {
        "api": "healthy",
        "vector_store": "healthy",
        "llm_router": "healthy"
    }
    
    return HealthCheck(
        status="healthy",
        version=settings.version,
        timestamp=datetime.now(),
        services=services
    )

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(query_request: QueryRequest):
    """Process a text query"""
    try:
        logger.info(f"Processing query: {query_request.query[:100]}...")
        
        # Convert preferred_llm to proper enum if it's a string
        preferred_llm = None
        if query_request.preferred_llm:
            if isinstance(query_request.preferred_llm, str):
                try:
                    preferred_llm = LLMProvider(query_request.preferred_llm.lower())
                except ValueError:
                    logger.warning(f"Invalid LLM provider: {query_request.preferred_llm}")
                    preferred_llm = None
            else:
                preferred_llm = query_request.preferred_llm
        
        # Convert to dict for orchestrator
        request_dict = {
            "query": query_request.query,
            "query_type": query_request.query_type,
            "preferred_llm": preferred_llm,
            "session_id": query_request.session_id,
            "max_tokens": query_request.max_tokens,
            "temperature": query_request.temperature,
            "context": query_request.context
        }
        
        response = await orchestrator.process_query(request_dict)
        logger.info(f"Query processed successfully for session: {query_request.session_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v1/query/image", response_model=QueryResponse)
async def process_image_query(image_upload: ImageUpload):
    """Process an image query"""
    try:
        logger.info(f"Processing image query: {image_upload.query[:100]}...")
        
        # Process image
        from app.multi_modal.handlers import MultiModalHandler
        handler = MultiModalHandler()
        
        image_result = handler.process_image(
            image_upload.image_data, 
            image_upload.image_type
        )
        
        if not image_result["processed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to process image: {image_result.get('error', 'Unknown error')}"
            )
        
        # Create query request
        request_dict = {
            "query": f"Analyze this image: {image_upload.query}",
            "query_type": QueryType.IMAGE,
            "preferred_llm": LLMProvider.OPENAI,  # Use GPT-4V for image analysis
            "session_id": f"img_{datetime.now().timestamp()}",
            "max_tokens": 1000,
            "temperature": 0.7,
            "context": {"image_metadata": image_result["metadata"]}
        }
        
        response = await orchestrator.process_query(request_dict)
        logger.info("Image query processed successfully")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image query: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/v1/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form("{}")
):
    """Upload and process a document (legacy endpoint)"""
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Check file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.allowed_file_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process document and get text content
        from app.multi_modal.processors import DocumentProcessor
        doc_processor = DocumentProcessor()
        
        if file_extension == "pdf":
            text_content = doc_processor.process_pdf(file_content)
        elif file_extension == "docx":
            text_content = doc_processor.process_docx(file_content)
        elif file_extension == "txt":
            text_content = doc_processor.process_text(file_content)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Processing for {file_extension} not implemented"
            )
        
        # Check if text extraction was successful
        if not text_content.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text content could be extracted from the file"
            )
        
        # Add to vector store
        import json
        doc_metadata = json.loads(metadata) if metadata != "{}" else {}
        doc_metadata.update({
            "filename": file.filename,
            "file_type": file_extension,
            "upload_time": datetime.now().isoformat()
        })
        
        await vector_store.add_documents(
            [text_content],  # Pass as list of strings
            [doc_metadata]
        )
        
        logger.info(f"Document {file.filename} processed and indexed successfully")
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "file_type": file_extension,
            "text_length": len(text_content),
            "metadata": doc_metadata
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/v1/models")
async def get_available_models():
    """Get list of available LLM models"""
    return {
        "available_models": settings.available_models,
        "default_model": settings.default_llm,
        "providers": {
            "openai": ["gpt-4", "gpt-4o-mini", "gpt-4-vision-preview"],
            "google": ["gemini-pro", "gemini-pro-vision"],
            "groq": ["mixtral-8x7b-32768", "llama3.2:latest-70b-4096"],
            "ollama": ["llama3.2:latest", "deepseek-coder", "mistral"]
        }
    }

@app.get("/api/v1/documents")
async def list_documents():
    """List indexed documents"""
    try:
        # Get document count from vector store
        doc_count = vector_store.index.ntotal if vector_store.index else 0
        
        return {
            "document_count": doc_count,
            "index_info": {
                "total_vectors": doc_count,
                "dimension": vector_store.dimension,
                "index_type": "FAISS"
            }
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/documents/clear")
async def clear_documents():
    """Clear all documents from vector store"""
    try:
        # Reset vector store
        vector_store.index = vector_store.index.__class__(vector_store.dimension)
        vector_store.documents = []
        vector_store.metadata = []
        vector_store.save_index()
        
        logger.info("All documents cleared from vector store")
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/search")
async def search_documents(query: str, limit: int = 5):
    """Search documents in vector store"""
    try:
        results = await vector_store.similarity_search(query, k=limit)
        
        return {
            "query": query,
            "results": results,
            "result_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v1/ingest", response_model=Dict[str, Any])
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("recursive"),
    metadata: str = Form("{}")
):
    """Enhanced document ingestion with advanced chunking"""
    try:
        # Read file content
        content = await file.read()
        
        # Create document processor instance
        from app.multi_modal.processors import DocumentProcessor
        doc_processor = DocumentProcessor()
        text_content = doc_processor.process_pdf(content)
        # Determine file type and extract text
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            text_content = doc_processor.process_pdf(content)
        elif file_extension == 'docx':
            text_content = doc_processor.process_docx(content)
        elif file_extension == 'txt':
            text_content = doc_processor.process_text(content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Check if text extraction was successful
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")
        
        # Parse metadata
        import json
        doc_metadata = json.loads(metadata) if metadata != "{}" else {}
        doc_metadata.update({
            "filename": file.filename,
            "file_type": file_extension,
            "upload_time": datetime.now().isoformat(),
            "file_size": len(content),
            "text_length": len(text_content)
        })
        
        # Process document with enhanced chunking
        from app.ingestion.document_processor import DocumentProcessor as IngestionProcessor
        processor = IngestionProcessor()
        
        result = await processor.process_document(
            text_content,  # Now passing string, not dict
            doc_metadata, 
            chunking_strategy
        )
        
        if result.success:
            # Store in vector database
            from app.retrieval.advanced_vector_store import AdvancedVectorStore
            vector_store = AdvancedVectorStore()
            await vector_store.add_chunks(result.chunks)
            
            return {
                "success": True,
                "document_hash": result.document_hash,
                "total_chunks": result.total_chunks,
                "chunking_strategy": chunking_strategy,
                "metadata": result.metadata,
                "text_length": len(text_content),
                "processing_time": result.processing_time
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v1/refine-query", response_model=Dict[str, Any])
async def refine_query(request: Dict[str, Any]):
    """LLM-based query refinement"""
    try:
        from app.query_refinement.query_refiner import QueryRefiner
        from app.models.schemas import QueryRefinementRequest
        
        refiner = QueryRefiner()
        
        refinement_request = QueryRefinementRequest(
            query=request["query"],
            context=request.get("context"),
            refinement_type=request.get("refinement_type"),
            user_preferences=request.get("user_preferences", {})
        )
        
        result = await refiner.refine_query(refinement_request)
        
        return {
            "original_query": result.original_query,
            "refined_query": result.refined_query,
            "sub_queries": result.sub_queries,
            "refinement_type": result.refinement_type,
            "confidence_score": result.confidence_score,
            "reasoning": result.reasoning
        }
        
    except Exception as e:
        logger.error(f"Query refinement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/advanced-search", response_model=Dict[str, Any])
async def advanced_search(request: Dict[str, Any]):
    """Advanced search with multiple ranking strategies"""
    try:
        query = request["query"]
        search_type = request.get("search_type", "hybrid")
        filters = request.get("filters", {})
        k = request.get("k", 10)
        
        # Generate query embedding
        from app.ingestion.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        query_embedding = await processor.embedding_model.aembed_query(query)
        
        # Perform advanced search
        from app.retrieval.advanced_vector_store import AdvancedVectorStore
        vector_store = AdvancedVectorStore()
        
        results = await vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            k=k,
            search_type=search_type,
            filters=filters
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.chunk.content,
                "metadata": result.chunk.metadata,
                "similarity_score": result.similarity_score,
                "rank": result.rank,
                "search_type": result.search_type
            })
        
        return {
            "query": query,
            "search_type": search_type,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Advanced search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/complete-rag", response_model=QueryResponse)
async def complete_rag_pipeline(request: Dict[str, Any]):
    """Complete RAG pipeline with optional query refinement"""
    try:
        original_query = request["query"]
        use_refinement = request.get("use_refinement", False)
        refinement_type = request.get("refinement_type")
        search_type = request.get("search_type", "hybrid")
        preferred_llm = request.get("preferred_llm", "gpt-4")
        
        # Convert string model name to provider enum
        from app.llm_routing.router import LLMRouter
        router = LLMRouter()
        provider_enum = router._convert_string_to_provider(preferred_llm)
        
        # Step 1: Optional query refinement
        if use_refinement:
            refinement_request = {
                "query": original_query,
                "context": request.get("context"),
                "refinement_type": refinement_type
            }
            
            refinement_result = await refine_query(refinement_request)
            
            # Use refined query or sub-queries
            if refinement_result["sub_queries"]:
                search_queries = refinement_result["sub_queries"]
            else:
                search_queries = [refinement_result["refined_query"]]
        else:
            search_queries = [original_query]
        
        # Step 2: Advanced retrieval
        all_results = []
        for query in search_queries:
            search_request = {
                "query": query,
                "search_type": search_type,
                "k": request.get("k", 5)
            }
            
            search_results = await advanced_search(search_request)
            all_results.extend(search_results["results"])
        
        # Step 3: Deduplicate and rank results
        unique_results = {}
        for result in all_results:
            chunk_id = result["metadata"].get("chunk_id", "unknown")
            if chunk_id not in unique_results or result["similarity_score"] > unique_results[chunk_id]["similarity_score"]:
                unique_results[chunk_id] = result
        
        # Get top results
        top_results = sorted(unique_results.values(), 
                           key=lambda x: x["similarity_score"], 
                           reverse=True)[:10]
        
        # Step 4: Generate final response
        context_text = "\n\n".join([
            f"Source: {result['metadata'].get('filename', 'Unknown')}\n{result['content']}" 
            for result in top_results
        ])
        
        # Use LLM to generate final response
        rag_prompt = f"""
        Based on the following context, please provide a comprehensive answer to the user's question.

        Context:
        {context_text}

        Question: {original_query}

        Please provide a detailed answer based on the context provided. If the context doesn't contain enough information, please indicate that clearly.
        """
        
        # Generate response with proper model handling
        llm_response = await router.generate_response(
            rag_prompt,
            preferred_llm,  # Pass the string directly
            max_tokens=request.get("max_tokens", 1000),
            temperature=request.get("temperature", 0.7)
        )
        
        return QueryResponse(
            response=llm_response["content"],
            model_used=llm_response.get("model", preferred_llm),
            session_id=request.get("session_id", "default"),
            context_used=RetrievalContext(
                chunks=[r["content"] for r in top_results],
                sources=[r["metadata"].get("filename", "Unknown") for r in top_results],
                relevance_scores=[r["similarity_score"] for r in top_results]
            ),
            metadata={
                "original_query": original_query,
                "refined_queries": search_queries if use_refinement else [],
                "search_type": search_type,
                "use_refinement": use_refinement,
                "total_chunks_found": len(top_results),
                "llm_response_success": llm_response.get("success", False)
            }
        )
        
    except Exception as e:
        logger.error(f"Complete RAG pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))





# Add these imports at the top
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from app.voice.google_voice_service import GoogleVoiceService
from app.models.schemas import VoiceRequest, VoiceResponse, SpeechToTextRequest, SpeechToTextResponse

# Initialize voice service
voice_service = GoogleVoiceService()

# Add these endpoints to your existing main.py

@app.post("/api/v1/text-to-speech", response_model=Dict[str, Any])
async def text_to_speech(request: VoiceRequest):
    """Convert text to speech using Google Cloud TTS"""
    try:
        result = await voice_service.text_to_speech(
            text=request.text,
            language_code=request.language_code,
            voice_name=request.voice_name,
            voice_gender=request.voice_gender,
            speaking_rate=request.speaking_rate,
            pitch=request.pitch,
            audio_encoding=request.audio_encoding
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Text-to-speech endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/speech-to-text", response_model=Dict[str, Any])
async def speech_to_text(
    file: UploadFile = File(...),
    language_code: str = Form("en-US"),
    audio_encoding: str = Form("WEBM_OPUS"),
    sample_rate: int = Form(16000)
):
    """Convert speech to text using Google Cloud Speech-to-Text"""
    try:
        # Read audio file
        audio_data = await file.read()
        
        # Convert to text
        result = await voice_service.speech_to_text(
            audio_data=audio_data,
            language_code=language_code,
            audio_encoding=audio_encoding,
            sample_rate=sample_rate
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Speech-to-text endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/voices", response_model=Dict[str, Any])
async def get_available_voices(language_code: str = None):
    """Get available voices"""
    try:
        voices = voice_service.get_available_voices(language_code)
        supported_languages = voice_service.get_supported_languages()
        
        return {
            "success": True,
            "voices": voices,
            "supported_languages": supported_languages,
            "total_voices": len(voices)
        }
        
    except Exception as e:
        logger.error(f"Get voices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from app.models.schemas import QueryResponse, VoiceRequest  # import your models



@app.post("/api/v1/rag-with-voice", response_model=Dict[str, Any])
async def rag_with_voice(request: Dict[str, Any]):
    """Complete RAG pipeline with voice response"""
    try:
        # 1️⃣ Run the RAG pipeline
        rag_result: QueryResponse = await complete_rag_pipeline(request)
        
        # 2️⃣ Decide whether to create TTS output
        generate_voice = request.get("generate_voice", False)
        if generate_voice and rag_result.response:
            voice_request = VoiceRequest(
                text=rag_result.response,
                language_code=request.get("voice_language", "en-US"),
                voice_name=request.get("voice_name"),
                # voice_name=request.get("voice_name", "en-US-Wavenet-D"),
                voice_gender=request.get("voice_gender", "NEUTRAL"),
                speaking_rate=request.get("speaking_rate", 1.0),
                pitch=request.get("pitch", 0.0),
                audio_encoding=request.get("audio_encoding", "MP3")
            )
            
            voice_result = await voice_service.text_to_speech(
                text=voice_request.text,
                language_code=voice_request.language_code,
                voice_name=voice_request.voice_name,
                voice_gender=voice_request.voice_gender,
                speaking_rate=voice_request.speaking_rate,
                pitch=voice_request.pitch,
                audio_encoding=voice_request.audio_encoding
            )
            
            # Convert QueryResponse to dict and add voice data
            result_dict = rag_result.model_dump()  # Use model_dump() for Pydantic v2
            result_dict["voice_response"] = voice_result
            
            return result_dict
        
        # Return the QueryResponse as a dictionary
        return rag_result.model_dump()  # Use model_dump() for Pydantic v2
        
    except Exception as e:
        logger.error(f"RAG with voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag-with-voice", response_model=Dict[str, Any])
async def rag_with_voice(request: Dict[str, Any]):
    """Complete RAG pipeline with automatic voice response"""
    try:
        # 1️⃣ Run the RAG pipeline
        rag_result: QueryResponse = await complete_rag_pipeline(request)
        
        # 2️⃣ Generate voice response if requested
        generate_voice = request.get("generate_voice", False)
        voice_result = None
        
        if generate_voice and rag_result.response:
            try:
                voice_result = await voice_service.text_to_speech(
                    text=rag_result.response,
                    language_code=request.get("voice_language", "en-US"),
                    voice_name=voice_service.get_voice_by_language(
                        request.get("voice_language", "en-US")
                    ),
                    voice_gender=request.get("voice_gender", "NEUTRAL"),
                    speaking_rate=request.get("speaking_rate", 1.0),
                    pitch=request.get("pitch", 0.0),
                    audio_encoding=request.get("audio_encoding", "MP3")
                )
            except Exception as e:
                logger.error(f"Voice generation failed: {e}")
                voice_result = {
                    "success": False,
                    "error": f"Voice generation failed: {str(e)}",
                    "audio_base64": None
                }
        
        # 3️⃣ Prepare response
        result_dict = rag_result.model_dump()
        if voice_result:
            result_dict["voice_response"] = voice_result
        
        return result_dict
        
    except Exception as e:
        logger.error(f"RAG with voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/v1/test-voice-auth")
async def test_voice_auth():
    """Test voice service authentication"""
    try:
        # Test if service account works
        from google.cloud import texttospeech
        client = texttospeech.TextToSpeechClient()
        
        # Try to list voices (doesn't cost anything)
        voices = client.list_voices()
        voice_count = len(voices.voices)
        
        return {
            "success": True,
            "message": "Service account authentication successful",
            "voice_count": voice_count,
            "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        }


@app.post("/api/v1/rag-with-voice")
async def rag_with_voice(req: VoiceRAGRequest):
    rag = await complete_rag_pipeline(req.model_dump())
    voice: dict | None = None
    if req.generate_voice:
        voice = await voice_service.text_to_speech(
            text=rag.response,
            language_code=req.voice_settings.language,
            voice_name=req.voice_settings.voice_name,
            voice_gender=req.voice_settings.voice_gender,
            speaking_rate=req.voice_settings.voice_speed,
            pitch=req.voice_settings.voice_pitch,
            audio_encoding="MP3"
        )
    return {
        "response": rag.response,
        "model_used": rag.model_used,
        "context_used": rag.context_used,
        "voice_response": voice,
    }



@app.get("/api/v1/documents/stats")
async def get_document_stats():
    """Get document statistics and metrics"""
    try:
        # Get basic document count
        docs_response = await list_documents()
        
        # Calculate additional statistics
        total_docs = docs_response.get("document_count", 0)
        index_info = docs_response.get("index_info", {})
        total_vectors = index_info.get("total_vectors", 0)
        
        # Get recent uploads if available
        recent_uploads_count = 0
        if hasattr(st.session_state, 'recent_uploads'):
            recent_uploads_count = len(st.session_state.recent_uploads)
        
        # Calculate storage metrics
        storage_metrics = {
            "total_documents": total_docs,
            "total_vectors": total_vectors,
            "vector_dimension": index_info.get("dimension", 1536),
            "index_type": index_info.get("index_type", "FAISS"),
            "recent_uploads": recent_uploads_count
        }
        
        # Performance metrics (you can expand these based on your needs)
        performance_metrics = {
            "avg_query_time": "0.25s",  # You can implement actual tracking
            "cache_hit_rate": "78%",
            "search_accuracy": "92%",
            "embedding_generation_time": "1.2s"
        }
        
        # Category distribution (mock data - implement based on your metadata)
        category_distribution = {
            "Research": 45,
            "Technical": 30,
            "Business": 15,
            "Other": 10
        }
        
        return {
            "success": True,
            "storage_metrics": storage_metrics,
            "performance_metrics": performance_metrics,
            "category_distribution": category_distribution,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "storage_metrics": {},
            "performance_metrics": {},
            "category_distribution": {}
        }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.debug)

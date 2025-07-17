import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.config.settings import settings
from app.models.schemas import DocumentChunk, IngestionResult

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Advanced chunking strategies
        self.text_splitters = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            ),
            "semantic": RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len
            ),
            "small": RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
        }
    
    async def process_document(self, content: str, metadata: Dict[str, Any], 
                         chunking_strategy: str = "recursive") -> IngestionResult:
        """Process document with advanced chunking and embedding"""
        try:
            # Validate input
            if not isinstance(content, str):
                raise ValueError(f"Content must be a string, got {type(content)}")
            
            if not content.strip():
                raise ValueError("Content cannot be empty")
            
            # Generate document hash for deduplication
            doc_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Select chunking strategy
            splitter = self.text_splitters.get(chunking_strategy, self.text_splitters["recursive"])
            
            # Create chunks with enhanced metadata
            chunks = await self._create_enhanced_chunks(content, metadata, splitter, doc_hash)
            
            # Generate embeddings for all chunks
            embeddings = await self._generate_embeddings([chunk.content for chunk in chunks])
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            return IngestionResult(
                success=True,
                document_hash=doc_hash,
                chunks=chunks,
                total_chunks=len(chunks),
                processing_time=datetime.now().timestamp(),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return IngestionResult(
                success=False,
                error=str(e),
                chunks=[],
                total_chunks=0
            )

        

        
    
    async def _create_enhanced_chunks(self, content: str, metadata: Dict[str, Any], 
                                    splitter, doc_hash: str) -> List[DocumentChunk]:
        """Create chunks with enhanced metadata and context"""
        text_chunks = splitter.split_text(content)
        enhanced_chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            # Create enhanced metadata for each chunk
            chunk_metadata = {
                **metadata,
                "chunk_id": f"{doc_hash}_{i}",
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "chunk_size": len(chunk_text),
                "document_hash": doc_hash,
                "created_at": datetime.now().isoformat(),
                "chunk_type": self._classify_chunk_type(chunk_text),
                "keywords": self._extract_keywords(chunk_text)
            }
            
            # Add context from neighboring chunks
            context = self._get_chunk_context(text_chunks, i)
            
            enhanced_chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                context=context,
                embedding=None  # Will be added later
            ))
        
        return enhanced_chunks
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify chunk type based on content"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ["def ", "class ", "function", "import"]):
            return "code"
        elif any(keyword in text_lower for keyword in ["table", "figure", "chart"]):
            return "data"
        elif text.count("?") > 2:
            return "faq"
        elif len(text.split()) < 50:
            return "summary"
        else:
            return "content"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract basic keywords from text"""
        # Simple keyword extraction - in production, use NLP libraries
        words = text.lower().split()
        # Filter out common words and keep longer words
        keywords = [word for word in words if len(word) > 4 and word.isalpha()]
        return list(set(keywords))[:10]  # Top 10 unique keywords
    
    def _get_chunk_context(self, all_chunks: List[str], current_index: int) -> Dict[str, str]:
        """Get context from neighboring chunks"""
        context = {}
        
        if current_index > 0:
            context["previous"] = all_chunks[current_index - 1][-200:]  # Last 200 chars
        
        if current_index < len(all_chunks) - 1:
            context["next"] = all_chunks[current_index + 1][:200]  # First 200 chars
        
        return context
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        try:
            embeddings = await self.embedding_model.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

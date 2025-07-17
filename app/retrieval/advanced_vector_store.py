import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime

from app.config.settings import settings
from app.models.schemas import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)

class AdvancedVectorStore:
    def __init__(self):
        self.dimension = 1536  # OpenAI embedding dimension
        self.chunks: List[DocumentChunk] = []
        self.metadata_index: Dict[str, Any] = {}
        
        # Initialize multiple indexes for different search strategies
        self.indexes = {
            "flat": faiss.IndexFlatIP(self.dimension),  # Exact search
            "ivf": None,  # Will be built when we have enough data
            "hnsw": faiss.IndexHNSWFlat(self.dimension, 32)  # Fast approximate search
        }
        
        # Configure HNSW parameters
        self.indexes["hnsw"].hnsw.efConstruction = 200
        self.indexes["hnsw"].hnsw.efSearch = 50
        
        self.load_index()
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks to vector store with advanced indexing"""
        try:
            if not chunks:
                return False
            
            # Extract embeddings and metadata
            embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)
            
            # Add to all indexes
            for index_name, index in self.indexes.items():
                if index is not None:
                    index.add(embeddings)
            
            # Store chunks and build metadata index
            start_idx = len(self.chunks)
            self.chunks.extend(chunks)
            
            # Build metadata index for filtering
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.metadata.get("chunk_id", f"chunk_{start_idx + i}")
                self.metadata_index[chunk_id] = {
                    "index": start_idx + i,
                    "metadata": chunk.metadata,
                    "chunk_type": chunk.metadata.get("chunk_type", "content")
                }
            
            # Build IVF index if we have enough data
            if len(self.chunks) >= 1000 and self.indexes["ivf"] is None:
                await self._build_ivf_index()
            
            self.save_index()
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            return False
    
    async def hybrid_search(self, query_embedding: List[float], 
                          query_text: str, 
                          k: int = 10,
                          search_type: str = "hybrid",
                          filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Advanced hybrid search with multiple ranking strategies"""
        try:
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Vector similarity search
            vector_results = await self._vector_search(query_vector, k * 2)
            
            # Keyword-based filtering
            keyword_results = await self._keyword_search(query_text, k * 2)
            
            # Metadata filtering
            if filters:
                vector_results = self._apply_filters(vector_results, filters)
                keyword_results = self._apply_filters(keyword_results, filters)
            
            # Hybrid ranking
            if search_type == "hybrid":
                final_results = await self._hybrid_rank(vector_results, keyword_results, query_text)
            elif search_type == "vector":
                final_results = vector_results
            elif search_type == "keyword":
                final_results = keyword_results
            else:
                final_results = vector_results
            
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _vector_search(self, query_vector: np.ndarray, k: int) -> List[RetrievalResult]:
        """Perform vector similarity search"""
        results = []
        
        # Try different indexes based on data size
        if len(self.chunks) < 1000:
            # Use flat index for small datasets
            distances, indices = self.indexes["flat"].search(query_vector, k)
        else:
            # Use HNSW for larger datasets
            distances, indices = self.indexes["hnsw"].search(query_vector, k)
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append(RetrievalResult(
                    chunk=chunk,
                    similarity_score=float(distance),
                    rank=i + 1,
                    search_type="vector"
                ))
        
        return results
    
    async def _keyword_search(self, query_text: str, k: int) -> List[RetrievalResult]:
        """Perform keyword-based search"""
        query_keywords = set(query_text.lower().split())
        keyword_scores = []
        
        for i, chunk in enumerate(self.chunks):
            # Calculate keyword overlap score
            chunk_keywords = set(chunk.metadata.get("keywords", []))
            chunk_text_words = set(chunk.content.lower().split())
            
            # Calculate different scoring metrics
            exact_matches = len(query_keywords.intersection(chunk_text_words))
            keyword_matches = len(query_keywords.intersection(chunk_keywords))
            
            # Combined score
            score = (exact_matches * 2 + keyword_matches) / len(query_keywords)
            
            if score > 0:
                keyword_scores.append((score, i))
        
        # Sort by score and return top k
        keyword_scores.sort(reverse=True)
        
        results = []
        for score, idx in keyword_scores[:k]:
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk=chunk,
                similarity_score=score,
                rank=len(results) + 1,
                search_type="keyword"
            ))
        
        return results
    
    async def _hybrid_rank(self, vector_results: List[RetrievalResult], 
                         keyword_results: List[RetrievalResult], 
                         query_text: str) -> List[RetrievalResult]:
        """Combine and rank results from different search methods"""
        # Create a unified scoring system
        combined_results = {}
        
        # Add vector results with weight
        for result in vector_results:
            chunk_id = result.chunk.metadata.get("chunk_id", "unknown")
            combined_results[chunk_id] = {
                "result": result,
                "vector_score": result.similarity_score * 0.7,  # 70% weight
                "keyword_score": 0,
                "final_score": 0
            }
        
        # Add keyword results with weight
        for result in keyword_results:
            chunk_id = result.chunk.metadata.get("chunk_id", "unknown")
            if chunk_id in combined_results:
                combined_results[chunk_id]["keyword_score"] = result.similarity_score * 0.3  # 30% weight
            else:
                combined_results[chunk_id] = {
                    "result": result,
                    "vector_score": 0,
                    "keyword_score": result.similarity_score * 0.3,
                    "final_score": 0
                }
        
        # Calculate final scores
        for chunk_id, data in combined_results.items():
            data["final_score"] = data["vector_score"] + data["keyword_score"]
        
        # Sort by final score
        sorted_results = sorted(combined_results.values(), 
                              key=lambda x: x["final_score"], 
                              reverse=True)
        
        # Return as RetrievalResult objects
        final_results = []
        for i, data in enumerate(sorted_results):
            result = data["result"]
            result.similarity_score = data["final_score"]
            result.rank = i + 1
            result.search_type = "hybrid"
            final_results.append(result)
        
        return final_results
    
    def _apply_filters(self, results: List[RetrievalResult], 
                      filters: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply metadata filters to search results"""
        filtered_results = []
        
        for result in results:
            metadata = result.chunk.metadata
            include = True
            
            for filter_key, filter_value in filters.items():
                if filter_key in metadata:
                    if isinstance(filter_value, list):
                        if metadata[filter_key] not in filter_value:
                            include = False
                            break
                    else:
                        if metadata[filter_key] != filter_value:
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    async def _build_ivf_index(self):
        """Build IVF index for large datasets"""
        try:
            # Create IVF index with clustering
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(100, len(self.chunks) // 10)  # Adaptive number of clusters
            
            self.indexes["ivf"] = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Get all embeddings
            embeddings = np.array([chunk.embedding for chunk in self.chunks], dtype=np.float32)
            
            # Train the index
            self.indexes["ivf"].train(embeddings)
            self.indexes["ivf"].add(embeddings)
            
            logger.info(f"Built IVF index with {nlist} clusters")
            
        except Exception as e:
            logger.error(f"Failed to build IVF index: {e}")
    
    def save_index(self):
        """Save vector store to disk"""
        try:
            import os
            os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
            
            # Save FAISS indexes
            for index_name, index in self.indexes.items():
                if index is not None:
                    faiss.write_index(index, f"{settings.faiss_index_path}_{index_name}.index")
            
            # Save chunks and metadata
            with open(f"{settings.faiss_index_path}_chunks.json", 'w') as f:
                # Convert chunks to serializable format
                serializable_chunks = []
                for chunk in self.chunks:
                    serializable_chunks.append({
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "context": chunk.context,
                        "embedding": chunk.embedding
                    })
                json.dump(serializable_chunks, f)
            
            with open(f"{settings.faiss_index_path}_metadata.json", 'w') as f:
                json.dump(self.metadata_index, f)
            
            logger.info("Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def load_index(self):
        """Load vector store from disk"""
        try:
            import os
            
            # Load FAISS indexes
            for index_name in self.indexes.keys():
                index_path = f"{settings.faiss_index_path}_{index_name}.index"
                if os.path.exists(index_path):
                    self.indexes[index_name] = faiss.read_index(index_path)
            
            # Load chunks
            chunks_path = f"{settings.faiss_index_path}_chunks.json"
            if os.path.exists(chunks_path):
                with open(chunks_path, 'r') as f:
                    chunk_data = json.load(f)
                    self.chunks = [DocumentChunk(**data) for data in chunk_data]
            
            # Load metadata
            metadata_path = f"{settings.faiss_index_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata_index = json.load(f)
            
            logger.info(f"Loaded vector store with {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
# Updated import - use langchain-openai instead of langchain-community
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config.settings import settings
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(
            api_key=settings.openai_api_key,  # Use api_key instead of openai_api_key
            model=settings.embedding_model
        )
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = 1536  # OpenAI embedding dimension
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.load_index()
    
    
    def load_index(self):
        """Load existing FAISS index or create new one"""
        try:
            if os.path.exists(f"{settings.faiss_index_path}.index"):
                self.index = faiss.read_index(f"{settings.faiss_index_path}.index")
                with open(f"{settings.faiss_index_path}.metadata", 'rb') as f:
                    self.metadata = pickle.load(f)
                with open(f"{settings.faiss_index_path}.documents", 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def save_index(self):
        """Save FAISS index to disk"""
        try:
            os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
            faiss.write_index(self.index, f"{settings.faiss_index_path}.index")
            with open(f"{settings.faiss_index_path}.metadata", 'wb') as f:
                pickle.dump(self.metadata, f)
            with open(f"{settings.faiss_index_path}.documents", 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    async def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector store"""
        try:
            all_chunks = []
            all_metadata = []
            
            for i, doc in enumerate(documents):
                chunks = self.text_splitter.split_text(doc)
                all_chunks.extend(chunks)
                
                doc_metadata = metadata[i] if metadata else {}
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "source_doc_index": i,
                        "chunk_text": chunk
                    }
                    all_metadata.append(chunk_metadata)
            
            # Generate embeddings
            embeddings = await self.embedding_model.aembed_documents(all_chunks)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            self.documents.extend(all_chunks)
            self.metadata.extend(all_metadata)
            
            self.save_index()
            logger.info(f"Added {len(all_chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = await self.embedding_model.aembed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search FAISS index
            distances, indices = self.index.search(query_vector, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.metadata[idx],
                        "similarity_score": float(distances[0][i])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

# src/backend/app/services/retrieval/bm25_retriever.py

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import asyncio
import pickle
import os
from src.backend.app.config import settings
from src.backend.app.utils.logger import get_logger
import re
from underthesea import word_tokenize

logger = get_logger(__name__)

class BM25Retriever:
    def __init__(self):
        self.bm25_index = None
        self.documents = []
        self.k1 = settings.BM25_K1
        self.b = settings.BM25_B
        self.index_path = "data/bm25_index.pkl"
        self.docs_path = "data/bm25_docs.pkl"
        self._index_ready = False
        self._load_or_build_index()
    
    def _load_or_build_index(self):
        """Load existing BM25 index or mark for building"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
                logger.info("Loading existing BM25 index ...")
                self._load_index()
                self._index_ready = True
            else:
                logger.info("BM25 index not found, will build on first use")
                self._index_ready = False
        
        except Exception as e:
            logger.error(f"Error loading BM25 index: {str(e)}")
            self._index_ready = False
    
    def _load_index(self):
        """Load BM25 index from disk"""
        try:
            with open(self.index_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Loaded BM25 index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading BM25 index files: {str(e)}")
            raise
    
    def _save_index(self):
        """Save BM25 index to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info("BM25 index saved successfully")
        except Exception as e:
            logger.error(f"Error saving BM25 index: {str(e)}")
            raise
    
    async def _build_index_from_qdrant(self):
        """Build BM25 index from documents in Qdrant"""
        try:
            # Import here to avoid circular imports
            from src.backend.app.services.retrieval.qdrant_retriever import QdrantRetriever
            
            logger.info("Building BM25 index from Qdrant...")
            qdrant = QdrantRetriever()
            
            # Get all documents from Qdrant with pagination
            all_points = []
            offset = None
            
            while True:
                search_results = await qdrant.client.scroll(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    limit=1000,  # Adjust based on memory constraints
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # We don't need vectors for BM25
                )
                
                points, next_offset = search_results
                all_points.extend(points)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            if not all_points:
                logger.warning("No documents found in Qdrant to build BM25 index")
                return False
            
            documents = []
            tokenized_docs = []
            
            for point in all_points:
                content = point.payload.get("content", "")
                if not content.strip():  # Skip empty documents
                    continue
                    
                doc = {
                    "content": content,
                    "metadata": point.payload.get("metadata", {}),
                    "source": point.payload.get("source", "unknown"),
                    "id": str(point.id),
                    "chunk_index": point.payload.get("chunk_index", 0)
                }
                documents.append(doc)
                
                # Tokenize document for BM25
                tokens = self._tokenize_text(content)
                if tokens:  # Only add if we have tokens
                    tokenized_docs.append(tokens)
                else:
                    # Remove the document if no valid tokens
                    documents.pop()
            
            if not tokenized_docs:
                logger.error("No valid documents found for BM25 indexing")
                return False
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
            self.documents = documents
            
            # Save index
            self._save_index()
            self._index_ready = True
            
            logger.info(f"Built BM25 index with {len(documents)} documents")
            return True
        
        except Exception as e:
            logger.error(f"Error building BM25 index: {str(e)}")
            self._index_ready = False
            return False
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text for BM25
        """
        try:
            if not text or not text.strip():
                return []
            
            # Clean text
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                return []
            
            # Use underthesea for Vietnamese tokenization
            tokens = word_tokenize(text, format="list")
            
            # Filter out short tokens and numbers
            tokens = [
                token.strip() for token in tokens 
                if len(token.strip()) > 1 and not token.strip().isdigit()
            ]
            
            return tokens
        
        except Exception as e:
            logger.error(f"Error tokenizing text: {str(e)}")
            return []
    
    async def _ensure_index_ready(self):
        """Ensure BM25 index is ready for use"""
        if not self._index_ready:
            success = await self._build_index_from_qdrant()
            if not success:
                raise Exception("Failed to build BM25 index")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 scoring
        """
        try:
            # Ensure index is ready
            await self._ensure_index_ready()
            
            if not self.bm25_index or not self.documents:
                logger.warning("BM25 index not available")
                return []
            
            logger.info(f"BM25 search for: '{query[:50]}...'")
            
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            
            if not query_tokens:
                logger.warning("No valid tokens found in query")
                return []
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Create results with scores
            results = []
            for i, score in enumerate(scores):
                if score > score_threshold:
                    doc = self.documents[i].copy()
                    doc["score"] = float(score)
                    results.append(doc)
            
            # Sort by score descending and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
            
            logger.info(f"Found {len(results)} BM25 matches (threshold: {score_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {str(e)}")
            return []
    
    async def health_check(self) -> bool:
        """Check if BM25 retriever is healthy"""
        try:
            await self._ensure_index_ready()
            
            if not self.bm25_index or not self.documents:
                return False
            
            # Test with simple query
            test_results = await self.retrieve("test", top_k=1)
            return True
            
        except Exception as e:
            logger.error(f"BM25 health check failed: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index"""
        if not self.bm25_index or not self.documents:
            return {
                "status": "not_loaded", 
                "doc_count": 0,
                "index_ready": self._index_ready
            }
        
        return {
            "status": "loaded",
            "doc_count": len(self.documents),
            "avg_doc_length": getattr(self.bm25_index, 'avgdl', 0),
            "k1": self.k1,
            "b": self.b,
            "index_ready": self._index_ready
        }
    
    async def rebuild_index(self) -> bool:
        """Manually rebuild the BM25 index"""
        logger.info("Manually rebuilding BM25 index...")
        self._index_ready = False
        return await self._build_index_from_qdrant()
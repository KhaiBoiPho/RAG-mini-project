# src/backend/app/services/retrieval/bm25_retriever.py
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import asyncio
import pickle
import os
from ...config import settings
from ...utils.logger import get_logger
import jieba
import re

logger = get_logger(__name__)

class BM25Retriever:
    def __init__(self):
        self.bm25_index = None
        self.documents = []
        self.k1 = settings.bm25_k1
        self.b = settings.bm25_b
        self.index_path = "data/bm25_index.pkl"
        self.docs_path = "data/bm25_docs.pkl"
        self._load_or_build_index()
    
    def _load_or_build_index(self):
        """Load existing BM25 index or build new one"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
                logger.info("Loading existing BM25 index...")
                self._load_index()
            else:
                logger.info("Building new BM25 index...")
                asyncio.create_task(self._build_index_from_qdrant())
        except Exception as e:
            logger.error(f"Error with BM25 index: {str(e)}")
    
    def _load_index(self):
        """Load BM25 index from disk"""
        with open(self.index_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        with open(self.docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        logger.info(f"Loaded BM25 index with {len(self.documents)} documents")
    
    def _save_index(self):
        """Save BM25 index to disk"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    async def _build_index_from_qdrant(self):
        """Build BM25 index from documents in Qdrant"""
        try:
            from .qdrant_retriever import QdrantRetriever
            qdrant = QdrantRetriever()
            
            # Get all documents from Qdrant
            # Note: In production, implement pagination for large collections
            search_results = await qdrant.client.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=10000  # Adjust based on your collection size
            )
            
            documents = []
            tokenized_docs = []
            
            for point in search_results[0]:
                doc = {
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "source": point.payload.get("source", "unknown"),
                    "id": str(point.id)
                }
                documents.append(doc)
                
                # Tokenize document for BM25
                tokens = self._tokenize_text(doc["content"])
                tokenized_docs.append(tokens)
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
            self.documents = documents
            
            # Save index
            self._save_index()
            
            logger.info(f"Built BM25 index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {str(e)}")
            raise
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text for BM25
        """
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use jieba for Vietnamese tokenization (you might want to use a Vietnamese-specific tokenizer)
        tokens = list(jieba.cut(text))
        
        # Filter out short tokens and numbers
        tokens = [token for token in tokens if len(token) > 1 and not token.isdigit()]
        
        return tokens
    
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
            if not self.bm25_index or not self.documents:
                logger.warning("BM25 index not available")
                return []
            
            logger.info(f"BM25 search for: {query[:50]}...")
            
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            
            if not query_tokens:
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
            
            logger.info(f"Found {len(results)} BM25 matches")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {str(e)}")
            return []
    
    async def health_check(self) -> bool:
        """Check if BM25 retriever is healthy"""
        try:
            if not self.bm25_index or not self.documents:
                return False
            
            # Test with simple query
            test_results = await self.retrieve("test", top_k=1)
            return True
            
        except Exception as e:
            logger.error(f"BM25 health check failed: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index"""
        if not self.bm25_index or not self.documents:
            return {"status": "not_loaded", "doc_count": 0}
        
        return {
            "status": "loaded",
            "doc_count": len(self.documents),
            "avg_doc_length": self.bm25_index.avgdl,
            "k1": self.k1,
            "b": self.b
        }
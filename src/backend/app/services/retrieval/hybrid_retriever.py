# src/backend/app/services/retrieval/hybrid_retriever.py
from typing import List, Dict, Any, Optional
from .qdrant_retriever import QdrantRetriever
from .bm25_retriever import BM25Retriever
from .cache_service import CacheService
from ...config import settings
from ...utils.logger import get_logger
import asyncio

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self):
        self.vector_retriever = QdrantRetriever()
        self.bm25_retriever = BM25Retriever()
        self.cache_service = CacheService()
        self.vector_weight = settings.vector_weight
        self.bm25_weight = settings.bm25_weight
        
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid approach (vector + BM25)
        """
        try:
            top_k = top_k or settings.top_k
            score_threshold = score_threshold or settings.score_threshold
            
            # Check cache first
            if use_cache:
                cached_result = await self.cache_service.get(query)
                if cached_result:
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return cached_result
            
            logger.info(f"Performing hybrid retrieval for: {query[:50]}...")
            
            # Run both retrievers in parallel
            vector_task = self.vector_retriever.retrieve(query, top_k * 2)
            bm25_task = self.bm25_retriever.retrieve(query, top_k * 2)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_task, 
                bm25_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.error(f"Vector retrieval failed: {vector_results}")
                vector_results = []
            if isinstance(bm25_results, Exception):
                logger.error(f"BM25 retrieval failed: {bm25_results}")
                bm25_results = []
            
            # Combine and re-rank results
            combined_results = self._hybrid_ranking(
                vector_results, 
                bm25_results, 
                query
            )
            
            # Filter by score threshold and limit results
            filtered_results = [
                doc for doc in combined_results 
                if doc["score"] >= score_threshold
            ][:top_k]
            
            # Cache the results
            if use_cache and filtered_results:
                await self.cache_service.set(query, filtered_results)
            
            logger.info(f"Retrieved {len(filtered_results)} documents")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            raise
    
    def _hybrid_ranking(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Combine and re-rank results from vector and BM25 retrievers
        """
        # Create a map of document content to combined scores
        doc_scores = {}
        
        # Add vector scores
        for doc in vector_results:
            content = doc["content"]
            vector_score = doc["score"]
            
            if content not in doc_scores:
                doc_scores[content] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "bm25_score": 0.0
                }
            
            doc_scores[content]["vector_score"] = vector_score
        
        # Add BM25 scores
        for doc in bm25_results:
            content = doc["content"]
            bm25_score = doc["score"]
            
            if content not in doc_scores:
                doc_scores[content] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "bm25_score": 0.0
                }
            
            doc_scores[content]["bm25_score"] = bm25_score
        
        # Calculate hybrid scores and create final results
        final_results = []
        for content, scores in doc_scores.items():
            # Normalize scores (assuming they're already between 0-1)
            vector_score = scores["vector_score"]
            bm25_score = scores["bm25_score"]
            
            # Weighted combination
            hybrid_score = (
                self.vector_weight * vector_score + 
                self.bm25_weight * bm25_score
            )
            
            # Create result document
            result_doc = scores["doc"].copy()
            result_doc["score"] = hybrid_score
            result_doc["scores_breakdown"] = {
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "hybrid_score": hybrid_score
            }
            
            final_results.append(result_doc)
        
        # Sort by hybrid score descending
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return final_results
    
    async def health_check(self) -> bool:
        """Check if retriever is healthy"""
        try:
            # Test with a simple query
            test_results = await self.retrieve("test", top_k=1, use_cache=False)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise
# src/backend/app/services/retrieval/hybrid_retriever.py

from typing import List, Dict, Any, Tuple, Optional
import asyncio
import math
from src.backend.app.config import settings
from src.backend.app.utils.logger import get_logger
from src.backend.app.services.retrieval.qdrant_retriever import QdrantRetriever
from src.backend.app.services.retrieval.bm25_retriever import BM25Retriever

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining vector search and BM25 for better results
    """
    
    def __init__(self):
        self.qdrant_retriever = QdrantRetriever()
        self.bm25_retriever = BM25Retriever()
        
        # Weights for combining scores
        self.vector_weight = getattr(settings, 'VECTOR_WEIGHT', 0.7)
        self.bm25_weight = getattr(settings, 'BM25_WEIGHT', 0.3)
        
        # Ensure weights sum to 1
        total_weight = self.vector_weight + self.bm25_weight
        if total_weight != 1.0:
            self.vector_weight = self.vector_weight / total_weight
            self.bm25_weight = self.bm25_weight / total_weight
        
        # Minimum scores
        self.min_vector_score = getattr(settings, 'MIN_VECTOR_SCORE', 0.1)
        self.min_bm25_score = getattr(settings, 'MIN_BM25_SCORE', 0.1)
        
        # RRF parameter
        self.rrf_k = getattr(settings, 'RRF_K', 60)
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        method: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve relevant documents using hybrid approach
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: "hybrid", "vector", or "bm25"
            filters: Optional filters for vector search
            
        Returns:
            Tuple of (documents, method_used)
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided")
                return [], method
            
            logger.info(f"Retrieving documents for query: '{query[:50]}...' using {method}")
            
            if method == "vector":
                docs = await self._vector_search(query, top_k, filters)
                return docs, "vector"
            
            elif method == "bm25":
                docs = await self._bm25_search(query, top_k)
                return docs, "bm25"
            
            else:  # hybrid
                docs = await self._hybrid_search(query, top_k, filters)
                return docs, "hybrid"
                
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return [], method
    
    async def _hybrid_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Combine vector and BM25 search results"""
        try:
            # Get more results for better fusion
            search_multiplier = max(2, min(5, top_k))
            vector_limit = top_k * search_multiplier
            bm25_limit = top_k * search_multiplier
            
            # Run both searches concurrently
            tasks = [
                self._vector_search(query, vector_limit, filters),
                self._bm25_search(query, bm25_limit)
            ]
            
            try:
                vector_docs, bm25_docs = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                if isinstance(vector_docs, Exception):
                    logger.error(f"Vector search failed: {vector_docs}")
                    vector_docs = []
                if isinstance(bm25_docs, Exception):
                    logger.error(f"BM25 search failed: {bm25_docs}")
                    bm25_docs = []
                
            except Exception as e:
                logger.error(f"Error in concurrent searches: {str(e)}")
                vector_docs, bm25_docs = [], []
            
            # If both failed, return empty
            if not vector_docs and not bm25_docs:
                logger.warning("Both vector and BM25 searches failed")
                return []
            
            # If only one succeeded, return that result
            if not vector_docs:
                logger.info("Using BM25 results only")
                return bm25_docs[:top_k]
            
            if not bm25_docs:
                logger.info("Using vector results only")
                return vector_docs[:top_k]
            
            # Combine and re-rank results
            combined_docs = self._fusion_ranking(vector_docs, bm25_docs)
            
            logger.info(f"Hybrid search completed: {len(combined_docs)} results")
            return combined_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to vector search only
            try:
                return await self._vector_search(query, top_k, filters)
            except:
                return []
    
    async def _vector_search(
        self, 
        query: str, 
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Pure vector search"""
        try:
            results = await self.qdrant_retriever.retrieve(
                query=query, 
                top_k=top_k,
                filters=filters
            )
            
            # Filter by minimum score and ensure valid structure
            filtered_results = []
            for doc in results:
                score = doc.get("score", 0)
                if score >= self.min_vector_score and doc.get("content"):
                    # Normalize vector scores (typically 0-1 range)
                    doc["normalized_score"] = min(1.0, max(0.0, score))
                    filtered_results.append(doc)
            
            logger.info(f"Vector search: {len(filtered_results)} results after filtering")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    async def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Pure BM25 search"""
        try:
            results = await self.bm25_retriever.retrieve(
                query=query, 
                top_k=top_k,
                score_threshold=self.min_bm25_score
            )
            
            # Normalize BM25 scores
            if results:
                max_score = max(doc.get("score", 0) for doc in results)
                if max_score > 0:
                    for doc in results:
                        original_score = doc.get("score", 0)
                        # Normalize to 0-1 range
                        doc["normalized_score"] = original_score / max_score
                else:
                    for doc in results:
                        doc["normalized_score"] = 0
            
            logger.info(f"BM25 search: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def _fusion_ranking(
        self, 
        vector_docs: List[Dict[str, Any]], 
        bm25_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine and re-rank documents using Reciprocal Rank Fusion (RRF) + Score Fusion
        """
        try:
            # Create document registry
            doc_registry = {}
            
            # Process vector search results
            for rank, doc in enumerate(vector_docs):
                doc_id = self._get_doc_id(doc)
                doc_registry[doc_id] = {
                    "doc": doc,
                    "vector_rank": rank + 1,
                    "vector_score": doc.get("normalized_score", 0),
                    "bm25_rank": None,
                    "bm25_score": 0
                }
            
            # Process BM25 search results
            for rank, doc in enumerate(bm25_docs):
                doc_id = self._get_doc_id(doc)
                if doc_id in doc_registry:
                    # Document found in both searches
                    doc_registry[doc_id]["bm25_rank"] = rank + 1
                    doc_registry[doc_id]["bm25_score"] = doc.get("normalized_score", 0)
                else:
                    # Document only in BM25 results
                    doc_registry[doc_id] = {
                        "doc": doc,
                        "vector_rank": None,
                        "vector_score": 0,
                        "bm25_rank": rank + 1,
                        "bm25_score": doc.get("normalized_score", 0)
                    }
            
            # Calculate fusion scores
            fusion_results = []
            for doc_id, info in doc_registry.items():
                # Reciprocal Rank Fusion (RRF)
                rrf_score = 0
                
                if info["vector_rank"] is not None:
                    rrf_score += 1.0 / (self.rrf_k + info["vector_rank"])
                
                if info["bm25_rank"] is not None:
                    rrf_score += 1.0 / (self.rrf_k + info["bm25_rank"])
                
                # Score fusion (weighted normalized scores)
                score_fusion = (
                    self.vector_weight * info["vector_score"] + 
                    self.bm25_weight * info["bm25_score"]
                )
                
                # Combined final score (RRF + Score fusion)
                final_score = 0.5 * rrf_score + 0.5 * score_fusion
                
                # Create result document
                doc = info["doc"].copy()
                doc.update({
                    "score": final_score,
                    "fusion_score": final_score,
                    "rrf_score": rrf_score,
                    "score_fusion": score_fusion,
                    "vector_score": info["vector_score"],
                    "bm25_score": info["bm25_score"],
                    "found_in": self._get_search_sources(info)
                })
                
                fusion_results.append(doc)
            
            # Sort by final score
            fusion_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Fusion ranking: {len(fusion_results)} documents combined")
            return fusion_results
            
        except Exception as e:
            logger.error(f"Error in fusion ranking: {str(e)}")
            # Fallback to vector results
            return vector_docs[:len(bm25_docs) + len(vector_docs)]
    
    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """Get unique identifier for document"""
        # Try different ID fields in order of preference
        for field in ["id", "chunk_id", "doc_id"]:
            if field in doc:
                return str(doc[field])
            
            # Check in metadata
            metadata = doc.get("metadata", {})
            if field in metadata:
                return str(metadata[field])
        
        # Fallback to source + content hash for uniqueness
        source = doc.get("source", "unknown")
        content = doc.get("content", "")
        chunk_index = doc.get("chunk_index", 0)
        
        # Create stable ID from source and chunk info
        return f"{source}#{chunk_index}#{hash(content[:200]) % 10000}"
    
    def _get_search_sources(self, info: Dict[str, Any]) -> List[str]:
        """Determine which search methods found this document"""
        sources = []
        if info["vector_rank"] is not None:
            sources.append("vector")
        if info["bm25_rank"] is not None:
            sources.append("bm25")
        return sources
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of retrieval components"""
        try:
            results = await asyncio.gather(
                self.qdrant_retriever.health_check(),
                self.bm25_retriever.health_check(),
                return_exceptions=True
            )
            
            vector_healthy = results[0] if not isinstance(results[0], Exception) else False
            bm25_healthy = results[1] if not isinstance(results[1], Exception) else False
            
            return {
                "vector_search": vector_healthy,
                "bm25_search": bm25_healthy,
                "overall": vector_healthy and bm25_healthy,
                "can_fallback": vector_healthy or bm25_healthy
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "vector_search": False,
                "bm25_search": False,
                "overall": False,
                "can_fallback": False
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        try:
            # Get stats from both retrievers
            tasks = []
            
            # Vector stats
            if hasattr(self.qdrant_retriever, 'get_collection_info'):
                tasks.append(self.qdrant_retriever.get_collection_info())
            else:
                tasks.append(asyncio.sleep(0))  # No-op
            
            # BM25 stats
            if hasattr(self.bm25_retriever, 'get_index_stats'):
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(self.bm25_retriever.get_index_stats)
                ))
            else:
                tasks.append(asyncio.sleep(0))  # No-op
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            vector_stats = results[0] if not isinstance(results[0], Exception) else {}
            bm25_stats = results[1] if not isinstance(results[1], Exception) else {}
            
            return {
                "vector_search": vector_stats,
                "bm25_search": bm25_stats,
                "fusion_config": {
                    "vector_weight": self.vector_weight,
                    "bm25_weight": self.bm25_weight,
                    "rrf_k": self.rrf_k,
                    "min_vector_score": self.min_vector_score,
                    "min_bm25_score": self.min_bm25_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "error": str(e),
                "fusion_config": {
                    "vector_weight": self.vector_weight,
                    "bm25_weight": self.bm25_weight,
                    "rrf_k": self.rrf_k
                }
            }
    
    def update_weights(self, vector_weight: float, bm25_weight: float):
        """Update fusion weights dynamically"""
        total = vector_weight + bm25_weight
        if total > 0:
            self.vector_weight = vector_weight / total
            self.bm25_weight = bm25_weight / total
            logger.info(f"Updated weights - Vector: {self.vector_weight:.2f}, BM25: {self.bm25_weight:.2f}")
        else:
            logger.warning("Invalid weights provided, keeping current values")
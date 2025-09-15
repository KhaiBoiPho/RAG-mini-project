# src/backend/app/services/graph/rag_graph.py
from typing import List, Dict, Any, Optional
from ..retrieval.hybrid_retriever import HybridRetriever
from ..generation.openai_service import OpenAIService
from ...models.chat_models import Message
from ...utils.logger import get_logger
import asyncio

logger = get_logger(__name__)

class RAGGraph:
    """
    Orchestrates the RAG pipeline with different strategies
    """
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = OpenAIService()
    
    async def process(
        self,
        query: str,
        conversation_history: List[Message],
        retrieval_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main processing pipeline for RAG
        """
        try:
            logger.info("Starting RAG graph processing")
            
            # Step 1: Query analysis and preprocessing
            processed_query = await self._preprocess_query(query, conversation_history)
            
            # Step 2: Retrieval strategy selection
            retrieval_strategy = self._select_retrieval_strategy(
                processed_query, 
                retrieval_config
            )
            
            # Step 3: Document retrieval
            retrieved_docs = await self._retrieve_documents(
                processed_query, 
                retrieval_strategy
            )
            
            # Step 4: Document relevance filtering and reranking
            filtered_docs = await self._filter_and_rerank(
                processed_query, 
                retrieved_docs
            )
            
            # Step 5: Response generation
            response = await self._generate_response(
                query, 
                filtered_docs, 
                conversation_history
            )
            
            # Step 6: Response post-processing
            final_response = await self._postprocess_response(
                response, 
                filtered_docs
            )
            
            return {
                "response": final_response,
                "retrieved_documents": [
                    {
                        "content": doc["content"],
                        "score": doc["score"],
                        "metadata": doc.get("metadata", {}),
                        "source": doc.get("source")
                    }
                    for doc in filtered_docs
                ],
                "processing_steps": {
                    "processed_query": processed_query,
                    "retrieval_strategy": retrieval_strategy,
                    "docs_retrieved": len(retrieved_docs),
                    "docs_filtered": len(filtered_docs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG graph processing: {str(e)}")
            raise
    
    async def _preprocess_query(
        self, 
        query: str, 
        conversation_history: List[Message]
    ) -> str:
        """
        Preprocess and potentially expand the query based on context
        """
        try:
            # Simple preprocessing for now - can be enhanced with query expansion
            processed = query.strip()
            
            # If we have conversation history, we might want to add context
            if conversation_history:
                # For now, just use the original query
                # TODO: Implement query expansion based on conversation context
                pass
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in query preprocessing: {str(e)}")
            return query
    
    def _select_retrieval_strategy(
        self, 
        query: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Select retrieval strategy based on query characteristics
        """
        # Default strategy
        strategy = {
            "method": "hybrid",  # hybrid, vector_only, bm25_only
            "top_k": config.get("top_k", 5),
            "score_threshold": config.get("score_threshold", 0.7),
            "vector_weight": config.get("vector_weight", 0.7),
            "bm25_weight": config.get("bm25_weight", 0.3)
        }
        
        # Query analysis for strategy selection
        query_length = len(query.split())
        
        if query_length < 3:
            # Short queries might benefit from BM25
            strategy["bm25_weight"] = 0.6
            strategy["vector_weight"] = 0.4
        elif query_length > 20:
            # Long queries might benefit more from vector similarity
            strategy["vector_weight"] = 0.8
            strategy["bm25_weight"] = 0.2
        
        # Check for specific legal terms or patterns
        legal_keywords = ["điều", "khoản", "luật", "nghị định", "quy định", "pháp luật"]
        if any(keyword in query.lower() for keyword in legal_keywords):
            # Legal-specific queries might need different handling
            strategy["top_k"] = min(strategy["top_k"] + 2, 10)
        
        return strategy
    
    async def _retrieve_documents(
        self, 
        query: str, 
        strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on selected strategy
        """
        try:
            if strategy["method"] == "hybrid":
                return await self.retriever.retrieve(
                    query=query,
                    top_k=strategy["top_k"],
                    score_threshold=strategy["score_threshold"]
                )
            elif strategy["method"] == "vector_only":
                return await self.retriever.vector_retriever.retrieve(
                    query=query,
                    top_k=strategy["top_k"],
                    score_threshold=strategy["score_threshold"]
                )
            elif strategy["method"] == "bm25_only":
                return await self.retriever.bm25_retriever.retrieve(
                    query=query,
                    top_k=strategy["top_k"]
                )
            else:
                logger.warning(f"Unknown retrieval method: {strategy['method']}")
                return await self.retriever.retrieve(query=query)
                
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            raise
    
    async def _filter_and_rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter and rerank retrieved documents
        """
        try:
            # Simple filtering for now
            filtered_docs = []
            
            for doc in documents:
                # Filter by minimum content length
                if len(doc["content"]) < 50:
                    continue
                
                # Filter duplicates based on content similarity
                is_duplicate = False
                for existing_doc in filtered_docs:
                    if self._is_similar_content(doc["content"], existing_doc["content"]):
                        # Keep the one with higher score
                        if doc["score"] > existing_doc["score"]:
                            filtered_docs.remove(existing_doc)
                        else:
                            is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_docs.append(doc)
            
            # Sort by score descending
            filtered_docs.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Filtered {len(documents)} -> {len(filtered_docs)} documents")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in document filtering: {str(e)}")
            return documents
    
    def _is_similar_content(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """
        Simple content similarity check
        """
        # Simple Jaccard similarity for now
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold
    
    async def _generate_response(
        self,
        original_query: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Message]
    ) -> str:
        """
        Generate response using retrieved documents
        """
        try:
            response = await self.generator.generate(
                query=original_query,
                context_docs=documents,
                conversation_history=conversation_history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            raise
    
    async def _postprocess_response(
        self, 
        response: str, 
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Post-process the generated response
        """
        try:
            # Simple post-processing
            processed_response = response.strip()
            
            # Add document references if not present
            if documents and "tài liệu" not in processed_response.lower():
                processed_response += f"\n\n(Dựa trên {len(documents)} tài liệu liên quan)"
            
            return processed_response
            
        except Exception as e:
            logger.error(f"Error in response post-processing: {str(e)}")
            return response
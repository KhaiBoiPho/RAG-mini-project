# src/backend/app/services/pipeline_service.py
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from src.backend.app.services.retrieval.hybrid_retriever import HybridRetriever
from src.backend.app.services.retrieval.qdrant_retriever import QdrantRetriever
from src.backend.app.services.retrieval.bm25_retriever import BM25Retriever
from src.backend.app.services.retrieval.cache_service import CacheService
from src.backend.app.services.generation.openai_service import OpenAIService
from src.backend.app.models.chat_models import Message, SearchResult
from src.backend.app.config import settings
from src.backend.app.utils.logger import get_logger
import asyncio
import uuid
import time
from datetime import datetime

logger = get_logger(__name__)

class PipelineService:
    def __init__(self):
        # Initialize all retrieval services
        self.hybrid_retriever = HybridRetriever()
        self.qdrant_retriever = QdrantRetriever()
        self.bm25_retriever = BM25Retriever()
        self.cache_service = CacheService()
        
        # Generation service
        self.generator = OpenAIService()
        
        # Conversation storage (in production, use proper database)
        self.conversation_store = {}
        
        # Configuration
        self.default_top_k = getattr(settings, 'DEFAULT_TOP_K', 5)
        self.enable_caching = getattr(settings, 'ENABLE_CACHE', True)
        self.retrieval_timeout = getattr(settings, 'RETRIEVAL_TIMEOUT', 30)
        
        logger.info("Pipeline service initialized with all retrievers")
    
    async def process_query(
        self,
        query: str,
        conversation_id: str,
        method: str = "hybrid",
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        use_history: bool = True,
        stream_response: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline
        """
        start_time = time.time()
        top_k = top_k or self.default_top_k
        
        # Initialize default metadata
        metadata = {
            "retrieval_method": "unknown",
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0,
            "document_count": 0,
            "error": False
        }
        
        try:
            logger.info(f"Processing query for conversation: {conversation_id}, method: {method}")
            
            # Get conversation history if needed
            conversation_history = []
            if use_history and conversation_id in self.conversation_store:
                conversation_history = self.conversation_store[conversation_id][-10:]  # Last 10 messages
            
            # Retrieve relevant documents
            retrieved_docs, retrieval_method, retrieval_time = await self._retrieve_documents(
                query=query,
                method=method,
                top_k=top_k,
                filters=filters,
                use_cache=use_cache and self.enable_caching
            )
            
            # Update metadata
            metadata.update({
                "retrieval_method": retrieval_method,
                "retrieval_time": retrieval_time,
                "document_count": len(retrieved_docs)
            })
            
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query: {query[:50]}...")
                metadata["total_time"] = time.time() - start_time
                return {
                    "response": "Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn.",
                    "retrieved_documents": [],
                    "conversation_id": conversation_id,
                    "metadata": metadata
                }
            
            # Generate response
            generation_start = time.time()
            
            if stream_response:
                # Return generator for streaming
                return self._stream_response(
                    query=query,
                    retrieved_docs=retrieved_docs,
                    conversation_history=conversation_history,
                    conversation_id=conversation_id,
                    metadata=metadata
                )
            else:
                # Generate complete response
                logger.info("Starting response generation...")
                
                generation_response = await self.generator.generate(
                    query=query,
                    context_docs=retrieved_docs,
                    conversation_history=conversation_history
                )
                
                logger.info(f"Generation response keys: {list(generation_response.keys())}")
                logger.info(f"Generated content length: {len(generation_response.get('content', ''))}")
                
                generation_time = time.time() - generation_start
                total_time = time.time() - start_time
                
                # Update final metadata
                metadata.update({
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "model_used": generation_response.get("model", "unknown"),
                    "finish_reason": generation_response.get("finish_reason", "unknown")
                })
                
                response_content = generation_response.get("content", "Không thể tạo phản hồi.")
                
                # Update conversation history
                self._update_conversation_history(conversation_id, query, response_content)
                
                return {
                    "response": response_content,
                    "retrieved_documents": [
                        SearchResult(
                            content=doc["content"],
                            score=doc["score"],
                            metadata=doc.get("metadata", {}),
                            source=doc.get("source", "unknown")
                        )
                        for doc in retrieved_docs
                    ],
                    "conversation_id": conversation_id,
                    "metadata": metadata
                }
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}")
            metadata.update({
                "total_time": time.time() - start_time,
                "error": True,
                "error_message": str(e)
            })
            
            return {
                "response": "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý. Vui lòng thử lại.",
                "retrieved_documents": [],
                "conversation_id": conversation_id,
                "metadata": metadata
            }
    
    async def _retrieve_documents(
        self,
        query: str,
        method: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        use_cache: bool
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Retrieve documents using specified method with caching
        """
        retrieval_start = time.time()
        
        try:
            # Check cache first
            cached_results = None
            if use_cache:
                cached_results = await self.cache_service.get_cached_results(
                    query=query,
                    method=method,
                    filters=filters,
                    top_k=top_k
                )
                
                if cached_results:
                    retrieval_time = time.time() - retrieval_start
                    logger.info(f"Retrieved {len(cached_results)} cached documents")
                    return cached_results, f"{method}_cached", retrieval_time
            
            # Retrieve from appropriate service
            retrieved_docs = []
            actual_method = method
            
            try:
                # Add timeout for retrieval
                retrieval_task = self._perform_retrieval(query, method, top_k, filters)
                retrieved_docs, actual_method = await asyncio.wait_for(
                    retrieval_task, 
                    timeout=self.retrieval_timeout
                )
                
            except asyncio.TimeoutError:
                logger.warning(f"Retrieval timeout for method {method}, falling back to vector search")
                try:
                    retrieved_docs, actual_method = await self._perform_retrieval(
                        query, "vector", top_k, filters
                    )
                    actual_method = f"{method}_timeout_fallback"
                except Exception as fallback_error:
                    logger.error(f"Fallback retrieval also failed: {fallback_error}")
                    retrieved_docs, actual_method = [], f"{method}_failed"
            
            retrieval_time = time.time() - retrieval_start
            
            # Cache results if successful
            if use_cache and retrieved_docs:
                try:
                    await self.cache_service.set_cached_results(
                        query=query,
                        results=retrieved_docs,
                        method=method,
                        filters=filters,
                        top_k=top_k
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to cache results: {cache_error}")
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using {actual_method}")
            return retrieved_docs, actual_method, retrieval_time
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            retrieval_time = time.time() - retrieval_start
            return [], f"{method}_error", retrieval_time
    
    async def _perform_retrieval(
        self,
        query: str,
        method: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Perform actual retrieval using specified method
        """
        if method == "hybrid":
            docs, actual_method = await self.hybrid_retriever.retrieve(
                query=query,
                top_k=top_k,
                method="hybrid",
                filters=filters
            )
            return docs, actual_method
            
        elif method == "vector":
            docs = await self.qdrant_retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters
            )
            return docs, "vector"
            
        elif method == "bm25":
            docs = await self.bm25_retriever.retrieve(
                query=query,
                top_k=top_k
            )
            return docs, "bm25"
            
        else:
            # Fallback to hybrid
            logger.warning(f"Unknown retrieval method: {method}, falling back to hybrid")
            docs, actual_method = await self.hybrid_retriever.retrieve(
                query=query,
                top_k=top_k,
                method="hybrid",
                filters=filters
            )
            return docs, "hybrid_fallback"
    
    async def _stream_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        conversation_history: List[Message],
        conversation_id: str,
        metadata: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response generation
        """
        try:
            # Send initial status with documents
            yield {
                "type": "documents_retrieved",
                "data": {
                    "count": len(retrieved_docs),
                    "method": metadata.get("retrieval_method", "unknown"),
                    "time": metadata.get("retrieval_time", 0),
                    "preview": [
                        {
                            "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                            "score": round(doc.get("score", 0), 3),
                            "source": doc.get("source", "unknown")
                        }
                        for doc in retrieved_docs[:3]
                    ]
                }
            }
            
            yield {"type": "status", "message": "Generating response..."}
            
            # Stream the response
            full_response = ""
            generation_start = time.time()
            
            try: 
                async for chunk in self.generator.stream_generate(
                    query=query,
                    context_docs=retrieved_docs,
                    conversation_history=conversation_history
                ):
                    if isinstance(chunk, str):
                        if chunk.startswith("Error:"):
                            yield {"type": "error", "message": chunk}
                            return
                        else:
                            full_response += chunk
                            yield {"type": "response_chunk", "content": chunk}
                    else:
                        logger.warning(f"Unexpected chunk type: {type(chunk)}")
                        
            except Exception as stream_error:
                logger.error(f"Error in streaming generation: {stream_error}")
                yield {"type": "error", "message": f"Streaming error: {str(stream_error)}"}
                return
            
            generation_time = time.time() - generation_start
            
            # Update conversation history
            self._update_conversation_history(conversation_id, query, full_response)
            
            # Update metadata
            metadata.update({
                "generation_time": generation_time,
                "total_time": metadata.get("retrieval_time", 0) + generation_time
            })
            
            # Send completion with metadata
            yield {
                "type": "complete",
                "conversation_id": conversation_id,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield {"type": "error", "message": str(e)}
    
    def _update_conversation_history(
        self, 
        conversation_id: str, 
        user_message: str, 
        ai_response: str
    ):
        """Update conversation history in memory store"""
        if conversation_id not in self.conversation_store:
            self.conversation_store[conversation_id] = []
        
        # Ensure ai_response is string
        if isinstance(ai_response, dict):
            ai_response = ai_response.get("content", str(ai_response))
        
        self.conversation_store[conversation_id].extend([
            Message(
                role="user",
                content=user_message,
                timestamp=datetime.now()
            ),
            Message(
                role="assistant",
                content=ai_response,
                timestamp=datetime.now()
            )
        ])
        
        # Keep only last N messages to avoid memory issues
        max_history = getattr(settings, 'MAX_CONVERSATION_HISTORY', 30)
        if len(self.conversation_store[conversation_id]) > max_history:
            self.conversation_store[conversation_id] = \
                self.conversation_store[conversation_id][-max_history:]
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a given conversation ID"""
        if conversation_id not in self.conversation_store:
            return []
        
        return [
            {
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in self.conversation_store[conversation_id]
        ]
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history"""
        if conversation_id in self.conversation_store:
            del self.conversation_store[conversation_id]
            logger.info(f"Cleared conversation history for: {conversation_id}")
            return True
        return False
    
    async def invalidate_cache_for_query(self, query: str) -> int:
        """Invalidate cache entries for a specific query"""
        return await self.cache_service.invalidate_query(query)
    
    async def warm_cache(self, common_queries: List[str]):
        """Pre-warm cache with common queries"""
        async def retrieval_func(query: str):
            docs, method, time_taken = await self._retrieve_documents(
                query=query,
                method="hybrid",
                top_k=self.default_top_k,
                filters=None,
                use_cache=False  # Don't check cache during warming
            )
            return docs, method
        
        await self.cache_service.warm_cache(common_queries, retrieval_func)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all services"""
        services_status = {}
        overall_healthy = True
        
        # Check individual retrievers
        try:
            hybrid_health = await self.hybrid_retriever.health_check()
            services_status["hybrid_retriever"] = hybrid_health
            if isinstance(hybrid_health, dict) and not hybrid_health.get("overall", False):
                overall_healthy = False
            elif not hybrid_health:
                overall_healthy = False
        except Exception as e:
            services_status["hybrid_retriever"] = {"status": "error", "error": str(e)}
            overall_healthy = False
        
        try:
            qdrant_health = await self.qdrant_retriever.health_check()
            services_status["qdrant_retriever"] = qdrant_health
            if not qdrant_health:
                overall_healthy = False
        except Exception as e:
            services_status["qdrant_retriever"] = {"status": "error", "error": str(e)}
            overall_healthy = False
        
        try:
            bm25_health = await self.bm25_retriever.health_check()
            services_status["bm25_retriever"] = bm25_health
            if not bm25_health:
                overall_healthy = False
        except Exception as e:
            services_status["bm25_retriever"] = {"status": "error", "error": str(e)}
            overall_healthy = False
        
        # Check generator
        try:
            generator_health = await self.generator.health_check()
            services_status["generator"] = generator_health
            if not generator_health:
                overall_healthy = False
        except Exception as e:
            services_status["generator"] = {"status": "error", "error": str(e)}
            overall_healthy = False
        
        # Cache service stats
        try:
            services_status["cache_service"] = self.cache_service.get_stats()
        except Exception as e:
            services_status["cache_service"] = {"status": "error", "error": str(e)}
        
        return {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "services": services_status,
            "conversation_store": {
                "active_conversations": len(self.conversation_store),
                "total_messages": sum(len(conv) for conv in self.conversation_store.values())
            }
        }
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        try:
            # Get retriever stats
            retriever_stats = {}
            
            try:
                retriever_stats["hybrid"] = await self.hybrid_retriever.get_stats()
            except Exception as e:
                retriever_stats["hybrid"] = {"error": str(e)}
            
            try:
                retriever_stats["qdrant"] = await self.qdrant_retriever.get_collection_info()
            except Exception as e:
                retriever_stats["qdrant"] = {"error": str(e)}
            
            try:
                retriever_stats["bm25"] = self.bm25_retriever.get_index_stats()
            except Exception as e:
                retriever_stats["bm25"] = {"error": str(e)}
            
            return {
                "pipeline_config": {
                    "default_top_k": self.default_top_k,
                    "caching_enabled": self.enable_caching,
                    "retrieval_timeout": self.retrieval_timeout
                },
                "retrievers": retriever_stats,
                "cache": self.cache_service.get_stats(),
                "conversations": {
                    "active_count": len(self.conversation_store),
                    "total_messages": sum(len(conv) for conv in self.conversation_store.values())
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {str(e)}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown pipeline services"""
        logger.info("Shutting down pipeline service...")
        
        try:
            await self.cache_service.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down cache service: {e}")
        
        # Clear conversation store
        self.conversation_store.clear()
        
        logger.info("Pipeline service shut down complete")
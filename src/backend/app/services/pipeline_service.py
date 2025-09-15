# src/backend/app/services/pipeline_service.py
from typing import List, Dict, Any, Optional, AsyncGenerator
from ..services.retrieval.hybrid_retriever import HybridRetriever
from ..services.generation.openai_service import OpenAIService
from ..services.graph.rag_graph import RAGGraph
from ..models.chat_models import RetrievedDocument, Message
from ..utils.logger import get_logger
from ..config import settings
import asyncio
import uuid
from datetime import datetime

logger = get_logger(__name__)

class PipelineService:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = OpenAIService()
        self.rag_graph = RAGGraph()
        self.conversation_store = {}  # In production, use proper database
        
    async def process_query(
        self,
        query: str,
        conversation_id: str,
        use_history: bool = True,
        retrieval_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline
        """
        try:
            logger.info(f"Processing query for conversation: {conversation_id}")
            
            # Get conversation history if needed
            conversation_history = []
            if use_history and conversation_id in self.conversation_store:
                conversation_history = self.conversation_store[conversation_id]
            
            # Use RAG graph for orchestration
            result = await self.rag_graph.process(
                query=query,
                conversation_history=conversation_history,
                retrieval_config=retrieval_config or {}
            )
            
            # Update conversation history
            self._update_conversation_history(
                conversation_id, 
                query, 
                result["response"]
            )
            
            return {
                "response": result["response"],
                "retrieved_documents": [
                    RetrievedDocument(
                        content=doc["content"],
                        score=doc["score"],
                        metadata=doc.get("metadata", {}),
                        source=doc.get("source")
                    )
                    for doc in result["retrieved_documents"]
                ],
                "conversation_id": conversation_id
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}")
            raise
    
    async def stream_query(
        self,
        query: str,
        conversation_id: str,
        use_history: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response for a query
        """
        try:
            logger.info(f"Starting streaming for conversation: {conversation_id}")
            
            # Get conversation history
            conversation_history = []
            if use_history and conversation_id in self.conversation_store:
                conversation_history = self.conversation_store[conversation_id]
            
            # First, retrieve relevant documents
            yield {"type": "status", "message": "Retrieving relevant documents..."}
            
            retrieved_docs = await self.retriever.retrieve(query)
            
            yield {
                "type": "documents", 
                "data": [
                    {
                        "content": doc["content"][:200] + "...",  # Preview
                        "score": doc["score"],
                        "source": doc.get("source")
                    }
                    for doc in retrieved_docs[:3]  # Show top 3
                ]
            }
            
            # Generate streaming response
            yield {"type": "status", "message": "Generating response..."}
            
            full_response = ""
            async for chunk in self.generator.stream_generate(
                query=query,
                context_docs=retrieved_docs,
                conversation_history=conversation_history
            ):
                full_response += chunk.get("content", "")
                yield {"type": "response", "content": chunk.get("content", "")}
            
            # Update conversation history
            self._update_conversation_history(conversation_id, query, full_response)
            
            yield {"type": "complete", "conversation_id": conversation_id}
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
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
        max_history = 20
        if len(self.conversation_store[conversation_id]) > max_history:
            self.conversation_store[conversation_id] = \
                self.conversation_store[conversation_id][-max_history:]
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all services"""
        services_status = {}
        
        try:
            # Check retriever
            await self.retriever.health_check()
            services_status["retriever"] = "healthy"
        except Exception as e:
            services_status["retriever"] = f"unhealthy: {str(e)}"
        
        try:
            # Check generator
            await self.generator.health_check()
            services_status["generator"] = "healthy"
        except Exception as e:
            services_status["generator"] = f"unhealthy: {str(e)}"
        
        return services_status
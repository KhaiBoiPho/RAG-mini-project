# src/backend/app/utils/langsmith_logger.py
import os
from typing import Dict, Any, Optional
from langsmith import Client
from ..config import settings
from .logger import get_logger

logger = get_logger(__name__)

class LangSmithLogger:
    """Optional LangSmith integration for tracing"""
    
    def __init__(self):
        self.client = None
        self.enabled = bool(settings.langsmith_api_key)
        
        if self.enabled:
            try:
                self.client = Client(
                    api_key=settings.langsmith_api_key,
                    api_url="https://api.smith.langchain.com"
                )
                logger.info("LangSmith logging enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith: {e}")
                self.enabled = False
    
    async def log_rag_session(
        self,
        session_id: str,
        query: str,
        retrieved_docs: list,
        response: str,
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log RAG session to LangSmith"""
        if not self.enabled:
            return
        
        try:
            run_data = {
                "name": "rag_query",
                "inputs": {
                    "query": query,
                    "session_id": session_id
                },
                "outputs": {
                    "response": response,
                    "retrieved_docs_count": len(retrieved_docs),
                    "processing_time": processing_time
                },
                "session_id": session_id,
                "extra": metadata or {}
            }
            
            # Log to LangSmith (simplified - actual implementation would use proper tracing)
            logger.info(f"Would log to LangSmith: {session_id}")
            
        except Exception as e:
            logger.error(f"Error logging to LangSmith: {e}")
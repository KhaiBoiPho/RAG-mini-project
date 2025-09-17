# src/backend/app/routers/chat.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from src.backend.app.models.chat_models import (
    ChatRequest, ChatResponse, ChatChoice, ChatUsage,
    HealthResponse, ErrorResponse, Message, MessageRole,
    SearchResult
)
from src.backend.app.services.pipeline_service import PipelineService
from src.backend.app.utils.logger import get_logger
import uuid
import json
import time
from datetime import datetime


router = APIRouter()
logger = get_logger(__name__)

# Single instance of pipeline service
_pipeline_service = None

async def get_pipeline_service() -> PipelineService:
    """Get singleton pipeline service instance"""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    
    return _pipeline_service

@router.post("/chat", response_model=ChatResponse, description="chat_model")
async def chat(
    request: ChatRequest,
    pipeline: PipelineService = Depends(get_pipeline_service)
):
    """
    Process a chat message and return AI response with retrieved context
    """
    try:
        start_time = time.time()
        
        # Generate conversation ID if not provieded
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
            
        logger.info(f"Preprocessing chat request: {request.conversation_id}")
        
        # Extract the last user message for processing
        user_message = request.messages[-1].content
        
        # Process the request through RAG pipeline
        result = await pipeline.process_query(
            query=user_message,
            conversation_id=request.conversation_id,
            method="hybrid", # Default to hybrid retrieval
            top_k=request.top_k,
            filters=None,
            use_cache=True,
            use_history=False,
            stream_response=False
        )
        
        # Extract metadata
        metadata = result.get("metadata", {})
        retrieval_time = metadata.get("retrieval_time", 0.0)
        generation_time = metadata.get("generation_time", 0.0)
        
        # Create assistant message
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=result["response"],
            timestamp=datetime.now()
        )
        
        # Create chat choice
        choice = ChatChoice(
            index=0,
            message=assistant_message,
            finish_reason="stop"
        )
        
        # Create usage information (mock values since pipeline doesn't return token counts)
        usage = ChatUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
        
        # Convert retrieved documents to SearchResult format
        source = []
        for doc in result.get("retrieved_documents", []):
            search_result = SearchResult(
                content=doc.content,
                source=doc.score,
                metadata=doc.metadata or {},
                chunk_id=doc.metadata.get("chunk_id") if doc.metadata else None,
                source=doc.source
            )
            source.append(search_result)
        
        # Create reponse
        response = ChatResponse(
            conversation_id=request.conversation_id,
            model=request.model,
            choices=[choice],
            usage=usage,
            sources=source,
            search_method=metadata.get("retrieval_method", "unknown"),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            created=int(time.time())
        )
        
        total_time = retrieval_time + generation_time
        logger.info(f"Chat request completed in {total_time:.2f}s")
        return response
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    pipeline: PipelineService = Depends(get_pipeline_service)
):
    """
    Stream chat reponse following OpenAI streaming format
    """
    try:
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
        
        logger.info(f"Starting streaming chat: {request.conversation_id}")
        
        # Validate streaming is requested
        if not request.stream:
            raise HTTPException(status_code=400, detail="Streaming not requested")
        
        user_message = request.messages[-1].content
        
        async def generate_response():
            try:
                # Process with streaming enabled
                result_generator = await pipeline.process_query(
                    query=user_message,
                    conversation_id=request.conversation_id,
                    method="hybrid",
                    top_k=request.top_k,
                    filters=None,
                    use_cache=True,
                    use_history=True,
                    stream_response=True
                )
                
                # Handle streaming generator
                if hasattr(result_generator, '__aiter__'):
                    async for chunk in result_generator:
                        chunk_type = chunk.get("type")
                        
                        if chunk_type == "documents_retrieved":
                            # Send source information
                            data = chunk.get("data", {})
                            sources = []
                            for preview in data.get("preview", []):
                                search_result = SearchResult(
                                    content=preview.get("content", ""),
                                    score=preview.get("score", 0.0),
                                    metadata={},
                                    source=preview.get("source")
                                )
                                sources.append(search_result)
                            
                            sources_chunk = {
                                "conversation_id": request.conversation_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "sources": [source.dict() for source in sources],
                                "search_method": data.get("method"),
                                "retrieval_time": data.get("time", 0)
                            }
                            yield f"data: {json.dumps(sources_chunk)}\n\n"
                        
                        elif chunk_type == "response_chunk":
                            # Stream response content
                            content = chunk.get("content", "")
                            stream_chunk = {
                                "conversation_id": request.conversation_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": content
                                    },
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(stream_chunk)}\n\n"
                        
                        elif chunk_type == "complete":
                            # Send final chunk
                            metadata = chunk.get("metadata", {})
                            final_chunk = {
                                "conversation_id": request.conversation_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0
                                },
                                "metadata": metadata
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                        
                        elif chunk_type == "error":
                            # Handle error
                            error_chunk = {
                                "error": {
                                    "message": chunk.get("message", "Unknown error"),
                                    "type": "server_error",
                                    "code": "internal_error"
                                }
                            }
                            yield f"data {json.dumps(error_chunk)}\n\n"
                            return
                else:
                    # Non-streaming fallback
                    logger.warning("Expected streaming generator but not got regular result")
                    error_chunk = {
                        "error": {
                            "message": "Streaming not available",
                            "type": "server_error",
                            "code": "streaming_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                
                yield "data: [DONE]\n\n"
            
            except Exception as e:
                logger.error(f"Error in streaming generator: {str(e)}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "internal_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no" # Disable nginx buffering
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in streaming chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("health", response_model=HealthResponse)
async def health_check(pipeline: PipelineService = Depends(get_pipeline_service)):
    """
    Check the health of all service
    """
    try:
        health_result = await pipeline.health_check()
        
        # Extract service status
        services_status = health_result.get("services", {})
        overall_status = health_result.get("overall_status", "unknown")
        
        # Convert service status to boolean format
        services_bool = {}
        for service, status in services_status.items():
            if isinstance(status, dict):
                services_bool[service] = status.get("status") == "healthy" or status.get("overall", False)
            else:
                services_bool[service] = bool(status)
        
        return HealthResponse(
            status=overall_status,
            services=services_bool,
            timestamp=int(time.time()),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed {str(e)}")
        return HealthResponse(
            status="unhealthy",
            services={"pipeline": False, "error": str(e)},
            timestamp=int(time.time()),
            version="1.0.0"
        )

@router.get("/models")
async def list_models():
    """
    List avaiable models
    """
    allowed_models = [
        "gpt-5-nano",
        "gpt-4o-mini", 
        "gpt-5o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "gpt-5-turbo"
    ]
    
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization"
            } for model in allowed_models
        ]
    }

@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history(
    coversation_id: str,
    pipeline: PipelineService = Depends(get_pipeline_service)
):
    """
    Get conversation history for a specific conversation ID
    """
    try:
        history = await pipeline.get_conversation_history(coversation_id)
        return {
            "conversation_id": coversation_id,
            "messages": history,
            "total_messages": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    pipeline: PipelineService = Depends(get_pipeline_service)
):
    """
    Clear conversation history for a specific conversation ID
    """
    try:
        success = await pipeline.clear_conversation(conversation_id)
        if success:
            return {"message": "Conversation cleared successfully", "conversation_id": conversation_id}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_pipeline_stats(
    pipeline: PipelineService = Depends(get_pipeline_service)
):
    """
    Get comprehensive pipeline statistics
    """
    try:
        stats = await pipeline.get_pipeline_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting pipeline stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/invalidate")
async def invalidate_cache(
    query: str,
    pipeline: PipelineService = Depends(get_pipeline_service)
):
    """
    Invalidate cache entries for a specific query
    """
    try:
        count = await pipeline.invalidate_cache_for_query(query)
        return {
            "message": f"Invalidated {count} cache entries",
            "query": query,
            "invalidated_count": count
        }
    except Exception as e:
        logger.error(f"Error invalidating cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
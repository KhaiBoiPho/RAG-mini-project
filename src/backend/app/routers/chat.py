# src/backend/app/routers/chat.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from ..models.chat_models import (
    ChatRequest, ChatResponse, ChatChoice, ChatUsage, 
    HealthResponse, ErrorResponse, Message, MessageRole,
    SearchResult
)
from ..services.pipeline_service import PipelineService
from ..utils.logger import get_logger
import uuid
import json
import time
from datetime import datetime

router = APIRouter()
logger = get_logger(__name__)

# Dependency to get pipeline service
async def get_pipeline_service() -> PipelineService:
    return PipelineService()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    pipeline: PipelineService = Depends(get_pipeline_service)
):
    """
    Process a chat message and return AI response with retrieved context
    """
    try:
        start_time = time.time()
        
        # Generate conversation ID if not provided
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
        
        logger.info(f"Processing chat request: {request.conversation_id}")
        
        # Extract the last user message for processing
        user_message = request.messages[-1].content
        
        # Process the request through RAG pipeline
        result = await pipeline.process_query(
            query=user_message,
            conversation_id=request.conversation_id,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            messages=request.messages  # Pass full conversation history
        )
        
        # Calculate processing times
        retrieval_time = result.get("retrieval_time", 0.0)
        generation_time = time.time() - start_time - retrieval_time
        
        # Create assistant message
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=result["response"],
            timestamp=datetime.utcnow()
        )
        
        # Create chat choice
        choice = ChatChoice(
            index=0,
            message=assistant_message,
            finish_reason=result.get("finish_reason", "stop")
        )
        
        # Create usage information
        usage = ChatUsage(
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0)
        )
        
        # Convert retrieved documents to SearchResult format
        sources = []
        if "retrieved_documents" in result:
            for doc in result["retrieved_documents"]:
                search_result = SearchResult(
                    content=doc.get("content", ""),
                    score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {}),
                    chunk_id=doc.get("chunk_id"),
                    source=doc.get("source")
                )
                sources.append(search_result)
        
        # Create response
        response = ChatResponse(
            conversation_id=request.conversation_id,
            model=request.model,
            choices=[choice],
            usage=usage,
            sources=sources,
            search_method=result.get("search_method"),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            created=int(time.time())
        )
        
        logger.info(f"Chat request completed in {generation_time + retrieval_time:.2f}s")
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
    Stream chat response following OpenAI streaming format
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
                # First send the sources/context if available
                retrieval_start = time.time()
                
                # Get initial context/sources
                context_result = await pipeline.get_context(
                    query=user_message,
                    conversation_id=request.conversation_id,
                    top_k=request.top_k
                )
                
                retrieval_time = time.time() - retrieval_start
                
                # Send sources information first
                if "retrieved_documents" in context_result:
                    sources = []
                    for doc in context_result["retrieved_documents"]:
                        search_result = SearchResult(
                            content=doc.get("content", ""),
                            score=doc.get("score", 0.0),
                            metadata=doc.get("metadata", {}),
                            chunk_id=doc.get("chunk_id"),
                            source=doc.get("source")
                        )
                        sources.append(search_result)
                    
                    # Send sources chunk
                    sources_chunk = {
                        "conversation_id": request.conversation_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "sources": [source.dict() for source in sources],
                        "search_method": context_result.get("search_method"),
                        "retrieval_time": retrieval_time
                    }
                    yield f"data: {json.dumps(sources_chunk)}\n\n"
                
                # Stream the actual response
                async for chunk in pipeline.stream_query(
                    query=user_message,
                    conversation_id=request.conversation_id,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    messages=request.messages,
                    context=context_result.get("context", "")
                ):
                    # Format chunk according to OpenAI streaming format
                    stream_chunk = {
                        "conversation_id": request.conversation_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant" if chunk.get("role") else None,
                                "content": chunk.get("content", "")
                            },
                            "finish_reason": chunk.get("finish_reason")
                        }]
                    }
                    yield f"data: {json.dumps(stream_chunk)}\n\n"
                    
                # Send final chunk with usage
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
                        "prompt_tokens": context_result.get("prompt_tokens", 0),
                        "completion_tokens": context_result.get("completion_tokens", 0),
                        "total_tokens": context_result.get("total_tokens", 0)
                    }
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
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
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in streaming chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check(pipeline: PipelineService = Depends(get_pipeline_service)):
    """
    Check the health of all services
    """
    try:
        services_status = await pipeline.health_check()
        
        # Convert service status to boolean
        services_bool = {}
        for service, status in services_status.items():
            services_bool[service] = status == "healthy"
        
        overall_status = "healthy" if all(services_bool.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            services=services_bool,
            timestamp=int(time.time()),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            services={"pipeline": False, "error": str(e)},
            timestamp=int(time.time()),
            version="1.0.0"
        )

@router.get("/models")
async def list_models():
    """
    List available models
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

# # Error handlers
# @router.exception_handler(ValueError)
# async def validation_exception_handler(request, exc):
#     """Handle validation errors"""
#     return ErrorResponse(
#         error="ValidationError",
#         message=str(exc),
#         timestamp=int(time.time()),
#         request_id=str(uuid.uuid4())
#     )

# @router.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     """Handle HTTP exceptions"""
#     return ErrorResponse(
#         error=f"HTTPError_{exc.status_code}",
#         message=exc.detail,
#         timestamp=int(time.time()),
#         request_id=str(uuid.uuid4())
#     )
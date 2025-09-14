#!/usr/bin/env python3
"""
Pydantic models for chat API
"""

import os
import time
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class MessageRole(str, Enum):
    """Message role enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Chat message model"""
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=10000)
    
    @field_validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty or only whitespace")
        return v.strip()


class ChatRequest(BaseModel):
    """Chat completion request model"""
    messages: List[Message] = Field(..., min_items=1, max_items=50)
    model: Optional[str] = Field(default="gpt-5-nano", max_length=100)
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=4000)
    stream: Optional[bool] = False
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    
    @field_validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("At least one message is required")
        
        # Check that the last message is from user
        if v[-1].role != MessageRole.USER:
            raise ValueError("Last message must be from user")
        
        return v
    
    @field_validator('model')
    def validate_model(cls, v):
        allowed_models = [
            "gpt-5-nano",
            "gpt-4o-mini",
            "gpt-5o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "gpt-5-turbo"
        ]
        if v not in allowed_models:
            raise ValueError(f"Model must be one of: {", ".join(allowed_models)}")
        
        return v


class ChatChoice(BaseModel):
    """Chat completion choice model"""
    index: int = 0
    message: Message
    finish_reason: Literal["stop", "length", "content_filter", "null"] = "stop"


class ChatUsage(BaseModel):
    """Tolen usage information"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class SearchResult(BaseModel):
    """Search result model"""
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    source: Optional[str] = Field(None, description="Source document")


class DocumentChunk(BaseModel):
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content")
    source: str = Field(..., description="Source document path")
    chunk_index: int = Field(..., description="Index of chunk in document")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ChatResponse(BaseModel):
    """Chat completion response model"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{os.urandom(12).hex()}")
    object: str = "chat.completion"
    sources: List[SearchResult] = Field(default_factory=list, description="Source documents used")
    search_method: str = Field(None, description="Search method used (hybrid/vector/bm25)")
    retrieval_time: float = Field(None, description="Time taken for retrieval (seconds)")
    generation_time: float = Field(None, description="Time taken for generation (seconds)")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage = Field(default_factory=ChatUsage)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chatcmpl-abcdxyz567",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-5-nano",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Xin Xin chào! Tôi có thể giúp gì cho bạn về luật giao thông?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 36,
                    "completion_tokens": 18,
                    "total_tokens": 54,
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    services: Dict[str, bool]
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    request_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request format",
                "timestamp": 123456333,
                "request_id": "req-abcdxyz567"
            }
        }
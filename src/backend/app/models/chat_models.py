#!/usr/bin/env python3
"""
Pydantic models for chat API
"""

import os
import time
from pydantic import ValidationError
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime

class MessageRole(str, Enum):
    """Message role enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Chat message model"""
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty or only whitespace")
        return v.strip()


class ChatRequest(BaseModel):
    """Chat completion request model"""
    messages: List[Message] = Field(..., min_items=1)
    conversation_id: Optional[str] = None
    model: Optional[str] = Field(default="gpt-5-nano")
    # gpt-5-nano not support settings temperature
    temperature: Optional[float] = Field(default=1)
    max_completion_tokens: Optional[int] = Field(default=2048)
    stream: Optional[bool] = False
    top_k: Optional[int] = Field(default=5)

    # Validator chỉ cần check message cuối cùng là user
    @field_validator('messages')
    def check_last_message_is_user(cls, v):
        if not v:
            raise ValueError("At least one message is required")
        if v[-1].role != MessageRole.USER:
            raise ValueError("Last message must be from user")
        return v


class ChatChoice(BaseModel):
    """Chat completion choice model"""
    index: int = 0
    message: Message
    finish_reason: Literal["stop", "length", "content_filter", "null"] = "stop"


class ChatUsage(BaseModel):
    """Token usage information"""
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


class ChatResponse(BaseModel):
    """Chat completion response model"""
    conversation_id: str = Field(..., description="Unique conversation identifier")
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
                "model": "gpt-4o-mini",
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


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime


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
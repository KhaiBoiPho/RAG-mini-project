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
    role: str
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
    
    
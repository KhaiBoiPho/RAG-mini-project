#!/usr/bin/env python3
"""
Configuration management for RAG Backend API
"""

import os
from typing import Any, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with enviroment variable support"""
    
    # API Configuration
    API_TITLE: str = "Legal RAG API"
    API_DESCRIPTION: str = "API for Legal RAG Chatbot"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(env="LOG_LEVEL")
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = Field(env="OPENAI_API_KEY")
    DEFAULT_MODEL: str = Field(default="gpt-5-nano", env="DEFAULT_MODEL")
    MAX_OUTPUT_TOKENS: int = Field(default=2048, env="MAX_OUTPUT_TOKENS")
    TEMPERATURE_GENERATION: float = Field(env="TEMPERATURE_GENERATION")
    CONVERSATION_HISTORY: Optional[List[Any]] = []
    OPENAI_TIMEOUT: int = Field(env="OPENAI_TIMEOUT")
    STREAM_BOOL: bool = Field(env="STREAM_BOOL")
    
    # Qdrant Configuration
    QDRANT_URL: Optional[str] = Field(env="QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = Field(env="QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = Field(default="legal_rag", env="QDRANT_COLLECTION_NAME")
    QDRANT_TIMEOUT: int = Field(default=30, env="QDRANT_TIMEOUT")
    QDRANT_DISTANCE: str = Field(env="QDRANT_DISTANCE")
    
    # HuggingFace Configuration
    HUGGINGFACE_API_KEY: Optional[str] = Field(env="HUGGINGFACE_API_KEY")
    EMBEDDING_DIMENSION: int = Field(env="EMBEDDING_DIMENSION")
    MAX_INPUT_TOKENS: int = Field(env="MAX_INPUT_TOKENS")
    EMBEDDING_BATCH_SIZE: int = Field(env="EMBEDDING_BATCH_SIZE")
    EMBEDDINGS_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        env="EMBEDDINGS_MODEL_NAME"
    )
    
    # BM25 Configuration
    BM25_K1: float = Field(default=1.5, env="BM25_K1")
    BM25_B: float = Field(default=0.75, env="BM25_B")
    
    # Search Configuration
    SEMANTIC_SEARCH_TOP_K: int = Field(default=10, env="SEMANTIC_SEARCH_TOP_K")
    SCORE_THRESHOLD: float = Field(default=0.5, env="SCORE_THRESHOLD")
    BM25_WEIGHT: float = Field(default=0.3, env="BM25_WEIGHT")
    VECTOR_WEIGHT: float = Field(default=0.7, env="VECTOR_WEIGHT")
    HIGH_CONFIDENCE_THRESHOLD: float = Field(default=0.85, env="HIGH_CONFIDENCE_THRESHOLD")
    
    # Performance Configuration
    MAX_CONTEXT_LENGTH: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    REQUEST_TIMEOUT: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="rag_backend.log", env="LOG_FILE")
    
    # CORS Configuration
    ALLOWED_ORIGINS: list = Field(default=["*"], env="ALLOWED_ORIGINS")
    ALLOWED_METHODS: list = Field(default=["*"], env="ALLOWED_METHODS")
    ALLOWED_HEADERS: list = Field(default=["*"], env="ALLOWED_HEADERS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra="ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings

def vaidate_required_settings():
    """Validate that all required settings are present"""
    required_settings = [
        ("QDRANT_URL", settings.QDRANT_URL),
        ("HUGGINGFACE_API_KEY", settings.HUGGINGFACE_API_KEY),
        ("OPENAI_API_KEY", settings.OPENAI_API_KEY),
    ]
    
    missing_settings = []
    for setting_name, setting_value in required_settings:
        if not setting_value:
            missing_settings.append(setting_name)
    
    if missing_settings:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_settings)}"
        )
    
    return True
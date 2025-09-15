#!/usr/bin/env python3
"""
Configuration management for RAG Backend API
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Load config with qdrant
CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class Settings(BaseSettings):
    """Application settings with enviroment variable support"""
    
    # Qdrant Configuration
    QDRANT_URL: Optional[str] = Field(env="QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = Field(env="QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = Field(default="legal_rag", env="COLLECTION_NAME")
    QDRANT_HOST: str = Field(default=config["qdrant"]["host"], env="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=config["qdrant"]["port"], env="QDRANT_PORT")
    # QDRANT_TIMEOUT: int = Field(default=30, env="QDRANT_TIMEOUT")
    
    # HuggingFace Configuration
    HUGGINGFACE_API_KEY: Optional[str] = Field(env="HUGGINGFACE_API_KEY")
    EMBEDDING_DIMENSION: int = Field(env="EMBEDDING_DIMENSION")
    MAX_INPUT_TOKENS: int = Field(env="MAX_INPUT_TOKENS")
    EMBEDDING_BATCH_SIZE: int = Field(env="EMBEDDING_BATCH_SIZE")
    EMBEDDINGS_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        env="EMBEDDINGS_MODEL_NAME"
    )
    
    # CPU Configuration
    EMBEDDING_MAX_WORKERS: int = Field(env="EMBEDDING_MAX_WORKERS")
    
    # Text processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Search Configuration
    SEMANTIC_SEARCH_TOP_K: int = Field(default=10, env="SEMANTIC_SEARCH_TOP_K")
    SCORE_THRESHOLD: float = Field(default=0.5, env="SCORE_THRESHOLD")
    BM25_WEIGHT: float = Field(default=0.3, env="BM25_WEIGHT")
    VECTOR_WEIGHT: float = Field(default=0.7, env="VECTOR_WEIGHT")
    HIGH_CONFIDENCE_THRESHOLD: float = Field(default=0.85, env="HIGH_CONFIDENCE_THRESHOLD")
    
    # Performance Configuration
    MAX_CONTEXT_LENGTH: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    REQUEST_TIMEOUT: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # case_sensitive = True
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
# src/backend/app/main.py

import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import routers and services
from src.backend.app.routers import chat
from src.backend.app.services.pipeline_service import PipelineService
from src.backend.app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Global pipeline service instance
pipeline_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global pipeline_service
    
    # Startup
    logger.info("Starting up RAG Chat API...")
    try:
        # Initialize pipeline service
        pipeline_service = PipelineService()
        logger.info("Pipeline service initialized successfully")
        
        # Warm up cache with common queries (optional)
        common_queries = [
            "Luật giao thông Việt Nam",
            "Phạt vi phạm giao thông",
            "Bằng lái xe",
            "Đăng ký xe",
            "Bảo hiểm xe"
        ]
        
        try:
            await pipeline_service.warm_cache(common_queries)
            logger.info("Cache warmed up successfully")
        except Exception as e:
            logger.warning(f"Cache warm-up failed: {e}")
        
        # Health check
        health = await pipeline_service.health_check()
        logger.info(f"System health: {health.get('overall_status', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline service: {e}")
        # Continue anyway, let individual requests handle failures
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chat API...")
    try:
        if pipeline_service:
            await pipeline_service.shutdown()
        logger.info("Pipeline service shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="Retrieval-Augmented Generation Chat API with Vietnamese Traffic Law Knowledge",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(
    chat.router,
    prefix="/api/v1",
    tags=["chat"]
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Chat API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation Chat API",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "endpoints": {
            "chat": "/api/v1/chat",
            "chat_stream": "/api/v1/chat/stream",
            "models": "/api/v1/models",
            "health": "/api/v1/health"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if os.getenv("DEBUG") == "true" else "Internal server error"
        }
    )

# Health check endpoint at root level
@app.get("/health")
async def simple_health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "message": "RAG Chat API is running",
        "timestamp": asyncio.get_event_loop().time()
    }

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
        use_colors=True
    )
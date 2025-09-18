# src/backend/app/main.py

import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
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

# Dependency function for shared pipeline instance
async def get_shared_pipeline_service() -> PipelineService:
    """Get the shared pipeline service instance"""
    global pipeline_service
    if pipeline_service is None:
        logger.error("Pipeline service not initialized!")
        raise HTTPException(status_code=503, detail="Service not available")
    return pipeline_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global pipeline_service
    
    # Startup
    logger.info("Starting up RAG Chat API...")
    try:
        # Initialize pipeline service
        pipeline_service = PipelineService()
        
        # Override chat router's pipeline dependency
        chat.get_pipeline_service = get_shared_pipeline_service
        
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
        # Don't continue with None pipeline service
        raise RuntimeError(f"Cannot start without pipeline service: {e}")
    
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

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for request validation errors (422)
    """
    logger.error(f"Validation error on {request.method} {request.url}")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Validation errors: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request Validation Error",
            "errors": exc.errors(),
            "body": exc.body,
            "url": str(request.url),
            "method": request.method
        }
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """
    Custom handler for Pydantic validation errors
    """
    logger.error(f"Pydantic validation error on {request.method} {request.url}")
    logger.error(f"Validation errors: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Model Validation Error",
            "errors": exc.errors(),
            "url": str(request.url),
            "method": request.method
        }
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

# Remove duplicate health endpoint (handled by chat router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception on {request.method} {request.url}: {str(exc)}", exc_info=True)
    
    # Safe debug check
    debug_mode = os.getenv("DEBUG", "false").lower() in ["true", "1", "yes"]
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if debug_mode else "Internal server error",
            "url": str(request.url),
            "method": request.method
        }
    )

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() in ["true", "1", "yes"]
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Debug mode: {os.getenv('DEBUG', 'false')}")
    logger.info(f"Reload: {reload}")
    
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
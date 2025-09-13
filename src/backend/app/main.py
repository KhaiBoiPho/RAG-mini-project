import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat
from .services.qdrant_service import QdrantService
from .services.embeddings_service import EmbeddingsService
from .services.hybrid_search_service import HybridSearchService
from .services.openai_service import OpenAIService
from .services.langsmith_logger import LangSmithLogger
from .config import Config
from qdrant_client import QdrantClient
import logging

logger = logging.getLogger("rag-backend")

app = FastAPI(title="Legal RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(chat.router)

@app.on_event("startup")
async def startup_event():
    app.state.qdrant = QdrantService(QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY), Config.QDRANT_COLLECTION)
    app.state.embeddings = EmbeddingsService(Config.HUGGINGFACE_API_KEY, Config.EMBEDDINGS_MODEL_NAME)
    app.state.hybrid_search = HybridSearchService(app.state.qdrant, app.state.embeddings)
    app.state.openai = OpenAIService(Config.OPENAI_API_KEY)
    app.state.langsmith = LangSmithLogger(os.getenv("LANGSMITH_API_KEY"))
    logger.info("All services initialized")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
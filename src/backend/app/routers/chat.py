from fastapi import APIRouter, Depends
from ..models.chat_models import ChatRequest, ChatResponse
from ..services.retrieval.hybrid_retriever import HybridSearchService
from ..services.generation.openai_service import OpenAIService
from ..utils.langsmith_logger import LangSmithLogger
import logging

logger = logging.getLogger("rag-backend.chat")
router = APIRouter(prefix="/v1/chat", tags=["chat"])

@router.post("/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    hybrid_search: HybridSearchService = Depends(),
    openai_svc: OpenAIService = Depends(),
    logger_svc: LangSmithLogger = Depends()
):
    user_message = request.messages[-1].content
    logger.info(f"Processing query: {user_message}")

    # Step 1 + 2: Hybrid search
    search_results = hybrid_search.search(user_message)
    context = [r.payload.get("page_content", "") for r in search_results if r]

    # Step 3: OpenAI
    system_content = "Bạn là trợ lý AI..." + "\n\n".join(context)
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_message}]
    openai_response = await openai_svc.get_response(messages, model=request.model, temperature=request.temperature)

    response_text = openai_response.choices[0].message.content

    # Step 4: Logging
    logger_svc.log_interaction(user_message, response_text, metadata={"context_count": len(context)})

    return ChatResponse(
        model=openai_response.model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}]
    )
# tests/test_chat_routes.py

import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from unittest.mock import AsyncMock
from src.backend.app.routers import chat
from src.backend.app.routers.chat import PipelineService
from src.backend.app.models.chat_models import ChatRequest, Message, MessageRole

# Create test app and include router
app = FastAPI()
app.include_router(chat.router)

# Mock pipeline service
@pytest.fixture
def mock_pipeline(monkeypatch):
    mock = AsyncMock(spec=PipelineService)
    # Mock process_query
    mock.process_query.return_value = {
        "response": "Test response",
        "retrieved_documents": [{"content": "doc1", "score": 0.9}],
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15,
        "finish_reason": "stop",
        "retrieval_time": 0.1,
        "search_method": "vector"
    }
    # Mock get_context
    mock.get_context.return_value = {
        "retrieved_documents": [{"content": "doc1", "score": 0.9}],
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15,
        "search_method": "vector",
        "context": "doc1"
    }
    # Mock stream_query
    async def fake_stream_query(**kwargs):
        yield {"content": "chunk1", "role": None, "finish_reason": None}
        yield {"content": "chunk2", "role": None, "finish_reason": "stop"}
    mock.stream_query.side_effect = fake_stream_query
    monkeypatch.setattr(chat, "get_pipeline_service", lambda: mock)
    return mock

@pytest.mark.asyncio
async def test_chat_endpoint(mock_pipeline):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        request_payload = ChatRequest(
            conversation_id=None,
            model="gpt-5-nano",
            temperature=0.7,
            max_tokens=50,
            top_k=3,
            messages=[Message(role=MessageRole.USER, content="Hello")]
        )
        response = await ac.post("/chat", json=request_payload.dict())
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert data["choices"][0]["message"]["content"] == "Test response"
        assert data["usage"]["total_tokens"] == 15

@pytest.mark.asyncio
async def test_health_endpoint(mock_pipeline):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]

@pytest.mark.asyncio
async def test_models_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert data["data"][0]["id"].startswith("gpt")

@pytest.mark.asyncio
async def test_chat_stream_endpoint(mock_pipeline):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        request_payload = ChatRequest(
            conversation_id=None,
            model="gpt-5-nano",
            temperature=0.7,
            max_tokens=50,
            top_k=3,
            messages=[Message(role=MessageRole.USER, content="Hello")],
            stream=True
        )
        async with ac.stream("POST", "/chat/stream", json=request_payload.dict()) as response:
            assert response.status_code == 200
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    chunks.append(line)
            assert any("chunk1" in c or "chunk2" in c for c in chunks)
            assert chunks[-1].endswith("[DONE]")

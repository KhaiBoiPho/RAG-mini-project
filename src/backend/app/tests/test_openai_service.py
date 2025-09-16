import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import types
from src.backend.app.services.generation.openai_service import OpenAIService

# Fake async generator for streaming chunks
async def fake_stream(*args, **kwargs):
    for text in ["chunk1", "chunk2"]:
        yield {
            "choices": [
                {
                    "delta": {"content": text},
                    "finish_reason": None
                }
            ]
        }

@pytest.mark.asyncio
async def test_generate_success(monkeypatch):
    service = OpenAIService()

    # Mock OpenAI response
    mock_response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Fake answer"))]
    )

    monkeypatch.setattr(
        "openai.ChatCompletion.acreate",
        AsyncMock(return_value=mock_response)
    )

    query = "Sample query"
    context_docs = [{"content": "Legal text", "source": "Civil Code"}]

    result = await service.generate(query, context_docs)

    assert result == "Fake answer"


@pytest.mark.asyncio
async def test_generate_error(monkeypatch):
    service = OpenAIService()

    # Force OpenAI to raise an error
    monkeypatch.setattr(
        "openai.ChatCompletion.acreate",
        AsyncMock(side_effect=RuntimeError("API error"))
    )

    with pytest.raises(RuntimeError, match="API error"):
        await service.generate("test", [{"content": "doc"}])


@pytest.mark.asyncio
async def test_health_check_success(monkeypatch):
    service = OpenAIService()

    # Mock successful health check
    monkeypatch.setattr(
        "openai.ChatCompletion.acreate",
        AsyncMock(return_value="ok")
    )

    ok = await service.health_check()
    assert ok is True


def test_format_context_and_prompt():
    service = OpenAIService()

    # Prepare fake docs
    docs = [{"content": "Article 1: Some rule", "source": "Law A"}]
    context = service._format_context(docs)

    # Check context contains content and source
    assert "Article 1" in context
    assert "Law A" in context

    # Build chat prompt
    query = "What does the law say?"
    prompt = service._format_chat_prompt(query, context)

    assert query in prompt
    assert context in prompt


@pytest.mark.asyncio
async def test_stream_generate(monkeypatch):
    service = OpenAIService()
    service.logger = AsyncMock()  # Fake logger so no AttributeError

    async def fake_stream(*args, **kwargs):
        for text in ["chunk1", "chunk2"]:
            yield types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=text),
                    finish_reason=None
                )]
            )

    monkeypatch.setattr(
        "openai.ChatCompletion.acreate",
        AsyncMock(return_value=fake_stream())
    )

    results = []
    async for chunk in service.stream_generate("test", [{"content": "doc"}]):
        results.append(chunk)

    assert results == ["chunk1", "chunk2"]
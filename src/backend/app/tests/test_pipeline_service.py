import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.backend.app.services.pipeline_service import PipelineService


@pytest.fixture
def pipeline_service():
    service = PipelineService()

    # Mock retrievers
    service.hybrid_retriever = AsyncMock()
    service.qdrant_retriever = AsyncMock()
    service.bm25_retriever = AsyncMock()
    
    # Mock cache service
    service.cache_service = AsyncMock()
    service.cache_service.get_stats.return_value = {"hits": 0, "misses": 0}
    
    # Mock generator
    service.generator = AsyncMock()
    service.generator.generate = AsyncMock(return_value="AI response")
    service.generator.stream_generate = AsyncMock(
        return_value=aiter([{"content": "Hello "}, {"content": "World"}])
    )
    
    return service


def aiter(iterable):
    async def gen():
        for item in iterable:
            yield item
    return gen()


@pytest.mark.asyncio
async def test_process_query_happy_path(pipeline_service):
    pipeline_service._retrieve_documents = AsyncMock(
        return_value=([{"content": "doc1", "score": 0.9, "metadata": {}, "source": "s"}], "hybrid", 0.01)
    )
    result = await pipeline_service.process_query("test", "conv1")
    assert result["response"] == "AI response"
    assert len(result["retrieved_documents"]) == 1
    assert result["conversation_id"] == "conv1"
    assert result["metadata"]["retrieval_method"] == "hybrid"


@pytest.mark.asyncio
async def test_process_query_no_documents(pipeline_service):
    pipeline_service._retrieve_documents = AsyncMock(return_value=([], "hybrid", 0.01))
    result = await pipeline_service.process_query("test", "conv1")
    assert "Xin lỗi" in result["response"]
    assert result["retrieved_documents"] == []


@pytest.mark.asyncio
async def test_process_query_with_streaming(pipeline_service):
    pipeline_service._retrieve_documents = AsyncMock(
        return_value=([{"content": "doc1", "score": 1.0, "metadata": {}, "source": "s"}], "hybrid", 0.01)
    )
    stream = await pipeline_service.process_query("test", "conv1", stream_response=True)
    chunks = [c async for c in stream]
    types = [c["type"] for c in chunks]
    assert "documents_retrieved" in types
    assert "status" in types
    assert "response_chunk" in types
    assert "complete" in types


@pytest.mark.asyncio
async def test_retrieve_documents_cache_hit(pipeline_service):
    pipeline_service.cache_service.get_cached_results.return_value = [{"content": "cached", "score": 0.5}]
    docs, method, t = await pipeline_service._retrieve_documents("q", "hybrid", 5, None, True)
    assert docs[0]["content"] == "cached"
    assert "cached" in method


@pytest.mark.asyncio
async def test_perform_retrieval_methods(pipeline_service):
    pipeline_service.hybrid_retriever.retrieve.return_value = (["hdoc"], "hybrid")
    pipeline_service.qdrant_retriever.retrieve.return_value = ["vdoc"]
    pipeline_service.bm25_retriever.retrieve.return_value = ["bdoc"]

    docs, m = await pipeline_service._perform_retrieval("q", "hybrid", 5, None)
    assert docs == ["hdoc"] and m == "hybrid"

    docs, m = await pipeline_service._perform_retrieval("q", "vector", 5, None)
    assert docs == ["vdoc"] and m == "vector"

    docs, m = await pipeline_service._perform_retrieval("q", "bm25", 5, None)
    assert docs == ["bdoc"] and m == "bm25"

    docs, m = await pipeline_service._perform_retrieval("q", "unknown", 5, None)
    assert m == "hybrid_fallback"


@pytest.mark.asyncio
async def test_conversation_history(pipeline_service):
    pipeline_service._update_conversation_history("conv1", "hello", "hi")
    history = await pipeline_service.get_conversation_history("conv1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    cleared = await pipeline_service.clear_conversation("conv1")
    assert cleared is True
    assert await pipeline_service.get_conversation_history("conv1") == []


@pytest.mark.asyncio
async def test_health_check_and_stats(pipeline_service):
    pipeline_service.hybrid_retriever.health_check.return_value = {"overall": True}
    pipeline_service.qdrant_retriever.health_check.return_value = True
    pipeline_service.bm25_retriever.health_check.return_value = True
    pipeline_service.generator.health_check.return_value = True

    status = await pipeline_service.health_check()
    assert status["overall_status"] == "healthy"

    pipeline_service.hybrid_retriever.get_stats.return_value = {"docs": 10}
    pipeline_service.qdrant_retriever.get_collection_info.return_value = {"coll": "info"}
    pipeline_service.bm25_retriever.get_index_stats.return_value = {"bm25": "ok"}
    stats = await pipeline_service.get_pipeline_stats()
    assert "retrievers" in stats
    assert "hybrid" in stats["retrievers"]


@pytest.mark.asyncio
async def test_shutdown(pipeline_service):
    await pipeline_service.shutdown()
    assert pipeline_service.conversation_store == {}
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

import src.backend.app.services.retrieval.hybrid_retriever as hybrid


@pytest.fixture
def retriever():
    r = hybrid.HybridRetriever()
    # Mock retrievers
    r.qdrant_retriever = AsyncMock()
    r.bm25_retriever = AsyncMock()
    return r


@pytest.mark.asyncio
async def test_vector_search_filters_by_min_score(retriever):
    retriever.qdrant_retriever.search.return_value = [
        {"id": 1, "score": 0.05},  # dưới ngưỡng
        {"id": 2, "score": 0.5},   # hợp lệ
    ]
    results = await retriever._vector_search("query", 5)
    assert len(results) == 1
    assert results[0]["id"] == 2


@pytest.mark.asyncio
async def test_bm25_search_filters_by_min_score(retriever):
    retriever.bm25_retriever.search.return_value = [
        {"id": "a", "score": 0.05},
        {"id": "b", "score": 0.2},
    ]
    results = await retriever._bm25_search("query", 5)
    assert len(results) == 1
    assert results[0]["id"] == "b"


@pytest.mark.asyncio
async def test_hybrid_search_combines_results(retriever):
    retriever.qdrant_retriever.search.return_value = [
        {"id": "1", "score": 0.8, "content": "doc1"},
        {"id": "2", "score": 0.7, "content": "doc2"},
    ]
    retriever.bm25_retriever.search.return_value = [
        {"id": "2", "score": 0.9, "content": "doc2"},
        {"id": "3", "score": 0.6, "content": "doc3"},
    ]

    results = await retriever._hybrid_search("query", top_k=2)
    assert len(results) == 2
    # doc2 xuất hiện trong cả hai → fusion_score phải cao
    ids = [d["id"] for d in results]
    assert "2" in ids


@pytest.mark.asyncio
async def test_retrieve_with_method_vector(retriever):
    retriever._vector_search = AsyncMock(return_value=[{"id": "x"}])
    docs, method = await retriever.retrieve("hello", method="vector")
    assert method == "vector"
    assert docs[0]["id"] == "x"


@pytest.mark.asyncio
async def test_retrieve_with_method_bm25(retriever):
    retriever._bm25_search = AsyncMock(return_value=[{"id": "y"}])
    docs, method = await retriever.retrieve("hello", method="bm25")
    assert method == "bm25"
    assert docs[0]["id"] == "y"


@pytest.mark.asyncio
async def test_retrieve_with_hybrid(retriever):
    retriever._hybrid_search = AsyncMock(return_value=[{"id": "z"}])
    docs, method = await retriever.retrieve("hello", method="hybrid")
    assert method == "hybrid"
    assert docs[0]["id"] == "z"


@pytest.mark.asyncio
async def test_retrieve_handles_exception(retriever):
    retriever._vector_search = AsyncMock(side_effect=Exception("fail"))
    docs, method = await retriever.retrieve("hello", method="vector")
    assert docs == []


@pytest.mark.asyncio
async def test_health_check(retriever):
    retriever.qdrant_retriever.health_check.return_value = True
    retriever.bm25_retriever.health_check.return_value = True
    ok = await retriever.health_check()
    assert ok is True


@pytest.mark.asyncio
async def test_get_stats(retriever):
    retriever.qdrant_retriever.get_stats.return_value = {"docs": 10}
    retriever.bm25_retriever.get_stats.return_value = {"docs": 20}
    stats = await retriever.get_stats()
    assert stats["vector_search"]["docs"] == 10
    assert stats["bm25_search"]["docs"] == 20
    assert "fusion_weights" in stats
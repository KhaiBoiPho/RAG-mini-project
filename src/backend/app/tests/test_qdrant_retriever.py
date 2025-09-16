# tests/test_qdrant_retriever.py

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from app.services.retrieval.qdrant_retriever import QdrantRetriever
from qdrant_client.models import Filter, FieldCondition


@pytest.fixture
def mock_settings():
    """Mock settings fixture"""
    with patch('app.services.retrieval.qdrant_retriever.settings') as mock_settings:
        mock_settings.QDRANT_URL = "http://localhost:6333"
        mock_settings.QDRANT_API_KEY = "test-key"
        mock_settings.QDRANT_COLLECTION_NAME = "test_collection"
        mock_settings.SEMANTIC_SEARCH_TOP_K = 5
        mock_settings.SCORE_THRESHOLD = 0.7
        yield mock_settings


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model"""
    mock_embedding = Mock()
    mock_embedding.create_single_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    return mock_embedding


@pytest.fixture
def sample_search_results():
    """Sample Qdrant search results"""
    results = []
    for i in range(3):
        result = Mock()
        result.id = f"doc_{i}"
        result.score = 0.9 - (i * 0.1)  # 0.9, 0.8, 0.7
        result.payload = {
            "content": f"This is document {i} content",
            "source": f"doc_{i}.pdf",
            "chunk_index": i,
            "metadata": {"type": "test"}
        }
        results.append(result)
    return results


class TestQdrantRetriever:
    """Test class for QdrantRetriever"""

    @patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient')
    @patch('app.services.retrieval.qdrant_retriever.EmbeddingModel')
    def test_init(self, mock_embedding_cls, mock_client_cls, mock_settings):
        """Test QdrantRetriever initialization"""
        mock_client = Mock()
        mock_embedding = Mock()
        mock_client_cls.return_value = mock_client
        mock_embedding_cls.return_value = mock_embedding
        
        retriever = QdrantRetriever()
        
        mock_client_cls.assert_called_once_with(
            url="http://localhost:6333",
            api_key="test-key"
        )
        assert retriever.collection_name == "test_collection"
        assert retriever.client == mock_client
        assert retriever.embedding == mock_embedding

    @pytest.mark.asyncio
    @patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient')
    @patch('app.services.retrieval.qdrant_retriever.EmbeddingModel')
    async def test_retrieve_success(self, mock_embedding_cls, mock_client_cls, 
                                   mock_settings, sample_search_results):
        """Test successful document retrieval"""
        # Setup mocks
        mock_client = AsyncMock()
        mock_embedding = Mock()
        mock_client.search = AsyncMock(return_value=sample_search_results)
        mock_embedding.create_single_embedding.return_value = [0.1, 0.2, 0.3]
        
        mock_client_cls.return_value = mock_client
        mock_embedding_cls.return_value = mock_embedding
        
        retriever = QdrantRetriever()
        
        # Test retrieval
        query = "test query"
        results = await retriever.retrieve(query, top_k=3, score_threshold=0.7)
        
        # Assertions
        assert len(results) == 3
        mock_embedding.create_single_embedding.assert_called_once_with(query)
        mock_client.search.assert_called_once()
        
        # Check result format
        for i, result in enumerate(results):
            assert "content" in result
            assert "score" in result
            assert "metadata" in result
            assert "source" in result
            assert result["score"] == 0.9 - (i * 0.1)

    @pytest.mark.asyncio
    @patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient')
    @patch('app.services.retrieval.qdrant_retriever.EmbeddingModel')
    async def test_retrieve_with_filters(self, mock_embedding_cls, mock_client_cls, 
                                        mock_settings, sample_search_results):
        """Test retrieval with filters"""
        mock_client = AsyncMock()
        mock_embedding = Mock()
        mock_client.search = AsyncMock(return_value=sample_search_results)
        mock_embedding.create_single_embedding.return_value = [0.1, 0.2, 0.3]
        
        mock_client_cls.return_value = mock_client
        mock_embedding_cls.return_value = mock_embedding
        
        retriever = QdrantRetriever()
        
        # Test with filters
        filters = {"source": "doc_1.pdf"}
        await retriever.retrieve("test", filters=filters)
        
        # Verify filter was built and passed
        call_args = mock_client.search.call_args
        assert call_args[1]["query_filter"] is not None

    def test_build_qdrant_filter_simple(self, mock_settings):
        """Test building simple Qdrant filter"""
        with patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient'), \
             patch('app.services.retrieval.qdrant_retriever.EmbeddingModel'):
            retriever = QdrantRetriever()
        
        filters = {"source": "test.pdf", "type": "document"}
        qdrant_filter = retriever._build_qdrant_filter(filters)
        
        assert isinstance(qdrant_filter, Filter)
        assert len(qdrant_filter.must) == 2

    def test_build_qdrant_filter_range(self, mock_settings):
        """Test building Qdrant filter with range"""
        with patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient'), \
             patch('app.services.retrieval.qdrant_retriever.EmbeddingModel'):
            retriever = QdrantRetriever()
        
        filters = {"score": {"gte": 0.5, "lte": 1.0}}
        qdrant_filter = retriever._build_qdrant_filter(filters)
        
        assert isinstance(qdrant_filter, Filter)
        assert len(qdrant_filter.must) == 1

    @pytest.mark.asyncio
    @patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient')
    @patch('app.services.retrieval.qdrant_retriever.EmbeddingModel')
    async def test_get_document_by_id_success(self, mock_embedding_cls, mock_client_cls, mock_settings):
        """Test successful document retrieval by ID"""
        # Setup mock result
        mock_point = Mock()
        mock_point.payload = {
            "content": "Test document content",
            "source": "test.pdf",
            "metadata": {"type": "test"}
        }
        
        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(return_value=[mock_point])
        
        mock_client_cls.return_value = mock_client
        mock_embedding_cls.return_value = Mock()
        
        retriever = QdrantRetriever()
        
        # Test retrieval
        result = await retriever.get_document_by_id("doc_123")
        
        assert result is not None
        assert result["content"] == "Test document content"
        assert result["source"] == "test.pdf"
        mock_client.retrieve.assert_called_once_with(
            collection_name="test_collection",
            ids=["doc_123"]
        )

    @pytest.mark.asyncio
    @patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient')
    @patch('app.services.retrieval.qdrant_retriever.EmbeddingModel')
    async def test_get_document_by_id_not_found(self, mock_embedding_cls, mock_client_cls, mock_settings):
        """Test document retrieval by ID when not found"""
        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(return_value=[])
        
        mock_client_cls.return_value = mock_client
        mock_embedding_cls.return_value = Mock()
        
        retriever = QdrantRetriever()
        
        result = await retriever.get_document_by_id("nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    @patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient')
    @patch('app.services.retrieval.qdrant_retriever.EmbeddingModel')
    async def test_health_check_success(self, mock_embedding_cls, mock_client_cls, mock_settings):
        """Test successful health check"""
        # Mock collections response
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(return_value=mock_collections_response)
        
        mock_client_cls.return_value = mock_client
        mock_embedding_cls.return_value = Mock()
        
        retriever = QdrantRetriever()
        
        health = await retriever.health_check()
        
        assert health is True
        mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient')
    @patch('app.services.retrieval.qdrant_retriever.EmbeddingModel')
    async def test_health_check_collection_not_found(self, mock_embedding_cls, mock_client_cls, mock_settings):
        """Test health check when collection doesn't exist"""
        # Mock collections response without our collection
        mock_collection = Mock()
        mock_collection.name = "other_collection"
        
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(return_value=mock_collections_response)
        
        mock_client_cls.return_value = mock_client
        mock_embedding_cls.return_value = Mock()
        
        retriever = QdrantRetriever()
        
        with pytest.raises(Exception, match="Collection test_collection not found"):
            await retriever.health_check()

    @pytest.mark.asyncio
    @patch('app.services.retrieval.qdrant_retriever.AsyncQdrantClient')
    @patch('app.services.retrieval.qdrant_retriever.EmbeddingModel')
    async def test_retrieve_error_handling(self, mock_embedding_cls, mock_client_cls, mock_settings):
        """Test error handling in retrieve method"""
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(side_effect=Exception("Connection error"))
        mock_embedding = Mock()
        mock_embedding.create_single_embedding.return_value = [0.1, 0.2, 0.3]
        
        mock_client_cls.return_value = mock_client
        mock_embedding_cls.return_value = mock_embedding
        
        retriever = QdrantRetriever()
        
        with pytest.raises(Exception):
            await retriever.retrieve("test query")
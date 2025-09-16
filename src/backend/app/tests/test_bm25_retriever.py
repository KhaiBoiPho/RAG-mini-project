import pytest
import asyncio
import pickle
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Import your modules
from app.services.retrieval.bm25_retriever import BM25Retriever
from rank_bm25 import BM25Okapi


@pytest.fixture
def mock_settings():
    """Mock settings fixture"""
    with patch('app.services.retrieval.bm25_retriever.settings') as mock_settings:
        mock_settings.BM25_K1 = 1.5
        mock_settings.BM25_B = 0.75
        mock_settings.QDRANT_COLLECTION_NAME = "test_collection"
        yield mock_settings


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "content": "This is a test document about machine learning",
            "metadata": {"source": "test1.pdf"},
            "source": "test1.pdf",
            "id": "doc1"
        },
        {
            "content": "Another document about artificial intelligence",
            "metadata": {"source": "test2.pdf"},
            "source": "test2.pdf", 
            "id": "doc2"
        },
        {
            "content": "Python programming language tutorial",
            "metadata": {"source": "test3.pdf"},
            "source": "test3.pdf",
            "id": "doc3"
        }
    ]


@pytest.fixture
def mock_bm25_index(sample_documents):
    """Mock BM25 index"""
    mock_index = Mock(spec=BM25Okapi)
    mock_index.avgdl = 10.5
    mock_index.get_scores.return_value = [0.8, 0.6, 0.3]
    return mock_index


class TestBM25Retriever:
    """Test class for BM25Retriever"""

    @patch('app.services.retrieval.bm25_retriever.os.path.exists')
    @patch('app.services.retrieval.bm25_retriever.asyncio.create_task')
    def test_init_with_existing_index(self, mock_create_task, mock_exists, mock_settings):
        """Test initialization when index files exist"""
        mock_exists.return_value = True
        
        with patch.object(BM25Retriever, '_load_index') as mock_load:
            retriever = BM25Retriever()
            mock_load.assert_called_once()
            assert retriever.k1 == 1.5
            assert retriever.b == 0.75
            # Should not create task when index exists
            mock_create_task.assert_not_called()

    @patch('app.services.retrieval.bm25_retriever.os.path.exists')
    @patch('app.services.retrieval.bm25_retriever.asyncio.create_task')
    def test_init_without_existing_index(self, mock_create_task, mock_exists, mock_settings):
        """Test initialization when index files don't exist"""
        mock_exists.return_value = False
        
        retriever = BM25Retriever()
        mock_create_task.assert_called_once()

    @patch('builtins.open')
    @patch('pickle.load')
    def test_load_index_success(self, mock_pickle_load, mock_open, mock_settings, 
                               sample_documents, mock_bm25_index):
        """Test successful index loading"""
        mock_pickle_load.side_effect = [mock_bm25_index, sample_documents]
        
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=True):
            with patch.object(BM25Retriever, '_load_index') as mock_load_method:
                retriever = BM25Retriever()
                # Manually call _load_index for testing
                retriever._load_index()
        
        assert mock_open.call_count == 2

    @patch('builtins.open')
    @patch('pickle.dump')
    @patch('app.services.retrieval.bm25_retriever.os.makedirs')
    def test_save_index_success(self, mock_makedirs, mock_pickle_dump, mock_open, 
                               mock_settings, mock_bm25_index, sample_documents):
        """Test successful index saving"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
                retriever.bm25_index = mock_bm25_index
                retriever.documents = sample_documents
                retriever._save_index()
        
        mock_makedirs.assert_called_once()
        assert mock_pickle_dump.call_count == 2
        assert mock_open.call_count == 2

    def test_tokenize_text_vietnamese(self, mock_settings):
        """Test Vietnamese text tokenization - FIXED"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
        
        with patch('app.services.retrieval.bm25_retriever.word_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["machine", "learning", "is", "great", "123", "a"]
            
            text = "Machine learning is great! 123"
            tokens = retriever._tokenize_text(text)
            
            # Should filter out short tokens and numbers
            expected_tokens = ["machine", "learning", "is", "great"]
            assert tokens == expected_tokens
            
            # FIX: The text is cleaned before tokenization, so "123" is still included
            # Updated assertion to match actual cleaned text
            mock_tokenize.assert_called_once_with("machine learning is great 123", format="list")

    @pytest.mark.asyncio
    async def test_build_index_from_qdrant_success(self, mock_settings, sample_documents):
        """Test successful index building from Qdrant - FIXED with proper mocking"""
        # Mock Qdrant points
        mock_points = []
        for doc in sample_documents:
            point = Mock()
            point.payload = {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "source": doc["source"]
            }
            point.id = doc["id"]
            mock_points.append(point)

        mock_scroll_result = (mock_points, None)
        
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                # FIX: Mock the import at the function level instead of class level
                with patch('app.services.retrieval.bm25_retriever.QdrantRetriever') as mock_qdrant_cls:
                    mock_qdrant = AsyncMock()
                    mock_qdrant.client.scroll = AsyncMock(return_value=mock_scroll_result)
                    mock_qdrant_cls.return_value = mock_qdrant
                    
                    retriever = BM25Retriever()
                    
                    with patch.object(retriever, '_save_index') as mock_save:
                        with patch('app.services.retrieval.bm25_retriever.BM25Okapi') as mock_bm25_cls:
                            mock_bm25_instance = Mock()
                            mock_bm25_cls.return_value = mock_bm25_instance
                            
                            await retriever._build_index_from_qdrant()
                            
                            mock_save.assert_called_once()
                            assert len(retriever.documents) == 3
                            assert retriever.bm25_index == mock_bm25_instance

    @pytest.mark.asyncio
    async def test_retrieve_success(self, mock_settings, sample_documents, mock_bm25_index):
        """Test successful document retrieval"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
                retriever.bm25_index = mock_bm25_index
                retriever.documents = sample_documents
        
        query = "machine learning"
        results = await retriever.retrieve(query, top_k=2, score_threshold=0.5)
        
        # Should return top 2 results above threshold
        assert len(results) == 2
        assert results[0]["score"] == 0.8
        assert results[1]["score"] == 0.6
        assert all("content" in doc for doc in results)

    @pytest.mark.asyncio
    async def test_retrieve_no_index(self, mock_settings):
        """Test retrieval when no index is available"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
        
        results = await retriever.retrieve("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_empty_query_tokens(self, mock_settings, mock_bm25_index, sample_documents):
        """Test retrieval with query that produces no tokens"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
                retriever.bm25_index = mock_bm25_index
                retriever.documents = sample_documents
        
        with patch.object(retriever, '_tokenize_text', return_value=[]):
            results = await retriever.retrieve("123 !@# $%^")
            assert results == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_settings, mock_bm25_index, sample_documents):
        """Test successful health check"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
                retriever.bm25_index = mock_bm25_index
                retriever.documents = sample_documents
        
        with patch.object(retriever, 'retrieve', return_value=[{"test": "doc"}]):
            health = await retriever.health_check()
            assert health is True

    @pytest.mark.asyncio
    async def test_health_check_no_index(self, mock_settings):
        """Test health check when no index is available"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
        
        health = await retriever.health_check()
        assert health is False

    def test_get_index_stats_loaded(self, mock_settings, mock_bm25_index, sample_documents):
        """Test getting stats when index is loaded"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
                retriever.bm25_index = mock_bm25_index
                retriever.documents = sample_documents
        
        stats = retriever.get_index_stats()
        
        assert stats["status"] == "loaded"
        assert stats["doc_count"] == 3
        assert stats["avg_doc_length"] == 10.5
        assert stats["k1"] == 1.5
        assert stats["b"] == 0.75

    def test_get_index_stats_not_loaded(self, mock_settings):
        """Test getting stats when index is not loaded"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
        
        stats = retriever.get_index_stats()
        
        assert stats["status"] == "not_loaded"
        assert stats["doc_count"] == 0

    # NEW TEST: Test async task creation issue
    def test_async_task_creation_in_sync_context(self, mock_settings):
        """Test handling of async task creation in synchronous context"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            # This should not raise an exception even if no event loop is running
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task') as mock_task:
                mock_task.side_effect = RuntimeError("no running event loop")
                
                # Should handle the error gracefully
                retriever = BM25Retriever()
                assert retriever.bm25_index is None
                assert retriever.documents == []

    # NEW TEST: Test text cleaning edge cases
    def test_tokenize_text_edge_cases(self, mock_settings):
        """Test text tokenization with edge cases"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                retriever = BM25Retriever()
        
        with patch('app.services.retrieval.bm25_retriever.word_tokenize') as mock_tokenize:
            # Test empty string
            mock_tokenize.return_value = []
            assert retriever._tokenize_text("") == []
            
            # Test only punctuation
            mock_tokenize.return_value = []
            assert retriever._tokenize_text("!@#$%^&*()") == []
            
            # Test mixed content
            mock_tokenize.return_value = ["valid", "token", "123", "a", "bb"]
            tokens = retriever._tokenize_text("Valid token! 123 a bb")
            assert tokens == ["valid", "token", "bb"]  # Filters out numbers and short tokens

    @pytest.mark.asyncio
    async def test_build_index_error_handling(self, mock_settings):
        """Test error handling during index building"""
        with patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False):
            with patch('app.services.retrieval.bm25_retriever.asyncio.create_task'):
                with patch('app.services.retrieval.bm25_retriever.QdrantRetriever') as mock_qdrant_cls:
                    # Mock connection error
                    mock_qdrant_cls.side_effect = ConnectionError("Cannot connect to Qdrant")
                    
                    retriever = BM25Retriever()
                    
                    with pytest.raises(ConnectionError):
                        await retriever._build_index_from_qdrant()


# Additional utility test for fixing the import issue
class TestBM25RetrieverImports:
    """Test import handling in BM25Retriever"""
    
    @patch('app.services.retrieval.bm25_retriever.os.path.exists', return_value=False)
    @patch('app.services.retrieval.bm25_retriever.asyncio.create_task')
    def test_qdrant_import_in_build_method(self, mock_create_task, mock_exists, mock_settings):
        """Test that QdrantRetriever is imported inside the build method, not at class level"""
        # This should not trigger any imports
        retriever = BM25Retriever()
        
        # QdrantRetriever should only be imported when _build_index_from_qdrant is called
        with patch('app.services.retrieval.bm25_retriever.QdrantRetriever') as mock_import:
            # This import happens inside the method, so should be mockable
            asyncio.run(retriever._build_index_from_qdrant())
            mock_import.assert_called_once()
# vector_database/test/test_embedding_pytest.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import sys

# --------------------------
# Patch settings before import
# --------------------------
mock_settings = MagicMock()
mock_settings.EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
mock_settings.EMBEDDING_DIMENSION = 5
mock_settings.MAX_INPUT_TOKENS = 512
mock_settings.EMBEDDING_BATCH_SIZE = 2

with patch.dict(sys.modules, {'vector_database.config.settings': mock_settings}):
    from vector_database.ingest.embedding import EmbeddingModel

# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def embedding_service():
    service = EmbeddingModel()
    service.dimension = 5  # small dimension for testing
    return service

@pytest.fixture
def mock_sentence_transformer():
    with patch('vector_database.ingest.embedding.SentenceTransformer') as mock_class:
        mock_model = MagicMock()
        # Default encode returns ones vector
        mock_model.encode.return_value = np.ones(5)
        mock_class.return_value = mock_model
        yield mock_class

# --------------------------
# Tests
# --------------------------
def test_create_single_embedding_length(embedding_service, mock_sentence_transformer):
    """Test single embedding returns correct length"""
    result = embedding_service.create_single_embedding("hello world")
    assert len(result) == 5
    assert all(v == 1.0 for v in result)

def test_create_single_embedding_empty_text(embedding_service):
    """Empty text should return zero vector"""
    result = embedding_service.create_single_embedding("")
    assert result == [0.0] * embedding_service.dimension

def test_create_embeddings_batch(embedding_service, mock_sentence_transformer):
    """Batch embeddings return correct number of vectors"""
    texts = ["a", "b", "c"]
    embeddings = embedding_service.create_embeddings_batch(texts, batch_size=2)
    assert len(embeddings) == 3
    assert all(len(e) == 5 for e in embeddings)

def test_cosine_similarity(embedding_service):
    """Cosine similarity calculations"""
    vec1 = [1,0,0,0,0]
    vec2 = [0,1,0,0,0]
    assert embedding_service.cosine_similarity(vec1, vec2) == 0.0

    vec3 = [1,1,0,0,0]
    sim = embedding_service.cosine_similarity(vec3, vec3)
    assert pytest.approx(sim) == 1.0

def test_get_model_info(embedding_service):
    """Check model info returns required fields"""
    info = embedding_service.get_model_info()
    assert "model" in info
    assert "dimension" in info
    assert "max_input_tokens" in info
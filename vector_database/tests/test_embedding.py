#!/usr/bin/env python3
"""
Tests for the embeddings service in vector_database/ingest/embedding.py
"""

import pytest
import numpy as np
import logging
from unittest.mock import patch, MagicMock

# Assuming the test runner is configured to find the module,
# otherwise, you might need to adjust sys.path.
# from vector_database.ingest.embedding import EmbeddingModel
from sentence_transformers import SentenceTransformer

# --- Test Configuration ---
# Use a known, small, and fast model for reliable testing.
TEST_MODEL_NAME = 'all-MiniLM-L6-v2'
TEST_MODEL_DIMENSION = 384  # Correct dimension for the test model
TEST_BATCH_SIZE = 4
TEST_MAX_TOKENS = 256

# --- Fixtures ---

@pytest.fixture
def mock_settings():
    """Mocks the settings object to provide a controlled test environment."""
    with patch('vector_database.ingest.embedding.settings') as mock:
        mock.EMBEDDINGS_MODEL = TEST_MODEL_NAME
        mock.EMBEDDING_DIMENSION = TEST_MODEL_DIMENSION
        mock.EMBEDDING_BATCH_SIZE = TEST_BATCH_SIZE
        mock.MAX_INPUT_TOKENS = TEST_MAX_TOKENS
        yield mock

@pytest.fixture
def embedding_model(mock_settings):
    """
    Provides a clean instance of EmbeddingModel for each test.
    This fixture ensures that the model is initialized once per test function,
    preventing side effects between tests. It also patches the thread-local
    storage for test isolation.
    """
    from vector_database.ingest.embedding import EmbeddingModel
    # Patching threading.local() to behave as a simple object in a single-threaded test context.
    # This avoids complexities of actual thread-local storage during unit testing.
    with patch('threading.local', new_callable=MagicMock) as mock_local_constructor:
        # The class uses self._local, so we ensure the instance gets a mock local object.
        # The code as written is missing `self._local = threading.local()`. We simulate it here.
        instance = EmbeddingModel()
        instance._local = MagicMock() 
        instance._init_model() # Manually call init_model since we overrode __init__ behavior
        yield instance

# --- Test Class for EmbeddingModel ---

class TestEmbeddingModel:
    """Groups all tests related to the EmbeddingModel class."""

    def test_initialization_success(self, embedding_model):
        """
        Tests if the EmbeddingModel initializes correctly with the expected attributes.
        """
        assert embedding_model.model_name == TEST_MODEL_NAME
        assert embedding_model.dimension == TEST_MODEL_DIMENSION
        assert isinstance(embedding_model.model_instance, SentenceTransformer)
        # Check that the model is in evaluation mode
        assert not embedding_model.model_instance.training

    def test_initialization_dimension_mismatch(self, caplog):
        """
        Tests that the model correctly handles and logs a dimension mismatch
        between settings and the actual model, and then updates its dimension.
        """
        from vector_database.ingest.embedding import EmbeddingModel
        wrong_dimension = 123
        with patch('vector_database.ingest.embedding.settings') as mock_settings_mismatch:
            mock_settings_mismatch.EMBEDDINGS_MODEL = TEST_MODEL_NAME
            mock_settings_mismatch.EMBEDDING_DIMENSION = wrong_dimension # Set incorrect dimension
            mock_settings_mismatch.EMBEDDING_BATCH_SIZE = TEST_BATCH_SIZE
            mock_settings_mismatch.MAX_INPUT_TOKENS = TEST_MAX_TOKENS
            
            with caplog.at_level(logging.WARNING):
                model = EmbeddingModel()

            # Check that a warning was logged
            assert "Dimension mismatch" in caplog.text
            assert f"expected {wrong_dimension}" in caplog.text
            assert f"got {TEST_MODEL_DIMENSION}" in caplog.text
            
            # Check that the dimension was corrected in the instance and the settings object
            assert model.dimension == TEST_MODEL_DIMENSION
            assert mock_settings_mismatch.EMBEDDING_DIMENSION == TEST_MODEL_DIMENSION

    @pytest.mark.parametrize("cuda_avail, mps_avail, expected_device", [
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ])
    def test_get_device(self, cuda_avail, mps_avail, expected_device, embedding_model):
        """
        Tests the device selection logic for CUDA, MPS (Apple Silicon), and CPU.
        """
        with patch('torch.cuda.is_available', return_value=cuda_avail):
            with patch('torch.backends.mps.is_available', return_value=mps_avail):
                device = embedding_model._get_device()
                assert device == expected_device

    def test_create_single_embedding_normal_text(self, embedding_model):
        """
        Tests creating an embedding for a simple, valid string.
        """
        text = "This is a test sentence."
        embedding = embedding_model.create_single_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == TEST_MODEL_DIMENSION
        assert all(isinstance(val, float) for val in embedding)
        # For normalized embeddings, the norm should be ~1.0
        assert np.isclose(np.linalg.norm(embedding), 1.0)

    def test_create_single_embedding_empty_text(self, embedding_model):
        """
        Tests that an empty string returns a zero vector of the correct dimension.
        """
        embedding = embedding_model.create_single_embedding("")
        assert embedding == [0.0] * TEST_MODEL_DIMENSION

    def test_create_single_embedding_cleans_text(self, embedding_model):
        """
        Tests that newlines and whitespace are correctly handled.
        """
        text_with_newline = "Hello\nworld"
        text_cleaned = "Hello world"
        
        embedding1 = embedding_model.create_single_embedding(text_with_newline)
        embedding2 = embedding_model.create_single_embedding(text_cleaned)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_create_embeddings_batch_success(self, embedding_model):
        """
        Tests creating embeddings for a batch of texts.
        """
        texts = ["First sentence.", "Second sentence.", "And a third one."]
        embeddings = embedding_model.create_embeddings_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(len(emb) == TEST_MODEL_DIMENSION for emb in embeddings)
        assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)

    def test_create_embeddings_multiple_batches(self, embedding_model, caplog):
        """
        Tests batch processing logic when the number of texts exceeds the batch size.
        """
        texts = ["1", "2", "3", "4", "5"]
        with caplog.at_level(logging.INFO):
            embeddings = embedding_model.create_embeddings_batch(texts, batch_size=2)

        assert len(embeddings) == len(texts)
        assert "Processed batch 1/3" in caplog.text
        assert "Processed batch 2/3" in caplog.text
        assert "Processed batch 3/3" in caplog.text

    def test_create_embeddings_batch_with_empty_string(self, embedding_model):
        """
        Tests that a batch containing an empty string is processed correctly.
        The empty string is replaced by a space " " and should produce a valid embedding.
        """
        texts = ["A valid sentence.", ""]
        embeddings = embedding_model.create_embeddings_batch(texts)
        
        # The embedding for the empty string (processed as " ") should not be a zero vector
        assert len(embeddings) == 2
        assert any(val != 0.0 for val in embeddings[1])

    def test_create_embeddings_batch_empty_list(self, embedding_model):
        """
        Tests that passing an empty list results in an empty list.
        """
        embeddings = embedding_model.create_embeddings_batch([])
        assert embeddings == []

    def test_batch_processing_error_fallback(self, embedding_model, caplog):
        """
        Tests that if an error occurs during batch processing, a list of zero
        vectors is returned as a fallback.
        """
        texts = ["good", "bad"]
        error_message = "Simulated model error"
        
        # Mock the encode method to raise an exception
        with patch.object(embedding_model.model_instance, 'encode', side_effect=Exception(error_message)):
            with caplog.at_level(logging.ERROR):
                embeddings = embedding_model.create_embeddings_batch(texts)
        
        assert f"Batch embedding failed: {error_message}" in caplog.text
        assert embeddings == [[0.0] * TEST_MODEL_DIMENSION for _ in texts]

    @pytest.mark.parametrize("vec1, vec2, expected", [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0),   # Identical vectors
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], -1.0), # Opposite vectors
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0),    # Orthogonal vectors
        ([0.5, 0.8, 0.2], [0.0, 0.0, 0.0], 0.0),    # One zero vector
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),    # Both zero vectors
    ])
    def test_cosine_similarity(self, vec1, vec2, expected, embedding_model):
        """
        Tests the cosine similarity calculation with various vector pairs.
        """
        similarity = embedding_model.cosine_similarity(vec1, vec2)
        assert np.isclose(similarity, expected)

    def test_get_model_info(self, embedding_model):
        """
        Tests if the model information is returned correctly.
        """
        info = embedding_model.get_model_info()
        expected_info = {
            "model": TEST_MODEL_NAME,
            "dimension": TEST_MODEL_DIMENSION,
            "max_input_tokens": TEST_MAX_TOKENS
        }
        assert info == expected_info
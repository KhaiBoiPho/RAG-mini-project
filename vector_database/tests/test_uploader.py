# test_uploader.py

import pytest
import json
from unittest.mock import patch, mock_open
from vector_database.ingest.uploader import Uploader

TEST_HOST = "localhost"
TEST_PORT = 6333
TEST_COLLECTION = "test_collection"
TEST_VECTORS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
TEST_METADATAS = [{"id": 1}, {"id": 2}]
TEST_FILEPATH = "test_embeddings.json"

# --- Fixtures ---

@pytest.fixture
def uploader():
    """Provide a clean Uploader instance with patched QdrantClient"""
    with patch("vector_database.ingest.uploader.QdrantClient") as mock_client_class:
        instance = mock_client_class.return_value
        uploader = Uploader(host=TEST_HOST, port=TEST_PORT, collection=TEST_COLLECTION)
        uploader.client = instance  # inject mock client
        yield uploader

# --- Tests ---

class TestUploader:

    def test_initialization(self):
        """Test Uploader initializes with correct attributes"""
        with patch("vector_database.ingest.uploader.QdrantClient") as mock_client_class:
            uploader = Uploader(host=TEST_HOST, port=TEST_PORT, collection=TEST_COLLECTION)
            assert uploader.host == TEST_HOST
            assert uploader.port == TEST_PORT
            assert uploader.collection == TEST_COLLECTION
            mock_client_class.assert_called_once_with(host=TEST_HOST, port=TEST_PORT)

    def test_upload_success(self, uploader):
        """Test uploading vectors calls QdrantClient.upsert"""
        uploader.upload(TEST_VECTORS, TEST_METADATAS)
        assert uploader.client.upsert.called
        args, kwargs = uploader.client.upsert.call_args
        assert kwargs["collection_name"] == TEST_COLLECTION
        points = kwargs["points"]
        assert len(points) == len(TEST_VECTORS)
        assert all(hasattr(p, "id") and hasattr(p, "vector") and hasattr(p, "payload") for p in points)

    def test_upload_error(self, uploader):
        """Test upload raises and logs exception"""
        uploader.client.upsert.side_effect = Exception("Upload failed")
        with pytest.raises(Exception) as e:
            uploader.upload(TEST_VECTORS, TEST_METADATAS)
        assert "Upload failed" in str(e.value)

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_to_file_success(self, mock_json_dump, mock_file, uploader):
        """Test saving embeddings to file"""
        uploader.save_to_file(TEST_VECTORS, TEST_METADATAS, TEST_FILEPATH)
        mock_file.assert_called_once_with(TEST_FILEPATH, "w", encoding="utf-8")
        mock_json_dump.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_load_from_file_success(self, mock_json_load, mock_file, uploader):
        """Test loading embeddings from file"""
        mock_json_load.return_value = [
            {"vector": v, "metadata": m} for v, m in zip(TEST_VECTORS, TEST_METADATAS)
        ]
        vectors, metadatas = uploader.load_from_file(TEST_FILEPATH)
        mock_file.assert_called_once_with(TEST_FILEPATH, "r", encoding="utf-8")
        mock_json_load.assert_called_once()
        assert vectors == TEST_VECTORS
        assert metadatas == TEST_METADATAS

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump", side_effect=Exception("File write failed"))
    def test_save_to_file_error(self, mock_json_dump, mock_file, uploader):
        """Test save_to_file raises on error"""
        with pytest.raises(Exception) as e:
            uploader.save_to_file(TEST_VECTORS, TEST_METADATAS, TEST_FILEPATH)
        assert "File write failed" in str(e.value)

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load", side_effect=Exception("File read failed"))
    def test_load_from_file_error(self, mock_json_load, mock_file, uploader):
        """Test load_from_file raises on error"""
        with pytest.raises(Exception) as e:
            uploader.load_from_file(TEST_FILEPATH)
        assert "File read failed" in str(e.value)
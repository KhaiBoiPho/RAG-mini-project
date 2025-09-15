import pytest
from unittest.mock import patch, MagicMock
from vector_database.ingest.text_loader import DocumentLoader
from langchain_core.documents import Document

@pytest.mark.parametrize(
    "filepath, expected_loader",
    [
        ("test.txt", "TextLoader"),
        ("test.pdf", "PyPDFLoader"),
        ("test.docx", "UnstructuredWordDocumentLoader"),
        ("test.html", "UnstructuredHTMLLoader"),
        ("test.md", "UnstructuredMarkdownLoader"),
        ("test.pptx", "UnstructuredPowerPointLoader"),
        ("test.xlsx", "UnstructuredExcelLoader"),
        ("test.json", "JSONLoader"),
        ("test.csv", "CSVLoader"),
        ("test.epub", "UnstructuredEPubLoader"),
    ]
)
def test_document_loader_load(filepath, expected_loader):
    """Test that DocumentLoader calls the correct loader class based on file suffix"""
    
    mock_docs = [Document(page_content="dummy text", metadata={"source": filepath})]

    # Patch the loader class so that its instance returns mock_docs on load()
    with patch(f"vector_database.ingest.text_loader.{expected_loader}") as MockLoaderClass:
        mock_instance = MagicMock()
        mock_instance.load.return_value = mock_docs
        MockLoaderClass.return_value = mock_instance

        loader_instance = DocumentLoader(filepath)
        docs = loader_instance.load()

        # Ensure the loader class was instantiated correctly
        MockLoaderClass.assert_called()
        # Ensure returned docs match our mock
        assert docs == mock_docs
        # Ensure metadata source is correct
        assert docs[0].metadata["source"] == filepath

def test_document_loader_unsupported_file():
    """Test that unsupported file types raise ValueError"""
    loader = DocumentLoader("file.unsupported")
    with pytest.raises(ValueError):
        loader.load()

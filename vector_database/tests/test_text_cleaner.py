import pytest
from vector_database.ingest.text_cleaner import TextCleaner
from langchain_core.documents import Document

@pytest.fixture
def sample_docs():
    """Create sample documents for testing purposes"""
    doc1 = Document(
        page_content="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\nCHƯƠNG I\nĐiều 1. Nội dung điều 1",
        metadata={"source": "doc1"}
    )
    doc2 = Document(
        page_content="QUỐC HỘI\nCHƯƠNG II\nĐiều 2: Nội dung điều 2  ",
        metadata={"source": "doc2"}
    )
    return [doc1, doc2]

def test_clean_headers(sample_docs):
    """Test that unwanted headers are removed"""
    cleaned_docs = TextCleaner.clean(sample_docs)
    for doc in cleaned_docs:
        assert "QUỐC HỘI" not in doc.page_content
        assert "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" not in doc.page_content
        assert "Độc lập - Tự do - Hạnh phúc" not in doc.page_content

def test_normalize_whitespace(sample_docs):
    """Test that extra whitespaces are normalized"""
    cleaned_docs = TextCleaner.clean(sample_docs)
    for doc in cleaned_docs:
        # No double spaces should exist
        assert "  " not in doc.page_content
        # Text should start with CHƯƠNG or Điều after stripping leading spaces/newlines
        stripped_content = doc.page_content.lstrip()
        assert stripped_content.startswith("CHƯƠNG") or stripped_content.startswith("Điều")

def test_format_dieu_chuong(sample_docs):
    """Test that 'Điều' and 'CHƯƠNG' are correctly formatted with newlines"""
    cleaned_docs = TextCleaner.clean(sample_docs)
    content1 = cleaned_docs[0].page_content
    content2 = cleaned_docs[1].page_content

    # There should be a newline before CHƯƠNG and Điều
    assert "\nCHƯƠNG I" in content1
    assert "\nĐiều 1." in content1
    assert "\nCHƯƠNG II" in content2
    assert "\nĐiều 2" in content2

def test_metadata_preserved(sample_docs):
    """Test that metadata is preserved after cleaning"""
    cleaned_docs = TextCleaner.clean(sample_docs)
    for original, cleaned in zip(sample_docs, cleaned_docs):
        assert original.metadata == cleaned.metadata
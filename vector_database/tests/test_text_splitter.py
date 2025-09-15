#!/usr/bin/env python3
"""
Tests for TextSplitter and DocumentProcessor
"""

import pytest
from vector_database.ingest.text_splitter import TextSplitter, DocumentProcessor

# --- Fixtures ---

@pytest.fixture
def splitter():
    return TextSplitter(chunk_size=20, chunk_overlap=5)

@pytest.fixture
def processor():
    return DocumentProcessor()


# --- Tests for TextSplitter ---

class TestTextSplitter:

    def test_clean_text_removes_html_and_emoji(self, splitter):
        text = "Hello <b>world</b>! 😀 &amp; test"
        cleaned = splitter._clean_text(text)
        assert "<b>" not in cleaned
        assert "😀" not in cleaned
        assert "&" not in cleaned  # HTML entity decoded
        assert "Hello world ! test" in cleaned

    def test_split_sentences_basic(self, splitter):
        text = "Hello world. How are you? I'm fine!"
        sentences = splitter._split_sentences(text)
        assert sentences == ["Hello world", "How are you", "I'm fine"]

    def test_split_text_chunks_with_overlap(self, splitter):
        text = "This is a test. Another sentence. And one more sentence."
        chunks = splitter.split_text(text)
        # Check that chunks are <= chunk_size or contain overlap
        for chunk in chunks:
            assert len(chunk) <= splitter.chunk_size + splitter.chunk_overlap

    def test_get_overlap_returns_correct_length(self, splitter):
        text = "This is a test chunk"
        overlap = splitter._get_overlap(text)
        assert len(overlap) <= splitter.chunk_overlap
        assert overlap in text

    def test_split_text_empty_string(self, splitter):
        chunks = splitter.split_text("")
        assert chunks == []


# --- Tests for DocumentProcessor ---

class TestDocumentProcessor:

    def test_process_text_splits_into_chunks(self, processor):
        text = "Sentence one. Sentence two. Sentence three."
        source_name = "test_doc"
        chunks = processor.process_text(text, source_name)
        assert isinstance(chunks, list)
        assert all('content' in c and 'id' in c for c in chunks)
        assert all(c['source'] == source_name for c in chunks)
        assert len(chunks) > 0

    def test_process_text_metadata(self, processor):
        text = "Sentence one. Sentence two."
        source_name = "doc1"
        chunks = processor.process_text(text, source_name)
        for c in chunks:
            meta = c['metadata']
            assert 'chunk_size' in meta
            assert 'total_chunks' in meta
            assert meta['chunk_size'] == len(c['content'])
            assert meta['total_chunks'] == len(chunks)

    def test_process_file_reads_file(self, tmp_path, processor):
        # Create a temporary text file
        file_path = tmp_path / "test.txt"
        content = "Line one. Line two."
        file_path.write_text(content, encoding="utf-8")

        chunks = processor.process_file(str(file_path))
        assert len(chunks) > 0
        for c in chunks:
            assert 'content' in c
            assert 'id' in c
            assert c['source'] == str(file_path)
            assert c['metadata']['file_path'] == str(file_path)

    def test_process_file_invalid_path_raises(self, processor):
        with pytest.raises(Exception) as e:
            processor.process_file("non_existing_file.txt")
        assert "Error processing file" in str(e.value)

    def test_process_text_empty_string(self, processor):
        chunks = processor.process_text("", "source")
        assert chunks == []
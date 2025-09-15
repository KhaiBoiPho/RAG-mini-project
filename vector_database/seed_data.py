#!/usr/bin/env python3
"""
Seed Qdrant with legal documents
"""

import logging
from ingest.text_loader import DocumentLoader
from ingest.text_cleaner import TextCleaner
from ingest.text_splitter import DocumentProcessor
from ingest.embedding import EmbeddingsModel
from ingest.uploader import Uploader
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_data(filepath: str, raw_text: str = None):
    """
    Full pipeline: load → clean → split → embed → upload
    - If raw_text != None → process raw text
    - If raw_text == None → process a file
    """
    cleaner = TextCleaner()
    processor = DocumentProcessor()

    if raw_text:  
        logger.info("Processing raw text")
        cleaned = cleaner.clean(raw_text)
        processed_chunks = processor.process_text(cleaned, source_name="raw_input")
    else:
        logger.info(f"Loading and processing file: {filepath}")
        loader = DocumentLoader(filepath)
        docs = loader.load()
        cleaned_texts = [cleaner.clean(doc.page_content) for doc in docs]
        # Use process_file so metadata includes file_path
        processed_chunks = processor.process_file(filepath, source_name=filepath)

    logger.info("Extracting content + metadata")
    splits = [chunk["content"] for chunk in processed_chunks]
    metadatas = [chunk["metadata"] | {
        "id": chunk["id"],
        "source": chunk["source"],
        "chunk_index": chunk["chunk_index"],
    } for chunk in processed_chunks]

    logger.info("Creating embeddings")
    embedder = EmbeddingsModel()
    vectors = embedder.create_embeddings_batch(splits)

    logger.info("Uploading to Qdrant")
    uploader = Uploader(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        collection=settings.qdrant_config.get("collection_name", settings.QDRANT_COLLECTION_NAME)
    )
    uploader.upload(vectors, metadatas)

    logger.info("Seeding complete.")

if __name__ == "__main__":
    # Example: seed from a file
    seed_data("dataset/luat2008.txt")

    # Example: seed directly from raw text
    # seed_data(filepath=None, raw_text="Civil Code 2015 states ...")
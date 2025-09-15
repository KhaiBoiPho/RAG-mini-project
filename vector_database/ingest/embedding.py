#!/usr/bin/env python3
"""
Embeddings service using HuggingFace models
All dataset will be convert to embeddings in local with HuggingFace models
"""

import sys
from typing import List
import numpy as np
from vector_database.config import settings
import logging
from sentence_transformers import SentenceTransformer
import torch

# sys.path.insert(0, '/home/khai/Desktop/RAG-mini-project/')

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        self.model_name = settings.EMBEDDINGS_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.max_input_tokens = settings.MAX_INPUT_TOKENS
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

        # Store actual SentenceTransformer model
        self.model_instance = None

        # Initialize model
        self._init_model()

    def _init_model(self):
        """Initialize SentenceTransformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            device = self._get_device()
            logger.info(f"Using device: {device}")

            self.model_instance = SentenceTransformer(self.model_name, device=device)
            self.model_instance.eval()

            # Verify dimension
            test_embedding = self.model_instance.encode("test", convert_to_numpy=True)
            actual_dimension = len(test_embedding)

            if actual_dimension != self.dimension:
                logger.warning(f"Dimension mismatch: expected {self.dimension}, got {actual_dimension}")
                self.dimension = actual_dimension
                settings.EMBEDDING_DIMENSION = actual_dimension

            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise e

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_model(self) -> SentenceTransformer:
        """Return the model instance"""
        return self.model_instance

    def create_single_embedding(self, text: str) -> List[float]:
        try:
            text = text.replace("\n", " ").strip()
            if not text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension

            with torch.no_grad():
                embedding = self.model_instance.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return [0.0] * self.dimension

    def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        batch_size = batch_size or self.batch_size or settings.EMBEDDING_BATCH_SIZE
        embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings.extend(self._process_batch(batch))
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            return [[0.0] * self.dimension for _ in texts]

    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            cleaned_texts = [(text.replace("\n", " ").strip() or " ") for text in texts]
            with torch.no_grad():
                embeddings = self.model_instance.encode(
                    cleaned_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            return [[0.0] * self.dimension for _ in texts]

    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    def get_model_info(self) -> dict:
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "max_input_tokens": self.max_input_tokens
        }

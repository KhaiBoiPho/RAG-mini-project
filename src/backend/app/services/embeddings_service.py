#!/usr/bin/env python3
"""
Embeddings service using HuggingFace models
"""

import openai
from typing import List, Union
import numpy as np
from app.config import settings
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = settings.EMBEDDINGS_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        
        try:
            self.model = SentenceTransformer(self.model)
        except Exception as e:
            logger.error(f"Error loading local embedding model: {e}")
            raise e
        
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        try:
            # Clean text
            text = text.replace("\n", " ").strip()
            
            if not text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            if len(embedding) != self.dimension:
                logger.warning(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")
            
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.dimension
        
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Create embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._process_batch(batch)
            embeddings.extend(batch_embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return embeddings
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts"""
        try:
            # Clean texts
            cleaned_texts = []
            for text in texts:
                cleaned_text = text.replace("\n", " ").strip()
                if not cleaned_text:
                    cleaned_text = " "
                cleaned_texts.append(cleaned_text)
            
            embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimension for _ in texts]
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "max_input_tokens": settings.MAX_INPUT_TOKENS
        }
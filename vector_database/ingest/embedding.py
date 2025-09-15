#!/usr/bin/env python3
"""
Embeddings service using HuggingFace models
All dataset will be convert to embeddings in local with HuggingFace models
"""

from typing import List, Union
import numpy as np
from ..config import settings
import logging
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        self.model = settings.EMBEDDINGS_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.max_input_tokens = settings.MAX_INPUT_TOKENS
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(
            max_workers=settings.EMBEDDING_MAX_WORKERS,
            thread_name_prefix="embedding_worker"
        )
        
        # Thread-local storage for model instances
        self._local = threading.local()
        
        # Initialize main for model instances
        self._init_model()
    
    def _init_model(self):
        """Initialize SentenceTransformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model}")
            
            # Configure device
            device = self._get_device()
            logger.info(f"Using device: {device}")
            
            # Load model
            model = SentenceTransformer(self.model, device=device)
            
            # Set model to evaluation mode
            model.eval()
            
            # Store in thread-local storage
            self._local.model = model
            
            # Verify dimension
            test_embedding = model.encode("test", convert_to_numpy=True)
            actual_dimension = len(test_embedding)
            
            if actual_dimension != self.dimension:
                logger.warning(f"Dimension mismatch: expected {self.dimension}, got {actual_dimension}")
                
                # Update dimension to actual
                self.dimension = actual_dimension
                settings.EMBEDDING_DIMENSION = actual_dimension
                
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise e
        
    def _get_device(self) -> str:
        """Get optimal device for inference"""
        if torch.cuda.is_available(): # Nvidia GPU
            return "cuda"
        elif torch.backends.mps.is_available(): # Apple Sillicon
            return "mps"
        else: 
            return "cpu"
        
    def _get_model(self) -> SentenceTransformer:
        """Get thread-local model instance"""
        if not hasattr(self._local, 'model'):
            device = self._get_device()
            self._local.model = SentenceTransformer(self.model, device=device)
            self._local.model.eval()
        return self._local.model
    
    def create_single_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text in thread pool"""
        try:
            # Clean text
            text = text.replace("\n", " ").strip()
            
            # get model
            model = self._get_model()
            
            if not text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension
            
            with torch.no_grad():
                embedding = model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True # L2 normalization for better similarity
                )
            
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error in thread pool embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.dimension
        
    def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Create embeddings for multiple texts in batches"""
        batch_size = batch_size or self.batch_size or settings.EMBEDDING_BATCH_SIZE
        embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._process_batch(batch)
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            return embeddings
                
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            # If error -> return zero vector fallback
            return [[0.0] * self.dimension for _ in texts]
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts"""
        model = self._get_model()
        
        try:
            # Clean texts
            cleaned_texts = []
            for text in texts:
                cleaned_text = text.replace("\n", " ").strip()
                if not cleaned_text:
                    cleaned_text = " "
                cleaned_texts.append(cleaned_text)
            
            with torch.no_grad():
                embeddings = model.encode(
                    cleaned_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            
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
            "max_input_tokens": self.max_input_tokens
        }
# uploader.py

import json
import logging
import uuid
from ..config import settings 
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

class Uploader:
    def __init__(self, host=None, port=None, collection="legal_docs"):
        self.port = settings.QDRANT_PORT or port
        self.host = settings.QDRANT_HOST or host
        self.collection = collection
        
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}, collection={self.collection}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def upload(self, vectors, metadatas):
        """Upload embeddings and metadata to Qdrant"""
        try:
            points = [
                PointStruct(id=str(uuid.uuid4().hex()), vector=vec, payload=meta)
                for vec, meta in zip(vectors, metadatas)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Uploaded {len(points)} vectors to Qdrant collection '{self.collection}'")
        except Exception as e:
            logger.error(f"Error uploading to Qdrant: {e}")
            raise

    def save_to_file(self, vectors, metadatas, filepath="embeddings.json"):
        """Save embeddings + metadata to local JSON file"""
        try:
            data = [{"vector": v, "metadata": m} for v, m in zip(vectors, metadatas)]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(data)} embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings to file {filepath}: {e}")
            raise

    def load_from_file(self, filepath="embeddings.json"):
        """Load embeddings + metadata from local JSON file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            vectors = [item["vector"] for item in data]
            metadatas = [item["metadata"] for item in data]
            logger.info(f"Loaded {len(data)} embeddings from {filepath}")
            return vectors, metadatas
        except Exception as e:
            logger.error(f"Error loading embeddings from file {filepath}: {e}")
            raise
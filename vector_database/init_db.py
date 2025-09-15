#!/usr/bin/env python3
"""
Initialize Qdrant database and collection for RAG
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_qdrant():
    """Initialize Qdrant collection"""
    
    # client = QdrantClient(
    #     host=settings.QDRANT_HOST,
    #     port=settings.QDRANT_PORT,
    #     api_key=settings.QDRANT_API_KEY,
    # )
    
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=settings.QDRANT_TIMEOUT
    )

    collection_name = settings.QDRANT_COLLECTION_NAME
    host = settings.QDRANT_HOST
    port = settings.QDRANT_PORT
    timeout = settings.QDRANT_TIMEOUT
    api_key = settings.QDRANT_API_KEY
    vector_size = settings.EMBEDDING_DIMENSION
    distance = settings.QDRANT_DISTANCE

    # Map string distance to Qdrant enum
    distance_map = {
        "Cosine": Distance.COSINE,
        "Dot": Distance.DOT,
        "Euclid": Distance.EUCLID,
    }
    distance_metric = distance_map.get(distance, Distance.COSINE)

    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists in Qdrant.")
    else:
        logger.info(f"Creating collection '{collection_name}' in Qdrant...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_metric),
        )
        logger.info(f"Collection '{collection_name}' created successfully.")

if __name__ == "__main__":
    init_qdrant()
#!/usr/bin/env python3
"""
Reset Qdrant collection (drop & recreate)
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_qdrant():
    """Drop and recreate Qdrant collection"""
    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )

    collection_name = settings.qdrant_config.get("collection_name", settings.QDRANT_COLLECTION_NAME)
    vector_size = settings.qdrant_config.get("vector_size", settings.EMBEDDING_DIMENSION)
    distance = settings.qdrant_config.get("distance", "Cosine")

    distance_map = {
        "Cosine": Distance.COSINE,
        "Dot": Distance.DOT,
        "Euclid": Distance.EUCLID,
    }
    distance_metric = distance_map.get(distance, Distance.COSINE)

    # Delete if exists
    if client.collection_exists(collection_name):
        logger.info(f"Dropping existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    logger.info(f"Recreating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance_metric),
    )
    logger.info(f"Collection '{collection_name}' reset successfully.")

if __name__ == "__main__":
    reset_qdrant()
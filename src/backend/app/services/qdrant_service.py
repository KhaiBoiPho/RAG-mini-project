#!/usr/bin/env python3
"""
Qdrant vector database service
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import uuid
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = settings.QDRANT_COLLECTION
        self.dimension = settings.EMBEDDING_DIMENSION
    
    async def initialize(self) -> None:
        """Initialize Qdrant client and verify connection"""
        try:
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
                timeout=settings.QDRANT_TIMEOUT
            )
            
            # Check if collection exists
            existing_collections = await self.client.get_collections()
            collection_names = [c.name for c in existing_collections.collections]
            
            if self.collection_name not in collection_names:
                await self._create_collection()
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
            
            # Test connection by getting collection info
            collection_info = await self.client.get_collection(
                collection_name=self.collection_name
            )
            
            logger.info(f"Connected to Qdrant collection: {self.collection_name}")
            logger.info(f"Vector size: {collection_info.config.params.vectors.size}")
            logger.info(f"Points count: {collection_info.points_count}")
            
            # Log sample point structure
            await self._log_sample_point()
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise
    
    async def _log_sample_point(self) -> None:
        """Log structure of a sample point for debugging"""
        try:
            first_point = None
            async for point in self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=True
            ):
                first_point = point
                break

            if first_point:
                logger.info("Sample point structure:")
                logger.info(f"Point ID: {first_point.id}")
                logger.info(f"Payload keys: {list(first_point.payload.keys())}")

                if 'metadata' in first_point.payload:
                    metadata_keys = list(first_point.payload['metadata'].keys())
                    logger.info(f"Metadata keys: {metadata_keys}")

        except Exception as e:
            logger.warning(f"Could not log sample point: {str(e)}")

    
    async def _create_collection(self):
        try:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection {self.collection_name} created")
        except Exception as e:
            logger.error(f"Failed to create collection '{self.collection_name}': {str(e)}")
            raise
    
    async def insert_documents(self, documents: List[Dict[str, Any]]):
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=doc["embedding"],
                payload={
                    "content": doc["content"],
                    "source": doc["source"]
                }
            ) for doc in documents
        ]
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            await self.client.upsert(collection_name=self.collection_name, points=batch)
            logger.info(f"Inserted batch {i//batch_size + 1}")
    
    async def search_similar(self, query_vector: List[float], top_k: int = None, 
                             score_threshold: float = None, collection_name: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of scored points        
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        collection_name = collection_name or self.collection_name
        top_k = top_k or settings.SEMANTIC_SEARCH_TOP_K
        score_threshold = score_threshold or settings.SCORE_THRESHOLD
        
        try:
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=score_threshold
            )
            
            logger.info(f"Semantic search returned {len(search_results)} results")
            if search_results:
                logger.debug(f"Top result score: {search_results[0].score:.4f}")
            
            results = []
            for hit in search_results:
                result = {
                    'content': hit.payload.get('content'),
                    'score': hit.score,
                    'source': hit.payload.get('source'),
                    'chunk_id': hit.payload.get('chunk_id'),
                    'metadata': hit.payload.get('metadata', {})
                }
                results.append(result)
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    async def search_with_filter(self, query_vector: List[float], filter_conditions: Dict[str, Any],
                          limit: int = None, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search with additional filters"""
        collection_name = collection_name or self.collection_name
        limit = limit or settings.MAX_RESULTS
        
        try:
            search_result = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                ),
                limit=limit,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                result = {
                    'content': hit.payload.get('content'),
                    'score': hit.score,
                    'source': hit.payload.get('source'),
                    'chunk_id': hit.payload.get('chunk_id'),
                    'metadata': hit.payload.get('metadata', {})
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in filtered search: {str(e)}")
            return []
    
    async def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get collection information"""
        collection_name = collection_name or self.collection_name
        
        try:
            info = await self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'points_count': info.points_count,
                'status': info.status,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    async def delete_collection(self, collection_name: str = None) -> bool:
        """Delete collection"""
        collection_name = collection_name or self.collection_name
        
        try:
            await self.client.delete_collection(collection_name)
            logger.info(f"Collection {collection_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
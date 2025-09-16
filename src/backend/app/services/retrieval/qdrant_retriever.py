# src/backend/app/services/retrieval/qdrant_retriever.py
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from src.backend.app.config import settings
from src.backend.app.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantRetriever:
    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        # Import embedding model here to avoid issues
        self._embedding_model = None
    
    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            try:
                # Adjust import path based on your project structure
                from vector_database.ingest.embedding import EmbeddingModel
                self._embedding_model = EmbeddingModel()
            except ImportError:
                # Fallback - try alternative import paths
                try:
                    from vector_database.ingest.embedding import EmbeddingModel
                    self._embedding_model = EmbeddingModel()
                except ImportError:
                    logger.error("Could not import EmbeddingModel")
                    raise ImportError("EmbeddingModel not found in expected locations")
        return self._embedding_model
    
    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents from Qdrant vector database
        """
        try:
            # Use default values if not provided
            if top_k is None:
                top_k = getattr(settings, 'SEMANTIC_SEARCH_TOP_K', 10)
            if score_threshold is None:
                score_threshold = getattr(settings, 'SCORE_THRESHOLD', 0.5)
            
            logger.info(f"Vector search for: '{query[:50]}...' (top_k={top_k}, threshold={score_threshold})")
            
            # Generate query embedding from local Huggingface model
            embedding_model = self._get_embedding_model()
            query_embedding = embedding_model.create_single_embedding(query)
            
            # Prepare filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_qdrant_filter(filters)
                logger.info(f"Applied filters: {filters}")
            
            # Search in Qdrant
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False  # Don't return vectors unless needed
            )
            
            # Convert results to standard format
            documents = []
            for result in search_results:
                # Clean metadata structure to avoid duplication
                metadata = result.payload.copy()
                content = metadata.pop("content", "")
                source = metadata.pop("source", "unknown")
                
                doc = {
                    "content": content,
                    "score": float(result.score),
                    "source": source,
                    "metadata": {
                        "id": str(result.id),
                        "chunk_index": metadata.get("chunk_index"),
                        **{k: v for k, v in metadata.items() 
                           if k not in ["id", "chunk_index"]}
                    }
                }
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} vector matches")
            return documents
        
        except Exception as e:
            logger.error(f"Error in vector retrieval: {str(e)}")
            raise
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Build Qdrant filter from dictionary"""
        try:
            conditions = []
            
            for field, value in filters.items():
                if isinstance(value, dict):
                    # Handle range filters
                    if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                        conditions.append(
                            FieldCondition(
                                key=field,
                                range=Range(
                                    gte=value.get("gte"),
                                    lte=value.get("lte"),
                                    gt=value.get("gt"),
                                    lt=value.get("lt")
                                )
                            )
                        )
                elif isinstance(value, list):
                    # Handle multiple values (OR condition)
                    for v in value:
                        conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=v))
                        )
                else:
                    # Handle exact match
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )

            return Filter(must=conditions) if conditions else None
        
        except Exception as e:
            logger.error(f"Error building Qdrant filter: {str(e)}")
            return None
            
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific document by ID"""
        try:
            result = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True,
                with_vectors=False
            )

            if result:
                point = result[0]
                return {
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload,
                    "source": point.payload.get("source", "unknown"),
                    "id": str(point.id)
                }
            return None

        except Exception as e:
            logger.error(f"Error retrieving document by ID {doc_id}: {str(e)}")
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = await self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.name,
                "status": collection_info.status,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "config": {
                    "vector_size": collection_info.config.params.vector_size,
                    "distance": collection_info.config.params.distance.value
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            # Check if client can connect
            collections = await self.client.get_collections()
            
            # Check if our collection exists
            collection_exists = any(
                col.name == self.collection_name for col in collections.collections
            )
            
            if not collection_exists:
                logger.error(f"Collection {self.collection_name} not found")
                return False
            
            # Check if embedding model is working
            embedding_model = self._get_embedding_model()
            test_embedding = embedding_model.create_single_embedding("test")
            
            if not test_embedding or len(test_embedding) == 0:
                logger.error("Embedding model not working")
                return False
            
            logger.info("Qdrant health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False
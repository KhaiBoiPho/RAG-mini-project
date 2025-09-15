# src/backend/app/services/retrieval/qdrant_retriever.py
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
import openai
from ...config import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)

class QdrantRetriever:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection_name
        self.embedding_model = settings.embedding_model
        openai.api_key = settings.openai_api_key
        
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents from Qdrant vector database
        """
        try:
            logger.info(f"Vector search for: {query[:50]}...")
            
            # Generate query embedding
            query_embedding = await self._get_embedding(query)
            
            # Prepare filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_qdrant_filter(filters)
            
            # Search in Qdrant
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter
            )
            
            # Convert results to standard format
            documents = []
            for result in search_results:
                doc = {
                    "content": result.payload.get("content", ""),
                    "score": float(result.score),
                    "metadata": {
                        "id": str(result.id),
                        "chunk_id": result.payload.get("chunk_id"),
                        "source": result.payload.get("source"),
                        **result.payload.get("metadata", {})
                    },
                    "source": result.payload.get("source", "unknown")
                }
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} vector matches")
            return documents
            
        except Exception as e:
            logger.error(f"Error in vector retrieval: {str(e)}")
            raise
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = await openai.Embedding.acreate(
                input=text,
                model=self.embedding_model
            )
            return response['data'][0]['embedding']
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary"""
        conditions = []
        
        for field, value in filters.items():
            if isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value:
                    conditions.append(
                        FieldCondition(
                            key=field,
                            range=Range(
                                gte=value.get("gte"),
                                lte=value.get("lte")
                            )
                        )
                    )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(key=field, match={"value": value})
                )
        
        return Filter(must=conditions) if conditions else None
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific document by ID"""
        try:
            result = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id]
            )
            
            if result:
                point = result[0]
                return {
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "source": point.payload.get("source", "unknown")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document by ID: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            collections = await self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                raise Exception(f"Collection {self.collection_name} not found")
            
            return True
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            raise
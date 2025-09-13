from qdrant_client import QdrantClient, models
import logging

logger = logging.getLogger("rag-backend.qdrant")

class QdrantService:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def search_exact(self, query: str):
        try:
            scroll_filter = models.Filter(
                must=[models.FieldCondition(
                    key="metadata.question",
                    match=models.MatchValue(value=query)
                )]
            )
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1,
                with_payload=True
            )
            if results and len(results[0]) > 0:
                return results[0][0]
            return None
        except Exception as e:
            logger.error(f"Exact search error: {e}")
            return None

    def search_semantic(self, vector, top_k=5):
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
                score_threshold=0.5
            )
            return results
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

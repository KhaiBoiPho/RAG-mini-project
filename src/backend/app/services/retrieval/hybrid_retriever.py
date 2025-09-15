from .qdrant_retriever import QdrantService
from ..embeddings_service import EmbeddingsService

class HybridSearchService:
    def __init__(self, qdrant_service: QdrantService, embeddings_service: EmbeddingsService):
        self.qdrant = qdrant_service
        self.embeddings = embeddings_service

    def search(self, query: str, top_k=5):
        # Exact match
        exact = self.qdrant.search_exact(query)
        results = [exact] if exact else []

        # Semantic search
        vector = self.embeddings.embed(query)
        semantic_results = self.qdrant.search_semantic(vector, top_k)
        results.extend(semantic_results)

        # Optional: ranking / filtering hybrid
        results = [r for r in results if r is not None]
        return results

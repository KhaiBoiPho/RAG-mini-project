import asyncio
from src.backend.app.services.retrieval.bm25_retriever import BM25Retriever
from src.backend.app.services.retrieval.qdrant_retriever import QdrantRetriever
from src.backend.app.services.retrieval.hybrid_retriever import HybridRetriever

bm25_retriever = BM25Retriever()
qdrant_retriever = QdrantRetriever()
hybrid = HybridRetriever()

async def main():
    # BM25 & Qdrant
    bm25_results = await bm25_retriever.retrieve("pháp luật", top_k=5)
    vector_results = await qdrant_retriever.retrieve("pháp luật", top_k=5)

    bm25_healthy = await bm25_retriever.health_check()
    qdrant_healthy = await qdrant_retriever.health_check()
    print("BM25 healthy:", bm25_healthy)
    print("Qdrant healthy:", qdrant_healthy)

    # Hybrid
    docs, method = await hybrid.retrieve("pháp luật", top_k=5, method="hybrid")
    print("Hybrid results:", len(docs), "via", method)

    docs, method = await hybrid.retrieve("pháp luật", top_k=5, method="vector")
    print("Vector results:", len(docs), "via", method)

    docs, method = await hybrid.retrieve("pháp luật", top_k=5, method="bm25")
    print("BM25 results:", len(docs), "via", method)

    # Health check hybrid
    health = await hybrid.health_check()
    if health["overall"]:
        print("All systems go!")
    elif health["can_fallback"]:
        print("Can work with degraded performance")

    # Dynamic weight adjustment
    hybrid.update_weights(vector_weight=0.6, bm25_weight=0.4)

if __name__ == "__main__":
    asyncio.run(main())
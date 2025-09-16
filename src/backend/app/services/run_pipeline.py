# Usage examples for Pipeline Service
from pipeline_service import PipelineService
import asyncio
import uuid

# Initialize pipeline
pipeline = PipelineService()

async def example_basic_query():
    """Basic query processing"""
    conversation_id = str(uuid.uuid4())
    
    result = await pipeline.process_query(
        query="Quy hoạch giao thông vận tải đường bộ được lập cho ít nhất bao nhiêu năm?",
        conversation_id=conversation_id,
        method="hybrid",  # or "vector", "bm25"
        top_k=5,
        use_cache=True
    )
    
    print(f"Response: {result['response']}")
    print(f"Documents found: {len(result['retrieved_documents'])}")
    print(f"Retrieval method: {result['metadata']['retrieval_method']}")
    print(f"Total time: {result['metadata']['total_time']:.2f}s")

async def example_streaming_query():
    """Streaming response"""
    conversation_id = str(uuid.uuid4())
    
    async for chunk in pipeline.process_query(
        query="Phá hoại đường, cầu, hầm thì bị gì?",
        conversation_id=conversation_id,
        method="hybrid",
        stream_response=True
    ):
        if chunk["type"] == "documents_retrieved":
            print(f"Found {chunk['data']['count']} documents")
            for doc in chunk['data']['preview']:
                print(f"- {doc['source']}: {doc['content'][:100]}...")
        
        elif chunk["type"] == "response_chunk":
            print(chunk["content"], end="", flush=True)
        
        elif chunk["type"] == "complete":
            print(f"\nCompleted in {chunk['metadata']['total_time']:.2f}s")

async def example_conversation_flow():
    """Multi-turn conversation"""
    conversation_id = str(uuid.uuid4())
    
    # First query
    result1 = await pipeline.process_query(
        query="Người tham gia giao thông phải đi như thế nào?",
        conversation_id=conversation_id,
        method="hybrid"
    )
    print("Bot:", result1["response"])
    
    # Follow-up query with history
    result2 = await pipeline.process_query(
        query="Nếu có thì bị phạt như thế nào?",
        conversation_id=conversation_id,
        method="hybrid",
        use_history=True  # Use conversation context
    )
    print("Bot:", result2["response"])
    
    # Get conversation history
    history = await pipeline.get_conversation_history(conversation_id)
    print(f"Conversation has {len(history)} messages")

async def example_different_methods():
    """Compare different retrieval methods"""
    query = "Công trình đường bộ gồm những gì?"
    conversation_id = str(uuid.uuid4())
    
    # Try different methods
    methods = ["hybrid", "vector", "bm25"]
    
    for method in methods:
        result = await pipeline.process_query(
            query=query,
            conversation_id=f"{conversation_id}_{method}",
            method=method,
            top_k=3
        )
        
        print(f"\n=== Method: {method} ===")
        print(f"Documents: {len(result['retrieved_documents'])}")
        print(f"Time: {result['metadata']['total_time']:.2f}s")
        print(f"Response preview: {result['response'][:200]}...")

async def example_with_filters():
    """Query with document filters"""
    result = await pipeline.process_query(
        query="Biển hiệu lệnh để làm gì?",
        conversation_id=str(uuid.uuid4()),
        method="vector",
        filters={
            "source": "luat2008.txt",  # Filter by document source
            "chunk_index": {"gte": 0, "lte": 100}  # First 100 chunks only
        },
        top_k=5
    )
    
    print(f"Filtered search found {len(result['retrieved_documents'])} documents")

async def example_cache_management():
    """Cache management examples"""
    
    # Warm cache with common queries
    common_queries = [
        "Pháp luật",
        "Hàng hóa", 
        "Đường bộ",
    ]
    
    print("Warming cache...")
    await pipeline.warm_cache(common_queries)
    
    # Query will use cached results
    result = await pipeline.process_query(
        query="Biển hiệu lệnh để làm gì?",
        conversation_id=str(uuid.uuid4()),
        use_cache=True
    )
    
    # Check if result came from cache
    if "cached" in result['metadata']['retrieval_method']:
        print("Result served from cache!")
    
    # Invalidate cache for specific query
    invalidated = await pipeline.invalidate_cache_for_query("Biển hiệu lệnh để làm gì?")
    print(f"Invalidated {invalidated} cache entries")

async def example_health_monitoring():
    """Health check and statistics"""
    
    # Health check
    health = await pipeline.health_check()
    print("=== Health Status ===")
    print(f"Overall: {health['overall_status']}")
    
    for service, status in health['services'].items():
        if isinstance(status, dict) and 'status' in status:
            print(f"{service}: {status['status']}")
        else:
            print(f"{service}: {'healthy' if status else 'unhealthy'}")
    
    # Detailed statistics
    stats = await pipeline.get_pipeline_stats()
    print("\n=== Pipeline Statistics ===")
    print(f"Default top_k: {stats['pipeline_config']['default_top_k']}")
    print(f"Caching enabled: {stats['pipeline_config']['caching_enabled']}")
    
    # Cache stats
    cache_stats = stats['cache']
    if cache_stats['enabled']:
        print(f"Cache items: {cache_stats['total_items']}")
        print(f"Cache hits: {cache_stats['total_cache_hits']}")
        print(f"Memory usage: {cache_stats['memory_usage_estimate']}")
    
    # Conversation stats
    conv_stats = stats['conversations']
    print(f"Active conversations: {conv_stats['active_count']}")
    print(f"Total messages: {conv_stats['total_messages']}")

async def example_error_handling():
    """Error handling and fallbacks"""
    
    try:
        # Query with invalid method (will fallback to hybrid)
        result = await pipeline.process_query(
            query="Test query",
            conversation_id=str(uuid.uuid4()),
            method="invalid_method"
        )
        
        print(f"Fallback method used: {result['metadata']['retrieval_method']}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    # Query when services might be down
    result = await pipeline.process_query(
        query="Another test query",
        conversation_id=str(uuid.uuid4()),
        method="hybrid"
    )
    
    if "error" in result:
        print(f"Pipeline error: {result['error']}")
    else:
        print("Pipeline handled gracefully")

# FastAPI endpoint examples
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel

app = FastAPI()
pipeline = PipelineService()

class QueryRequest(BaseModel):
    query: str
    conversation_id: str
    method: str = "hybrid"
    top_k: int = 5
    use_cache: bool = True
    use_history: bool = True

class QueryResponse(BaseModel):
    response: str
    conversation_id: str
    document_count: int
    retrieval_method: str
    processing_time: float

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """REST API endpoint for queries"""
    try:
        result = await pipeline.process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            method=request.method,
            top_k=request.top_k,
            use_cache=request.use_cache,
            use_history=request.use_history
        )
        
        return QueryResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            document_count=len(result["retrieved_documents"]),
            retrieval_method=result["metadata"]["retrieval_method"],
            processing_time=result["metadata"]["total_time"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_endpoint():
    """Health check endpoint"""
    return await pipeline.health_check()

@app.get("/stats")
async def stats_endpoint():
    """Statistics endpoint"""
    return await pipeline.get_pipeline_stats()

@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """WebSocket endpoint for streaming queries"""
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            
            # Stream response
            async for chunk in pipeline.process_query(
                query=data["query"],
                conversation_id=data["conversation_id"],
                method=data.get("method", "hybrid"),
                stream_response=True
            ):
                await websocket.send_json(chunk)
                
    except Exception as e:
        await websocket.send_json({
            "type": "error", 
            "message": str(e)
        })
    finally:
        await websocket.close()

# Run examples
async def main():
    print("=== Pipeline Service Examples ===\n")
    
    await pipeline.cache_service.start()
    
    # Wait for services to initialize
    await asyncio.sleep(2)
    
    await example_basic_query()
    print("\n" + "="*50 + "\n")
    
    await example_conversation_flow()
    print("\n" + "="*50 + "\n")
    
    await example_different_methods()
    print("\n" + "="*50 + "\n")
    
    await example_cache_management()
    print("\n" + "="*50 + "\n")
    
    await example_health_monitoring()
    
    # Cleanup
    await pipeline.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
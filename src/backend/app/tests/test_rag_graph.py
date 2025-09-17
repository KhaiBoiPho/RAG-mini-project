#!/usr/bin/env python3
"""
Test script for LangGraph RAG implementation
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.backend.app.services.graph.rag_graph import RAGGraph
from src.backend.app.models.chat_models import Message, MessageRole
from datetime import datetime

async def test_langgraph_rag():
    """Test the LangGraph RAG implementation"""
    
    print("Testing LangGraph RAG Implementation")
    print("=" * 50)
    
    # Initialize RAG graph
    try:
        rag_graph = RAGGraph()
        print("RAG Graph initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG Graph: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "name": "Simple Legal Query",
            "query": "Bằng lái xe A1 được lái xe gì?",
            "conversation_history": [],
            "config": {
                "top_k": 5,
                "score_threshold": 0.7
            }
        },
        {
            "name": "Complex Legal Query",
            "query": "Luật giao thông quy định như thế nào về việc xử phạt vi phạm tốc độ trên đường cao tốc và đường thường?",
            "conversation_history": [],
            "config": {
                "top_k": 7,
                "score_threshold": 0.6
            }
        },
        {
            "name": "Conversation Context Query",
            "query": "Còn về mức phạt thì sao?",
            "conversation_history": [
                Message(
                    role=MessageRole.USER,
                    content="Bằng lái xe A1 được lái xe gì?",
                    timestamp=datetime.now()
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Bằng lái xe A1 được phép lái xe máy có dung tích xi-lanh từ 50cm³ đến dưới 175cm³.",
                    timestamp=datetime.now()
                )
            ],
            "config": {
                "top_k": 5,
                "score_threshold": 0.7
            }
        }
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        
        if test_case['conversation_history']:
            print("Conversation History:")
            for msg in test_case['conversation_history']:
                role = msg.role.value.upper()
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"  {role}: {content}")
        
        try:
            # Process query
            start_time = asyncio.get_event_loop().time()
            
            result = await rag_graph.process(
                query=test_case['query'],
                conversation_history=test_case['conversation_history'],
                retrieval_config=test_case['config']
            )
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            # Display results
            print(f"Processing Time: {processing_time:.2f}s")
            
            if result.get('error'):
                print(f"Error: {result['error']}")
                continue
            
            print(f"Retrieved Documents: {len(result.get('retrieved_documents', []))}")
            
            # Show processing steps
            processing_steps = result.get('processing_steps', {})
            if processing_steps:
                print("Processing Steps:")
                for step in processing_steps.get('steps_completed', []):
                    print(f"  ✓ {step}")
                
                if 'query_analysis' in processing_steps:
                    analysis = processing_steps['query_analysis']
                    print(f"Query Analysis:")
                    print(f"  - Length: {analysis.get('length')} words")
                    print(f"  - Has legal terms: {analysis.get('has_legal_terms')}")
                    print(f"  - Question type: {analysis.get('question_type')}")
                    print(f"  - Complexity: {analysis.get('complexity')}")
            
            # Show response preview
            response = result.get('response', '')
            response_preview = response[:300] + "..." if len(response) > 300 else response
            print(f"Response Preview:\n{response_preview}")
            
            # Show document sources preview
            docs = result.get('retrieved_documents', [])
            if docs:
                print("Top Document Sources:")
                for j, doc in enumerate(docs[:3], 1):
                    score = doc.get('score', 0)
                    content_preview = doc.get('content', '')[:100] + "..."
                    source = doc.get('source', 'Unknown')
                    print(f"  {j}. Score: {score:.3f} | Source: {source}")
                    print(f"     Content: {content_preview}")
            
            print("Test completed successfully")
            
        except Exception as e:
            print(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🔍 Testing Health Check")
    
    try:
        health = await rag_graph.health_check()
        print("Health Check Results:")
        for service, status in health.items():
            status_icon = "good" if status else "failed"
            print(f"  {service}: {status_icon}")
        
        print("Health check completed")
        
    except Exception as e:
        print(f"Health check failed: {str(e)}")
    
    print("\nAll tests completed!")

async def test_graph_visualization():
    """Test graph structure visualization"""
    print("\nTesting Graph Structure")
    
    try:
        rag_graph = RAGGraph()
        
        # Get graph information
        graph = rag_graph.graph
        
        print("Graph Nodes:")
        if hasattr(graph, 'nodes'):
            for node in graph.nodes:
                print(f"  - {node}")
        
        print("Graph compiled successfully")
        
    except Exception as e:
        print(f"Graph structure test failed: {str(e)}")

async def benchmark_performance():
    """Benchmark performance across different query types"""
    print("\n⚡ Performance Benchmarking")
    
    rag_graph = RAGGraph()
    
    queries = [
        "Luật giao thông",
        "Bằng lái xe A1 được lái xe máy bao nhiêu phân khối?",
        "Thủ tục đăng ký xe ô tô mới cần những giấy tờ gì và mức phí là bao nhiêu?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nBenchmark {i}: {query[:50]}...")
        
        times = []
        for run in range(3):  # Run 3 times for average
            try:
                start_time = asyncio.get_event_loop().time()
                
                result = await rag_graph.process(
                    query=query,
                    conversation_history=[],
                    retrieval_config={"top_k": 5}
                )
                
                end_time = asyncio.get_event_loop().time()
                processing_time = end_time - start_time
                times.append(processing_time)
                
                if not result.get('error'):
                    docs_count = len(result.get('retrieved_documents', []))
                    print(f"  Run {run + 1}: {processing_time:.2f}s ({docs_count} docs)")
                else:
                    print(f"  Run {run + 1}: Error - {result['error']}")
                    
            except Exception as e:
                print(f"  Run {run + 1}: Exception - {str(e)}")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"  Average: {avg_time:.2f}s")

def main():
    """Main test runner"""
    print("LangGraph RAG Test Suite")
    print("=" * 50)
    
    # Check if we're in the right environment
    try:
        import langgraph
        print(f"LangGraph version: {langgraph.__version__}")
    except ImportError:
        print("LangGraph not installed. Please install with:")
        print("pip install langgraph")
        return
    
    # Run tests
    try:
        asyncio.run(test_langgraph_rag())
        asyncio.run(test_graph_visualization())
        asyncio.run(benchmark_performance())
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
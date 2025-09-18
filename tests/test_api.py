#!/usr/bin/env python3
"""
Test script for RAG Chat API
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAG-Chat-API-Tester/1.0'
        })
    
    def test_root(self):
        """Test root endpoint"""
        print("\n=== Testing Root Endpoint ===")
        try:
            response = self.session.get(f"{self.base_url}/")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            return response.status_code == 200
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def test_health(self):
        """Test health check endpoints"""
        print("\n=== Testing Health Endpoints ===")
        
        # Test simple health
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"Simple Health Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            simple_ok = response.status_code == 200
        except Exception as e:
            print(f"Simple health error: {e}")
            simple_ok = False
        
        # Test detailed health
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            print(f"Detailed Health Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            detailed_ok = response.status_code == 200
        except Exception as e:
            print(f"Detailed health error: {e}")
            detailed_ok = False
        
        return simple_ok and detailed_ok
    
    def test_models(self):
        """Test models endpoint"""
        print("\n=== Testing Models Endpoint ===")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models")
            print(f"Status: {response.status_code}")
            data = response.json()
            print(f"Available models: {len(data.get('data', []))}")
            for model in data.get('data', [])[:3]:  # Show first 3
                print(f"  - {model.get('id')}")
            return response.status_code == 200
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        print("\n=== Testing Basic Chat ===")
        
        test_cases = [
            {
                "name": "Traffic Law Question",
                "query": "Luật giao thông Việt Nam quy định gì về phạt vi phạm tốc độ?"
            },
            {
                "name": "License Question", 
                "query": "Bằng lái xe A1 có được phép lái xe máy bao nhiêu phân khối?"
            },
            {
                "name": "Registration Question",
                "query": "Thủ tục đăng ký xe máy mới cần những giấy tờ gì?"
            }
        ]
        
        success_count = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": test_case["query"]
                    }
                ],
                "model": "gpt-4o-mini",
                "temperature": 1,
                "top_k": 5
            }
            
            try:
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/api/v1/chat", json=payload)
                end_time = time.time()
                
                print(f"Status: {response.status_code}")
                print(f"Response time: {end_time - start_time:.2f}s")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"Conversation ID: {data.get('conversation_id')}")
                    print(f"Model: {data.get('model')}")
                    print(f"Sources: {len(data.get('sources', []))}")
                    print(f"Search method: {data.get('search_method')}")
                    print(f"Retrieval time: {data.get('retrieval_time', 0):.3f}s")
                    print(f"Generation time: {data.get('generation_time', 0):.3f}s")
                    
                    # Show response preview
                    response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    print(f"Response preview: {response_text[:200]}...")
                    
                    success_count += 1
                else:
                    print(f"Error response: {response.text}")
                    
            except Exception as e:
                print(f"Error in test {i}: {e}")
        
        print(f"\nBasic Chat Tests: {success_count}/{len(test_cases)} passed")
        return success_count == len(test_cases)
    
    def test_streaming_chat(self):
        """Test streaming chat"""
        print("\n=== Testing Streaming Chat ===")
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Quy định về đăng kiểm xe ô tô trong luật giao thông?"
                }
            ],
            "model": "gpt-4o-mini",
            "stream": True,
            "top_k": 3
        }
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/chat/stream", json=payload, stream=True)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                print("Streaming chunks:")
                chunk_count = 0
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            data = decoded_line[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                print("Stream completed")
                                break
                            
                            try:
                                chunk = json.loads(data)
                                chunk_count += 1
                                
                                # Show different types of chunks
                                if 'sources' in chunk:
                                    print(f"  Sources chunk: {len(chunk['sources'])} sources")
                                elif 'choices' in chunk:
                                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        print(f"  Content chunk {chunk_count}: '{content[:50]}...'")
                                elif 'usage' in chunk:
                                    print(f"  Final chunk with usage: {chunk['usage']}")
                                    
                            except json.JSONDecodeError as e:
                                print(f"  Invalid JSON chunk: {data[:100]}")
                
                print(f"Total chunks received: {chunk_count}")
                return chunk_count > 0
            else:
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def test_conversation_management(self):
        """Test conversation management endpoints"""
        print("\nTesting Conversation Management")
        
        # First create a conversation
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Test conversation management"
                }
            ],
            "model": "gpt-4o-mini"
        }
        
        try:
            # Create conversation
            response = self.session.post(f"{self.base_url}/api/v1/chat", json=payload)
            if response.status_code != 200:
                print("Failed to create conversation for testing")
                return False
            
            conversation_id = response.json().get('conversation_id')
            print(f"Created conversation: {conversation_id}")
            
            # Test get history
            history_response = self.session.get(f"{self.base_url}/api/v1/conversations/{conversation_id}/history")
            print(f"Get history status: {history_response.status_code}")
            
            if history_response.status_code == 200:
                history_data = history_response.json()
                print(f"History messages: {len(history_data.get('messages', []))}")
            
            # Test clear conversation
            clear_response = self.session.delete(f"{self.base_url}/api/v1/conversations/{conversation_id}")
            print(f"Clear conversation status: {clear_response.status_code}")
            
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def test_pipeline_stats(self):
        """Test pipeline statistics"""
        print("\n=== Testing Pipeline Stats ===")
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/stats")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                stats = response.json()
                print("Pipeline statistics:")
                print(f"  Pipeline config: {stats.get('pipeline_config', {})}")
                print(f"  Cache stats available: {'cache' in stats}")
                print(f"  Conversations: {stats.get('conversations', {})}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling"""
        print("\nTesting Error Handling")
        
        test_cases = [
            {
                "name": "Invalid model",
                "payload": {
                    "messages": [{"role": "user", "content": "test"}],
                    "model": "invalid-model"
                },
                "expected_status": 422
            },
            {
                "name": "Empty messages",
                "payload": {
                    "messages": [],
                    "model": "gpt-4o-mini"
                },
                "expected_status": 422
            },
            {
                "name": "Invalid message role",
                "payload": {
                    "messages": [{"role": "invalid", "content": "test"}],
                    "model": "gpt-4o-mini"
                },
                "expected_status": 422
            }
        ]
        
        success_count = 0
        
        for test_case in test_cases:
            try:
                response = self.session.post(f"{self.base_url}/api/v1/chat", json=test_case["payload"])
                if response.status_code == test_case["expected_status"]:
                    print(f"{test_case['name']}: Expected {test_case['expected_status']}, got {response.status_code}")
                    success_count += 1
                else:
                    print(f"{test_case['name']}: Expected {test_case['expected_status']}, got {response.status_code}")
            except Exception as e:
                print(f"{test_case['name']}: Error {e}")
        
        return success_count == len(test_cases)
    
    def run_all_tests(self):
        """Run all tests"""
        print("Starting RAG Chat API Tests...")
        print(f"Base URL: {self.base_url}")
        
        tests = [
            ("Root Endpoint", self.test_root),
            ("Health Check", self.test_health),
            ("Models List", self.test_models),
            ("Basic Chat", self.test_basic_chat),
            ("Streaming Chat", self.test_streaming_chat),
            ("Conversation Management", self.test_conversation_management),
            ("Pipeline Stats", self.test_pipeline_stats),
            ("Error Handling", self.test_error_handling)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                status = "PASS" if result else "✗ FAIL"
                print(f"\n{status}: {test_name}")
            except Exception as e:
                results.append((test_name, False))
                print(f"\nERROR: {test_name} - {e}")
        
        # Summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "PASS" if result else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("All tests passed!")
        else:
            print(f"{total - passed} test(s) failed")
        
        return passed == total

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG Chat API")
    parser.add_argument("--url", default=BASE_URL, help="Base URL for API")
    parser.add_argument("--test", choices=["all", "health", "chat", "stream", "models"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test == "all":
        success = tester.run_all_tests()
    elif args.test == "stream":
        success = tester.test_streaming_chat()
    elif args.test == "models":
        success = tester.test_models()
    else:
        print(f"Unknown test: {args.test}")
        success = False
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
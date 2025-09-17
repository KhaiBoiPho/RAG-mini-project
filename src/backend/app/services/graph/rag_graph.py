# src/backend/app/services/graph/rag_graph.py
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from src.backend.app.services.retrieval.hybrid_retriever import HybridRetriever
from src.backend.app.services.generation.openai_service import OpenAIService
from src.backend.app.models.chat_models import Message
from src.backend.app.utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

# Define the state schema for our graph
class RAGState(TypedDict):
    # Messages in the conversation
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Original user query
    original_query: str
    
    # Processed/expanded query
    processed_query: str
    
    # Retrieval configuration
    retrieval_config: Dict[str, Any]
    
    # Retrieved documents
    retrieved_documents: List[Dict[str, Any]]
    
    # Filtered and reranked documents
    filtered_documents: List[Dict[str, Any]]
    
    # Processing metadata
    processing_metadata: Dict[str, Any]
    
    # Final response
    response: Optional[str]
    
    # Error information if any
    error: Optional[str]


class RAGGraph:
    """
    LangGraph-based RAG pipeline with state management
    """
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = OpenAIService()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow
        """
        # Create a new graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("preprocess_query", self._preprocess_query_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("filter_rerank", self._filter_rerank_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("postprocess_response", self._postprocess_response_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the flow
        workflow.set_entry_point("analyze_query")
        
        # Normal flow
        workflow.add_edge("analyze_query", "preprocess_query")
        workflow.add_edge("preprocess_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "filter_rerank")
        workflow.add_edge("filter_rerank", "generate_response")
        workflow.add_edge("generate_response", "postprocess_response")
        workflow.add_edge("postprocess_response", END)
        
        # Error handling edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._check_for_errors,
            {
                "error": "handle_error",
                "continue": "preprocess_query"
            }
        )
        
        workflow.add_conditional_edges(
            "retrieve_documents",
            self._check_retrieval_results,
            {
                "no_results": "handle_error",
                "has_results": "filter_rerank"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def process(
        self,
        query: str,
        conversation_history: List[Message],
        retrieval_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main processing method using LangGraph
        """
        try:
            logger.info("Starting LangGraph RAG processing")
            
            # Convert conversation history to LangChain messages
            messages = self._convert_to_langchain_messages(conversation_history)
            messages.append(HumanMessage(content=query))
            
            # Initialize state
            initial_state = RAGState(
                messages=messages,
                original_query=query,
                processed_query="",
                retrieval_config=retrieval_config,
                retrieved_documents=[],
                filtered_documents=[],
                processing_metadata={
                    "start_time": datetime.now().isoformat(),
                    "steps_completed": []
                },
                response=None,
                error=None
            )
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Format response
            return self._format_response(final_state)
            
        except Exception as e:
            logger.error(f"Error in RAG graph processing: {str(e)}")
            return {
                "response": "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý.",
                "retrieved_documents": [],
                "processing_steps": {},
                "error": str(e)
            }
    
    async def _analyze_query_node(self, state: RAGState) -> RAGState:
        """
        Analyze the incoming query
        """
        try:
            logger.info("Analyzing query")
            
            query = state["original_query"]
            metadata = state["processing_metadata"]
            
            # Query analysis
            analysis = {
                "length": len(query.split()),
                "has_legal_terms": self._has_legal_terms(query),
                "question_type": self._classify_question_type(query),
                "complexity": self._assess_complexity(query)
            }
            
            metadata["query_analysis"] = analysis
            metadata["steps_completed"].append("analyze_query")
            
            return {
                **state,
                "processing_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in query analysis: {str(e)}")
            return {
                **state,
                "error": f"Query analysis failed: {str(e)}"
            }
    
    async def _preprocess_query_node(self, state: RAGState) -> RAGState:
        """
        Preprocess and expand query if needed
        """
        try:
            logger.info("Preprocessing query")
            
            query = state["original_query"]
            analysis = state["processing_metadata"].get("query_analysis", {})
            
            # Query preprocessing based on analysis
            processed_query = query.strip()
            
            # Expand query for legal terms
            if analysis.get("has_legal_terms", False):
                processed_query = self._expand_legal_query(processed_query)
            
            # Add context from conversation if needed
            if len(state["messages"]) > 1:
                processed_query = self._add_conversation_context(
                    processed_query, 
                    state["messages"][:-1]  # Exclude current query
                )
            
            metadata = state["processing_metadata"]
            metadata["processed_query"] = processed_query
            metadata["steps_completed"].append("preprocess_query")
            
            return {
                **state,
                "processed_query": processed_query,
                "processing_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in query preprocessing: {str(e)}")
            return {
                **state,
                "error": f"Query preprocessing failed: {str(e)}"
            }
    
    async def _retrieve_documents_node(self, state: RAGState) -> RAGState:
        """
        Retrieve documents using hybrid retrieval
        """
        try:
            logger.info("Retrieving documents")
            
            query = state["processed_query"]
            config = state["retrieval_config"]
            analysis = state["processing_metadata"].get("query_analysis", {})
            
            # Adjust retrieval parameters based on query analysis
            adjusted_config = self._adjust_retrieval_config(config, analysis)
            
            # Perform retrieval
            retrieved_docs = await self.retriever.retrieve(
                query=query,
                top_k=adjusted_config.get("top_k", 5),
                method=adjusted_config.get("method", "hybrid"),
                filters=adjusted_config.get("filters")
            )
            
            metadata = state["processing_metadata"]
            metadata["retrieval_config"] = adjusted_config
            metadata["docs_retrieved"] = len(retrieved_docs)
            metadata["steps_completed"].append("retrieve_documents")
            
            return {
                **state,
                "retrieved_documents": retrieved_docs,
                "processing_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            return {
                **state,
                "error": f"Document retrieval failed: {str(e)}"
            }
    
    async def _filter_rerank_node(self, state: RAGState) -> RAGState:
        """
        Filter and rerank retrieved documents
        """
        try:
            logger.info("Filtering and reranking documents")
            
            docs = state["retrieved_documents"]
            query = state["processed_query"]
            
            # Apply filtering
            filtered_docs = self._apply_document_filters(docs, query)
            
            # Apply reranking
            reranked_docs = await self._rerank_documents(filtered_docs, query)
            
            metadata = state["processing_metadata"]
            metadata["docs_after_filtering"] = len(filtered_docs)
            metadata["docs_final"] = len(reranked_docs)
            metadata["steps_completed"].append("filter_rerank")
            
            return {
                **state,
                "filtered_documents": reranked_docs,
                "processing_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in filtering/reranking: {str(e)}")
            return {
                **state,
                "filtered_documents": state["retrieved_documents"],  # Fallback
                "error": f"Filtering/reranking failed: {str(e)}"
            }
    
    async def _generate_response_node(self, state: RAGState) -> RAGState:
        """
        Generate response using filtered documents
        """
        try:
            logger.info("Generating response")
            
            query = state["original_query"]
            docs = state["filtered_documents"]
            conversation_history = self._convert_from_langchain_messages(state["messages"][:-1])
            
            # Generate response
            response = await self.generator.generate(
                query=query,
                context_docs=docs,
                conversation_history=conversation_history
            )
            
            metadata = state["processing_metadata"]
            metadata["response_length"] = len(response)
            metadata["steps_completed"].append("generate_response")
            
            return {
                **state,
                "response": response,
                "processing_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            return {
                **state,
                "error": f"Response generation failed: {str(e)}"
            }
    
    async def _postprocess_response_node(self, state: RAGState) -> RAGState:
        """
        Post-process the generated response
        """
        try:
            logger.info("Post-processing response")
            
            response = state["response"]
            docs = state["filtered_documents"]
            
            # Post-process response
            processed_response = self._postprocess_response(response, docs)
            
            metadata = state["processing_metadata"]
            metadata["end_time"] = datetime.now().isoformat()
            metadata["steps_completed"].append("postprocess_response")
            
            return {
                **state,
                "response": processed_response,
                "processing_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in response post-processing: {str(e)}")
            return {
                **state,
                "error": f"Response post-processing failed: {str(e)}"
            }
    
    async def _handle_error_node(self, state: RAGState) -> RAGState:
        """
        Handle errors in the pipeline
        """
        logger.error(f"Handling error: {state.get('error')}")
        
        # Provide fallback response
        fallback_response = "Xin lỗi, tôi gặp khó khăn trong việc xử lý câu hỏi của bạn. Vui lòng thử lại hoặc đặt câu hỏi khác."
        
        metadata = state["processing_metadata"]
        metadata["error_handled"] = True
        metadata["steps_completed"].append("handle_error")
        
        return {
            **state,
            "response": fallback_response,
            "processing_metadata": metadata
        }
    
    def _check_for_errors(self, state: RAGState) -> str:
        """
        Check if there are any errors in the state
        """
        return "error" if state.get("error") else "continue"
    
    def _check_retrieval_results(self, state: RAGState) -> str:
        """
        Check if retrieval returned any results
        """
        docs = state.get("retrieved_documents", [])
        return "has_results" if docs else "no_results"
    
    # Helper methods
    def _convert_to_langchain_messages(self, messages: List[Message]) -> List[AnyMessage]:
        """Convert Message objects to LangChain messages"""
        langchain_messages = []
        for msg in messages:
            if msg.role.value == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role.value == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role.value == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
        return langchain_messages
    
    def _convert_from_langchain_messages(self, messages: List[AnyMessage]) -> List[Message]:
        """Convert LangChain messages back to Message objects"""
        from ...models.chat_models import MessageRole
        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = MessageRole.USER
            elif isinstance(msg, AIMessage):
                role = MessageRole.ASSISTANT
            elif isinstance(msg, SystemMessage):
                role = MessageRole.SYSTEM
            else:
                continue
            
            converted.append(Message(
                role=role,
                content=msg.content,
                timestamp=datetime.now()
            ))
        return converted
    
    def _has_legal_terms(self, query: str) -> bool:
        """Check if query contains legal terms"""
        legal_keywords = [
            "điều", "khoản", "luật", "nghị định", "quy định", "pháp luật",
            "bộ luật", "thông tư", "quyết định", "vi phạm", "xử phạt"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in legal_keywords)
    
    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["là gì", "định nghĩa", "khái niệm"]):
            return "definition"
        elif any(word in query_lower for word in ["thủ tục", "cách", "làm thế nào"]):
            return "procedure"
        elif any(word in query_lower for word in ["phạt", "mức phạt", "xử phạt"]):
            return "penalty"
        elif any(word in query_lower for word in ["được phép", "có được", "được không"]):
            return "permission"
        else:
            return "general"
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        if word_count < 5:
            return "simple"
        elif word_count < 15:
            return "medium"
        else:
            return "complex"
    
    def _expand_legal_query(self, query: str) -> str:
        """Expand legal queries with synonyms"""
        # Simple expansion - can be enhanced with more sophisticated methods
        expansions = {
            "phạt": "phạt vi phạm hành chính xử phạt",
            "bằng lái": "bằng lái xe giấy phép lái xe",
            "đăng ký": "đăng ký xe cơ giới"
        }
        
        expanded_query = query
        for term, expansion in expansions.items():
            if term in query.lower():
                expanded_query += f" {expansion}"
        
        return expanded_query
    
    def _add_conversation_context(self, query: str, previous_messages: List[AnyMessage]) -> str:
        """Add relevant context from conversation history"""
        if not previous_messages:
            return query
        
        # Get last user message for context
        for msg in reversed(previous_messages):
            if isinstance(msg, HumanMessage):
                # Simple context addition
                return f"Trong bối cảnh câu hỏi trước: '{msg.content[:100]}...', {query}"
        
        return query
    
    def _adjust_retrieval_config(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust retrieval configuration based on query analysis"""
        adjusted = config.copy()
        
        # Adjust top_k based on complexity
        complexity = analysis.get("complexity", "medium")
        if complexity == "complex":
            adjusted["top_k"] = min(adjusted.get("top_k", 5) + 2, 10)
        
        # Adjust method for legal terms
        if analysis.get("has_legal_terms", False):
            adjusted["method"] = "hybrid"  # Prefer hybrid for legal queries
        
        return adjusted
    
    def _apply_document_filters(self, docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply various filters to documents"""
        filtered = []
        
        for doc in docs:
            # Minimum content length filter
            if len(doc["content"]) < 50:
                continue
            
            # Score threshold filter
            if doc.get("score", 0) < 0.3:
                continue
            
            # Duplicate content filter
            is_duplicate = any(
                self._is_similar_content(doc["content"], existing["content"]) 
                for existing in filtered
            )
            
            if not is_duplicate:
                filtered.append(doc)
        
        return filtered
    
    async def _rerank_documents(self, docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance"""
        # Simple reranking by score for now
        # Can be enhanced with cross-encoder reranking
        return sorted(docs, key=lambda x: x.get("score", 0), reverse=True)
    
    def _is_similar_content(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """Check content similarity using Jaccard similarity"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold
    
    def _postprocess_response(self, response: str, documents: List[Dict[str, Any]]) -> str:
        """Post-process the generated response"""
        processed = response.strip()
        
        # Add source reference if not present
        if documents and "nguồn" not in processed.lower() and "tài liệu" not in processed.lower():
            processed += f"\n\n(Thông tin dựa trên {len(documents)} tài liệu pháp luật liên quan)"
        
        return processed
    
    def _format_response(self, state: RAGState) -> Dict[str, Any]:
        """Format the final response"""
        return {
            "response": state.get("response", "Không thể tạo phản hồi"),
            "retrieved_documents": [
                {
                    "content": doc["content"],
                    "score": doc["score"],
                    "metadata": doc.get("metadata", {}),
                    "source": doc.get("source")
                }
                for doc in state.get("filtered_documents", [])
            ],
            "processing_steps": state.get("processing_metadata", {}),
            "error": state.get("error")
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the RAG graph"""
        try:
            # Check if retriever is healthy
            retriever_health = await self.retriever.health_check()
            
            # Check if generator is healthy  
            generator_health = await self.generator.health_check()
            
            return {
                "rag_graph": True,
                "retriever": retriever_health,
                "generator": generator_health,
                "graph_compiled": self.graph is not None
            }
        except Exception as e:
            logger.error(f"RAG graph health check failed: {e}")
            return {
                "rag_graph": False,
                "error": str(e)
            }
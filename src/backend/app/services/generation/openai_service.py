# src/backend/app/services/generation/openai_service.py

from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI
from src.backend.app.config import settings
from src.backend.app.utils.logger import get_logger
from src.backend.app.models.chat_models import Message, MessageRole
import asyncio
import os
import time

logger = get_logger(__name__)

# src/backend/app/models/chat_models.py
class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.DEFAULT_MODEL
        self.system_prompt = self._load_system_prompt()
        self.chat_prompt_template = self._load_chat_prompt_template()
        self.temperature = settings.TEMPERATURE_GENERATION
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        try:
            prompt_path = os.path.join(
                os.path.dirname(__file__),
                "../../prompts/system_prompt.txt"
            )
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        except FileNotFoundError:
            logger.warning("System prompt file not found, using default")
            return """Bạn là một trợ lý AI chuyên về pháp luật Việt Nam. 
            Hãy trả lời câu hỏi dựa trên thông tin được cung cấp từ các văn bản pháp luật.
            Nếu không tìm thấy thông tin liên quan, hãy thông báo rõ ràng.
            Luôn trích dẫn điều khoản cụ thể khi có thể."""
        
    def _load_chat_prompt_template(self) -> str:
        """Load chat prompt template from file"""
        try:
            prompt_path = os.path.join(
                os.path.dirname(__file__), 
                "../../prompts/chat_prompt.txt"
            )
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning("Chat prompt template file not found, using default")
            return """Dựa trên thông tin sau đây từ các văn bản pháp luật:
                    {context}
                    Câu hỏi: {question}
                    Hãy trả lời câu hỏi dựa trên thông tin được cung cấp:
                    - Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp
                    - Nếu không có thông tin liên quan, hãy nói rõ
                    - Trích dẫn điều khoản cụ thể khi có thể
                    - Giải thích rõ ràng và dễ hiểu"""
    
    async def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Message]] = None,
        model: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI API
        Returns dict with content, usage info, and metadata
        """
        try:
            start_time = time.time()
            logger.info(f"Generating response for query: {query[:50]} ...")
            
            # Use provided parameter or defaults
            model = model or self.model
            max_tokens = max_tokens or settings.MAX_OUTPUT_TOKENS
            temperature = temperature or self.temperature
            
            # Prepare context from retrieved documents
            context = self._format_context(context_docs)
            
            # Format the main prompt
            user_prompt = self._format_chat_prompt(query, context)
            
            # Build messages for chat completion
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-3:]: # Last 3 messages
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Add current query
            messages.append({"role": "user", "content": user_prompt})
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=settings.OPENAI_TIMEOUT
            )
            
            generated_time = time.time() - start_time
            generated_text = response.choices[0].message.content.strip()
            
            logger.info(f"Response generated successfully in {generated_time:.2f}s")
            
            return {
                "content": generated_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": model,
                "finish_reason": response.choices[0].finish_reason,
                "generation_time": generated_time
            }
        
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise
    
    async def stream_generate(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Message]] = None,
        model: Optional[str] = None,
        temperature: float = None
    ) -> AsyncGenerator[str, None]:
        """Stream AI response text chunks"""
        try:
            logger.info(f"Starting streaming generation for: {query[:50]} ...")
            
            model = model or self.model
            temperature = temperature or self.temperature
            
            # Prepare context and messages
            context = self._format_context(context_docs)
            user_prompt = self._format_chat_prompt(query, context)
            
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if conversation_history:
                for msg in conversation_history[:-3]:
                    messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
            
            messages.append({"role": "user", "content": user_prompt})

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=temperature,
                timeout=settings.OPENAI_TIMEOUT
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.error("Error in streaming generation: %s", e)
            yield {"error": str(e)}

    
    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        if not docs:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "Không rõ nguồn")
            score = doc.get("score", 0)
            
            context_part = f"""[Tài liệu {i}] (Độ liên quan: {score:.3f})
                                Nguồn: {source}
                                Nội dung: {content}
                                ---"""
            context_parts.append(context_part)
                                        
            return "\n".join(context_parts)
    
    def _format_chat_prompt(self, query: str, context: str) -> str:
        """Format the chat prompt with query and context"""
        return self.chat_prompt_template.format(
            context=context,
            question=query
        )
    
    async def health_check(self) -> bool:
        """Check if OpenAI service is healthy"""
        try:
            # Simple test call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                timeout=10
            )
            return True
        
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            raise
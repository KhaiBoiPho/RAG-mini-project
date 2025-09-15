from openai import OpenAI
from fastapi import HTTPException
import logging

logger = logging.getLogger("rag-backend.openai")

class OpenAIService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    async def get_response(self, messages, model="gpt-3.5-turbo", temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048
            )
            return response
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger("rag-backend.embeddings")

class EmbeddingsService:
    def __init__(self, api_key: str, model_name: str):
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_name = model_name

    def embed(self, text: str):
        try:
            return self.client.feature_extraction(model=self.model_name, text=text)
        except Exception as e:
            logger.error(f"Embeddings error: {e}")
            raise
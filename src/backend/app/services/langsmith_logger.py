from langsmith import Client as LangSmithClient

class LangSmithLogger:
    def __init__(self, api_key: str):
        self.client = LangSmithClient(api_key=api_key)

    def log_interaction(self, user_message, response, metadata=None):
        self.client.log({
            "user_message": user_message,
            "response": response,
            "metadata": metadata or {}
        })

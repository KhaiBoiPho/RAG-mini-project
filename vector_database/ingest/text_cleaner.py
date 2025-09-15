# vector_database/ingest/text_cleaner.py
"""
    Clean text before after text_loader and before tex_splitter
"""

import re
from langchain_core.documents import Document
from typing import List

class TextCleaner:
    """Clean legal text content"""

    @staticmethod
    def clean(docs: List[Document]) -> List[Document]:
        cleaned_docs = []

        for doc in docs:
            text = doc.page_content

            # Remove unwanted headers
            text = re.sub(r"(QUỐC HỘI|CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM|Độc lập - Tự do - Hạnh phúc)", "", text)

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Ensure Điều/Chương formatting
            text = re.sub(r"(Điều\s+\d+\.?)", r"\n\1", text)
            text = re.sub(r"(CHƯƠNG\s+[IVXLC]+)", r"\n\1", text)

            cleaned_docs.append(Document(page_content=text, metadata=doc.metadata))

        return cleaned_docs

# Global service instance
text_cleaner = TextCleaner()
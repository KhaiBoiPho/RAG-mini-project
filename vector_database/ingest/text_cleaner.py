# vector_database/ingest/text_cleaner.py
"""
    Clean text before after text_loader and before tex_splitter
"""

import re
from langchain_core.documents import Document
from typing import List, Union

class TextCleaner:
    @staticmethod
    def clean(docs: List[Union[Document, str]]) -> List[Document]:
        cleaned_docs = []

        for doc in docs:
            if isinstance(doc, Document):
                text = doc.page_content
                metadata = doc.metadata
            else:
                text = doc
                metadata = {}

            # Xử lý text như trước
            text = re.sub(r"(QUỐC HỘI|CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM|Độc lập - Tự do - Hạnh phúc)", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"(Điều\s+\d+\.?)", r"\n\1", text)
            text = re.sub(r"(CHƯƠNG\s+[IVXLC]+)", r"\n\1", text)

            cleaned_docs.append(Document(page_content=text, metadata=metadata))

        return cleaned_docs

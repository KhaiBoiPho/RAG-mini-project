# vector_database/ingest/text_loader.py
"""
    Load data from dataset and convert it to raw text
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    JSONLoader,
    CSVLoader,
    UnstructuredEPubLoader,
)


class DocumentLoader:
    """Generic loader for legal documents in multiple formats"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load(self) -> List[Document]:
        """Load documents based on file type"""
        suffix = self.filepath.suffix.lower()

        if suffix == ".txt":
            loader = TextLoader(str(self.filepath), encoding="utf-8")

        elif suffix == ".pdf":
            loader = PyPDFLoader(str(self.filepath))

        elif suffix in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(str(self.filepath))

        elif suffix in [".html", ".htm"]:
            loader = UnstructuredHTMLLoader(str(self.filepath))

        elif suffix in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(str(self.filepath))

        elif suffix in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(str(self.filepath))

        elif suffix in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(str(self.filepath))

        elif suffix == ".json":
            loader = JSONLoader(
                file_path=str(self.filepath),
                jq_schema=".",
                text_content=False
            )

        elif suffix == ".csv":
            loader = CSVLoader(str(self.filepath))

        elif suffix == ".epub":
            loader = UnstructuredEPubLoader(str(self.filepath))

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        docs = loader.load()
        
        return docs


# Global service instance
document_loader = DocumentLoader()
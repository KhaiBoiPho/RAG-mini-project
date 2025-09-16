# src/backend/app/utils/helpers.py
from typing import List, Dict, Any, Optional
import re
import unicodedata
from datetime import datetime
import hashlib

def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text"""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    return text

def extract_legal_references(text: str) -> List[str]:
    """Extract legal references from text"""
    references = []
    
    # Pattern for "Điều X"
    article_pattern = r'[Đ|đ]iều\s+(\d+)'
    articles = re.findall(article_pattern, text)
    references.extend([f"Điều {art}" for art in articles])
    
    # Pattern for "Khoản X"
    clause_pattern = r'[Kk]hoản\s+(\d+)'
    clauses = re.findall(clause_pattern, text)
    references.extend([f"Khoản {clause}" for clause in clauses])
    
    return list(set(references))

def generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    timestamp = datetime.now().isoformat()
    return hashlib.md5(timestamp.encode()).hexdigest()[:12]

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard index"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union

def format_documents_for_display(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format documents for display in UI"""
    formatted_docs = []
    
    for doc in documents:
        formatted_doc = {
            "id": doc.get("metadata", {}).get("id", "unknown"),
            "content": truncate_text(doc["content"], 300),
            "full_content": doc["content"],
            "score": round(doc["score"], 3),
            "source": doc.get("source", "Không rõ nguồn"),
            "legal_refs": extract_legal_references(doc["content"]),
            "metadata": doc.get("metadata", {})
        }
        formatted_docs.append(formatted_doc)
    
    return formatted_docs

def validate_query(query: str) -> Dict[str, Any]:
    """Validate user query"""
    errors = []
    
    if not query or not query.strip():
        errors.append("Câu hỏi không được để trống")
    
    if len(query) > 1000:
        errors.append("Câu hỏi quá dài (tối đa 1000 ký tự)")
    
    if len(query.strip()) < 3:
        errors.append("Câu hỏi quá ngắn (tối thiểu 3 ký tự)")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "cleaned_query": normalize_vietnamese_text(query) if query else ""
    }
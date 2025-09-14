#!/usr/bin/env python3
"""
Text splitter use to chunk overlap
"""

import re
import html
import emoji
from typing import List
from app.config import settings

class TextSplitter:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        # Clean text
        text = self._clean_text(text)
        
        # Split by sentences first
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length = len(current_chunk)
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Decode HTML entities (example: &amp; -> &)
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove emojis (Unicode ranges for emoji)
        text = emoji.replace_emoji(text, replace=" ")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Vietnamese letters
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> str:
        """Split text into sentences"""
        # Simple sentence splitting for Vietnamese
        sentences = re.split(r'[\.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to get overlap at word boundary
        overlap_text = text[-self.chunk_overlap:]
        space_index = overlap_text.find(' ')
        
        if space_index != -1:
            return overlap_text[space_index:].strip()
        
        return overlap_text


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = TextSplitter()
        
    def process_file(self, file_path: str, source_name: str = None) -> List[dict]:
        """Process a document file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self.text_splitter.split_text(content)
            source_name = source_name or file_path
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': f"{source_name}_chunk_{i}",
                    'content': chunk,
                    'source': source_name,
                    'chunk_index': i,
                    'metadata': {
                        'file_path': file_path,
                        'chunk_size': len(chunk),
                        'total_chunks': len(chunks)
                    }
                }
                processed_chunks.append(chunk_data)
            
            return processed_chunks
        
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {str(e)}")
    
    def process_text(self, text: str, source_name: str) -> List[dict]:
        """Process raw text into chunks"""
        chunks = self.text_splitter.split_text(text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'id': f"{source_name}_chunk_{i}",
                'content': chunk,
                'source': source_name,
                'chunk_index': i,
                'metadata': {
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
            }
            processed_chunks.append(chunk_data)
        
        return processed_chunks
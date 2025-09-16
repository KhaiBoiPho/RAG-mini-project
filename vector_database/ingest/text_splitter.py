#!/usr/bin/env python3
"""
Intelligent Text Splitter with Recursive Chunking
"""

import re
import html
import emoji
from typing import List, Tuple, Optional
from enum import Enum
from ..config import settings

class SeparatorType(Enum):
    """Defines the precedence of the separators"""
    PARAGRAPH = "\n\n"
    LINE = "\n"
    SENTENCE = r'[\.!?]+'
    CLAUSE = r'[,;:]'
    WORD = " "
    CHARACTER = ""

class TextSplitter:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, 
                 min_chunk_size: int = None, max_recursion_depth: int = 5):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or max(50, self.chunk_size // 10)
        self.max_recursion_depth = max_recursion_depth
        
        # Define the priority order to split text
        self.separators = [
            SeparatorType.PARAGRAPH,
            SeparatorType.LINE, 
            SeparatorType.SENTENCE,
            SeparatorType.CLAUSE,
            SeparatorType.WORD,
            SeparatorType.CHARACTER
        ]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive approach"""
        # Clean text first
        text = self._clean_text(text)
        
        if not text:
            return []
        
        # Start recursive splitting
        chunks = self._recursive_split(text, 0)
        
        # Post-process chunks to ensure overlap
        final_chunks = self._add_overlap(chunks)
        
        return final_chunks
    
    def _recursive_split(self, text: str, depth: int) -> List[str]:
        """Recursively split text using different separators"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # If max depth is reached, force split by character
        if depth >= self.max_recursion_depth:
            return self._force_split(text)
        
        # Try each separator in order of priority
        for separator_type in self.separators:
            if depth >= len(self.separators):
                break
                
            splits = self._split_by_separator(text, separator_type)
            
            if len(splits) > 1:  # If splitable
                result_chunks = []
                
                for split in splits:
                    if not split.strip():
                        continue
                        
                    # If split is still too large, continue recursively
                    if len(split) > self.chunk_size:
                        sub_chunks = self._recursive_split(split, depth + 1)
                        result_chunks.extend(sub_chunks)
                    else:
                        result_chunks.append(split)
                
                # Merge adjacent small chunks if possible
                merged_chunks = self._merge_small_chunks(result_chunks)
                return merged_chunks
        
        # If no separator can be split, force split
        return self._force_split(text)
    
    def _split_by_separator(self, text: str, separator_type: SeparatorType) -> List[str]:
        """Split text by specific separator type"""
        if separator_type == SeparatorType.PARAGRAPH:
            return re.split(r'\n\s*\n', text)
        
        elif separator_type == SeparatorType.LINE:
            return text.split('\n')
        
        elif separator_type == SeparatorType.SENTENCE:
            # Split by sentence ending punctuation
            splits = re.split(r'([\.!?]+)', text)
            # Recombine punctuation with preceding text
            result = []
            for i in range(0, len(splits) - 1, 2):
                sentence = splits[i]
                if i + 1 < len(splits):
                    sentence += splits[i + 1]
                if sentence.strip():
                    result.append(sentence.strip())
            # Handle last part if exists
            if len(splits) % 2 == 1 and splits[-1].strip():
                result.append(splits[-1].strip())
            return result
        
        elif separator_type == SeparatorType.CLAUSE:
            # Split by clause separators but keep the separator
            splits = re.split(r'([,;:])', text)
            result = []
            for i in range(0, len(splits) - 1, 2):
                clause = splits[i]
                if i + 1 < len(splits):
                    clause += splits[i + 1]
                if clause.strip():
                    result.append(clause.strip())
            if len(splits) % 2 == 1 and splits[-1].strip():
                result.append(splits[-1].strip())
            return result
        
        elif separator_type == SeparatorType.WORD:
            return text.split(' ')
        
        else:  # CHARACTER
            return list(text)
    
    def _force_split(self, text: str) -> List[str]:
        """Force split text by character count when other methods fail"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Find the best break point in the overlap area
            if end < len(text):
                # Find nearest space to avoid cutting in the middle of word
                best_break = end
                for i in range(max(start, end - 100), min(len(text), end + 100)):
                    if text[i] in ' \n\t':
                        if abs(i - end) < abs(best_break - end):
                            best_break = i
                end = best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge adjacent small chunks to optimize size"""
        if not chunks:
            return []
        
        result = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            # If the current chunk is too small and can be merged with the next chunk
            if (len(current_chunk) < self.min_chunk_size and 
                len(current_chunk) + len(next_chunk) + 1 <= self.chunk_size):
                current_chunk = current_chunk + " " + next_chunk
            else:
                result.append(current_chunk)
                current_chunk = next_chunk
        
        result.append(current_chunk)
        return result
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks for better context"""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap = self._get_overlap_text(prev_chunk)
                
                # Add overlap to current chunk if it doesn't make it too long
                if overlap and len(overlap) + len(chunk) + 1 <= self.chunk_size * 1.1:
                    overlapped_chunk = overlap + " " + chunk
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get meaningful overlap text from the end of a chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Get the last part by overlap size
        overlap_text = text[-self.chunk_overlap:]
        
        # Find natural break points (space, punctuation)
        for i, char in enumerate(overlap_text):
            if char in ' \n\t.,;:!?':
                return overlap_text[i:].strip()
        
        # If no natural break point is found, return the entire overlap
        return overlap_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Replace emojis with space
        text = emoji.replace_emoji(text, replace=" ")
        
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        # Clean special characters but preserve Vietnamese
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\n]', ' ', text, flags=re.UNICODE)
        
        return text.strip()


class DocumentProcessor:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.text_splitter = TextSplitter(chunk_size, chunk_overlap)
        
    def process_file(self, file_path: str, source_name: str = None) -> List[dict]:
        """Process a document file into intelligent chunks"""
        try:
            # Try different encodings
            encodings = ['utf-16le', 'utf-8', 'utf-16', 'cp1252']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise Exception(f"Could not decode file with any supported encoding")
            
            chunks = self.text_splitter.split_text(content)
            source_name = source_name or file_path
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Analyze chunk quality
                chunk_stats = self._analyze_chunk(chunk)
                
                chunk_data = {
                    'id': f"{source_name}_chunk_{i:03d}",
                    'content': chunk,
                    'source': source_name,
                    'chunk_index': i,
                    'metadata': {
                        'file_path': file_path,
                        'encoding_used': used_encoding,
                        'chunk_size': len(chunk),
                        'word_count': chunk_stats['word_count'],
                        'sentence_count': chunk_stats['sentence_count'],
                        'total_chunks': len(chunks),
                        'completeness_score': chunk_stats['completeness_score']
                    }
                }
                processed_chunks.append(chunk_data)
            
            return processed_chunks
        
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {str(e)}")
    
    def process_text(self, text: str, source_name: str) -> List[dict]:
        """Process raw text into intelligent chunks"""
        chunks = self.text_splitter.split_text(text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_stats = self._analyze_chunk(chunk)
            
            chunk_data = {
                'id': f"{source_name}_chunk_{i:03d}",
                'content': chunk,
                'source': source_name,
                'chunk_index': i,
                'metadata': {
                    'chunk_size': len(chunk),
                    'word_count': chunk_stats['word_count'],
                    'sentence_count': chunk_stats['sentence_count'],
                    'total_chunks': len(chunks),
                    'completeness_score': chunk_stats['completeness_score']
                }
            }
            processed_chunks.append(chunk_data)
        
        return processed_chunks
    
    def _analyze_chunk(self, chunk: str) -> dict:
        """Analyze chunk quality and completeness"""
        word_count = len(chunk.split())
        sentences = re.split(r'[\.!?]+', chunk)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Calculate completeness score based on various factors
        completeness_score = 0.0
        
        # Check if chunk starts with complete sentence
        if chunk and chunk[0].isupper():
            completeness_score += 0.3
        
        # Check if chunk ends with sentence terminator
        if chunk and chunk[-1] in '.!?':
            completeness_score += 0.3
        
        # Check word count ratio
        if word_count >= 10:  # Reasonable minimum
            completeness_score += 0.2
        
        # Check sentence structure
        if sentence_count > 0 and word_count / sentence_count > 3:  # Reasonable words per sentence
            completeness_score += 0.2
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'completeness_score': round(completeness_score, 2)
        }
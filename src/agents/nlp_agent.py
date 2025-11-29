"""NLP Agent for text extraction and summarization"""

import spacy
from typing import List
from transformers import pipeline

from ..core.messages import ExtractionRequest, ExtractionResponse
from ..core.logger import logger
from ..tools.pdf_extractor import extract_text_from_pdf, extract_text_from_file


class NLPAgent:
    """Extracts and processes text from documents"""
    
    def __init__(self):
        self.logger = logger.get_logger()
        self.nlp = None
        self.summarizer = None
        self._load_models()
    
    def _load_models(self):
        """Load spaCy and HuggingFace models"""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                # Fallback: create a minimal nlp object
                self.nlp = None
            
            # Load HuggingFace summarization model
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU (M1 Mac compatible)
                )
            except Exception as e:
                self.logger.warning(f"Could not load BART model, using fallback: {e}")
                # Try smaller model
                try:
                    self.summarizer = pipeline(
                        "summarization",
                        model="facebook/bart-base",
                        device=-1
                    )
                except Exception as e2:
                    self.logger.warning(f"Could not load BART-base either: {e2}")
                    self.summarizer = None
        
        except Exception as e:
            self.logger.error(f"Error loading NLP models: {e}")
            self.nlp = None
            self.summarizer = None
    
    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """
        Extract text from file and return chunks.
        
        Args:
            request: ExtractionRequest with file path or content
        
        Returns:
            ExtractionResponse with chunks and optional summary
        """
        try:
            # Extract raw text
            if request.file_path:
                raw_text = extract_text_from_file(request.file_path)
            elif request.file_content:
                if request.file_type == "pdf":
                    raw_text = extract_text_from_pdf(file_content=request.file_content)
                else:
                    raw_text = request.file_content.decode('utf-8')
            else:
                return ExtractionResponse(
                    chunks=[],
                    success=False,
                    error="No file path or content provided"
                )
            
            # Clean and chunk text
            chunks = self.chunk_text(raw_text)
            
            # Generate summary if text is long
            summary = None
            if len(raw_text) > 1000 and self.summarizer:
                try:
                    summary = self.summarize(raw_text)
                except Exception as e:
                    self.logger.warning(f"Summarization failed: {e}")
            
            return ExtractionResponse(
                chunks=chunks,
                summary=summary,
                success=True
            )
        
        except Exception as e:
            self.logger.exception("Error in NLP extraction")
            return ExtractionResponse(
                chunks=[],
                success=False,
                error=str(e)
            )
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
        
        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary if possible
            if end < len(text) and self.nlp:
                try:
                    # Find sentence boundary near end
                    doc = self.nlp(text[start:end])
                    sentences = list(doc.sents)
                    if len(sentences) > 1:
                        # Use last complete sentence
                        last_sentence = sentences[-1]
                        end = start + last_sentence.end_char
                except Exception:
                    # Fallback to character-based chunking
                    pass
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        self.logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize long text using HuggingFace model.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
        
        Returns:
            Summary text
        """
        if not self.summarizer:
            # Fallback: return first few sentences
            sentences = text.split('.')
            return '. '.join(sentences[:3]) + '.'
        
        try:
            # Truncate if too long (models have token limits)
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            return summary[0]['summary_text']
        
        except Exception as e:
            self.logger.warning(f"Summarization error: {e}")
            # Fallback
            sentences = text.split('.')
            return '. '.join(sentences[:3]) + '.'
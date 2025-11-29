"""LLM Agent for content generation using OpenAI"""

import json
import os
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from ..core.messages import GenerationRequest, GenerationResponse, ContentType
from ..core.logger import logger

load_dotenv()


class LLMAgent:
    """Generates learning content using OpenAI GPT-4o-mini"""
    
    def __init__(self):
        self.logger = logger.get_logger()
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate content based on request type.
        
        Args:
            request: GenerationRequest with content type and chunks
        
        Returns:
            GenerationResponse with generated content
        """
        try:
            if request.content_type == ContentType.QUIZ:
                return self.generate_quiz(request)
            elif request.content_type == ContentType.FLASHCARD:
                return self.generate_flashcards(request)
            elif request.content_type == ContentType.INTERACTIVE:
                return self.generate_interactive(request)
            elif request.content_type == ContentType.MIXED:
                return self.generate_mixed_bundle(request)
            else:
                return GenerationResponse(
                    content_type=request.content_type,
                    data={},
                    success=False,
                    error=f"Unknown content type: {request.content_type}"
                )
        
        except Exception as e:
            self.logger.exception("Error in LLM generation")
            return GenerationResponse(
                content_type=request.content_type,
                data={},
                success=False,
                error=str(e)
            )
    
    def generate_quiz(self, request: GenerationRequest) -> GenerationResponse:
        """Generate quiz with MCQs"""
        # Validate chunks are present and non-empty
        chunks = request.chunks or []
        if not chunks or all(not chunk.strip() for chunk in chunks):
            self.logger.error("No chunks provided for quiz generation")
            return GenerationResponse(
                content_type=ContentType.QUIZ,
                data={},
                success=False,
                error="No content chunks available. Please ensure the PDF was properly extracted."
            )
        
        self.logger.info(f"Generating quiz with {len(chunks)} chunks, total length: {sum(len(c) for c in chunks)} chars")
        
        num_questions = request.num_items or 5
        # Create numbered chunks for reference
        chunks_with_numbers = []
        for i, chunk in enumerate(chunks, 1):
            if chunk.strip():  # Only include non-empty chunks
                chunks_with_numbers.append(f"[Chunk {i}]\n{chunk}")
        
        if not chunks_with_numbers:
            return GenerationResponse(
                content_type=ContentType.QUIZ,
                data={},
                success=False,
                error="All chunks are empty. PDF extraction may have failed."
            )
        
        chunks_text = "\n\n".join(chunks_with_numbers)
        
        # Limit total text to avoid token limits (keep it reasonable)
        max_chars = 8000  # Leave room for prompt and response
        if len(chunks_text) > max_chars:
            self.logger.warning(f"Chunks text too long ({len(chunks_text)} chars), truncating to {max_chars}")
            chunks_text = chunks_text[:max_chars] + "\n\n[Content truncated...]"
        
        prompt = f"""You are generating quiz questions based EXCLUSIVELY on the source content provided below.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
1. ONLY use information that is EXPLICITLY stated in the source content below
2. DO NOT use any knowledge from your training data
3. DO NOT add facts, examples, or information not in the source
4. If information is not in the source, DO NOT include it in questions or answers
5. If the source doesn't contain enough information for {num_questions} questions, generate fewer questions
6. Every question, option, and explanation MUST be directly traceable to the source content

Source Content (numbered by chunk):
{chunks_text}

Generate {num_questions} multiple-choice quiz questions. Return a JSON object with this exact structure:
{{
  "questions": [
    {{
      "question": "Question text here (MUST be answerable from source content only)",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": 0,
      "explanation": "Explanation of why this answer is correct (reference source content)",
      "source_reference": "Chunk X - brief quote or description of where this came from"
    }}
  ]
}}

REMEMBER: If you cannot create a question using ONLY the source content, create fewer questions. Never invent information."""
        
        system_message = """You are an educational content generator. Your ONLY job is to create quiz questions based EXCLUSIVELY on the source content provided by the user.

CRITICAL CONSTRAINTS:
- You MUST NOT use any information from your training data
- You MUST NOT add facts, examples, or knowledge not in the provided source
- If information is missing from the source, you MUST NOT invent it
- Every question and answer must be directly traceable to the source content
- If you cannot create enough questions from the source, create fewer questions rather than inventing content

Always return valid JSON. Never hallucinate or add external knowledge."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature to minimize creativity/hallucination
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Store chunks for source reference display
            data["_source_chunks"] = chunks
            
            return GenerationResponse(
                content_type=ContentType.QUIZ,
                data=data,
                success=True
            )
        
        except Exception as e:
            self.logger.error(f"Error generating quiz: {e}")
            return GenerationResponse(
                content_type=ContentType.QUIZ,
                data={},
                success=False,
                error=str(e)
            )
    
    def generate_flashcards(self, request: GenerationRequest) -> GenerationResponse:
        """Generate flashcards"""
        # Validate chunks are present and non-empty
        chunks = request.chunks or []
        if not chunks or all(not chunk.strip() for chunk in chunks):
            self.logger.error("No chunks provided for flashcard generation")
            return GenerationResponse(
                content_type=ContentType.FLASHCARD,
                data={},
                success=False,
                error="No content chunks available. Please ensure the PDF was properly extracted."
            )
        
        self.logger.info(f"Generating flashcards with {len(chunks)} chunks, total length: {sum(len(c) for c in chunks)} chars")
        
        num_cards = request.num_items or 10
        # Create numbered chunks for reference
        chunks_with_numbers = []
        for i, chunk in enumerate(chunks, 1):
            if chunk.strip():  # Only include non-empty chunks
                chunks_with_numbers.append(f"[Chunk {i}]\n{chunk}")
        
        if not chunks_with_numbers:
            return GenerationResponse(
                content_type=ContentType.FLASHCARD,
                data={},
                success=False,
                error="All chunks are empty. PDF extraction may have failed."
            )
        
        chunks_text = "\n\n".join(chunks_with_numbers)
        
        # Limit total text to avoid token limits
        max_chars = 8000
        if len(chunks_text) > max_chars:
            self.logger.warning(f"Chunks text too long ({len(chunks_text)} chars), truncating to {max_chars}")
            chunks_text = chunks_text[:max_chars] + "\n\n[Content truncated...]"
        
        prompt = f"""You are generating flashcards based EXCLUSIVELY on the source content provided below.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
1. ONLY use information that is EXPLICITLY stated in the source content below
2. DO NOT use any knowledge from your training data
3. DO NOT add facts, examples, or information not in the source
4. If information is not in the source, DO NOT include it in flashcards
5. If the source doesn't contain enough information for {num_cards} cards, generate fewer cards
6. Every front and back content MUST be directly traceable to the source content

Source Content (numbered by chunk):
{chunks_text}

Generate {num_cards} flashcards. Return a JSON object with this exact structure:
{{
  "cards": [
    {{
      "front": "Question or term on the front (MUST be from source content only)",
      "back": "Answer or definition on the back (MUST be from source content only)",
      "source_reference": "Chunk X - brief quote or description of where this came from"
    }}
  ]
}}

REMEMBER: If you cannot create a flashcard using ONLY the source content, create fewer cards. Never invent information."""
        
        system_message = """You are an educational content generator. Your ONLY job is to create flashcards based EXCLUSIVELY on the source content provided by the user.

CRITICAL CONSTRAINTS:
- You MUST NOT use any information from your training data
- You MUST NOT add facts, examples, or knowledge not in the provided source
- If information is missing from the source, you MUST NOT invent it
- Every flashcard front and back must be directly traceable to the source content
- If you cannot create enough cards from the source, create fewer cards rather than inventing content

Always return valid JSON. Never hallucinate or add external knowledge."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature to minimize creativity/hallucination
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Store chunks for source reference display
            data["_source_chunks"] = chunks
            
            return GenerationResponse(
                content_type=ContentType.FLASHCARD,
                data=data,
                success=True
            )
        
        except Exception as e:
            self.logger.error(f"Error generating flashcards: {e}")
            return GenerationResponse(
                content_type=ContentType.FLASHCARD,
                data={},
                success=False,
                error=str(e)
            )
    
    def generate_interactive(self, request: GenerationRequest) -> GenerationResponse:
        """Generate interactive lesson plan"""
        # Validate chunks are present and non-empty
        chunks = request.chunks or []
        if not chunks or all(not chunk.strip() for chunk in chunks):
            self.logger.error("No chunks provided for interactive generation")
            return GenerationResponse(
                content_type=ContentType.INTERACTIVE,
                data={},
                success=False,
                error="No content chunks available. Please ensure the PDF was properly extracted."
            )
        
        self.logger.info(f"Generating interactive content with {len(chunks)} chunks, total length: {sum(len(c) for c in chunks)} chars")
        
        num_steps = request.num_items or 3
        # Create numbered chunks for reference
        chunks_with_numbers = []
        for i, chunk in enumerate(chunks, 1):
            if chunk.strip():  # Only include non-empty chunks
                chunks_with_numbers.append(f"[Chunk {i}]\n{chunk}")
        
        if not chunks_with_numbers:
            return GenerationResponse(
                content_type=ContentType.INTERACTIVE,
                data={},
                success=False,
                error="All chunks are empty. PDF extraction may have failed."
            )
        
        chunks_text = "\n\n".join(chunks_with_numbers)
        
        # Limit total text to avoid token limits
        max_chars = 8000
        if len(chunks_text) > max_chars:
            self.logger.warning(f"Chunks text too long ({len(chunks_text)} chars), truncating to {max_chars}")
            chunks_text = chunks_text[:max_chars] + "\n\n[Content truncated...]"
        
        prompt = f"""You are generating an interactive lesson plan based EXCLUSIVELY on the source content provided below.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
1. ONLY use information that is EXPLICITLY stated in the source content below
2. DO NOT use any knowledge from your training data
3. DO NOT add facts, examples, or information not in the source
4. If information is not in the source, DO NOT include it in the lesson
5. If the source doesn't contain enough information for {num_steps} steps, generate fewer steps
6. Every step, checkpoint, and answer MUST be directly traceable to the source content

Source Content (numbered by chunk):
{chunks_text}

Generate an interactive {num_steps}-step lesson plan. Return a JSON object with this exact structure:
{{
  "title": "Lesson title (MUST be based on source content only)",
  "steps": [
    {{
      "step_number": 1,
      "title": "Step title (from source content)",
      "content": "Step content and instructions (ONLY from source content)",
      "checkpoint": "Question or task to check understanding (based on source content)",
      "checkpoint_answer": "Model answer or solution (ONLY from source content)",
      "source_reference": "Chunk X - brief quote or description of where this step's content came from"
    }}
  ]
}}

REMEMBER: If you cannot create a step using ONLY the source content, create fewer steps. Never invent information, examples, or facts."""
        
        system_message = """You are an educational content generator. Your ONLY job is to create interactive lesson plans based EXCLUSIVELY on the source content provided by the user.

CRITICAL CONSTRAINTS:
- You MUST NOT use any information from your training data
- You MUST NOT add facts, examples, or knowledge not in the provided source
- If information is missing from the source, you MUST NOT invent it
- Every step, checkpoint, and answer must be directly traceable to the source content
- If you cannot create enough steps from the source, create fewer steps rather than inventing content

Always return valid JSON. Never hallucinate or add external knowledge."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature to minimize creativity/hallucination
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Store chunks for source reference display
            data["_source_chunks"] = chunks
            
            return GenerationResponse(
                content_type=ContentType.INTERACTIVE,
                data=data,
                success=True
            )
        
        except Exception as e:
            self.logger.error(f"Error generating interactive content: {e}")
            return GenerationResponse(
                content_type=ContentType.INTERACTIVE,
                data={},
                success=False,
                error=str(e)
            )
    
    def generate_mixed_bundle(self, request: GenerationRequest) -> GenerationResponse:
        """Generate all three content types for mixed bundle"""
        # Generate each type
        quiz_request = GenerationRequest(
            content_type=ContentType.QUIZ,
            chunks=request.chunks,
            num_items=5
        )
        quiz_response = self.generate_quiz(quiz_request)
        
        flashcard_request = GenerationRequest(
            content_type=ContentType.FLASHCARD,
            chunks=request.chunks,
            num_items=10
        )
        flashcard_response = self.generate_flashcards(flashcard_request)
        
        interactive_request = GenerationRequest(
            content_type=ContentType.INTERACTIVE,
            chunks=request.chunks,
            num_items=3
        )
        interactive_response = self.generate_interactive(interactive_request)
        
        # Combine into mixed bundle
        data = {
            "quiz": quiz_response.data if quiz_response.success else {},
            "flashcards": flashcard_response.data if flashcard_response.success else {},
            "interactive": interactive_response.data if interactive_response.success else {}
        }
        
        success = quiz_response.success and flashcard_response.success and interactive_response.success
        
        return GenerationResponse(
            content_type=ContentType.MIXED,
            data=data,
            success=success,
            error=None if success else "Some content types failed to generate"
        )


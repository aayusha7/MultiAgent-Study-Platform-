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
        
        # Build feedback adaptation section
        feedback_section = ""
        if request.feedback_context and request.feedback_context.get("has_feedback"):
            fc = request.feedback_context
            feedback_section = f"""

FEEDBACK-BASED ADAPTATION:
Based on user feedback history ({fc.get('feedback_count', 0)} previous interactions):
- Average feedback: {fc.get('average_feedback', 0.5)*100:.1f}% positive
- Recent feedback trend: {fc.get('positive_rate', 0.5)*100:.1f}% positive

ADAPTATION GUIDELINES:
{fc.get('adaptation_instructions', 'Provide balanced, clear content.')}

Please adapt the quiz questions accordingly while still using ONLY the source content below.
"""
        
        prompt = f"""You are generating quiz questions based EXCLUSIVELY on the source content provided below.{feedback_section}

⚠️ CRITICAL ANTI-HALLUCINATION RULES - VIOLATION OF THESE RULES IS UNACCEPTABLE:
1. ONLY use information that is EXPLICITLY and VERBATIM stated in the source content below
2. DO NOT use ANY knowledge from your training data, general knowledge, or external sources
3. DO NOT add facts, examples, definitions, or information NOT in the source
4. DO NOT paraphrase or interpret - use the EXACT wording from the source when possible
5. If information is not in the source, DO NOT include it in questions, options, or answers
6. If the source doesn't contain enough information for {num_questions} questions, generate FEWER questions (even just 1-2 if needed)
7. Every question, option, and explanation MUST be directly traceable to a SPECIFIC chunk number
8. For each question, you MUST quote or reference the EXACT text from the source that supports it
9. If you cannot find the answer in the source, DO NOT create that question
10. DO NOT make assumptions or inferences beyond what is explicitly stated
11. CRITICAL: In source_reference, use the ACTUAL chunk number from the source (e.g., if the question uses content from [Chunk 2], write "Chunk 2" in source_reference, NOT "Chunk 1" or "Chunk X")
12. Different questions can come from different chunks - use the correct chunk number for each question

Source Content (numbered by chunk):
{chunks_text}

Generate {num_questions} multiple-choice quiz questions. Return a JSON object with this exact structure:
{{
  "questions": [
    {{
      "question": "Question text here (MUST be answerable DIRECTLY from source content - quote the relevant part)",
      "options": ["Option A (from source)", "Option B (from source)", "Option C (from source)", "Option D (from source)"],
      "correct_answer": 0,
      "explanation": "Explanation with DIRECT QUOTE from source content showing where this answer comes from",
      "source_reference": "Chunk [NUMBER] - EXACT quote: '...' from the source above. IMPORTANT: Use the ACTUAL chunk number from the source (e.g., if content is from [Chunk 2], write 'Chunk 2', not 'Chunk 1' or 'Chunk X')"
    }}
  ]
}}

VALIDATION CHECKLIST before generating each question:
- [ ] Can I find the answer EXPLICITLY stated in the source?
- [ ] Can I quote the EXACT text that supports this question?
- [ ] Are all options based on information in the source?
- [ ] Have I NOT added any external knowledge?

REMEMBER: If you cannot create a question using ONLY the source content, create fewer questions. NEVER invent information. It's better to have 1 accurate question than 5 invented ones."""
        
        system_message = """You are an educational content generator. Your ONLY job is to create quiz questions based EXCLUSIVELY on the source content provided by the user.

⚠️ CRITICAL ANTI-HALLUCINATION CONSTRAINTS:
- You MUST NOT use ANY information from your training data, general knowledge, or external sources
- You MUST NOT add facts, examples, definitions, or knowledge NOT in the provided source
- You MUST NOT paraphrase or interpret beyond what is explicitly stated
- If information is missing from the source, you MUST NOT invent it - create fewer questions instead
- Every question, option, and answer MUST be directly traceable to a SPECIFIC chunk with exact quotes
- If you cannot find explicit information in the source, DO NOT create that question
- Adapt question difficulty, clarity, and style based on user feedback, but ALWAYS use ONLY source content
- Quote directly from the source when possible - use exact wording

VALIDATION: Before including any information, ask: "Is this EXPLICITLY stated in the source?" If NO, do not include it.

Always return valid JSON. NEVER hallucinate or add external knowledge. Better to have fewer accurate questions than many invented ones."""
        
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
        
        # Build feedback adaptation section
        feedback_section = ""
        if request.feedback_context and request.feedback_context.get("has_feedback"):
            fc = request.feedback_context
            feedback_section = f"""

FEEDBACK-BASED ADAPTATION:
Based on user feedback history ({fc.get('feedback_count', 0)} previous interactions):
- Average feedback: {fc.get('average_feedback', 0.5)*100:.1f}% positive
- Recent feedback trend: {fc.get('positive_rate', 0.5)*100:.1f}% positive

ADAPTATION GUIDELINES:
{fc.get('adaptation_instructions', 'Provide balanced, clear content.')}

Please adapt the flashcards accordingly while still using ONLY the source content below.
"""
        
        prompt = f"""You are generating flashcards based EXCLUSIVELY on the source content provided below.{feedback_section}

⚠️ CRITICAL ANTI-HALLUCINATION RULES - VIOLATION OF THESE RULES IS UNACCEPTABLE:
1. ONLY use information that is EXPLICITLY and VERBATIM stated in the source content below
2. DO NOT use ANY knowledge from your training data, general knowledge, or external sources
3. DO NOT add facts, examples, definitions, or information NOT in the source
4. DO NOT paraphrase or interpret - use the EXACT wording from the source when possible
5. If information is not in the source, DO NOT include it in flashcards
6. If the source doesn't contain enough information for {num_cards} cards, generate FEWER cards (even just 1-2 if needed)
7. Every front and back content MUST be directly traceable to a SPECIFIC chunk number
8. For each flashcard, you MUST quote or reference the EXACT text from the source
9. If you cannot find the information in the source, DO NOT create that flashcard
10. DO NOT make assumptions or inferences beyond what is explicitly stated
11. CRITICAL: In source_reference, use the ACTUAL chunk number from the source (e.g., if the flashcard uses content from [Chunk 2], write "Chunk 2" in source_reference, NOT "Chunk 1" or "Chunk X")
12. Different flashcards can come from different chunks - use the correct chunk number for each flashcard

Source Content (numbered by chunk):
{chunks_text}

Generate {num_cards} flashcards. Return a JSON object with this exact structure:
{{
  "cards": [
    {{
      "front": "Question or term on the front (MUST be from source content only)",
      "back": "Answer or definition on the back (MUST be from source content only)",
      "source_reference": "Chunk [NUMBER] - brief quote or description. IMPORTANT: Use the ACTUAL chunk number from the source (e.g., if content is from [Chunk 2], write 'Chunk 2', not 'Chunk 1' or 'Chunk X')"
    }}
  ]
}}

REMEMBER: If you cannot create a flashcard using ONLY the source content, create fewer cards. Never invent information."""
        
        system_message = """You are an educational content generator. Your ONLY job is to create flashcards based EXCLUSIVELY on the source content provided by the user.

⚠️ CRITICAL ANTI-HALLUCINATION CONSTRAINTS:
- You MUST NOT use ANY information from your training data, general knowledge, or external sources
- You MUST NOT add facts, examples, definitions, or knowledge NOT in the provided source
- You MUST NOT paraphrase or interpret beyond what is explicitly stated
- If information is missing from the source, you MUST NOT invent it - create fewer cards instead
- Every flashcard front and back MUST be directly traceable to a SPECIFIC chunk with exact quotes
- If you cannot find explicit information in the source, DO NOT create that flashcard
- Adapt flashcard complexity and clarity based on user feedback, but ALWAYS use ONLY source content
- Quote directly from the source when possible - use exact wording

VALIDATION: Before including any information, ask: "Is this EXPLICITLY stated in the source?" If NO, do not include it.

Always return valid JSON. NEVER hallucinate or add external knowledge. Better to have fewer accurate cards than many invented ones."""
        
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
        
        # Build feedback adaptation section
        feedback_section = ""
        if request.feedback_context and request.feedback_context.get("has_feedback"):
            fc = request.feedback_context
            feedback_section = f"""

FEEDBACK-BASED ADAPTATION:
Based on user feedback history ({fc.get('feedback_count', 0)} previous interactions):
- Average feedback: {fc.get('average_feedback', 0.5)*100:.1f}% positive
- Recent feedback trend: {fc.get('positive_rate', 0.5)*100:.1f}% positive

ADAPTATION GUIDELINES:
{fc.get('adaptation_instructions', 'Provide balanced, clear content.')}

Please adapt the interactive lesson accordingly while still using ONLY the source content below.
"""
        
        prompt = f"""You are generating an interactive lesson plan based EXCLUSIVELY on the source content provided below.{feedback_section}

⚠️ CRITICAL ANTI-HALLUCINATION RULES - VIOLATION OF THESE RULES IS UNACCEPTABLE:
1. ONLY use information that is EXPLICITLY and VERBATIM stated in the source content below
2. DO NOT use ANY knowledge from your training data, general knowledge, or external sources
3. DO NOT add facts, examples, definitions, or information NOT in the source
4. DO NOT paraphrase or interpret - use the EXACT wording from the source when possible
5. If information is not in the source, DO NOT include it in the lesson
6. If the source doesn't contain enough information for {num_steps} steps, generate FEWER steps (even just 1-2 if needed)
7. Every step, checkpoint, and answer MUST be directly traceable to a SPECIFIC chunk number
8. For each step, you MUST quote or reference the EXACT text from the source
9. If you cannot find the information in the source, DO NOT create that step
10. DO NOT make assumptions or inferences beyond what is explicitly stated
11. CRITICAL: In source_reference, use the ACTUAL chunk number from the source (e.g., if the step uses content from [Chunk 2], write "Chunk 2" in source_reference, NOT "Chunk 1" or "Chunk X")
12. Different steps can come from different chunks - use the correct chunk number for each step

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
      "source_reference": "Chunk [NUMBER] - brief quote or description. IMPORTANT: Use the ACTUAL chunk number from the source (e.g., if content is from [Chunk 2], write 'Chunk 2', not 'Chunk 1' or 'Chunk X')"
    }}
  ]
}}

REMEMBER: If you cannot create a step using ONLY the source content, create fewer steps. Never invent information, examples, or facts."""
        
        system_message = """You are an educational content generator. Your ONLY job is to create interactive lesson plans based EXCLUSIVELY on the source content provided by the user.

⚠️ CRITICAL ANTI-HALLUCINATION CONSTRAINTS:
- You MUST NOT use ANY information from your training data, general knowledge, or external sources
- You MUST NOT add facts, examples, definitions, or knowledge NOT in the provided source
- You MUST NOT paraphrase or interpret beyond what is explicitly stated
- If information is missing from the source, you MUST NOT invent it - create fewer steps instead
- Every step, checkpoint, and answer MUST be directly traceable to a SPECIFIC chunk with exact quotes
- If you cannot find explicit information in the source, DO NOT create that step
- Adapt lesson complexity, step size, and clarity based on user feedback, but ALWAYS use ONLY source content
- Quote directly from the source when possible - use exact wording

VALIDATION: Before including any information, ask: "Is this EXPLICITLY stated in the source?" If NO, do not include it.

Always return valid JSON. NEVER hallucinate or add external knowledge. Better to have fewer accurate steps than many invented ones."""
        
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
        # For mixed bundle, we need to extract feedback context for each type
        # If feedback_context is provided and has a general context, use it
        # Otherwise, each type will use its own feedback history (handled in orchestrator)
        
        # Generate each type with feedback context
        # Note: feedback_context should contain context for each type if available
        quiz_feedback = None
        flashcard_feedback = None
        interactive_feedback = None
        
        # Default counts
        quiz_count = 5
        flashcard_count = 10
        interactive_count = 3
        
        if request.feedback_context:
            # If feedback_context has type-specific contexts, use them
            if isinstance(request.feedback_context, dict):
                quiz_feedback = request.feedback_context.get("quiz")
                flashcard_feedback = request.feedback_context.get("flashcard")
                interactive_feedback = request.feedback_context.get("interactive")
                
                # Get adaptive counts if available
                if quiz_feedback and "adaptive_count" in quiz_feedback:
                    quiz_count = quiz_feedback["adaptive_count"]
                if flashcard_feedback and "adaptive_count" in flashcard_feedback:
                    flashcard_count = flashcard_feedback["adaptive_count"]
                if interactive_feedback and "adaptive_count" in interactive_feedback:
                    interactive_count = interactive_feedback["adaptive_count"]
        
        quiz_request = GenerationRequest(
            content_type=ContentType.QUIZ,
            chunks=request.chunks,
            num_items=quiz_count,
            feedback_context=quiz_feedback
        )
        quiz_response = self.generate_quiz(quiz_request)
        
        flashcard_request = GenerationRequest(
            content_type=ContentType.FLASHCARD,
            chunks=request.chunks,
            num_items=flashcard_count,
            feedback_context=flashcard_feedback
        )
        flashcard_response = self.generate_flashcards(flashcard_request)
        
        interactive_request = GenerationRequest(
            content_type=ContentType.INTERACTIVE,
            chunks=request.chunks,
            num_items=interactive_count,
            feedback_context=interactive_feedback
        )
        interactive_response = self.generate_interactive(interactive_request)
        
        # Combine into mixed bundle
        # Include data even if generation had errors, as long as we have some content
        data = {
            "quiz": quiz_response.data if quiz_response.success and quiz_response.data else {},
            "flashcards": flashcard_response.data if flashcard_response.success and flashcard_response.data else {},
            "interactive": interactive_response.data if interactive_response.success and interactive_response.data else {}
        }
        
        # Success if at least one type generated successfully
        success = (quiz_response.success and quiz_response.data) or \
                  (flashcard_response.success and flashcard_response.data) or \
                  (interactive_response.success and interactive_response.data)
        
        # Log what was generated
        self.logger.info(
            f"Mixed bundle generation: "
            f"quiz={'✓' if (quiz_response.success and quiz_response.data) else '✗'}, "
            f"flashcards={'✓' if (flashcard_response.success and flashcard_response.data) else '✗'}, "
            f"interactive={'✓' if (interactive_response.success and interactive_response.data) else '✗'}"
        )
        
        return GenerationResponse(
            content_type=ContentType.MIXED,
            data=data,
            success=success,
            error=None if success else "Some content types failed to generate"
        )


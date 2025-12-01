"""LLM Agent for content generation using OpenAI"""

import json
import os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv

from ..core.messages import GenerationRequest, GenerationResponse, ContentType
from ..core.logger import logger

load_dotenv()


class LLMAgent:
    """Generates learning content using OpenAI GPT-4o-mini"""
    
    def __init__(self):
        self.logger = logger.get_logger()
        
        # Try to get from Streamlit secrets first (for Streamlit Cloud)
        api_key = None
        model = "gpt-4o-mini"
        
        try:
            import streamlit as st
            secrets = st.secrets
            api_key = secrets.get("OPENAI_API_KEY")
            model = secrets.get("OPENAI_MODEL", "gpt-4o-mini")
        except (ImportError, AttributeError, KeyError, FileNotFoundError, RuntimeError):
            # Streamlit not available or secrets not configured
            pass
        
        # Fall back to environment variables
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not model or model == "gpt-4o-mini":
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or environment variables. Please configure it in Streamlit Cloud secrets or .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger.info(f"LLM Agent initialized with model: {self.model}")
    
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
        self.logger.info(f"Quiz generation requested: {num_questions} questions")
        # Create numbered chunks for reference
        chunks_with_numbers = []
        chunk_objects = []  # Store (index, chunk) pairs for filtering
        
        for i, chunk in enumerate(chunks, 1):
            if chunk.strip():  # Only include non-empty chunks
                chunk_objects.append((i, chunk))
        
        if not chunk_objects:
            return GenerationResponse(
                content_type=ContentType.QUIZ,
                data={},
                success=False,
                error="All chunks are empty. PDF extraction may have failed."
            )
        
        # Filter out very short chunks or simple lists (likely not useful for questions)
        # Keep chunks that are substantial (more than just a list of terms)
        filtered_chunks = []
        for i, chunk in chunk_objects:
            # Skip chunks that are just comma-separated lists with no sentences
            # A chunk is likely a simple list if it has many commas but few periods/question marks
            comma_count = chunk.count(',')
            sentence_endings = chunk.count('.') + chunk.count('?') + chunk.count('!')
            word_count = len(chunk.split())
            
            # Keep chunk if it has substantial content (sentences) or is long enough
            if sentence_endings > 0 or word_count > 20 or comma_count < 5:
                filtered_chunks.append((i, chunk))
            else:
                self.logger.debug(f"Skipping Chunk {i} - appears to be a simple list")
        
        # If we filtered out too many, keep some of the filtered ones
        if len(filtered_chunks) < num_questions:
            # Add back some of the filtered chunks to ensure we have enough
            needed = num_questions - len(filtered_chunks)
            for i, chunk in chunk_objects:
                if (i, chunk) not in filtered_chunks and needed > 0:
                    filtered_chunks.append((i, chunk))
                    needed -= 1
        
        # For better diversity, sample chunks from across the document
        if len(filtered_chunks) > 20:  # If we have many chunks, sample strategically
            import random
            # Sample evenly across the document
            num_to_sample = min(50, len(filtered_chunks))
            if num_to_sample < len(filtered_chunks):
                step = max(1, len(filtered_chunks) // num_to_sample)
                sampled = [filtered_chunks[i] for i in range(0, len(filtered_chunks), step)][:num_to_sample]
                # Add some random chunks for extra diversity
                if len(filtered_chunks) > num_to_sample:
                    random_indices = random.sample(range(len(filtered_chunks)), min(10, len(filtered_chunks) - num_to_sample))
                    for idx in random_indices:
                        if filtered_chunks[idx] not in sampled:
                            sampled.append(filtered_chunks[idx])
                filtered_chunks = sampled
                self.logger.info(f"Sampled {len(filtered_chunks)} chunks from {len(chunk_objects)} total chunks for diversity")
        
        # Format chunks with numbers
        for i, chunk in filtered_chunks:
            chunks_with_numbers.append(f"[Chunk {i}]\n{chunk}")
        
        chunks_text = "\n\n".join(chunks_with_numbers)
        
        # Limit total text to avoid token limits (GPT-4o-mini has 128k context, so we can use more)
        # Using ~50k chars (~12k tokens) leaves plenty of room for prompt and response
        max_chars = 50000
        if len(chunks_text) > max_chars:
            self.logger.warning(f"Chunks text too long ({len(chunks_text)} chars), truncating to {max_chars}")
            # Instead of just truncating, try to include complete chunks
            truncated_chunks = []
            current_length = 0
            for chunk in chunks_with_numbers:
                if current_length + len(chunk) + 2 <= max_chars:  # +2 for "\n\n"
                    truncated_chunks.append(chunk)
                    current_length += len(chunk) + 2
                else:
                    break
            chunks_text = "\n\n".join(truncated_chunks)
            if len(chunks_with_numbers) > len(truncated_chunks):
                chunks_text += f"\n\n[Note: {len(chunks_with_numbers) - len(truncated_chunks)} additional chunks were truncated due to length limits]"
        
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
6. IMPORTANT: With {len(chunks_with_numbers)} chunks of content provided, you should be able to generate {num_questions} questions. Only generate fewer if you truly cannot find enough distinct information. Try to use different chunks for different questions.
7. Every question, option, and explanation MUST be directly traceable to a SPECIFIC chunk number
8. For each question, you MUST quote or reference the EXACT text from the source that supports it
9. If you cannot find the answer in the source, DO NOT create that question
10. DO NOT make assumptions or inferences beyond what is explicitly stated
11. CRITICAL: In source_reference, use the ACTUAL chunk number from the source (e.g., if the question uses content from [Chunk 2], write "Chunk 2" in source_reference, NOT "Chunk 1" or "Chunk X")
12. Different questions can come from different chunks - use the correct chunk number for each question

Source Content (numbered by chunk):
{chunks_text}

CRITICAL REQUIREMENT: You have {len(chunks_with_numbers)} chunks of content available (over 50,000 characters). This is MORE than enough material to create {num_questions} questions.

YOU MUST GENERATE EXACTLY {num_questions} QUESTIONS - NO FEWER, NO MORE.

⚠️ CRITICAL DIVERSITY REQUIREMENT - VIOLATION WILL RESULT IN POOR QUALITY:
- You MUST use DIFFERENT chunks for DIFFERENT questions - NO EXCEPTIONS
- DO NOT use the same chunk for multiple questions - each question MUST come from a different chunk
- Question 1 should use Chunk X, Question 2 should use Chunk Y (different from X), Question 3 should use Chunk Z (different from X and Y), etc.
- Spread questions across as many different chunks as possible (aim to use at least {min(num_questions, len(chunks_with_numbers))} different chunks)
- If you see a chunk that's just a simple list (like "Term1, Term2, Term3"), SKIP IT and use chunks with more detailed content
- Prioritize chunks with substantial content over simple lists or headings
- Before creating each question, check: "Have I already used this chunk number?" If YES, use a different chunk
- The source_reference MUST show different chunk numbers for different questions

Generate EXACTLY {num_questions} multiple-choice quiz questions. Return a JSON object with this exact structure:
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

FINAL REMINDER: 
- You have {len(chunks_with_numbers)} chunks with over 50,000 characters of content
- You MUST return EXACTLY {num_questions} questions in the "questions" array
- CRITICAL: Use DIFFERENT chunks for different questions - each question MUST have a different chunk number in source_reference
- DO NOT use the same chunk twice - if you see "Chunk 1" in one question, use "Chunk 2", "Chunk 3", etc. for other questions
- Skip simple list chunks - prioritize chunks with detailed explanations or substantial content
- NEVER invent information - only use what's explicitly in the source
- The JSON must have exactly {num_questions} items in the questions array
- DO NOT return fewer than {num_questions} questions
- Verify: Check that your source_reference values show different chunk numbers for each question"""
        
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
            max_retries = 1  # Reduced to 1 retry for speed (only retry if < 50% of requested)
            questions = []
            
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    # On retry, be more explicit about needing the exact number
                    retry_prompt = f"""You previously generated only {len(questions)} questions, but I need EXACTLY {num_questions} questions.

You have {len(chunks_with_numbers)} chunks of content available. This is MORE than enough to create {num_questions} questions.

REQUIREMENT: Generate EXACTLY {num_questions} questions. Use different chunks for different questions. Even if some chunks have less information, you can still create questions from them.

Here is the source content again:
{chunks_text}

Generate EXACTLY {num_questions} multiple-choice quiz questions. Return a JSON object with this exact structure:
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

You MUST return EXACTLY {num_questions} questions in the array."""
                    current_prompt = retry_prompt
                else:
                    current_prompt = prompt
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": current_prompt}
                    ],
                    temperature=0.1,  # Very low temperature to minimize creativity/hallucination
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Validate that we got questions
                questions = data.get("questions", [])
                if not questions or len(questions) == 0:
                    if attempt < max_retries:
                        self.logger.warning(f"Attempt {attempt + 1}: Quiz generation returned empty questions array, retrying...")
                        continue
                    else:
                        self.logger.error("Quiz generation returned empty questions array after all retries")
                        return GenerationResponse(
                            content_type=ContentType.QUIZ,
                            data={},
                            success=False,
                            error="No questions were generated. The source content may not contain enough information."
                        )
                
                # If we got enough questions, break
                if len(questions) >= num_questions:
                    break
                
                # Only retry if we got significantly fewer questions (< 50% of requested)
                # This avoids unnecessary retries for small differences
                if len(questions) < (num_questions * 0.5) and attempt < max_retries:
                    self.logger.warning(f"Attempt {attempt + 1}: Generated {len(questions)} questions but {num_questions} were requested. Retrying to get more...")
                elif attempt < max_retries:
                    # Got at least 50% but not all - accept it to save time
                    self.logger.info(f"Generated {len(questions)} questions (requested {num_questions}). Accepting result to save time.")
                    break
                else:
                    self.logger.warning(f"Generated {len(questions)} questions but {num_questions} were requested after {max_retries + 1} attempts.")
            
            # Log final result
            if len(questions) >= num_questions:
                self.logger.info(f"Successfully generated {len(questions)} quiz questions (requested: {num_questions})")
            else:
                self.logger.warning(f"Final result: Generated {len(questions)} questions but {num_questions} were requested.")
            
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
        
        # Limit total text to avoid token limits (GPT-4o-mini has 128k context, so we can use more)
        # Using ~50k chars (~12k tokens) leaves plenty of room for prompt and response
        max_chars = 50000
        if len(chunks_text) > max_chars:
            self.logger.warning(f"Chunks text too long ({len(chunks_text)} chars), truncating to {max_chars}")
            # Instead of just truncating, try to include complete chunks
            truncated_chunks = []
            current_length = 0
            for chunk in chunks_with_numbers:
                if current_length + len(chunk) + 2 <= max_chars:  # +2 for "\n\n"
                    truncated_chunks.append(chunk)
                    current_length += len(chunk) + 2
                else:
                    break
            chunks_text = "\n\n".join(truncated_chunks)
            if len(chunks_with_numbers) > len(truncated_chunks):
                chunks_text += f"\n\n[Note: {len(chunks_with_numbers) - len(truncated_chunks)} additional chunks were truncated due to length limits]"
        
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
        
        # Limit total text to avoid token limits (GPT-4o-mini has 128k context, so we can use more)
        # Using ~50k chars (~12k tokens) leaves plenty of room for prompt and response
        max_chars = 50000
        if len(chunks_text) > max_chars:
            self.logger.warning(f"Chunks text too long ({len(chunks_text)} chars), truncating to {max_chars}")
            # Instead of just truncating, try to include complete chunks
            truncated_chunks = []
            current_length = 0
            for chunk in chunks_with_numbers:
                if current_length + len(chunk) + 2 <= max_chars:  # +2 for "\n\n"
                    truncated_chunks.append(chunk)
                    current_length += len(chunk) + 2
                else:
                    break
            chunks_text = "\n\n".join(truncated_chunks)
            if len(chunks_with_numbers) > len(truncated_chunks):
                chunks_text += f"\n\n[Note: {len(chunks_with_numbers) - len(truncated_chunks)} additional chunks were truncated due to length limits]"
        
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
        # Validate chunks first
        chunks = request.chunks or []
        if not chunks or all(not chunk.strip() for chunk in chunks):
            self.logger.error("No valid chunks provided for mixed bundle generation")
            return GenerationResponse(
                content_type=ContentType.MIXED,
                data={},
                success=False,
                error="No content chunks available. Please ensure the PDF was properly extracted."
            )
        
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
        
        # Generate all three content types in parallel for speed
        quiz_request = GenerationRequest(
            content_type=ContentType.QUIZ,
            chunks=request.chunks,
            num_items=quiz_count,
            feedback_context=quiz_feedback
        )
        flashcard_request = GenerationRequest(
            content_type=ContentType.FLASHCARD,
            chunks=request.chunks,
            num_items=flashcard_count,
            feedback_context=flashcard_feedback
        )
        interactive_request = GenerationRequest(
            content_type=ContentType.INTERACTIVE,
            chunks=request.chunks,
            num_items=interactive_count,
            feedback_context=interactive_feedback
        )
        
        # Generate all three in parallel
        self.logger.info("Starting parallel generation of quiz, flashcards, and interactive content...")
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all three tasks
                quiz_future = executor.submit(self.generate_quiz, quiz_request)
                flashcard_future = executor.submit(self.generate_flashcards, flashcard_request)
                interactive_future = executor.submit(self.generate_interactive, interactive_request)
                
                # Wait for all to complete and get results
                try:
                    quiz_response = quiz_future.result(timeout=120)  # 2 minute timeout per task
                except Exception as e:
                    self.logger.error(f"Quiz generation failed with exception: {e}")
                    quiz_response = GenerationResponse(
                        content_type=ContentType.QUIZ,
                        data={},
                        success=False,
                        error=f"Exception: {str(e)}"
                    )
                
                try:
                    flashcard_response = flashcard_future.result(timeout=120)
                except Exception as e:
                    self.logger.error(f"Flashcard generation failed with exception: {e}")
                    flashcard_response = GenerationResponse(
                        content_type=ContentType.FLASHCARD,
                        data={},
                        success=False,
                        error=f"Exception: {str(e)}"
                    )
                
                try:
                    interactive_response = interactive_future.result(timeout=120)
                except Exception as e:
                    self.logger.error(f"Interactive generation failed with exception: {e}")
                    interactive_response = GenerationResponse(
                        content_type=ContentType.INTERACTIVE,
                        data={},
                        success=False,
                        error=f"Exception: {str(e)}"
                    )
        except Exception as e:
            self.logger.exception(f"Critical error in parallel generation: {e}")
            # Return error response
            return GenerationResponse(
                content_type=ContentType.MIXED,
                data={},
                success=False,
                error=f"Critical error during generation: {str(e)}"
            )
        
        self.logger.info("Parallel generation completed")
        
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
        
        # Build detailed error message
        errors = []
        if not (quiz_response.success and quiz_response.data):
            quiz_error = quiz_response.error or 'Failed to generate'
            errors.append(f"Quiz: {quiz_error}")
            self.logger.error(f"Quiz generation failed: {quiz_error}")
        if not (flashcard_response.success and flashcard_response.data):
            flashcard_error = flashcard_response.error or 'Failed to generate'
            errors.append(f"Flashcards: {flashcard_error}")
            self.logger.error(f"Flashcard generation failed: {flashcard_error}")
        if not (interactive_response.success and interactive_response.data):
            interactive_error = interactive_response.error or 'Failed to generate'
            errors.append(f"Interactive: {interactive_error}")
            self.logger.error(f"Interactive generation failed: {interactive_error}")
        
        # Always provide detailed error message if any failed
        if errors:
            error_msg = f"Failed to generate: {', '.join(errors)}"
            self.logger.error(f"Mixed bundle generation failed. Details: {error_msg}")
        else:
            error_msg = None
        
        return GenerationResponse(
            content_type=ContentType.MIXED,
            data=data,
            success=success,
            error=error_msg
        )


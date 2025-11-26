"""Manager Agent core orchestration logic"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from .messages import (
    ExtractionRequest, ExtractionResponse,
    GenerationRequest, GenerationResponse,
    RLUpdateRequest, RLRecommendation,
    ManagerCommand, ContentType, LearningMode
)
from .memory import load_state, save_state, reset_state, RLState
from .logger import logger


class ManagerAgent:
    """Orchestrates all agents and manages workflow"""
    
    def __init__(self, username: Optional[str] = None):
        self.logger = logger.get_logger()
        self.session_context: Dict[str, Any] = {}
        self.username = username
        self.state: RLState = load_state(username)
    
    def handle_command(self, command: ManagerCommand) -> Dict[str, Any]:
        """Handle a manager command synchronously"""
        try:
            # Add session_id to params if provided
            params = command.params.copy() if command.params else {}
            if command.session_id:
                params["session_id"] = command.session_id
            
            if command.action == "extract":
                return self._handle_extract(params)
            elif command.action == "generate":
                return self._handle_generate(params)
            elif command.action == "update_rl":
                return self._handle_update_rl(params)
            elif command.action == "recommend":
                return self._handle_recommend(params)
            elif command.action == "survey":
                return self._handle_survey(params)
            elif command.action == "reset_preferences":
                return self._handle_reset_preferences()
            else:
                return {"success": False, "error": f"Unknown action: {command.action}"}
        except Exception as e:
            self.logger.exception(f"Error handling command {command.action}")
            return {"success": False, "error": str(e)}
    
    async def handle_command_async(self, command: ManagerCommand) -> Dict[str, Any]:
        """Handle a manager command asynchronously (for parallel agent calls)"""
        # Future enhancement: use asyncio.gather for parallel operations
        return self.handle_command(command)
    
    def _handle_extract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route extraction request to NLP Agent"""
        from ..agents.nlp_agent import NLPAgent
        
        nlp_agent = NLPAgent()
        # Remove session_id from params as ExtractionRequest doesn't accept it
        extract_params = {k: v for k, v in params.items() if k in ['file_path', 'file_content', 'file_type']}
        request = ExtractionRequest(**extract_params)
        response = nlp_agent.extract(request)
        
        # Store in session context
        session_id = params.get("session_id")
        self.logger.info(f"Extract handler - session_id: {session_id}, chunks count: {len(response.chunks) if response.chunks else 0}")
        
        if session_id:
            self.session_context[session_id] = {
                "chunks": response.chunks,
                "summary": response.summary
            }
            self.logger.info(f"Stored {len(response.chunks)} chunks in session {session_id} (total {sum(len(c) for c in response.chunks)} chars)")
            self.logger.info(f"Session context keys: {list(self.session_context.keys())}")
        else:
            self.logger.warning("No session_id provided in extract params, chunks will not be stored in session context")
        
        return {
            "success": response.success,
            "chunks": response.chunks,
            "summary": response.summary,
            "error": response.error
        }
    
    def _handle_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route generation request to LLM Agent"""
        from ..agents.llm_agent import LLMAgent
        
        llm_agent = LLMAgent()
        
        # Determine content type
        content_type_str = params.get("content_type", "quiz")
        content_type = ContentType(content_type_str)
        
        # Get chunks from session or params
        chunks = params.get("chunks", [])
        session_id = params.get("session_id")
        
        if not chunks and session_id:
            self.logger.info(f"Generate handler - session_id: {session_id}")
            self.logger.info(f"Available session context keys: {list(self.session_context.keys())}")
            session_data = self.session_context.get(session_id, {})
            chunks = session_data.get("chunks", [])
            self.logger.info(f"Retrieved {len(chunks)} chunks from session {session_id}")
            if chunks:
                self.logger.info(f"First chunk preview: {chunks[0][:100]}...")
        
        # Fallback: if still no chunks, try to get from params (in case UI passed them directly)
        if not chunks:
            # Check if chunks were passed in params but with a different key
            if "extracted_chunks" in params:
                chunks = params.get("extracted_chunks", [])
                self.logger.info(f"Retrieved {len(chunks)} chunks from params.extracted_chunks")
        
        # Filter out empty chunks
        if chunks:
            chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
            self.logger.info(f"After filtering empty chunks: {len(chunks)} chunks remaining")
        
        # Validate chunks are present and non-empty
        if not chunks:
            error_msg = "No content chunks available. Please extract text from a file first."
            self.logger.error(error_msg)
            return {
                "success": False,
                "content_type": content_type.value,
                "data": {},
                "error": error_msg
            }
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
        
        if not chunks:
            error_msg = "All extracted chunks are empty. PDF extraction may have failed."
            self.logger.error(error_msg)
            return {
                "success": False,
                "content_type": content_type.value,
                "data": {},
                "error": error_msg
            }
        
        self.logger.info(f"Generating {content_type.value} with {len(chunks)} chunks (total {sum(len(c) for c in chunks)} chars)")
        
        # Reload state to get latest feedback before generating context
        self.state = load_state(self.username)
        
        # Get feedback context for content adaptation
        # For mixed bundle, get context for all types and calculate adaptive quantities
        if content_type == ContentType.MIXED:
            quiz_fc = self._get_feedback_context("quiz")
            flashcard_fc = self._get_feedback_context("flashcard")
            interactive_fc = self._get_feedback_context("interactive")
            
            feedback_context = {
                "quiz": quiz_fc,
                "flashcard": flashcard_fc,
                "interactive": interactive_fc
            }
            
            # Calculate adaptive quantities for each type in mixed bundle
            # Store in params so LLM agent can use them
            base_quiz = self._calculate_item_count(ContentType.QUIZ, len(chunks))
            base_flashcard = self._calculate_item_count(ContentType.FLASHCARD, len(chunks))
            base_interactive = self._calculate_item_count(ContentType.INTERACTIVE, len(chunks))
            
            self.logger.info(f"Base counts for {len(chunks)} chunks: quiz={base_quiz}, flashcard={base_flashcard}, interactive={base_interactive}")
            
            # Adjust based on feedback
            quiz_count = self._calculate_adaptive_count(base_quiz, quiz_fc, "quiz")
            flashcard_count = self._calculate_adaptive_count(base_flashcard, flashcard_fc, "flashcard")
            interactive_count = self._calculate_adaptive_count(base_interactive, interactive_fc, "interactive")
            
            if quiz_count != base_quiz:
                self.logger.info(f"Quiz count adjusted from {base_quiz} to {quiz_count} based on feedback")
            
            # Store in feedback_context for LLM agent to use
            feedback_context["quiz"]["adaptive_count"] = quiz_count
            feedback_context["flashcard"]["adaptive_count"] = flashcard_count
            feedback_context["interactive"]["adaptive_count"] = interactive_count
        else:
            feedback_context = self._get_feedback_context(content_type_str)
        
        # Calculate dynamic item count based on chunks AND feedback
        # For MIXED, num_items is not used (each type has its own adaptive count)
        num_items = params.get("num_items")
        if content_type != ContentType.MIXED:
            if num_items is None:
                base_count = self._calculate_item_count(content_type, len(chunks))
                # Adjust quantity based on feedback
                if feedback_context.get("has_feedback"):
                    avg_feedback = feedback_context.get("average_feedback", 0.5)
                    if avg_feedback < 0.4:  # Disliked content
                        # Reduce by 40-60% for disliked content
                        reduction_factor = 0.4 + (avg_feedback * 0.4)  # 0.4 to 0.56 range
                        num_items = max(2, int(base_count * reduction_factor))
                        self.logger.info(f"Reduced {content_type_str} items from {base_count} to {num_items} due to low feedback ({avg_feedback:.2f})")
                    elif avg_feedback > 0.7:  # Liked content - increase quantity
                        # Increase by 20-50% for liked content
                        increase_factor = 1.2 + ((avg_feedback - 0.7) * 1.0)  # 1.2 to 1.5 range
                        num_items = min(int(base_count * increase_factor), base_count * 2)  # Cap at 2x
                        self.logger.info(f"Increased {content_type_str} items from {base_count} to {num_items} due to high feedback ({avg_feedback:.2f})")
                    else:
                        num_items = base_count
                else:
                    num_items = base_count
            else:
                num_items = num_items
        else:
            # For MIXED, num_items is not used - adaptive counts are in feedback_context
            num_items = None
            self.logger.info(f"Mixed bundle: quiz={feedback_context.get('quiz', {}).get('adaptive_count', 'N/A')}, "
                           f"flashcard={feedback_context.get('flashcard', {}).get('adaptive_count', 'N/A')}, "
                           f"interactive={feedback_context.get('interactive', {}).get('adaptive_count', 'N/A')}")
        

        request = GenerationRequest(    
            content_type=content_type,
            chunks=chunks,
            num_items=num_items,
            context=params.get("context"),
            feedback_context=feedback_context
        )
        
        response = llm_agent.generate(request)
        
        return {
            "success": response.success,
            "content_type": response.content_type.value,
            "data": response.data,
            "error": response.error
        }
    
    def _handle_update_rl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route feedback update to RL Agent"""
        try:
            from ..agents.rl_agent import RLAgent
            
            rl_agent = RLAgent(username=self.username)
            request = RLUpdateRequest(**params)
            rl_agent.update_from_feedback(request)
            
            # Reload state after update
            self.state = load_state(self.username)
            
            return {"success": True}
        except Exception as e:
            self.logger.exception(f"Error updating RL state: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_recommend(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendation from RL Agent"""
        try:
            from ..agents.rl_agent import RLAgent
            
            rl_agent = RLAgent(username=self.username)
            recommendation = rl_agent.recommend_mode()
            
            return {
                "success": True,
                "recommended_mode": recommendation.recommended_mode,
                "probabilities": recommendation.probabilities,
                "confidence": recommendation.confidence
            }
        except Exception as e:
            self.logger.exception(f"Error getting RL recommendation: {e}")
            # Return default recommendation on error
            return {
                "success": True,
                "recommended_mode": "quiz",  # Safe default
                "probabilities": {"quiz": 0.5, "flashcard": 0.5, "interactive": 0.5},
                "confidence": 0.0
            }
    
    def _handle_survey(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle survey response"""
        preference = params.get("preference")
        
        self.state.survey_completed = True
        self.state.initial_preference = preference
        self.state.total_sessions += 1
        self.state.last_updated = datetime.now().isoformat()
        
        save_state(self.state, self.username)
        
        return {
            "success": True,
            "preference": preference,
            "survey_completed": True
        }
    
    def _handle_reset_preferences(self) -> Dict[str, Any]:
        """Reset user preferences and RL state"""
        from .memory import reset_state
        self.state = reset_state(self.username)
        self.session_context.clear()
        
        return {
            "success": True,
            "message": "Preferences reset successfully"
        }
    
    def _calculate_item_count(self, content_type: ContentType, num_chunks: int) -> int:
        """Dynamically calculate number of items based on document length"""
        if content_type == ContentType.QUIZ:
            # For large files (100+ chunks), generate more questions
            if num_chunks >= 100:
                return min(20, max(10, num_chunks // 10))  # 10-20 questions for large files
            elif num_chunks >= 50:
                return min(15, max(8, num_chunks // 8))  # 8-15 for medium files
            else:
                return min(10, max(3, num_chunks // 2))  # 3-10 for small files
        elif content_type == ContentType.FLASHCARD:
            # For large files, generate more flashcards
            if num_chunks >= 100:
                return min(25, max(15, num_chunks // 8))  # 15-25 for large files
            else:
                return min(15, max(5, num_chunks))  # 5-15 for smaller files
        elif content_type == ContentType.INTERACTIVE:
            # For large files, generate more steps
            if num_chunks >= 100:
                return min(8, max(5, num_chunks // 30))  # 5-8 for large files
            else:
                return min(5, max(2, num_chunks // 3))  # 2-5 for smaller files
        else:
            return 5
    
    def _calculate_adaptive_count(self, base_count: int, feedback_context: Dict[str, Any], content_type: str) -> int:
        """
        Calculate adaptive item count based on feedback.
        
        Args:
            base_count: Base count before feedback adjustment
            feedback_context: Feedback context for this content type
            content_type: Type of content (for logging)
        
        Returns:
            Adjusted item count
        """
        if not feedback_context.get("has_feedback"):
            return base_count
        
        avg_feedback = feedback_context.get("average_feedback", 0.5)
        
        if avg_feedback < 0.4:  # Disliked content
            # Reduce by 40-60% for disliked content
            reduction_factor = 0.4 + (avg_feedback * 0.4)  # 0.4 to 0.56 range
            adjusted = max(2, int(base_count * reduction_factor))
            self.logger.info(f"Reduced {content_type} items from {base_count} to {adjusted} due to low feedback ({avg_feedback:.2f})")
            return adjusted
        elif avg_feedback > 0.7:  # Liked content - increase quantity
            # Increase by 20-50% for liked content
            increase_factor = 1.2 + ((avg_feedback - 0.7) * 1.0)  # 1.2 to 1.5 range
            adjusted = min(int(base_count * increase_factor), base_count * 2)  # Cap at 2x
            self.logger.info(f"Increased {content_type} items from {base_count} to {adjusted} due to high feedback ({avg_feedback:.2f})")
            return adjusted
        else:
            return base_count
    
    def get_current_preference(self) -> Optional[str]:
        """Get current user preference from state"""
        if not self.state.survey_completed:
            return None
        return self.state.initial_preference
    
    def should_show_mixed_bundle(self) -> bool:
        """Determine if mixed bundle should be shown"""
        return (
            not self.state.survey_completed or
            self.state.initial_preference == LearningMode.UNKNOWN.value
        )
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context"""
        return self.session_context.get(session_id, {})
    
    def _get_feedback_context(self, content_type: str) -> Dict[str, Any]:
        """
        Analyze feedback history to create adaptation context for content generation.
        
        Args:
            content_type: Type of content being generated ("quiz", "flashcard", "interactive")
        
        Returns:
            Dictionary with feedback insights for content adaptation
        """
        feedback_context = {
            "has_feedback": False,
            "average_feedback": 0.5,
            "feedback_count": 0,
            "positive_rate": 0.5,
            "adaptation_instructions": ""
        }
        
        # Get feedback history for this content type
        mode_history = [entry for entry in self.state.mode_history if entry.get("mode") == content_type]
        
        if not mode_history:
            return feedback_context
        
        feedback_context["has_feedback"] = True
        feedback_context["feedback_count"] = len(mode_history)
        
        # Calculate average feedback
        feedbacks = [entry.get("feedback", 0.5) for entry in mode_history]
        feedback_context["average_feedback"] = sum(feedbacks) / len(feedbacks) if feedbacks else 0.5
        
        # Calculate positive rate
        positive_count = sum(1 for f in feedbacks if f > 0.5)
        feedback_context["positive_rate"] = positive_count / len(feedbacks) if feedbacks else 0.5
        
        # Get recent feedback (last 5)
        recent_feedbacks = feedbacks[-5:]
        recent_avg = sum(recent_feedbacks) / len(recent_feedbacks) if recent_feedbacks else 0.5
        
        # Create adaptation instructions based on feedback patterns
        adaptation_instructions = []
        
        if recent_avg > 0.7:
            # User likes current content - maintain or slightly enhance
            if content_type == "quiz":
                adaptation_instructions.append("User has been enjoying the quiz questions. Maintain similar difficulty and style.")
            elif content_type == "flashcard":
                adaptation_instructions.append("User finds flashcards helpful. Keep similar complexity and clarity.")
            else:
                adaptation_instructions.append("User enjoys interactive lessons. Maintain similar step-by-step approach.")
        elif recent_avg < 0.4:
            # User dislikes current content - adapt significantly
            if content_type == "quiz":
                adaptation_instructions.append("User found previous questions challenging. Make questions SIGNIFICANTLY EASIER and more straightforward.")
                adaptation_instructions.append("Focus on clear, direct questions with concise explanations. Make them more ENGAGING and INTERESTING with practical examples from the source.")
                adaptation_instructions.append("Prioritize questions that are fun, relatable, and easier to understand.")
            elif content_type == "flashcard":
                adaptation_instructions.append("User found previous flashcards difficult. Use MUCH SIMPLER language and more focused concepts.")
                adaptation_instructions.append("Make flashcards more INTERESTING and ENGAGING with memorable examples and clear connections.")
                adaptation_instructions.append("Keep concepts bite-sized and easy to remember.")
            else:
                adaptation_instructions.append("User found previous interactive content complex. Break down into MUCH SMALLER, CLEARER steps.")
                adaptation_instructions.append("Make the content MORE INTERESTING and ENGAGING with practical examples, analogies, and real-world connections from the source.")
                adaptation_instructions.append("Use a conversational, friendly tone. Make each step feel rewarding and easy to complete.")
        else:
            # Mixed feedback - balanced approach
            adaptation_instructions.append("User has mixed feedback. Provide balanced content with clear explanations.")
        
        # Add difficulty guidance based on feedback variance
        if len(recent_feedbacks) > 1:
            variance = sum((f - recent_avg) ** 2 for f in recent_feedbacks) / len(recent_feedbacks)
            if variance > 0.1:
                adaptation_instructions.append("User preferences vary. Provide a mix of difficulty levels.")
        
        feedback_context["adaptation_instructions"] = " ".join(adaptation_instructions)
        
        self.logger.info(
            f"Feedback context for {content_type}: "
            f"avg={feedback_context['average_feedback']:.2f}, "
            f"positive_rate={feedback_context['positive_rate']:.2f}, "
            f"count={feedback_context['feedback_count']}"
        )
        

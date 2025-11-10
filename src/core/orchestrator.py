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
    
    def __init__(self):
        self.logger = logger.get_logger()
        self.session_context: Dict[str, Any] = {}
        self.state: RLState = load_state()
    
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
        request = ExtractionRequest(**params)
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
        
        # Calculate dynamic item count based on chunks
        num_items = params.get("num_items")
        if num_items is None:
            num_items = self._calculate_item_count(content_type, len(chunks))
        
        request = GenerationRequest(
            content_type=content_type,
            chunks=chunks,
            num_items=num_items,
            context=params.get("context")
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
        from ..agents.rl_agent import RLAgent
        
        rl_agent = RLAgent()
        request = RLUpdateRequest(**params)
        rl_agent.update_from_feedback(request)
        
        # Reload state after update
        self.state = load_state()
        
        return {"success": True}
    
    def _handle_recommend(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendation from RL Agent"""
        from ..agents.rl_agent import RLAgent
        
        rl_agent = RLAgent()
        recommendation = rl_agent.recommend_mode()
        
        return {
            "success": True,
            "recommended_mode": recommendation.recommended_mode,
            "probabilities": recommendation.probabilities,
            "confidence": recommendation.confidence
        }
    
    def _handle_survey(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle survey response"""
        preference = params.get("preference")
        
        self.state.survey_completed = True
        self.state.initial_preference = preference
        self.state.total_sessions += 1
        self.state.last_updated = datetime.now().isoformat()
        
        save_state(self.state)
        
        return {
            "success": True,
            "preference": preference,
            "survey_completed": True
        }
    
    def _handle_reset_preferences(self) -> Dict[str, Any]:
        """Reset user preferences and RL state"""
        self.state = reset_state()
        self.session_context.clear()
        
        return {
            "success": True,
            "message": "Preferences reset successfully"
        }
    
    def _calculate_item_count(self, content_type: ContentType, num_chunks: int) -> int:
        """Dynamically calculate number of items based on document length"""
        if content_type == ContentType.QUIZ:
            return min(10, max(3, num_chunks // 2))
        elif content_type == ContentType.FLASHCARD:
            return min(15, max(5, num_chunks))
        elif content_type == ContentType.INTERACTIVE:
            return min(5, max(2, num_chunks // 3))
        else:
            return 5
    
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


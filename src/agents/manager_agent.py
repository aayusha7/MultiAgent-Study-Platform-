"""Manager Agent for high-level orchestration"""

from typing import Dict, Any, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

from ..core.orchestrator import ManagerAgent as Orchestrator
from ..core.messages import ManagerCommand, LearningMode
from ..core.logger import logger

load_dotenv()


class ManagerAgent:
    """
    High-level manager that coordinates all agents.
    Uses orchestrator for core logic and OpenAI for reasoning.
    """
    
    def __init__(self, username: Optional[str] = None):
        self.logger = logger.get_logger()
        self.username = username
        self.orchestrator = Orchestrator(username=username)
        self.openai_client = None
        
        # Initialize OpenAI client if available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI client: {e}")
    
    def process_user_request(
        self,
        action: str,
        params: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user request through orchestrator.
        
        Args:
            action: Action to perform
            params: Parameters for the action
            session_id: Optional session identifier
        
        Returns:
            Result dictionary
        """
        command = ManagerCommand(
            action=action,
            params=params,
            session_id=session_id
        )
        
        return self.orchestrator.handle_command(command)
    
    def reason_about_preference(
        self,
        user_feedback_history: list,
        current_preference: Optional[str]
    ) -> Dict[str, Any]:
        """
        Use OpenAI to reason about user preferences (with rule-based fallback).
        
        Args:
            user_feedback_history: List of feedback entries
            current_preference: Current user preference
        
        Returns:
            Reasoning result with suggested mode mix
        """
        if not self.openai_client:
            # Rule-based fallback
            return self._rule_based_reasoning(user_feedback_history, current_preference)
        
        try:
            # Build prompt for reasoning
            feedback_summary = self._summarize_feedback(user_feedback_history)
            
            prompt = f"""Based on the user's learning history, suggest the best learning mode mix.

Current preference: {current_preference or "Unknown"}
Feedback history: {feedback_summary}

Suggest a mode mix (percentages should sum to 100):
- quiz: percentage
- flashcard: percentage  
- interactive: percentage

Return JSON with this structure:
{{
  "quiz": 70,
  "flashcard": 20,
  "interactive": 10,
  "reasoning": "Brief explanation"
}}
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert learning advisor. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            self.logger.warning(f"OpenAI reasoning failed, using fallback: {e}")
            return self._rule_based_reasoning(user_feedback_history, current_preference)
    
    def _rule_based_reasoning(
        self,
        user_feedback_history: list,
        current_preference: Optional[str]
    ) -> Dict[str, Any]:
        """Rule-based fallback for preference reasoning"""
        # Get RL recommendation
        rl_recommendation = self.orchestrator._handle_recommend({})
        
        if rl_recommendation.get("success"):
            recommended = rl_recommendation.get("recommended_mode", "quiz")
            probs = rl_recommendation.get("probabilities", {})
            
            # Convert probabilities to percentages
            total = sum(probs.values())
            if total > 0:
                quiz_pct = int((probs.get("quiz", 0.33) / total) * 100)
                flashcard_pct = int((probs.get("flashcard", 0.33) / total) * 100)
                interactive_pct = 100 - quiz_pct - flashcard_pct
            else:
                quiz_pct = 33
                flashcard_pct = 33
                interactive_pct = 34
            
            return {
                "quiz": quiz_pct,
                "flashcard": flashcard_pct,
                "interactive": interactive_pct,
                "reasoning": f"Based on RL agent recommendation: {recommended}"
            }
        
        # Default mix
        return {
            "quiz": 40,
            "flashcard": 30,
            "interactive": 30,
            "reasoning": "Default balanced mix"
        }
    
    def _summarize_feedback(self, feedback_history: list) -> str:
        """Summarize feedback history for reasoning"""
        if not feedback_history:
            return "No feedback history"
        
        # Count feedback by mode
        mode_counts = {}
        total_feedback = len(feedback_history)
        
        for entry in feedback_history[-20:]:  # Last 20 entries
            mode = entry.get("mode", "unknown")
            feedback = entry.get("feedback", 0.5)
            
            if mode not in mode_counts:
                mode_counts[mode] = {"positive": 0, "negative": 0, "total": 0}
            
            mode_counts[mode]["total"] += 1
            if feedback > 0.5:
                mode_counts[mode]["positive"] += 1
            else:
                mode_counts[mode]["negative"] += 1
        
        summary_parts = []
        for mode, counts in mode_counts.items():
            pos_rate = counts["positive"] / counts["total"] if counts["total"] > 0 else 0
            summary_parts.append(
                f"{mode}: {counts['total']} interactions, "
                f"{pos_rate:.0%} positive"
            )
        
        return "; ".join(summary_parts)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current manager state"""
        return {
            "survey_completed": self.orchestrator.state.survey_completed,
            "current_preference": self.orchestrator.get_current_preference(),
            "should_show_mixed": self.orchestrator.should_show_mixed_bundle(),
            "total_sessions": self.orchestrator.state.total_sessions
        }


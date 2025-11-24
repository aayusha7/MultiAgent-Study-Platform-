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

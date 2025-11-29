"""Dataclasses for inter-agent communication"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class ContentType(str, Enum):
  """Types of learning content"""
  QUIZ = "quiz"
  FLASHCARD = "flashcard"
  INTERACTIVE = "interactive"
  MIXED = "mixed"


class LearningMode(str, Enum):
  """Learning style preferences"""
  QUIZ = "quiz"
  FLASHCARD = "flashcard"
  INTERACTIVE = "interactive"
  UNKNOWN = "i_dont_know"


@dataclass
class ExtractionRequest:
  """Request for NLP Agent to extract text"""
  file_path: Optional[str] = None
  file_content: Optional[str] = None
  file_type: str = "pdf" # pdf or text


@dataclass
class ExtractionReponse:
  """Response from NLP Agent with extracted chunks"""
  chunks: List[str]
  summary: Optional[str] = None
  success: bool = True
  error: Optional[str] = None


@dataclass
class GenerationRequest:
  """Request for LLM Agent to generate content"""
  content_type: ContentType
  chunks: List[str]
  num_items: Optional[int] = None  # Number of questions/cards/steps
  context: Optional[str] = None
  feedback_context: Optional[Dict[str, Any]] = None  # Feedback history and preferences for adaptation


@dataclass
class GenerationResponse:
  """Response from LLM Agent with generated content"""
  content_type: ContentType
  data: Dict[str, Any]  # JSON structure for quiz/flashcard/interactive
  success: bool = True
  error: Optional[str] = None



@dataclass
class RLUpdateRequest:
  """Request to update RL Agent with feedback"""
  mode: str  # "quiz", "flashcard", or "interactive"
  feedback: float  # 0.0 (dislike) to 1.0 (like), or binary 0.0/1.0
  session_id: Optional[str] = None


@dataclass
class RLRecommendation:
  """Recommendation from RL Agent"""
  recommended_mode: str
  probabilities: Dict[str, float]  # Mode -> probability
  confidence: float  # 0.0 to 1.0


@dataclass
class ManagerCommand:
  """Command for Manager Agent"""
  action: str  # "extract", "generate", "update_rl", "recommend", "survey"
  params: Dict[str, Any]
  session_id: Optional[str] = None


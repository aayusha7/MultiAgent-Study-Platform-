"""Dataclasses for inter-agent communication"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class ContentType(str, Enum)
class LearningMode(str, Enum)

@dataclass
class ExtractionRequest

@dataclass
class ExtractionReponse

@dataclass
class GenerationRequest

@dataclass
class GenerationResponse

@dataclass
class RLUpdateRequest

@dataclass
class RLRecommendation

@dataclass
class ManagerCommand

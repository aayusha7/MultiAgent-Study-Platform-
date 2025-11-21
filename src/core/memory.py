"""RL state persistence"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class RLState:
    """Reinforcement Learning state"""
    # Mode probabilities (Thompson Sampling parameters)
    mode_alpha: Dict[str, float]  # Success counts
    mode_beta: Dict[str, float]   # Failure counts
    
    # Mode history for analysis
    mode_history: List[Dict[str, Any]]  # List of {mode, feedback, timestamp}
    
    # Performance tracking per chunk/source
    chunk_performance: Optional[Dict[str, Dict[str, Any]]] = field(default_factory=dict)  # chunk_id -> {correct: int, incorrect: int, attempts: int, last_attempt: str}
    # Format: {"chunk_0": {"correct": 3, "incorrect": 1, "attempts": 4, "last_attempt": "2025-11-17T..."}, ...}
    
    # Survey state
    survey_completed: bool = False
    initial_preference: Optional[str] = None  # "quiz", "flashcard", "interactive", or None
    
    # Session tracking
    total_sessions: int = 0
    last_updated: Optional[str] = None


def get_state_path(username: Optional[str] = None) -> Path:
    """Get path to state file (user-specific if username provided)"""
    if username:
        from .auth import get_user_state_path
        return get_user_state_path(username)
    return Path(__file__).parent.parent.parent / ".ma_state.json"


def load_state(username: Optional[str] = None) -> RLState:
    """Load RL state from JSON file (user-specific if username provided)"""
    state_path = get_state_path(username)
    
    if not state_path.exists():
        # Initialize default state
        return RLState(
            mode_alpha={"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0},
            mode_beta={"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0},
            mode_history=[],
            chunk_performance={},
            survey_completed=False,
            initial_preference=None,
            total_sessions=0,
            last_updated=None
        )
    
    try:
        with open(state_path, 'r') as f:
            data = json.load(f)
        
        # Ensure all modes are present
        modes = ["quiz", "flashcard", "interactive"]
        for mode in modes:
            if mode not in data.get("mode_alpha", {}):
                data.setdefault("mode_alpha", {})[mode] = 1.0
            if mode not in data.get("mode_beta", {}):
                data.setdefault("mode_beta", {})[mode] = 1.0
        
        # Ensure chunk_performance exists
        if "chunk_performance" not in data:
            data["chunk_performance"] = {}
        
        return RLState(**data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # If file is corrupted, return default state
        return RLState(
            mode_alpha={"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0},
            mode_beta={"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0},
            mode_history=[],
            chunk_performance={},
            survey_completed=False,
            initial_preference=None,
            total_sessions=0,
            last_updated=None
        )


def save_state(state: RLState, username: Optional[str] = None) -> bool:
    """Save RL state to JSON file (user-specific if username provided)"""
    state_path = get_state_path(username)
    
    try:
        # Convert dataclass to dict
        data = asdict(state)
        
        # Ensure directory exists
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        return False


def reset_state(username: Optional[str] = None) -> RLState:
    """Reset RL state to initial values"""
    state = RLState(
        mode_alpha={"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0},
        mode_beta={"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0},
        mode_history=[],
        chunk_performance={},
        survey_completed=False,
        initial_preference=None,
        total_sessions=0,
        last_updated=None
    )
    save_state(state, username)
    return state


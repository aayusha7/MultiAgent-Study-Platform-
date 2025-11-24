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
    
    # File hash to filename mapping (for analytics display)
    file_mapping: Optional[Dict[str, str]] = field(default_factory=dict)  # file_hash -> filename


def get_state_path(username: Optional[str] = None) -> Path:
    """Get path to state file (user-specific if username provided)"""
    if username:
        from .auth import get_user_state_path
        return get_user_state_path(username)
    return Path(__file__).parent.parent.parent / ".ma_state.json"


def load_state(username: Optional[str] = None) -> RLState:
    """Load RL state - tries Supabase first, then SQLite, then JSON file"""
    # Try Supabase first (if available and username provided)
    if username:
        try:
            from .supabase_client import load_rl_state_supabase
            supabase_state = load_rl_state_supabase(username)
            if supabase_state:
                # Ensure all modes are present
                modes = ["quiz", "flashcard", "interactive"]
                for mode in modes:
                    if mode not in supabase_state.get("mode_alpha", {}):
                        supabase_state.setdefault("mode_alpha", {})[mode] = 1.0
                    if mode not in supabase_state.get("mode_beta", {}):
                        supabase_state.setdefault("mode_beta", {})[mode] = 1.0
                
                # Ensure chunk_performance exists
                if "chunk_performance" not in supabase_state:
                    supabase_state["chunk_performance"] = {}
                
                # Ensure file_mapping exists
                if "file_mapping" not in supabase_state:
                    supabase_state["file_mapping"] = {}
                
                return RLState(**supabase_state)
        except Exception as e:
            from .logger import logger
            logger.get_logger().debug(f"Supabase not available, trying other storage: {e}")
        
        # Try SQLite database (if available)
        try:
            from .database import load_rl_state
            db_state = load_rl_state(username)
            if db_state:
                # Ensure all modes are present
                modes = ["quiz", "flashcard", "interactive"]
                for mode in modes:
                    if mode not in db_state.get("mode_alpha", {}):
                        db_state.setdefault("mode_alpha", {})[mode] = 1.0
                    if mode not in db_state.get("mode_beta", {}):
                        db_state.setdefault("mode_beta", {})[mode] = 1.0
                
                # Ensure chunk_performance exists
                if "chunk_performance" not in db_state:
                    db_state["chunk_performance"] = {}
                
                # Ensure file_mapping exists
                if "file_mapping" not in db_state:
                    db_state["file_mapping"] = {}
                
                return RLState(**db_state)
        except Exception as e:
            from .logger import logger
            logger.get_logger().debug(f"Database not available, using file storage: {e}")
    
    # Fallback to JSON file
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
            last_updated=None,
            file_mapping={}
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
        
        # Ensure file_mapping exists
        if "file_mapping" not in data:
            data["file_mapping"] = {}
        
        return RLState(**data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # If file is corrupted, return default state
        from .logger import logger
        logger.get_logger().warning(f"Corrupted state file for user {username}. Using default state.")
        return RLState(
            mode_alpha={"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0},
            mode_beta={"quiz": 1.0, "flashcard": 1.0, "interactive": 1.0},
            mode_history=[],
            chunk_performance={},
            survey_completed=False,
            initial_preference=None,
            total_sessions=0,
            last_updated=None,
            file_mapping={}
        )


def save_state(state: RLState, username: Optional[str] = None) -> bool:
    """Save RL state - tries Supabase first, then SQLite, then JSON file"""
    # Try Supabase first (if available and username provided)
    if username:
        try:
            from .supabase_client import save_rl_state_supabase
            data = asdict(state)
            if save_rl_state_supabase(username, data):
                return True
        except Exception as e:
            from .logger import logger
            logger.get_logger().debug(f"Supabase not available, trying other storage: {e}")
        
        # Try SQLite database (if available)
        try:
            from .database import save_rl_state
            data = asdict(state)
            if save_rl_state(username, data):
                return True
        except Exception as e:
            from .logger import logger
            logger.get_logger().debug(f"Database not available, using file storage: {e}")
    
    # Fallback to JSON file
    state_path = get_state_path(username)
    
    try:
        # Convert dataclass to dict
        data = asdict(state)
        
        # Ensure directory exists
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first, then rename (atomic write)
        temp_path = state_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename (works on most filesystems)
        temp_path.replace(state_path)
        
        return True
    except Exception as e:
        from .logger import logger
        logger.get_logger().error(f"Failed to save RL state: {e}")
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
        last_updated=None,
        file_mapping={}
    )
    save_state(state, username)
    return state


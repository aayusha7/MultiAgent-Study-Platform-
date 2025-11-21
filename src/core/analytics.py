"""Analytics module for performance tracking and analysis"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from .memory import load_state, save_state
from .logger import logger


def record_quiz_answer(
    chunk_id: str,
    source_reference: str,
    is_correct: bool,
    question_text: str = "",
    username: Optional[str] = None
) -> None:
    """
    Record a quiz answer performance for a specific chunk.
    
    Args:
        chunk_id: Identifier for the chunk (e.g., "chunk_0", "chunk_1")
        source_reference: Reference to source content (e.g., "Chunk 1 - Introduction")
        is_correct: Whether the answer was correct
        question_text: The question text (optional, for reference)
        username: Username for user-specific state (optional)
    """
    state = load_state(username)
    
    # Handle case where chunk_performance might not exist (for old state files)
    if not hasattr(state, 'chunk_performance') or state.chunk_performance is None:
        state.chunk_performance = {}
    
    # Use source_reference as key if chunk_id is not available
    key = chunk_id if chunk_id else source_reference
    
    if key not in state.chunk_performance:
        state.chunk_performance[key] = {
            "correct": 0,
            "incorrect": 0,
            "attempts": 0,
            "last_attempt": None,
            "source_reference": source_reference,
            "questions": []
        }
    
    # Update performance
    state.chunk_performance[key]["attempts"] += 1
    if is_correct:
        state.chunk_performance[key]["correct"] += 1
    else:
        state.chunk_performance[key]["incorrect"] += 1
    
    state.chunk_performance[key]["last_attempt"] = datetime.now().isoformat()
    
    # Store question for reference
    if question_text:
        state.chunk_performance[key]["questions"].append({
            "question": question_text[:200],  # Truncate for storage
            "correct": is_correct,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 10 questions per chunk
        if len(state.chunk_performance[key]["questions"]) > 10:
            state.chunk_performance[key]["questions"] = state.chunk_performance[key]["questions"][-10:]
    
    save_state(state, username)
    
    logger.get_logger().info(
        f"Recorded quiz answer for {key}: {'correct' if is_correct else 'incorrect'}"
    )


def get_chunk_performance(chunk_id: str, username: Optional[str] = None) -> Dict[str, Any]:
    """
    Get performance statistics for a specific chunk.
    
    Args:
        chunk_id: Identifier for the chunk
        username: Username for user-specific state (optional)
        
    Returns:
        Dictionary with performance metrics
    """
    state = load_state(username)
    
    # Handle case where chunk_performance might not exist (for old state files)
    if not hasattr(state, 'chunk_performance') or state.chunk_performance is None:
        return {
            "correct": 0,
            "incorrect": 0,
            "attempts": 0,
            "accuracy": 0.0,
            "source_reference": ""
        }
    
    perf = state.chunk_performance.get(chunk_id, {})
    attempts = perf.get("attempts", 0)
    correct = perf.get("correct", 0)
    
    accuracy = (correct / attempts * 100) if attempts > 0 else 0.0
    
    return {
        "correct": correct,
        "incorrect": perf.get("incorrect", 0),
        "attempts": attempts,
        "accuracy": accuracy,
        "source_reference": perf.get("source_reference", ""),
        "last_attempt": perf.get("last_attempt")
    }


def get_all_chunk_performance(username: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get performance statistics for all chunks.
    
    Args:
        username: Username for user-specific state (optional)
    
    Returns:
        Dictionary mapping chunk_id to performance metrics
    """
    state = load_state(username)
    
    # Handle case where chunk_performance might not exist (for old state files)
    if not hasattr(state, 'chunk_performance') or state.chunk_performance is None:
        return {}
    
    result = {}
    for chunk_id, perf in state.chunk_performance.items():
        attempts = perf.get("attempts", 0)
        correct = perf.get("correct", 0)
        accuracy = (correct / attempts * 100) if attempts > 0 else 0.0
        
        result[chunk_id] = {
            "correct": correct,
            "incorrect": perf.get("incorrect", 0),
            "attempts": attempts,
            "accuracy": accuracy,
            "source_reference": perf.get("source_reference", ""),
            "last_attempt": perf.get("last_attempt")
        }
    
    return result


def get_weak_areas(threshold: float = 60.0, min_attempts: int = 2, username: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Identify weak areas based on performance.
    
    Args:
        threshold: Accuracy threshold below which an area is considered weak (default 60%)
        min_attempts: Minimum number of attempts required to be considered (default 2)
        username: Username for user-specific state (optional)
        
    Returns:
        List of weak areas with performance details, sorted by accuracy (worst first)
    """
    all_perf = get_all_chunk_performance(username)
    weak_areas = []
    
    for chunk_id, perf in all_perf.items():
        if perf["attempts"] >= min_attempts and perf["accuracy"] < threshold:
            weak_areas.append({
                "chunk_id": chunk_id,
                "source_reference": perf["source_reference"],
                "accuracy": perf["accuracy"],
                "correct": perf["correct"],
                "incorrect": perf["incorrect"],
                "attempts": perf["attempts"],
                "last_attempt": perf["last_attempt"]
            })
    
    # Sort by accuracy (worst first)
    weak_areas.sort(key=lambda x: x["accuracy"])
    
    return weak_areas


def get_strong_areas(threshold: float = 80.0, min_attempts: int = 2, username: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Identify strong areas based on performance.
    
    Args:
        threshold: Accuracy threshold above which an area is considered strong (default 80%)
        min_attempts: Minimum number of attempts required to be considered (default 2)
        username: Username for user-specific state (optional)
        
    Returns:
        List of strong areas with performance details, sorted by accuracy (best first)
    """
    all_perf = get_all_chunk_performance(username)
    strong_areas = []
    
    for chunk_id, perf in all_perf.items():
        if perf["attempts"] >= min_attempts and perf["accuracy"] >= threshold:
            strong_areas.append({
                "chunk_id": chunk_id,
                "source_reference": perf["source_reference"],
                "accuracy": perf["accuracy"],
                "correct": perf["correct"],
                "incorrect": perf["incorrect"],
                "attempts": perf["attempts"],
                "last_attempt": perf["last_attempt"]
            })
    
    # Sort by accuracy (best first)
    strong_areas.sort(key=lambda x: x["accuracy"], reverse=True)
    
    return strong_areas


def get_performance_summary(username: Optional[str] = None) -> Dict[str, Any]:
    """
    Get overall performance summary statistics.
    
    Args:
        username: Username for user-specific state (optional)
    
    Returns:
        Dictionary with summary statistics
    """
    all_perf = get_all_chunk_performance(username)
    
    if not all_perf:
        return {
            "total_chunks": 0,
            "total_attempts": 0,
            "total_correct": 0,
            "total_incorrect": 0,
            "overall_accuracy": 0.0,
            "chunks_with_data": 0
        }
    
    total_attempts = sum(p["attempts"] for p in all_perf.values())
    total_correct = sum(p["correct"] for p in all_perf.values())
    total_incorrect = sum(p["incorrect"] for p in all_perf.values())
    chunks_with_data = len([p for p in all_perf.values() if p["attempts"] > 0])
    
    overall_accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0.0
    
    return {
        "total_chunks": len(all_perf),
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "overall_accuracy": overall_accuracy,
        "chunks_with_data": chunks_with_data
    }


def extract_chunk_id_from_reference(source_reference: str, filename: Optional[str] = None) -> str:
    """
    Extract chunk identifier from source reference string.
    Includes filename to make it unique across different files.
    
    Args:
        source_reference: Reference string like "Chunk 1 - Introduction" or "Chunk X - ..."
        filename: Optional filename to make chunk ID unique per file
        
    Returns:
        Chunk identifier (e.g., "file1_chunk_1" or "chunk_1")
    """
    import re
    import hashlib
    
    # Try to extract chunk number
    match = re.search(r'[Cc]hunk\s*(\d+)', source_reference)
    chunk_num = match.group(1) if match else "unknown"
    
    # Create unique identifier using filename if available
    if filename:
        # Create a short hash of filename for uniqueness
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"{file_hash}_chunk_{chunk_num}"
    
    # Fallback: use first 50 chars as identifier
    return source_reference[:50].replace(" ", "_").lower()


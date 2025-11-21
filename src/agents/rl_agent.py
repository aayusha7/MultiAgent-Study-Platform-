"""RL Agent using Thompson Sampling for personalization"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional

from ..core.messages import RLUpdateRequest, RLRecommendation
from ..core.memory import load_state, save_state, RLState
from ..core.logger import logger


class RLAgent:
    """Reinforcement Learning agent using Thompson Sampling"""
    
    def __init__(self, username: Optional[str] = None):
        self.logger = logger.get_logger()
        self.username = username
        self.state: RLState = load_state(username)
        self.modes = ["quiz", "flashcard", "interactive"]
    
    def update_from_feedback(self, request: RLUpdateRequest):
        """
        Update RL state based on user feedback.
        
        Args:
            request: RLUpdateRequest with mode and feedback (0.0-1.0)
        """
        mode = request.mode.lower()
        
        if mode not in self.modes:
            self.logger.warning(f"Unknown mode: {mode}")
            return
        
        # Ensure mode exists in state
        if mode not in self.state.mode_alpha:
            self.state.mode_alpha[mode] = 1.0
        if mode not in self.state.mode_beta:
            self.state.mode_beta[mode] = 1.0
        
        # Update Thompson Sampling parameters
        # Alpha = successes, Beta = failures
        # Feedback: 1.0 = success, 0.0 = failure
        # For partial feedback (0.0-1.0), we can interpret as probability
        
        feedback = max(0.0, min(1.0, request.feedback))
        
        # Update alpha (successes) and beta (failures)
        # If feedback > 0.5, treat as success; otherwise failure
        if feedback > 0.5:
            self.state.mode_alpha[mode] += feedback
        else:
            self.state.mode_beta[mode] += (1.0 - feedback)
        
        # Record in history
        self.state.mode_history.append({
            "mode": mode,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "session_id": request.session_id
        })
        
        # Update metadata
        self.state.last_updated = datetime.now().isoformat()
        
        # Save state
        save_state(self.state, self.username)
        
        self.logger.info(
            f"Updated RL state for {mode}: "
            f"alpha={self.state.mode_alpha[mode]:.2f}, "
            f"beta={self.state.mode_beta[mode]:.2f}"
        )
    
    def recommend_mode(self) -> RLRecommendation:
        """
        Recommend best learning mode using Thompson Sampling.
        
        Returns:
            RLRecommendation with recommended mode and probabilities
        """
        # Reload state to get latest feedback
        self.state = load_state(self.username)
        
        # Sample from Beta distribution for each mode
        samples = {}
        probabilities = {}
        
        for mode in self.modes:
            # Ensure mode exists
            alpha = self.state.mode_alpha.get(mode, 1.0)
            beta = self.state.mode_beta.get(mode, 1.0)
            
            # Sample from Beta distribution
            sample = np.random.beta(alpha, beta)
            samples[mode] = sample
            
            # Calculate probability (normalized)
            # This represents the expected success rate
            if alpha + beta > 0:
                prob = alpha / (alpha + beta)
            else:
                prob = 0.5
            probabilities[mode] = prob
        
        # Recommend mode with highest sample (Thompson Sampling)
        recommended_mode = max(samples, key=samples.get)
        
        # Calculate confidence (difference between top two)
        sorted_samples = sorted(samples.values(), reverse=True)
        if len(sorted_samples) >= 2:
            confidence = sorted_samples[0] - sorted_samples[1]
        else:
            confidence = 1.0
        
        # Normalize confidence to 0-1
        confidence = min(1.0, max(0.0, confidence))
        
        self.logger.info(
            f"RL recommendation: {recommended_mode} "
            f"(confidence: {confidence:.2f}, probabilities: {probabilities})"
        )
        
        return RLRecommendation(
            recommended_mode=recommended_mode,
            probabilities=probabilities,
            confidence=confidence
        )
    
    def get_mode_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each mode.
        
        Returns:
            Dictionary mapping mode to statistics
        """
        stats = {}
        
        for mode in self.modes:
            alpha = self.state.mode_alpha.get(mode, 1.0)
            beta = self.state.mode_beta.get(mode, 1.0)
            
            total = alpha + beta
            success_rate = alpha / total if total > 0 else 0.5
            
            stats[mode] = {
                "success_rate": success_rate,
                "alpha": alpha,
                "beta": beta,
                "total_feedback": total - 2.0  # Subtract initial priors
            }
        
        return stats


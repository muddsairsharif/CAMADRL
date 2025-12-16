"""
Base Agent implementation for CAMADRL framework.

Provides the abstract base class that all agents should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the CAMADRL framework.
    
    This class defines the common interface that all agent implementations
    must follow, ensuring consistency across different agent types.
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary for agent parameters
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Training statistics
        self.total_steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        
    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            explore: Whether to use exploration strategy
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's policy based on a batch of experiences.
        
        Args:
            batch: Dictionary containing batch data (states, actions, rewards, etc.)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the agent's model and parameters.
        
        Args:
            filepath: Path to save the agent
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load the agent's model and parameters.
        
        Args:
            filepath: Path to load the agent from
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.episodes += 1
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information and statistics.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            "agent_id": self.agent_id,
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "total_reward": self.total_reward,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

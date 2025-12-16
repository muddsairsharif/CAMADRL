"""
Base agent class for CAMADRL framework.

This module provides the abstract base class for all agents in the framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """
    Abstract base class for all agents in CAMADRL framework.
    
    This class defines the interface that all agents must implement,
    including methods for action selection, learning, and state management.
    
    Attributes:
        agent_id: Unique identifier for the agent
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        device: PyTorch device (cpu or cuda)
        epsilon: Exploration rate for epsilon-greedy policies
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu"
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary with hyperparameters
            device: Device to run computations on ('cpu' or 'cuda')
        
        Raises:
            ValueError: If state_dim or action_dim are invalid
        """
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive integers")
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        self.device = torch.device(device)
        
        # Default hyperparameters
        self.epsilon = self.config.get("epsilon", 0.1)
        self.gamma = self.config.get("gamma", 0.99)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        
        # Training state
        self.training = True
        self.total_steps = 0
        
    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            explore: Whether to use exploration
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Update agent's policy based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def set_training_mode(self, mode: bool = True) -> None:
        """
        Set the agent to training or evaluation mode.
        
        Args:
            mode: True for training, False for evaluation
        """
        self.training = mode
        
    def save(self, path: str) -> None:
        """
        Save agent's parameters to file.
        
        Args:
            path: Path to save the agent
        """
        raise NotImplementedError("save() must be implemented by subclasses")
    
    def load(self, path: str) -> None:
        """
        Load agent's parameters from file.
        
        Args:
            path: Path to load the agent from
        """
        raise NotImplementedError("load() must be implemented by subclasses")
    
    def reset(self) -> None:
        """Reset agent's internal state for a new episode."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information and statistics.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            "agent_id": self.agent_id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "training": self.training,
        }

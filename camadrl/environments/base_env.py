"""
Base Environment implementation for CAMADRL framework.

Provides the abstract base class that all environments should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym


class BaseEnv(gym.Env, ABC):
    """
    Abstract base class for all environments in the CAMADRL framework.
    
    This class defines the common interface that all environment implementations
    must follow, ensuring consistency across different environment types.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the base environment.
        
        Args:
            config: Configuration dictionary for environment parameters
        """
        super().__init__()
        self.config = config or {}
        
        self.num_agents = self.config.get("num_agents", 1)
        self.max_steps = self.config.get("max_steps", 1000)
        
        self.current_step = 0
        self.episode_count = 0
        
        # These should be set by subclasses
        self.observation_space = None
        self.action_space = None
        
    @abstractmethod
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (initial observation, info dictionary)
        """
        self.current_step = 0
        self.episode_count += 1
        return None, {}
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        return None, 0.0, terminated, truncated, {}
    
    @abstractmethod
    def render(self) -> Any:
        """
        Render the environment.
        
        Returns:
            Rendered output (implementation-dependent)
        """
        pass
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary containing environment state
        """
        return {
            "current_step": self.current_step,
            "episode_count": self.episode_count,
            "num_agents": self.num_agents,
        }
    
    def seed(self, seed: int = None) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

"""
Base environment class for CAMADRL framework.

This module provides the abstract base class for all environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym


class BaseEnv(ABC, gym.Env):
    """
    Abstract base class for all environments in CAMADRL framework.
    
    This class extends gymnasium.Env and defines the interface for
    multi-agent environments with context awareness.
    
    Attributes:
        num_agents: Number of agents in the environment
        state_dim: Dimension of the state space per agent
        action_dim: Dimension of the action space per agent
        max_steps: Maximum steps per episode
    """
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base environment.
        
        Args:
            num_agents: Number of agents in the environment
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            
        Raises:
            ValueError: If num_agents, state_dim, or action_dim are invalid
        """
        super().__init__()
        
        if num_agents <= 0 or state_dim <= 0 or action_dim <= 0:
            raise ValueError("num_agents, state_dim, and action_dim must be positive")
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Episode management
        self.max_steps = self.config.get("max_steps", 1000)
        self.current_step = 0
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(action_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (list of initial states, info dict)
        """
        pass
    
    @abstractmethod
    def step(
        self,
        actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            Tuple of (states, rewards, dones, info)
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            Rendered frame if mode is 'rgb_array', None otherwise
        """
        pass
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def get_state(self, agent_id: int) -> np.ndarray:
        """
        Get state observation for specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            State observation
        """
        raise NotImplementedError("get_state() must be implemented by subclasses")
    
    def get_global_state(self) -> np.ndarray:
        """
        Get global state of the environment.
        
        Returns:
            Global state array
        """
        raise NotImplementedError("get_global_state() must be implemented by subclasses")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get environment information.
        
        Returns:
            Dictionary with environment info
        """
        return {
            "num_agents": self.num_agents,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }

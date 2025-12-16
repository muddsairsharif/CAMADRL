"""
Custom Environment for specialized EV charging scenarios.

A flexible environment that can be customized for specific
EV charging coordination research scenarios.
"""

from typing import Any, Dict, Tuple, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from camadrl.environments.base_env import BaseEnv


class CustomEnv(BaseEnv):
    """
    Custom environment for specialized EV charging scenarios.
    
    Provides a flexible framework that can be extended for
    specific research requirements and use cases.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Custom Environment.
        
        Args:
            config: Configuration dictionary with keys:
                - state_dim: Dimension of state space (default: 10)
                - action_dim: Dimension of action space (default: 4)
                - num_agents: Number of agents (default: 3)
                - reward_fn: Custom reward function (optional)
                - observation_fn: Custom observation function (optional)
                - max_steps: Maximum steps per episode (default: 100)
        """
        super().__init__(config)
        
        self.state_dim = self.config.get("state_dim", 10)
        self.action_dim = self.config.get("action_dim", 4)
        self.num_agents = self.config.get("num_agents", 3)
        
        # Custom functions
        self.reward_fn = self.config.get("reward_fn", self._default_reward)
        self.observation_fn = self.config.get("observation_fn", self._default_observation)
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Environment state
        self.state = None
        self.agent_states = None
        
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state
        self.state = np.random.randn(self.state_dim)
        self.agent_states = [
            np.random.randn(self.state_dim) for _ in range(self.num_agents)
        ]
        
        obs = self.observation_fn(self.state, self.agent_states, 0)
        info = self.get_state()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        agent_idx = self.current_step % self.num_agents
        
        # Update state based on action
        self._update_state(action, agent_idx)
        
        # Compute reward
        reward = self.reward_fn(self.state, self.agent_states, action, agent_idx)
        
        # Check termination
        terminated = self._check_termination()
        
        super().step(action)
        truncated = self.current_step >= self.max_steps
        
        obs = self.observation_fn(self.state, self.agent_states, agent_idx)
        info = self.get_state()
        info["agent_idx"] = agent_idx
        
        return obs, reward, terminated, truncated, info
    
    def _update_state(self, action: int, agent_idx: int) -> None:
        """Update environment state based on action."""
        # Simple state transition
        action_effect = np.zeros(self.state_dim)
        action_effect[action % self.state_dim] = 1.0
        
        self.agent_states[agent_idx] += 0.1 * action_effect
        self.state = np.mean(self.agent_states, axis=0)
        
        # Add some noise
        self.state += np.random.randn(self.state_dim) * 0.01
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if state exceeds bounds
        if np.any(np.abs(self.state) > 10):
            return True
        return False
    
    def _default_reward(
        self,
        state: np.ndarray,
        agent_states: list,
        action: int,
        agent_idx: int
    ) -> float:
        """Default reward function."""
        # Reward for keeping state near zero
        reward = -np.sum(state ** 2) * 0.01
        
        # Reward for coordination (similar states across agents)
        state_variance = np.var(agent_states, axis=0).mean()
        reward -= state_variance * 0.1
        
        return reward
    
    def _default_observation(
        self,
        state: np.ndarray,
        agent_states: list,
        agent_idx: int
    ) -> np.ndarray:
        """Default observation function."""
        # Return global state
        return state.astype(np.float32)
    
    def render(self) -> Dict[str, Any]:
        """Render the environment state."""
        return {
            "state": self.state.tolist() if self.state is not None else None,
            "agent_states": [s.tolist() for s in self.agent_states] if self.agent_states else None,
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        state = super().get_state()
        state.update({
            "state": self.state.tolist() if self.state is not None else None,
            "num_agents": self.num_agents,
        })
        return state
    
    def set_reward_function(self, reward_fn: Callable) -> None:
        """
        Set a custom reward function.
        
        Args:
            reward_fn: Function with signature (state, agent_states, action, agent_idx) -> reward
        """
        self.reward_fn = reward_fn
    
    def set_observation_function(self, observation_fn: Callable) -> None:
        """
        Set a custom observation function.
        
        Args:
            observation_fn: Function with signature (state, agent_states, agent_idx) -> observation
        """
        self.observation_fn = observation_fn

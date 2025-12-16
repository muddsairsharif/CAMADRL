"""
Custom environment template for CAMADRL framework.

This module provides a template for creating custom multi-agent environments.
Users can extend this class to implement their own environments.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from camadrl.environments.base_env import BaseEnv


class CustomEnv(BaseEnv):
    """
    Custom environment template.
    
    This class serves as a template for creating custom multi-agent
    reinforcement learning environments. Extend this class and implement
    the required methods for your specific use case.
    
    Attributes:
        Custom attributes specific to your environment
    """
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize custom environment.
        
        Args:
            num_agents: Number of agents
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        super().__init__(num_agents, state_dim, action_dim, config)
        
        # Initialize custom environment parameters
        self.custom_param = self.config.get("custom_param", 1.0)
        
        # Initialize state variables
        self.agent_states = [None] * num_agents
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Override this method to implement your custom reset logic.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (list of initial states, info dict)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # TODO: Implement your custom reset logic here
        # Initialize agent states
        states = []
        for i in range(self.num_agents):
            # Create initial state for each agent
            state = np.random.randn(self.state_dim).astype(np.float32)
            states.append(state)
            self.agent_states[i] = state
        
        info = self.get_info()
        
        return states, info
    
    def step(
        self,
        actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Override this method to implement your custom step logic.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            Tuple of (states, rewards, dones, info)
        """
        self.current_step += 1
        
        # TODO: Implement your custom step logic here
        # Process actions and update environment
        states = []
        rewards = []
        dones = []
        
        for i in range(self.num_agents):
            action = actions[i]
            
            # Update state based on action
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            self.agent_states[i] = next_state
            states.append(next_state)
            
            # Calculate reward
            reward = 0.0  # Implement your reward function
            rewards.append(reward)
            
            # Check if done
            done = self.current_step >= self.max_steps
            dones.append(done)
        
        info = self.get_info()
        
        return states, rewards, dones, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Override this method to implement custom visualization.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            Rendered frame if mode is 'rgb_array', None otherwise
        """
        # TODO: Implement your custom rendering logic
        if mode == "human":
            print(f"Step: {self.current_step}")
            for i in range(self.num_agents):
                print(f"  Agent {i}: {self.agent_states[i][:3]}...")  # Show first 3 values
        
        return None
    
    def get_state(self, agent_id: int) -> np.ndarray:
        """
        Get state observation for specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            State observation
        """
        if 0 <= agent_id < self.num_agents:
            return self.agent_states[agent_id]
        else:
            raise ValueError(f"Invalid agent_id: {agent_id}")
    
    def get_global_state(self) -> np.ndarray:
        """
        Get global state of the environment.
        
        Returns:
            Global state array
        """
        # Concatenate all agent states
        return np.concatenate(self.agent_states)
    
    def _custom_helper_method(self) -> Any:
        """
        Add custom helper methods as needed.
        
        Returns:
            Custom return value
        """
        # TODO: Implement custom helper methods
        pass

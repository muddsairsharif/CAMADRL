"""
Replay Buffer implementation for experience replay.

Implements efficient storage and sampling of agent experiences
for off-policy reinforcement learning algorithms.
"""

from typing import Dict, Tuple
import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    
    Implements a circular buffer for efficient memory management
    in off-policy RL algorithms like DQN.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        seed: int = None
    ):
        """
        Initialize Replay Buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize storage
        self.observations = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
        if seed is not None:
            np.random.seed(seed)
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: float
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Whether episode terminated
        """
        # Ensure correct shapes
        if np.isscalar(action):
            action = np.array([action])
        
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_observation
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary containing batch of transitions
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices]
        }
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self.size
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions based on their TD error for more efficient learning.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        seed: int = None
    ):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            alpha: Prioritization exponent
            beta: Importance sampling weight exponent
            seed: Random seed
        """
        super().__init__(capacity, state_dim, action_dim, seed)
        
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: float
    ) -> None:
        """Add transition with maximum priority."""
        super().add(observation, action, reward, next_observation, done)
        self.priorities[self.ptr - 1] = self.max_priority
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (batch, weights, indices)
        """
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        batch = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices]
        }
        
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions
            priorities: New priority values (TD errors)
        """
        self.priorities[indices] = priorities + 1e-6  # Add small constant to avoid zero priority
        self.max_priority = max(self.max_priority, priorities.max())


class MultiAgentReplayBuffer:
    """
    Replay Buffer for multi-agent scenarios.
    
    Stores transitions for multiple agents with support for
    centralized training paradigms.
    """
    
    def __init__(
        self,
        capacity: int,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        seed: int = None
    ):
        """
        Initialize Multi-Agent Replay Buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            num_agents: Number of agents
            state_dim: Dimension of state space per agent
            action_dim: Dimension of action space per agent
            seed: Random seed
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize storage for all agents
        self.observations = np.zeros((capacity, num_agents, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, num_agents, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
        if seed is not None:
            np.random.seed(seed)
    
    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        done: float
    ) -> None:
        """
        Add multi-agent transition.
        
        Args:
            observations: Observations for all agents
            actions: Actions for all agents
            rewards: Rewards for all agents
            next_observations: Next observations for all agents
            done: Whether episode terminated
        """
        self.observations[self.ptr] = observations
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_observations[self.ptr] = next_observations
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of multi-agent transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices]
        }
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self.size

"""
Experience replay buffer for reinforcement learning.

This module implements various replay buffer types for storing and
sampling experiences during training.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import random
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for DQN-style algorithms.
    
    Stores transitions (state, action, reward, next_state, done) and
    provides random sampling for training.
    
    Attributes:
        capacity: Maximum buffer capacity
        state_dim: Dimension of state space
        buffer: Deque storing experiences
        device: PyTorch device for tensors
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: str = "cpu"
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            state_dim: Dimension of state space
            device: Device for PyTorch tensors
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = torch.device(device)
        
        # Use deque for efficient FIFO operations
        self.buffer = deque(maxlen=capacity)
        
        # Statistics
        self.total_added = 0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.total_added += 1
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
            
        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Batch size ({batch_size}) cannot exceed buffer size ({len(self.buffer)})")
        
        # Random sampling
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples.
        
        Args:
            min_size: Minimum required size
            
        Returns:
            True if buffer size >= min_size
        """
        return len(self.buffer) >= min_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Samples experiences based on their TD-error for more efficient learning.
    Implements prioritized experience replay as in Rainbow DQN.
    
    Attributes:
        priorities: Priority values for each experience
        alpha: Priority exponent
        beta: Importance sampling exponent
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        device: str = "cpu"
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            state_dim: Dimension of state space
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            device: Device for PyTorch tensors
        """
        super().__init__(capacity, state_dim, device)
        
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Add experience with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            priority: Priority value (uses max if None)
        """
        super().add(state, action, reward, next_state, done)
        
        if priority is None:
            priority = self.max_priority
        
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample batch with prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Batch size ({batch_size}) cannot exceed buffer size ({len(self.buffer)})")
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, weights_tensor, indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)


class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent scenarios.
    
    Stores joint experiences from multiple agents.
    """
    
    def __init__(
        self,
        capacity: int,
        num_agents: int,
        state_dim: int,
        device: str = "cpu"
    ):
        """
        Initialize multi-agent replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            num_agents: Number of agents
            state_dim: Dimension of state space per agent
            device: Device for PyTorch tensors
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.device = torch.device(device)
        
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool]
    ) -> None:
        """
        Add joint experience from all agents.
        
        Args:
            states: List of states for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_states: List of next states for each agent
            dones: List of done flags for each agent
        """
        experience = (states, actions, rewards, next_states, dones)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of joint experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary containing batched tensors for all agents
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Batch size ({batch_size}) cannot exceed buffer size ({len(self.buffer)})")
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip and convert to tensors
        all_states, all_actions, all_rewards, all_next_states, all_dones = zip(*batch)
        
        # Transpose to get per-agent batches
        result = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        
        for agent_id in range(self.num_agents):
            agent_states = [s[agent_id] for s in all_states]
            agent_actions = [a[agent_id] for a in all_actions]
            agent_rewards = [r[agent_id] for r in all_rewards]
            agent_next_states = [ns[agent_id] for ns in all_next_states]
            agent_dones = [d[agent_id] for d in all_dones]
            
            result["states"].append(torch.FloatTensor(agent_states).to(self.device))
            result["actions"].append(torch.LongTensor(agent_actions).to(self.device))
            result["rewards"].append(torch.FloatTensor(agent_rewards).to(self.device))
            result["next_states"].append(torch.FloatTensor(agent_next_states).to(self.device))
            result["dones"].append(torch.FloatTensor(agent_dones).to(self.device))
        
        return result
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

"""
Deep Q-Network (DQN) Agent implementation.

This module provides a DQN agent with experience replay and target networks.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from camadrl.agents.base_agent import BaseAgent
from camadrl.networks.dqn_network import DQNNetwork
from camadrl.utils.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent.
    
    Implements DQN algorithm with experience replay and target network
    for stable training.
    
    Attributes:
        q_network: Main Q-network
        target_network: Target Q-network for stable targets
        replay_buffer: Experience replay buffer
        update_counter: Counter for target network updates
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
        Initialize DQN agent.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary
            device: Device to run computations on
        """
        super().__init__(agent_id, state_dim, action_dim, config, device)
        
        # Hyperparameters
        self.batch_size = self.config.get("batch_size", 64)
        self.buffer_size = self.config.get("buffer_size", 100000)
        self.target_update_freq = self.config.get("target_update_freq", 100)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        
        # Networks
        hidden_dim = self.config.get("hidden_dim", 256)
        self.q_network = DQNNetwork(
            state_dim,
            action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            state_dim,
            action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            state_dim,
            device=device
        )
        
        # Training state
        self.update_counter = 0
        
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            explore: Whether to use exploration
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if explore and self.training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Update Q-network using experience replay.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        # Store experience in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
        
        # Skip update if not enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze()
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_mean": current_q_values.mean().item(),
            "buffer_size": len(self.replay_buffer),
        }
    
    def save(self, path: str) -> None:
        """Save agent parameters."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_counter": self.update_counter,
            "total_steps": self.total_steps,
            "config": self.config,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.update_counter = checkpoint.get("update_counter", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
    
    def reset(self) -> None:
        """Reset agent for new episode."""
        pass

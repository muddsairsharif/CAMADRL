"""
Policy Gradient Agent implementation.

This module implements a policy gradient agent using REINFORCE algorithm
with baseline for variance reduction.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from camadrl.agents.base_agent import BaseAgent
from camadrl.networks.actor_critic_network import ActorCriticNetwork


class PolicyGradientAgent(BaseAgent):
    """
    Policy Gradient Agent using REINFORCE algorithm.
    
    Implements the REINFORCE algorithm with a value function baseline
    for variance reduction.
    
    Attributes:
        policy_network: Neural network for policy and value estimation
        trajectory: Buffer for storing episode trajectory
        entropy_coef: Coefficient for entropy regularization
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
        Initialize Policy Gradient agent.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary
            device: Device to run computations on
        """
        super().__init__(agent_id, state_dim, action_dim, config, device)
        
        # Hyperparameters
        self.entropy_coef = self.config.get("entropy_coef", 0.01)
        self.value_coef = self.config.get("value_coef", 0.5)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        
        # Network
        hidden_dim = self.config.get("hidden_dim", 256)
        self.policy_network = ActorCriticNetwork(
            state_dim,
            action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )
        
        # Trajectory storage
        self.trajectory = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> int:
        """
        Select action by sampling from policy distribution.
        
        Args:
            state: Current state observation
            explore: Whether to use exploration (not used in policy gradient)
            
        Returns:
            Sampled action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities and value
        action_probs, value = self.policy_network(state_tensor)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Store log probability, value, and entropy for training
        if self.training:
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.entropies.append(entropy)
        
        return action.item()
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Store reward and update policy at episode end.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        # Store reward
        self.rewards.append(reward)
        self.total_steps += 1
        
        # Update at episode end
        if done:
            return self._update_policy()
        
        return {}
    
    def _update_policy(self) -> Dict[str, float]:
        """
        Update policy using REINFORCE with baseline.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.rewards) == 0:
            return {}
        
        # Calculate discounted returns
        returns = []
        discounted_return = 0
        for reward in reversed(self.rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Stack stored values
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()
        entropies = torch.stack(self.entropies)
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Calculate losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=self.max_grad_norm
        )
        self.optimizer.step()
        
        # Calculate metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "mean_return": returns.mean().item(),
            "mean_value": values.mean().item(),
        }
        
        # Clear buffers
        self.trajectory.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save agent parameters."""
        torch.save({
            "policy_network": self.policy_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
    
    def reset(self) -> None:
        """Reset agent for new episode."""
        self.trajectory.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()

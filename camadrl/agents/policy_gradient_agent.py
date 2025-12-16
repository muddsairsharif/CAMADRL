"""
Policy Gradient Agent implementation.

Implements policy gradient methods including REINFORCE and Actor-Critic
for continuous and discrete action spaces in EV charging coordination.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from camadrl.agents.base_agent import BaseAgent


class PolicyGradientAgent(BaseAgent):
    """
    Policy Gradient Agent using Actor-Critic architecture.
    
    Supports both discrete and continuous action spaces for
    flexible EV charging control strategies.
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any] = None
    ):
        """
        Initialize Policy Gradient agent.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary
        """
        super().__init__(agent_id, state_dim, action_dim, config)
        
        # Hyperparameters
        self.actor_lr = self.config.get("actor_lr", 0.001)
        self.critic_lr = self.config.get("critic_lr", 0.001)
        self.gamma = self.config.get("gamma", 0.99)
        self.entropy_coef = self.config.get("entropy_coef", 0.01)
        
        # Build actor and critic networks
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
    def _build_actor(self) -> nn.Module:
        """Build the actor (policy) network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic(self) -> nn.Module:
        """Build the critic (value) network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action based on the current policy.
        
        Args:
            state: Current state observation
            explore: Whether to sample from policy (True) or use deterministic action (False)
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        
        if explore:
            # Sample from policy
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store for training
            self.episode_states.append(state)
            self.episode_actions.append(action.item())
            self.episode_log_probs.append(log_prob.item())
            
            return action.item()
        else:
            # Deterministic action
            return action_probs.argmax(dim=1).item()
    
    def store_reward(self, reward: float) -> None:
        """Store reward for the current step."""
        self.episode_rewards.append(reward)
    
    def update(self, batch: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """
        Update the agent using episode data or batch data.
        
        Args:
            batch: Optional dictionary containing batch data
            
        Returns:
            Dictionary of training metrics
        """
        if batch is not None:
            return self._update_from_batch(batch)
        else:
            return self._update_from_episode()
    
    def _update_from_episode(self) -> Dict[str, float]:
        """Update using episode data (REINFORCE with baseline)."""
        if len(self.episode_rewards) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        log_probs = torch.FloatTensor(self.episode_log_probs).to(self.device)
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute state values
        state_values = self.critic(states).squeeze()
        
        # Compute advantages
        advantages = returns - state_values.detach()
        
        # Update critic
        critic_loss = nn.MSELoss()(state_values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        actor_loss = -(new_log_probs * advantages).mean() - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.total_steps += len(self.episode_rewards)
        self.total_reward += sum(self.episode_rewards)
        
        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "episode_reward": sum(self.episode_rewards)
        }
        
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        return metrics
    
    def _update_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update using batch data (for compatibility with replay buffer)."""
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Compute state values
        state_values = self.critic(states).squeeze()
        next_state_values = self.critic(next_states).squeeze()
        
        # Compute TD targets
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = td_targets - state_values.detach()
        
        # Update critic
        critic_loss = nn.MSELoss()(state_values, td_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.total_steps += len(states)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }
    
    def save(self, filepath: str) -> None:
        """Save agent model and parameters."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "agent_info": self.get_info(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent model and parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        
        # Restore agent info
        info = checkpoint.get("agent_info", {})
        self.total_steps = info.get("total_steps", 0)
        self.episodes = info.get("episodes", 0)
        self.total_reward = info.get("total_reward", 0.0)

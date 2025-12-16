"""
Deep Q-Network (DQN) Agent implementation.

Implements a DQN agent with experience replay and target network
for stable learning in EV charging coordination.
"""

from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from camadrl.agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent with experience replay and target network.
    
    Uses double Q-learning to reduce overestimation bias and
    provides stable learning for continuous control tasks.
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary
        """
        super().__init__(agent_id, state_dim, action_dim, config)
        
        # Hyperparameters
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.gamma = self.config.get("gamma", 0.99)
        self.epsilon = self.config.get("epsilon", 1.0)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        self.target_update_freq = self.config.get("target_update_freq", 100)
        self.tau = self.config.get("tau", 0.001)  # Soft update parameter
        
        # Build Q-networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_counter = 0
        
    def _build_network(self) -> nn.Module:
        """Build the Q-network architecture."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            Selected action
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the DQN using a batch of experiences.
        
        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._soft_update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.total_steps += len(states)
        
        return {
            "loss": loss.item(),
            "q_value": current_q_values.mean().item(),
            "epsilon": self.epsilon
        }
    
    def _soft_update_target_network(self) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath: str) -> None:
        """Save agent model and parameters."""
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_counter": self.update_counter,
            "agent_info": self.get_info(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent model and parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.update_counter = checkpoint.get("update_counter", 0)
        
        # Restore agent info
        info = checkpoint.get("agent_info", {})
        self.total_steps = info.get("total_steps", 0)
        self.episodes = info.get("episodes", 0)
        self.total_reward = info.get("total_reward", 0.0)

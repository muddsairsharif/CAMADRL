"""
Context-Aware Deep Reinforcement Learning (CADRL) Agent.

Implements a CADRL agent that uses context information for decision making
in multi-agent EV charging coordination scenarios.
"""

from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from camadrl.agents.base_agent import BaseAgent


class CADRLAgent(BaseAgent):
    """
    Context-Aware Deep Reinforcement Learning Agent.
    
    This agent uses context information (e.g., grid state, other agents' states,
    time of day) to make informed decisions in EV charging coordination.
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        context_dim: int = 10,
        config: Dict[str, Any] = None
    ):
        """
        Initialize CADRL agent.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            context_dim: Dimension of the context information
            config: Configuration dictionary
        """
        super().__init__(agent_id, state_dim, action_dim, config)
        
        self.context_dim = context_dim
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.gamma = self.config.get("gamma", 0.99)
        self.epsilon = self.config.get("epsilon", 0.1)
        
        # Build context-aware network
        self.network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
    def _build_network(self) -> nn.Module:
        """Build the context-aware neural network."""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def select_action(
        self,
        state: np.ndarray,
        context: np.ndarray = None,
        explore: bool = True
    ) -> np.ndarray:
        """
        Select action based on state and context.
        
        Args:
            state: Current state observation
            context: Context information (grid state, time, etc.)
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            Selected action
        """
        if context is None:
            context = np.zeros(self.context_dim)
        
        # Epsilon-greedy exploration
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Concatenate state and context
        state_context = np.concatenate([state, context])
        state_context_tensor = torch.FloatTensor(state_context).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.network(state_context_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent using a batch of experiences.
        
        Args:
            batch: Dictionary containing states, actions, rewards, next_states, contexts, dones
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        contexts = batch.get("contexts", torch.zeros(states.shape[0], self.context_dim)).to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Concatenate states with contexts
        state_contexts = torch.cat([states, contexts], dim=1)
        next_state_contexts = torch.cat([next_states, contexts], dim=1)
        
        # Compute current Q values
        current_q_values = self.network(state_contexts).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.network(next_state_contexts).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_steps += len(states)
        
        return {
            "loss": loss.item(),
            "q_value": current_q_values.mean().item()
        }
    
    def save(self, filepath: str) -> None:
        """Save agent model and parameters."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "agent_info": self.get_info(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent model and parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore agent info
        info = checkpoint.get("agent_info", {})
        self.total_steps = info.get("total_steps", 0)
        self.episodes = info.get("episodes", 0)
        self.total_reward = info.get("total_reward", 0.0)

"""
Context-Aware Deep Reinforcement Learning (CADRL) Agent.

This module implements the CADRL agent with context awareness capabilities
for multi-agent coordination.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from camadrl.agents.base_agent import BaseAgent
from camadrl.networks.actor_critic_network import ActorCriticNetwork
from camadrl.networks.communication_network import CommunicationNetwork


class CADRLAgent(BaseAgent):
    """
    Context-Aware Deep Reinforcement Learning Agent.
    
    This agent uses context information from other agents and the environment
    to make informed decisions in multi-agent scenarios.
    
    Attributes:
        actor_critic: Actor-critic network for policy and value estimation
        communication_net: Network for processing context information
        context_dim: Dimension of context information
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        context_dim: int = 64,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu"
    ):
        """
        Initialize CADRL agent.
        
        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            context_dim: Dimension of context embedding
            config: Configuration dictionary
            device: Device to run computations on
        """
        super().__init__(agent_id, state_dim, action_dim, config, device)
        
        self.context_dim = context_dim
        
        # Network architectures
        hidden_dim = self.config.get("hidden_dim", 256)
        self.actor_critic = ActorCriticNetwork(
            state_dim + context_dim,
            action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.communication_net = CommunicationNetwork(
            state_dim,
            context_dim,
            hidden_dim=hidden_dim // 2
        ).to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.actor_critic.parameters()) + 
            list(self.communication_net.parameters()),
            lr=self.learning_rate
        )
        
        # Experience buffer
        self.trajectory = []
        self.context_history = []
        
    def select_action(
        self,
        state: np.ndarray,
        context: Optional[np.ndarray] = None,
        explore: bool = True
    ) -> Tuple[int, np.ndarray]:
        """
        Select action with context awareness.
        
        Args:
            state: Current state observation
            context: Context information from other agents
            explore: Whether to use exploration
            
        Returns:
            Tuple of (action, context_embedding)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Process context
            if context is None:
                context_embedding = torch.zeros(1, self.context_dim).to(self.device)
            else:
                context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
                context_embedding = self.communication_net(context_tensor)
            
            # Combine state and context
            combined_input = torch.cat([state_tensor, context_embedding], dim=1)
            
            # Get action distribution
            action_probs, _ = self.actor_critic(combined_input)
            
            if explore and self.training:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
            else:
                action = torch.argmax(action_probs, dim=1)
            
            return action.item(), context_embedding.cpu().numpy()
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        context: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Update agent using actor-critic algorithm with context.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            context: Context information
            
        Returns:
            Dictionary of training metrics
        """
        # Store experience
        self.trajectory.append((state, action, reward, next_state, done, context))
        self.total_steps += 1
        
        # Update periodically or at episode end
        if done or len(self.trajectory) >= self.config.get("update_frequency", 32):
            return self._update_policy()
        
        return {}
    
    def _update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected trajectory.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.trajectory) == 0:
            return {}
        
        # Prepare batch
        states, actions, rewards, next_states, dones, contexts = zip(*self.trajectory)
        
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Process contexts
        if contexts[0] is not None:
            contexts_tensor = torch.FloatTensor(np.array(contexts)).to(self.device)
            context_embeddings = self.communication_net(contexts_tensor)
        else:
            context_embeddings = torch.zeros(len(states), self.context_dim).to(self.device)
        
        # Combine states and contexts
        combined_states = torch.cat([states_tensor, context_embeddings], dim=1)
        
        # Get predictions
        action_probs, values = self.actor_critic(combined_states)
        
        # Calculate advantages
        with torch.no_grad():
            next_context_embeddings = torch.zeros(len(next_states), self.context_dim).to(self.device)
            combined_next_states = torch.cat([next_states_tensor, next_context_embeddings], dim=1)
            _, next_values = self.actor_critic(combined_next_states)
            
            advantages = rewards_tensor + self.gamma * next_values.squeeze() * (1 - dones_tensor) - values.squeeze()
        
        # Actor loss
        action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)) + 1e-10)
        actor_loss = -(action_log_probs.squeeze() * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = advantages.pow(2).mean()
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor_critic.parameters()) + list(self.communication_net.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Clear trajectory
        self.trajectory.clear()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "total_loss": total_loss.item(),
            "mean_value": values.mean().item(),
        }
    
    def save(self, path: str) -> None:
        """Save agent parameters."""
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "communication_net": self.communication_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.communication_net.load_state_dict(checkpoint["communication_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
    
    def reset(self) -> None:
        """Reset agent for new episode."""
        self.trajectory.clear()
        self.context_history.clear()

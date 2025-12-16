"""
Actor-Critic Network implementation.

Implements actor-critic architectures for policy gradient methods
in multi-agent EV charging coordination.
"""

from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from camadrl.networks.base_network import BaseNetwork


class ActorCriticNetwork(BaseNetwork):
    """
    Actor-Critic Network for policy gradient methods.
    
    Combines actor (policy) and critic (value) networks with shared
    feature extraction for efficient learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: list = None,
        continuous: bool = False,
        config: Dict[str, Any] = None
    ):
        """
        Initialize Actor-Critic Network.
        
        Args:
            input_dim: Dimension of input (state dimension)
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            continuous: Whether action space is continuous
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [256, 256, 128]
        self.continuous = continuous
        
        # Shared feature extraction
        self.shared_features = self._build_shared_layers()
        
        # Actor (policy) head
        if continuous:
            self.actor_mean = nn.Linear(self.hidden_dims[-1], action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], action_dim),
                nn.Softmax(dim=-1)
            )
        
        # Critic (value) head
        self.critic_head = nn.Linear(self.hidden_dims[-1], 1)
        
        self.to(self.device)
    
    def _build_shared_layers(self) -> nn.Module:
        """Build shared feature extraction layers."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (action distribution parameters, state value)
        """
        features = self.shared_features(x)
        
        # Actor output
        if self.continuous:
            action_mean = self.actor_mean(features)
            action_log_std = self.actor_log_std.expand_as(action_mean)
            action_params = (action_mean, action_log_std)
        else:
            action_params = self.actor_head(features)
        
        # Critic output
        value = self.critic_head(features)
        
        return action_params, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            state: Input state tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_params, value = self.forward(state)
        
        if self.continuous:
            action_mean, action_log_std = action_params
            action_std = action_log_std.exp()
            
            if deterministic:
                action = action_mean
                log_prob = None
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            if deterministic:
                action = action_params.argmax(dim=-1)
                log_prob = None
            else:
                dist = Categorical(action_params)
                action = dist.sample()
                log_prob = dist.log_prob(action).unsqueeze(-1)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            state: Input state tensor
            action: Actions to evaluate
            
        Returns:
            Tuple of (log_prob, entropy, value)
        """
        action_params, value = self.forward(state)
        
        if self.continuous:
            action_mean, action_log_std = action_params
            action_std = action_log_std.exp()
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        else:
            dist = Categorical(action_params)
            log_prob = dist.log_prob(action).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
        
        return log_prob, entropy, value


class AttentionActorCritic(BaseNetwork):
    """
    Actor-Critic with attention mechanism for context-aware learning.
    
    Uses multi-head attention to incorporate context from other agents
    and environmental factors.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 256,
        config: Dict[str, Any] = None
    ):
        """
        Initialize Attention Actor-Critic Network.
        
        Args:
            input_dim: Dimension of input (state dimension)
            action_dim: Dimension of action space
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension size
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic_head = nn.Linear(hidden_dim // 2, 1)
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.
        
        Args:
            x: Input state tensor [batch_size, input_dim]
            context: Context tensor [batch_size, num_context, input_dim]
            
        Returns:
            Tuple of (action probabilities, state value)
        """
        # Embed input
        x_embed = self.input_embedding(x)  # [batch_size, hidden_dim]
        
        # Add batch dimension for attention if needed
        if x_embed.dim() == 2:
            x_embed = x_embed.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply attention if context is provided
        if context is not None:
            context_embed = self.input_embedding(context)  # [batch_size, num_context, hidden_dim]
            attended, _ = self.attention(x_embed, context_embed, context_embed)
        else:
            attended = x_embed
        
        # Remove sequence dimension
        if attended.size(1) == 1:
            attended = attended.squeeze(1)
        
        # Process features
        features = self.feature_net(attended)
        
        # Actor and critic outputs
        action_probs = self.actor_head(features)
        value = self.critic_head(features)
        
        return action_probs, value

"""
Actor-Critic network architecture.

This module implements the Actor-Critic neural network architecture
for policy gradient algorithms.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from camadrl.networks.base_network import BaseNetwork


class ActorCriticNetwork(BaseNetwork):
    """
    Actor-Critic network architecture.
    
    Implements a shared feature extraction network with separate
    actor (policy) and critic (value) heads.
    
    Attributes:
        shared_layers: Shared feature extraction layers
        actor_head: Actor network (policy)
        critic_head: Critic network (value function)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            num_layers: Number of shared hidden layers
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, hidden_dim, config)
        
        self.num_layers = num_layers
        self.dropout_rate = self.config.get("dropout_rate", 0.0)
        
        # Shared feature extraction layers
        shared_layers = []
        current_dim = state_dim
        
        for i in range(num_layers):
            shared_layers.append(nn.Linear(current_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            
            if self.dropout_rate > 0:
                shared_layers.append(nn.Dropout(self.dropout_rate))
            
            current_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Actor head (policy network)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.initialize_weights(method="xavier")
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (action_probs, state_value)
            - action_probs: Action probabilities of shape (batch_size, action_dim)
            - state_value: State value of shape (batch_size, 1)
        """
        # Extract shared features
        features = self.shared_layers(state)
        
        # Compute action probabilities
        action_probs = self.actor_head(features)
        
        # Compute state value
        state_value = self.critic_head(features)
        
        return action_probs, state_value
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities from actor network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action probabilities
        """
        features = self.shared_layers(state)
        return self.actor_head(features)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value from critic network.
        
        Args:
            state: Input state tensor
            
        Returns:
            State value
        """
        features = self.shared_layers(state)
        return self.critic_head(features)
    
    def get_features(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extract features from state.
        
        Args:
            state: Input state tensor
            
        Returns:
            Feature tensor
        """
        return self.shared_layers(state)


class RecurrentActorCritic(BaseNetwork):
    """
    Recurrent Actor-Critic network with LSTM/GRU.
    
    Useful for partially observable environments or when temporal
    information is important.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        recurrent_type: str = "lstm",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Recurrent Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            recurrent_type: Type of recurrent cell ('lstm' or 'gru')
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, hidden_dim, config)
        
        self.recurrent_type = recurrent_type
        
        # Input processing
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        
        # Recurrent layer
        if recurrent_type == "lstm":
            self.recurrent_layer = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        elif recurrent_type == "gru":
            self.recurrent_layer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown recurrent_type: {recurrent_type}")
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.initialize_weights(method="xavier")
    
    def forward(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the recurrent network.
        
        Args:
            state: Input state tensor of shape (batch_size, seq_len, state_dim)
            hidden_state: Previous hidden state (for LSTM/GRU)
            
        Returns:
            Tuple of (action_probs, state_value, new_hidden_state)
        """
        # Process input
        x = F.relu(self.input_layer(state))
        
        # Recurrent layer
        if hidden_state is not None:
            x, new_hidden_state = self.recurrent_layer(x, hidden_state)
        else:
            x, new_hidden_state = self.recurrent_layer(x)
        
        # Take last output for actor-critic
        if len(x.shape) == 3:  # (batch, seq, features)
            x = x[:, -1, :]
        
        # Actor and critic outputs
        action_probs = self.actor_head(x)
        state_value = self.critic_head(x)
        
        return action_probs, state_value, new_hidden_state

"""
Deep Q-Network (DQN) architecture.

This module implements the DQN neural network architecture for
Q-value estimation in reinforcement learning.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from camadrl.networks.base_network import BaseNetwork


class DQNNetwork(BaseNetwork):
    """
    Deep Q-Network architecture.
    
    Implements a multi-layer perceptron for Q-value estimation.
    The network takes a state as input and outputs Q-values for all actions.
    
    Attributes:
        fc_layers: Fully connected layers
        output_layer: Final output layer
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of actions)
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, hidden_dim, config)
        
        self.num_layers = num_layers
        self.dropout_rate = self.config.get("dropout_rate", 0.0)
        self.use_dueling = self.config.get("use_dueling", False)
        
        # Build network layers
        layers = []
        current_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            current_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Dueling DQN architecture
        if self.use_dueling:
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        else:
            self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.initialize_weights(method="xavier")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        # Extract features
        features = self.fc_layers(state)
        
        if self.use_dueling:
            # Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Combine value and advantage streams
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            # Standard DQN
            q_values = self.output_layer(features)
        
        return q_values
    
    def get_features(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extract features from state without computing Q-values.
        
        Args:
            state: Input state tensor
            
        Returns:
            Feature tensor
        """
        return self.fc_layers(state)


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for exploration (used in Noisy DQN).
    
    Implements learnable noise for exploration as in Rainbow DQN.
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        """
        Initialize Noisy Linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            sigma_init: Initial value for noise standard deviation
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Register noise buffers
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self) -> None:
        """Initialize network parameters."""
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features ** 0.5)
    
    def reset_noise(self) -> None:
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

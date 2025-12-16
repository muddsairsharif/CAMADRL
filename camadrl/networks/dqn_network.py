"""
DQN Network implementation.

Implements Deep Q-Network architectures with various enhancements
like dueling architecture and noisy layers.
"""

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from camadrl.networks.base_network import BaseNetwork


class DQNNetwork(BaseNetwork):
    """
    Deep Q-Network with optional dueling architecture.
    
    Implements a flexible DQN that can use standard or dueling architecture
    for improved value estimation in EV charging scenarios.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = None,
        dueling: bool = False,
        config: Dict[str, Any] = None
    ):
        """
        Initialize DQN Network.
        
        Args:
            input_dim: Dimension of input (state dimension)
            output_dim: Dimension of output (action dimension)
            hidden_dims: List of hidden layer dimensions
            dueling: Whether to use dueling architecture
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [256, 256, 128]
        self.dueling = dueling
        
        # Build network layers
        self.features = self._build_feature_layers()
        
        if self.dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.hidden_dims[-1], 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
        else:
            # Standard DQN
            self.q_head = nn.Linear(self.hidden_dims[-1], output_dim)
        
        self.to(self.device)
    
    def _build_feature_layers(self) -> nn.Module:
        """Build feature extraction layers."""
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DQN.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        features = self.features(x)
        
        if self.dueling:
            # Dueling architecture
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine value and advantage
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN
            q_values = self.q_head(features)
        
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Input state tensor
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.output_dim, (state.shape[0],))
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=1)


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration.
    
    Implements factorized Gaussian noise for parameter space exploration
    as an alternative to epsilon-greedy.
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        """
        Initialize noisy linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            sigma_init: Initial value for noise parameter
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
        
        # Noise parameters (not learnable)
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self) -> None:
        """Initialize parameters."""
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features ** 0.5)
    
    def reset_noise(self) -> None:
        """Reset noise parameters."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

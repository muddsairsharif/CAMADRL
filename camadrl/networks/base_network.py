"""
Base network class for CAMADRL framework.

This module provides the abstract base class for all neural networks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base class for all neural networks in CAMADRL framework.
    
    This class extends PyTorch's nn.Module and defines common functionality
    for all network architectures.
    
    Attributes:
        input_dim: Dimension of input features
        output_dim: Dimension of output features
        hidden_dim: Dimension of hidden layers
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            hidden_dim: Dimension of hidden layers
            config: Configuration dictionary
            
        Raises:
            ValueError: If dimensions are invalid
        """
        super().__init__()
        
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive integers")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.config = config or {}
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def initialize_weights(self, method: str = "xavier") -> None:
        """
        Initialize network weights.
        
        Args:
            method: Initialization method ('xavier', 'kaiming', 'normal')
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == "normal":
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get network information.
        
        Returns:
            Dictionary containing network information
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_parameters": self.get_num_parameters(),
        }

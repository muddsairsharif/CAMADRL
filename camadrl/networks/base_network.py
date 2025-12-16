"""
Base Network implementation for CAMADRL framework.

Provides the abstract base class that all neural networks should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base class for all neural networks in the CAMADRL framework.
    
    This class defines the common interface that all network implementations
    must follow, ensuring consistency across different network types.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the base network.
        
        Args:
            config: Configuration dictionary for network parameters
        """
        super().__init__()
        self.config = config or {}
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
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
    
    def get_num_params(self) -> int:
        """
        Get the total number of parameters in the network.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """
        Get the number of trainable parameters in the network.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self) -> None:
        """Freeze all network parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all network parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get network information.
        
        Returns:
            Dictionary containing network information
        """
        return {
            "num_params": self.get_num_params(),
            "num_trainable_params": self.get_num_trainable_params(),
            "device": str(self.device),
        }

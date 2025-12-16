"""
Communication network for multi-agent context sharing.

This module implements neural networks for processing and sharing
context information between agents in multi-agent systems.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from camadrl.networks.base_network import BaseNetwork


class CommunicationNetwork(BaseNetwork):
    """
    Communication network for agent context sharing.
    
    Processes information from other agents to generate context embeddings
    that can be used for context-aware decision making.
    
    Attributes:
        encoder: Encoder network for processing agent information
        attention: Attention mechanism for aggregating information
        context_generator: Network for generating context embeddings
    """
    
    def __init__(
        self,
        agent_state_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Communication network.
        
        Args:
            agent_state_dim: Dimension of each agent's state
            context_dim: Dimension of output context embedding
            hidden_dim: Dimension of hidden layers
            num_attention_heads: Number of attention heads
            config: Configuration dictionary
        """
        super().__init__(agent_state_dim, context_dim, hidden_dim, config)
        
        self.num_attention_heads = num_attention_heads
        
        # Encoder for processing agent states
        self.encoder = nn.Sequential(
            nn.Linear(agent_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Multi-head attention for aggregating information
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        # Context generator
        self.context_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim)
        )
        
        # Initialize weights
        self.initialize_weights(method="xavier")
    
    def forward(
        self,
        agent_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the communication network.
        
        Args:
            agent_states: Tensor of agent states (batch_size, num_agents, state_dim)
                         or (batch_size, state_dim) for single agent
            mask: Optional attention mask
            
        Returns:
            Context embedding tensor (batch_size, context_dim)
        """
        # Handle single agent input
        if len(agent_states.shape) == 2:
            agent_states = agent_states.unsqueeze(1)
        
        # Encode agent states
        encoded = self.encoder(agent_states)  # (batch, num_agents, hidden_dim)
        
        # Apply attention to aggregate information
        attended, attention_weights = self.attention(
            encoded, encoded, encoded,
            key_padding_mask=mask
        )
        
        # Average pooling over agents
        aggregated = attended.mean(dim=1)  # (batch, hidden_dim)
        
        # Generate context embedding
        context = self.context_generator(aggregated)  # (batch, context_dim)
        
        return context
    
    def get_attention_weights(
        self,
        agent_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            agent_states: Tensor of agent states
            mask: Optional attention mask
            
        Returns:
            Attention weights tensor
        """
        if len(agent_states.shape) == 2:
            agent_states = agent_states.unsqueeze(1)
        
        encoded = self.encoder(agent_states)
        _, attention_weights = self.attention(
            encoded, encoded, encoded,
            key_padding_mask=mask
        )
        
        return attention_weights


class GraphCommunicationNetwork(BaseNetwork):
    """
    Graph-based communication network using Graph Neural Networks.
    
    Uses graph convolutions to propagate information between agents
    based on their connectivity structure.
    """
    
    def __init__(
        self,
        agent_state_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Graph Communication network.
        
        Args:
            agent_state_dim: Dimension of each agent's state
            context_dim: Dimension of output context embedding
            hidden_dim: Dimension of hidden layers
            num_layers: Number of graph convolution layers
            config: Configuration dictionary
        """
        super().__init__(agent_state_dim, context_dim, hidden_dim, config)
        
        self.num_layers = num_layers
        
        # Initial embedding
        self.embedding = nn.Linear(agent_state_dim, hidden_dim)
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim)
        )
        
        # Initialize weights
        self.initialize_weights(method="xavier")
    
    def forward(
        self,
        agent_states: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the graph communication network.
        
        Args:
            agent_states: Tensor of agent states (batch_size, num_agents, state_dim)
            adjacency_matrix: Adjacency matrix (batch_size, num_agents, num_agents)
            
        Returns:
            Context embedding tensor (batch_size, context_dim)
        """
        # Initial embedding
        x = F.relu(self.embedding(agent_states))
        
        # Apply graph convolutions
        for layer in self.graph_layers:
            x = layer(x, adjacency_matrix)
            x = F.relu(x)
        
        # Aggregate over agents (mean pooling)
        aggregated = x.mean(dim=1)
        
        # Generate context
        context = self.output_projection(aggregated)
        
        return context


class GraphConvLayer(nn.Module):
    """
    Graph convolution layer for message passing between agents.
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize graph convolution layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.self_linear = nn.Linear(in_features, out_features)
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through graph convolution.
        
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adjacency: Adjacency matrix (batch_size, num_nodes, num_nodes)
            
        Returns:
            Updated node features (batch_size, num_nodes, out_features)
        """
        # Normalize adjacency matrix
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1)
        adjacency_normalized = adjacency / degree
        
        # Aggregate neighbor features
        neighbor_features = torch.bmm(adjacency_normalized, x)
        
        # Combine neighbor and self features
        output = self.linear(neighbor_features) + self.self_linear(x)
        
        return output


class MessagePassingNetwork(nn.Module):
    """
    Message passing network for explicit agent-to-agent communication.
    """
    
    def __init__(
        self,
        message_dim: int,
        hidden_dim: int = 128
    ):
        """
        Initialize message passing network.
        
        Args:
            message_dim: Dimension of messages
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Message encoding
        self.message_encoder = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message aggregation
        self.message_aggregator = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Message decoding
        self.message_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
    
    def forward(
        self,
        messages: torch.Tensor
    ) -> torch.Tensor:
        """
        Process and aggregate messages.
        
        Args:
            messages: Incoming messages (batch_size, num_messages, message_dim)
            
        Returns:
            Aggregated message (batch_size, message_dim)
        """
        # Encode messages
        encoded = self.message_encoder(messages)
        
        # Aggregate messages
        _, hidden = self.message_aggregator(encoded)
        aggregated = hidden.squeeze(0)
        
        # Decode aggregated message
        output = self.message_decoder(aggregated)
        
        return output

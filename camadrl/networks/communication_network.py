"""
Communication Network for multi-agent coordination.

Implements communication mechanisms that allow agents to share
information and coordinate their actions.
"""

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

from camadrl.networks.base_network import BaseNetwork


class CommunicationNetwork(BaseNetwork):
    """
    Communication Network for multi-agent message passing.
    
    Enables agents to exchange information through learned communication
    protocols for improved coordination in EV charging scenarios.
    """
    
    def __init__(
        self,
        state_dim: int,
        message_dim: int,
        hidden_dim: int = 128,
        num_agents: int = 3,
        config: Dict[str, Any] = None
    ):
        """
        Initialize Communication Network.
        
        Args:
            state_dim: Dimension of agent state
            message_dim: Dimension of communication messages
            hidden_dim: Hidden layer dimension
            num_agents: Number of agents in the system
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.state_dim = state_dim
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        
        # Message encoder: state -> message
        self.message_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()
        )
        
        # Message aggregator: multiple messages -> aggregated message
        self.message_aggregator = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # State updater: state + aggregated message -> new state
        self.state_updater = nn.Sequential(
            nn.Linear(state_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.to(self.device)
    
    def forward(
        self,
        states: torch.Tensor,
        adjacency: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for communication.
        
        Args:
            states: Agent states [batch_size, num_agents, state_dim]
            adjacency: Communication graph [num_agents, num_agents]
                      1 if agents can communicate, 0 otherwise
            
        Returns:
            Tuple of (updated_states, messages)
        """
        batch_size, num_agents, _ = states.shape
        
        # Generate messages from all agents
        messages = self.message_encoder(states)  # [batch_size, num_agents, message_dim]
        
        # Default to fully connected if no adjacency provided
        if adjacency is None:
            adjacency = torch.ones(num_agents, num_agents).to(self.device)
            adjacency.fill_diagonal_(0)  # No self-communication
        
        # Aggregate messages for each agent
        aggregated_messages = []
        for i in range(num_agents):
            # Get messages from neighbors
            neighbor_mask = adjacency[i].unsqueeze(0).unsqueeze(-1)  # [1, num_agents, 1]
            neighbor_messages = messages * neighbor_mask  # [batch_size, num_agents, message_dim]
            
            # Sum messages from neighbors
            summed_messages = neighbor_messages.sum(dim=1)  # [batch_size, message_dim]
            
            # Apply aggregation function
            agg_message = self.message_aggregator(summed_messages)  # [batch_size, message_dim]
            aggregated_messages.append(agg_message)
        
        aggregated_messages = torch.stack(aggregated_messages, dim=1)  # [batch_size, num_agents, message_dim]
        
        # Update states with aggregated messages
        combined = torch.cat([states, aggregated_messages], dim=-1)  # [batch_size, num_agents, state_dim + message_dim]
        updated_states = self.state_updater(combined)  # [batch_size, num_agents, state_dim]
        
        return updated_states, messages
    
    def encode_message(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode a single agent's state to a message.
        
        Args:
            state: Agent state [batch_size, state_dim]
            
        Returns:
            Encoded message [batch_size, message_dim]
        """
        return self.message_encoder(state)


class GraphCommunicationNetwork(BaseNetwork):
    """
    Graph-based Communication Network using Graph Neural Networks.
    
    Uses graph convolutions to propagate information across agents
    based on their connectivity structure.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 3,
        config: Dict[str, Any] = None
    ):
        """
        Initialize Graph Communication Network.
        
        Args:
            node_dim: Dimension of node (agent) features
            edge_dim: Dimension of edge features (0 if no edge features)
            hidden_dim: Hidden layer dimension
            num_layers: Number of graph convolution layers
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Edge embedding (if edge features exist)
        if edge_dim > 0:
            self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, edge_dim > 0)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, node_dim)
        
        self.to(self.device)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through graph network.
        
        Args:
            node_features: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        # Embed node features
        x = self.node_embedding(node_features)
        
        # Embed edge features if provided
        edge_attr = None
        if edge_features is not None and self.edge_dim > 0:
            edge_attr = self.edge_embedding(edge_features)
        
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
        
        # Output layer
        output = self.output_layer(x)
        
        return output


class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer for message passing.
    """
    
    def __init__(self, in_dim: int, out_dim: int, use_edge_features: bool = False):
        """
        Initialize graph convolution layer.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            use_edge_features: Whether to use edge features
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_edge_features = use_edge_features
        
        # Message function
        if use_edge_features:
            self.message_net = nn.Sequential(
                nn.Linear(in_dim * 2 + in_dim, out_dim),  # source + target + edge
                nn.ReLU()
            )
        else:
            self.message_net = nn.Sequential(
                nn.Linear(in_dim * 2, out_dim),  # source + target
                nn.ReLU()
            )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through graph convolution.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, in_dim] (optional)
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        row, col = edge_index
        
        # Compute messages
        if self.use_edge_features and edge_attr is not None:
            messages = self.message_net(
                torch.cat([x[row], x[col], edge_attr], dim=-1)
            )
        else:
            messages = self.message_net(
                torch.cat([x[row], x[col]], dim=-1)
            )
        
        # Aggregate messages
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, self.out_dim).to(x.device)
        aggregated.index_add_(0, col, messages)
        
        # Update node features
        updated = self.update_net(torch.cat([x, aggregated], dim=-1))
        
        return updated

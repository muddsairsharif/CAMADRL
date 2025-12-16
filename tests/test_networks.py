"""Unit tests for network implementations."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camadrl.networks import DQNNetwork, ActorCriticNetwork, CommunicationNetwork

class TestDQNNetwork:
    """Tests for DQN Network."""
    
    def test_network_initialization(self):
        """Test network can be initialized."""
        net = DQNNetwork(state_dim=10, action_dim=5, hidden_dim=64)
        assert net.input_dim == 10
        assert net.output_dim == 5
    
    def test_forward_pass(self):
        """Test forward pass."""
        net = DQNNetwork(state_dim=10, action_dim=5, hidden_dim=64)
        state = torch.randn(2, 10)
        output = net(state)
        assert output.shape == (2, 5)

class TestActorCriticNetwork:
    """Tests for Actor-Critic Network."""
    
    def test_network_initialization(self):
        """Test network can be initialized."""
        net = ActorCriticNetwork(state_dim=10, action_dim=5, hidden_dim=64)
        assert net.input_dim == 10
        assert net.output_dim == 5
    
    def test_forward_pass(self):
        """Test forward pass."""
        net = ActorCriticNetwork(state_dim=10, action_dim=5, hidden_dim=64)
        state = torch.randn(2, 10)
        action_probs, value = net(state)
        assert action_probs.shape == (2, 5)
        assert value.shape == (2, 1)

class TestCommunicationNetwork:
    """Tests for Communication Network."""
    
    def test_network_initialization(self):
        """Test network can be initialized."""
        net = CommunicationNetwork(agent_state_dim=10, context_dim=32, hidden_dim=64)
        assert net.input_dim == 10
        assert net.output_dim == 32
    
    def test_forward_pass(self):
        """Test forward pass."""
        net = CommunicationNetwork(agent_state_dim=10, context_dim=32, hidden_dim=64)
        agent_states = torch.randn(2, 5, 10)  # batch_size=2, num_agents=5
        context = net(agent_states)
        assert context.shape == (2, 32)

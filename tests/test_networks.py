"""
Unit tests for CAMADRL networks.

Tests the functionality of base network, DQN network,
actor-critic network, and communication network implementations.
"""

import pytest
import torch

from camadrl.networks import (
    BaseNetwork, DQNNetwork, ActorCriticNetwork, CommunicationNetwork
)


class TestDQNNetwork:
    """Test suite for DQN Network."""
    
    def test_initialization(self):
        """Test DQN network initialization."""
        network = DQNNetwork(
            input_dim=10,
            output_dim=5,
            hidden_dims=[64, 64]
        )
        
        assert network.input_dim == 10
        assert network.output_dim == 5
        assert network.hidden_dims == [64, 64]
    
    def test_forward_pass(self):
        """Test forward pass."""
        network = DQNNetwork(input_dim=10, output_dim=5)
        x = torch.randn(32, 10)
        
        output = network(x)
        assert output.shape == (32, 5)
    
    def test_dueling_architecture(self):
        """Test dueling DQN architecture."""
        network = DQNNetwork(
            input_dim=10,
            output_dim=5,
            dueling=True
        )
        
        x = torch.randn(32, 10)
        output = network(x)
        assert output.shape == (32, 5)
    
    def test_get_action(self):
        """Test action selection."""
        network = DQNNetwork(input_dim=10, output_dim=5)
        x = torch.randn(4, 10)
        
        # Without exploration
        actions = network.get_action(x, epsilon=0.0)
        assert actions.shape == (4,)
        
        # With exploration
        actions = network.get_action(x, epsilon=0.5)
        assert actions.shape == (4,)
    
    def test_num_params(self):
        """Test parameter counting."""
        network = DQNNetwork(input_dim=10, output_dim=5)
        
        num_params = network.get_num_params()
        assert num_params > 0
        
        num_trainable = network.get_num_trainable_params()
        assert num_trainable == num_params


class TestActorCriticNetwork:
    """Test suite for Actor-Critic Network."""
    
    def test_initialization(self):
        """Test actor-critic network initialization."""
        network = ActorCriticNetwork(
            input_dim=10,
            action_dim=5,
            hidden_dims=[64, 64]
        )
        
        assert network.input_dim == 10
        assert network.action_dim == 5
    
    def test_forward_pass(self):
        """Test forward pass."""
        network = ActorCriticNetwork(input_dim=10, action_dim=5)
        x = torch.randn(32, 10)
        
        action_params, value = network(x)
        assert action_params.shape == (32, 5)
        assert value.shape == (32, 1)
    
    def test_continuous_actions(self):
        """Test continuous action space."""
        network = ActorCriticNetwork(
            input_dim=10,
            action_dim=3,
            continuous=True
        )
        
        x = torch.randn(32, 10)
        action_params, value = network(x)
        
        # action_params is (mean, log_std)
        assert isinstance(action_params, tuple)
        assert len(action_params) == 2
    
    def test_get_action(self):
        """Test action selection."""
        network = ActorCriticNetwork(input_dim=10, action_dim=5)
        x = torch.randn(4, 10)
        
        action, log_prob, value = network.get_action(x, deterministic=False)
        assert action.shape[0] == 4
        assert log_prob is not None
        assert value.shape == (4, 1)
    
    def test_evaluate_actions(self):
        """Test action evaluation."""
        network = ActorCriticNetwork(input_dim=10, action_dim=5)
        x = torch.randn(32, 10)
        actions = torch.randint(0, 5, (32,))
        
        log_prob, entropy, value = network.evaluate_actions(x, actions)
        assert log_prob.shape == (32, 1)
        assert entropy.shape == (32, 1)
        assert value.shape == (32, 1)


class TestCommunicationNetwork:
    """Test suite for Communication Network."""
    
    def test_initialization(self):
        """Test communication network initialization."""
        network = CommunicationNetwork(
            state_dim=10,
            message_dim=8,
            num_agents=3
        )
        
        assert network.state_dim == 10
        assert network.message_dim == 8
        assert network.num_agents == 3
    
    def test_forward_pass(self):
        """Test forward pass with message passing."""
        network = CommunicationNetwork(
            state_dim=10,
            message_dim=8,
            num_agents=3
        )
        
        states = torch.randn(2, 3, 10)  # batch_size=2, num_agents=3, state_dim=10
        
        updated_states, messages = network(states)
        assert updated_states.shape == (2, 3, 10)
        assert messages.shape == (2, 3, 8)
    
    def test_custom_adjacency(self):
        """Test with custom communication graph."""
        network = CommunicationNetwork(
            state_dim=10,
            message_dim=8,
            num_agents=3
        )
        
        states = torch.randn(2, 3, 10)
        adjacency = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=torch.float32)
        
        updated_states, messages = network(states, adjacency)
        assert updated_states.shape == (2, 3, 10)
    
    def test_encode_message(self):
        """Test message encoding."""
        network = CommunicationNetwork(
            state_dim=10,
            message_dim=8,
            num_agents=3
        )
        
        state = torch.randn(4, 10)
        message = network.encode_message(state)
        assert message.shape == (4, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

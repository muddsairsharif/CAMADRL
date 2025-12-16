"""
Unit tests for CAMADRL agents.

Tests the functionality of base agent, DQN agent, CADRL agent,
and policy gradient agent implementations.
"""

import pytest
import numpy as np
import torch
import tempfile
import os

from camadrl.agents import BaseAgent, DQNAgent, CADRLAgent, PolicyGradientAgent


class TestDQNAgent:
    """Test suite for DQN Agent."""
    
    def test_initialization(self):
        """Test DQN agent initialization."""
        agent = DQNAgent(
            agent_id=0,
            state_dim=10,
            action_dim=5,
            config={"learning_rate": 0.001}
        )
        
        assert agent.agent_id == 0
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.learning_rate == 0.001
    
    def test_select_action(self):
        """Test action selection."""
        agent = DQNAgent(agent_id=0, state_dim=10, action_dim=5)
        state = np.random.rand(10)
        
        # Test exploration
        action = agent.select_action(state, explore=True)
        assert 0 <= action < 5
        
        # Test exploitation
        action = agent.select_action(state, explore=False)
        assert 0 <= action < 5
    
    def test_update(self):
        """Test agent update."""
        agent = DQNAgent(agent_id=0, state_dim=10, action_dim=5)
        
        batch = {
            "states": torch.randn(32, 10),
            "actions": torch.randint(0, 5, (32,)),
            "rewards": torch.randn(32),
            "next_states": torch.randn(32, 10),
            "dones": torch.zeros(32)
        }
        
        metrics = agent.update(batch)
        assert "loss" in metrics
        assert "q_value" in metrics
        assert isinstance(metrics["loss"], float)
    
    def test_save_load(self):
        """Test saving and loading agent."""
        agent = DQNAgent(agent_id=0, state_dim=10, action_dim=5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "agent.pth")
            agent.save(filepath)
            
            new_agent = DQNAgent(agent_id=0, state_dim=10, action_dim=5)
            new_agent.load(filepath)
            
            assert new_agent.agent_id == agent.agent_id


class TestCADRLAgent:
    """Test suite for CADRL Agent."""
    
    def test_initialization(self):
        """Test CADRL agent initialization."""
        agent = CADRLAgent(
            agent_id=0,
            state_dim=10,
            action_dim=5,
            context_dim=8
        )
        
        assert agent.context_dim == 8
        assert agent.state_dim == 10
        assert agent.action_dim == 5
    
    def test_select_action_with_context(self):
        """Test action selection with context."""
        agent = CADRLAgent(
            agent_id=0,
            state_dim=10,
            action_dim=5,
            context_dim=8
        )
        
        state = np.random.rand(10)
        context = np.random.rand(8)
        
        action = agent.select_action(state, context=context, explore=False)
        assert 0 <= action < 5
    
    def test_update_with_context(self):
        """Test agent update with context."""
        agent = CADRLAgent(
            agent_id=0,
            state_dim=10,
            action_dim=5,
            context_dim=8
        )
        
        batch = {
            "states": torch.randn(32, 10),
            "actions": torch.randint(0, 5, (32,)),
            "rewards": torch.randn(32),
            "next_states": torch.randn(32, 10),
            "contexts": torch.randn(32, 8),
            "dones": torch.zeros(32)
        }
        
        metrics = agent.update(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)


class TestPolicyGradientAgent:
    """Test suite for Policy Gradient Agent."""
    
    def test_initialization(self):
        """Test policy gradient agent initialization."""
        agent = PolicyGradientAgent(
            agent_id=0,
            state_dim=10,
            action_dim=5
        )
        
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')
    
    def test_select_action(self):
        """Test action selection."""
        agent = PolicyGradientAgent(
            agent_id=0,
            state_dim=10,
            action_dim=5
        )
        
        state = np.random.rand(10)
        action = agent.select_action(state, explore=True)
        assert 0 <= action < 5
    
    def test_episode_update(self):
        """Test episode-based update."""
        agent = PolicyGradientAgent(
            agent_id=0,
            state_dim=10,
            action_dim=5
        )
        
        # Simulate episode
        for _ in range(10):
            state = np.random.rand(10)
            agent.select_action(state, explore=True)
            agent.store_reward(np.random.randn())
        
        # Update
        metrics = agent.update()
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics
        assert "entropy" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for agent implementations."""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camadrl.agents import DQNAgent, CADRLAgent, PolicyGradientAgent

class TestDQNAgent:
    """Tests for DQN Agent."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = DQNAgent(agent_id=0, state_dim=10, action_dim=5)
        assert agent.agent_id == 0
        assert agent.state_dim == 10
        assert agent.action_dim == 5
    
    def test_action_selection(self):
        """Test action selection."""
        agent = DQNAgent(agent_id=0, state_dim=10, action_dim=5)
        state = np.random.randn(10)
        action = agent.select_action(state)
        assert 0 <= action < 5
    
    def test_update(self):
        """Test agent update."""
        agent = DQNAgent(agent_id=0, state_dim=10, action_dim=5)
        state = np.random.randn(10)
        action = 0
        reward = 1.0
        next_state = np.random.randn(10)
        done = False
        
        metrics = agent.update(state, action, reward, next_state, done)
        assert isinstance(metrics, dict)

class TestCADRLAgent:
    """Tests for CADRL Agent."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = CADRLAgent(agent_id=0, state_dim=10, action_dim=5, context_dim=32)
        assert agent.agent_id == 0
        assert agent.context_dim == 32
    
    def test_action_selection_with_context(self):
        """Test action selection with context."""
        agent = CADRLAgent(agent_id=0, state_dim=10, action_dim=5, context_dim=32)
        state = np.random.randn(10)
        context = np.random.randn(20)
        action, context_emb = agent.select_action(state, context=context)
        assert 0 <= action < 5
        assert context_emb.shape[1] == 32

class TestPolicyGradientAgent:
    """Tests for Policy Gradient Agent."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = PolicyGradientAgent(agent_id=0, state_dim=10, action_dim=5)
        assert agent.agent_id == 0
    
    def test_action_selection(self):
        """Test action selection."""
        agent = PolicyGradientAgent(agent_id=0, state_dim=10, action_dim=5)
        state = np.random.randn(10)
        action = agent.select_action(state)
        assert 0 <= action < 5

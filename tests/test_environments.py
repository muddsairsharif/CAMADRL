"""Unit tests for environment implementations."""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camadrl.environments import GridWorld, TrafficSim, CustomEnv

class TestGridWorld:
    """Tests for GridWorld environment."""
    
    def test_env_initialization(self):
        """Test environment can be initialized."""
        env = GridWorld(num_agents=2, grid_size=10)
        assert env.num_agents == 2
        assert env.grid_size == 10
    
    def test_reset(self):
        """Test environment reset."""
        env = GridWorld(num_agents=2, grid_size=10)
        states, info = env.reset()
        assert len(states) == 2
        assert all(state.shape[0] == env.state_dim for state in states)
    
    def test_step(self):
        """Test environment step."""
        env = GridWorld(num_agents=2, grid_size=10)
        states, _ = env.reset()
        actions = [0, 1]
        next_states, rewards, dones, info = env.step(actions)
        assert len(next_states) == 2
        assert len(rewards) == 2
        assert len(dones) == 2

class TestTrafficSim:
    """Tests for TrafficSim environment."""
    
    def test_env_initialization(self):
        """Test environment can be initialized."""
        env = TrafficSim(num_agents=5)
        assert env.num_agents == 5
    
    def test_reset(self):
        """Test environment reset."""
        env = TrafficSim(num_agents=5)
        states, info = env.reset()
        assert len(states) == 5

class TestCustomEnv:
    """Tests for CustomEnv environment."""
    
    def test_env_initialization(self):
        """Test environment can be initialized."""
        env = CustomEnv(num_agents=3, state_dim=8, action_dim=4)
        assert env.num_agents == 3
        assert env.state_dim == 8
        assert env.action_dim == 4

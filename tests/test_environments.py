"""
Unit tests for CAMADRL environments.

Tests the functionality of base environment, GridWorld,
TrafficSim, and CustomEnv implementations.
"""

import pytest
import numpy as np

from camadrl.environments import BaseEnv, GridWorld, TrafficSim, CustomEnv


class TestGridWorld:
    """Test suite for GridWorld environment."""
    
    def test_initialization(self):
        """Test GridWorld initialization."""
        env = GridWorld({"grid_size": 10, "num_agents": 2})
        
        assert env.grid_size == 10
        assert env.num_agents == 2
        assert env.observation_space is not None
        assert env.action_space is not None
    
    def test_reset(self):
        """Test environment reset."""
        env = GridWorld({"grid_size": 10, "num_agents": 2})
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
    
    def test_step(self):
        """Test environment step."""
        env = GridWorld({"grid_size": 10, "num_agents": 2})
        env.reset()
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_episode(self):
        """Test complete episode."""
        env = GridWorld({"grid_size": 10, "num_agents": 2, "max_steps": 50})
        obs, _ = env.reset()
        
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        assert steps > 0
        assert isinstance(total_reward, (int, float))


class TestTrafficSim:
    """Test suite for TrafficSim environment."""
    
    def test_initialization(self):
        """Test TrafficSim initialization."""
        env = TrafficSim({
            "network_size": 20,
            "num_agents": 3,
            "num_charging_stations": 5
        })
        
        assert env.network_size == 20
        assert env.num_agents == 3
        assert env.num_charging_stations == 5
    
    def test_reset(self):
        """Test environment reset."""
        env = TrafficSim({"network_size": 20, "num_agents": 3})
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
    
    def test_step(self):
        """Test environment step."""
        env = TrafficSim({"network_size": 20, "num_agents": 3})
        env.reset()
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_traffic_dynamics(self):
        """Test traffic dynamics."""
        env = TrafficSim({"network_size": 20, "num_agents": 3})
        env.reset()
        
        # Test multiple steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
        
        assert hasattr(env, 'agent_positions')
        assert hasattr(env, 'agent_batteries')


class TestCustomEnv:
    """Test suite for CustomEnv."""
    
    def test_initialization(self):
        """Test CustomEnv initialization."""
        env = CustomEnv({
            "state_dim": 10,
            "action_dim": 4,
            "num_agents": 2
        })
        
        assert env.state_dim == 10
        assert env.action_dim == 4
        assert env.num_agents == 2
    
    def test_custom_reward_function(self):
        """Test custom reward function."""
        def custom_reward(state, agent_states, action, agent_idx):
            return -np.sum(state ** 2)
        
        env = CustomEnv({
            "state_dim": 10,
            "action_dim": 4,
            "reward_fn": custom_reward
        })
        
        assert env.reward_fn == custom_reward
    
    def test_custom_observation_function(self):
        """Test custom observation function."""
        def custom_obs(state, agent_states, agent_idx):
            return state * 2
        
        env = CustomEnv({
            "state_dim": 10,
            "action_dim": 4,
            "observation_fn": custom_obs
        })
        
        assert env.observation_fn == custom_obs
    
    def test_set_reward_function(self):
        """Test dynamically setting reward function."""
        env = CustomEnv({"state_dim": 10, "action_dim": 4})
        
        def new_reward(state, agent_states, action, agent_idx):
            return np.sum(state)
        
        env.set_reward_function(new_reward)
        assert env.reward_fn == new_reward
    
    def test_episode(self):
        """Test complete episode."""
        env = CustomEnv({
            "state_dim": 10,
            "action_dim": 4,
            "max_steps": 30
        })
        obs, _ = env.reset()
        
        done = False
        steps = 0
        
        while not done and steps < 30:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert steps > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

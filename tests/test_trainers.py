"""Unit tests for trainer implementations."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camadrl.environments import GridWorld
from camadrl.agents import DQNAgent
from camadrl.trainers import MultiAgentTrainer

class TestMultiAgentTrainer:
    """Tests for MultiAgentTrainer."""
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        env = GridWorld(num_agents=2, grid_size=5)
        agents = [
            DQNAgent(0, env.state_dim, env.action_dim),
            DQNAgent(1, env.state_dim, env.action_dim)
        ]
        trainer = MultiAgentTrainer(env, agents)
        assert trainer.num_episodes == 1000  # default
    
    def test_train_episode(self):
        """Test single episode training."""
        env = GridWorld(num_agents=2, grid_size=5)
        agents = [
            DQNAgent(0, env.state_dim, env.action_dim),
            DQNAgent(1, env.state_dim, env.action_dim)
        ]
        config = {"max_steps_per_episode": 10}
        trainer = MultiAgentTrainer(env, agents, config)
        
        metrics = trainer.train_episode()
        assert "episode_reward_mean" in metrics
        assert "episode_steps" in metrics

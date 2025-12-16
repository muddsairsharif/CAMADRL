"""
Unit tests for CAMADRL trainers.

Tests the functionality of base trainer, multi-agent trainer,
and distributed trainer implementations.
"""

import pytest
import numpy as np

from camadrl.agents import DQNAgent
from camadrl.environments import GridWorld
from camadrl.trainers import BaseTrainer, MultiAgentTrainer


class TestMultiAgentTrainer:
    """Test suite for Multi-Agent Trainer."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        env = GridWorld({"grid_size": 5, "num_agents": 2, "max_steps": 20})
        agents = [
            DQNAgent(0, env.observation_space.shape[0], env.action_space.n),
            DQNAgent(1, env.observation_space.shape[0], env.action_space.n)
        ]
        
        trainer = MultiAgentTrainer(
            agents=agents,
            env=env,
            config={"num_episodes": 10}
        )
        
        assert trainer.agents == agents
        assert trainer.env == env
        assert trainer.num_episodes == 10
    
    def test_train_episode(self):
        """Test training single episode."""
        env = GridWorld({"grid_size": 5, "num_agents": 2, "max_steps": 20})
        agents = [
            DQNAgent(0, env.observation_space.shape[0], env.action_space.n),
            DQNAgent(1, env.observation_space.shape[0], env.action_space.n)
        ]
        
        trainer = MultiAgentTrainer(
            agents=agents,
            env=env,
            config={"num_episodes": 5, "learning_starts": 0}
        )
        
        stats = trainer.train_episode()
        
        assert "episode_reward" in stats
        assert "episode_length" in stats
        assert isinstance(stats["episode_reward"], (int, float))
        assert stats["episode_length"] > 0
    
    def test_evaluate(self):
        """Test agent evaluation."""
        env = GridWorld({"grid_size": 5, "num_agents": 2, "max_steps": 20})
        agents = [
            DQNAgent(0, env.observation_space.shape[0], env.action_space.n),
            DQNAgent(1, env.observation_space.shape[0], env.action_space.n)
        ]
        
        trainer = MultiAgentTrainer(
            agents=agents,
            env=env,
            config={"num_episodes": 5}
        )
        
        eval_stats = trainer.evaluate(num_episodes=2)
        
        assert "mean_reward" in eval_stats
        assert "std_reward" in eval_stats
        assert "mean_length" in eval_stats
        assert isinstance(eval_stats["mean_reward"], (int, float))
    
    def test_get_statistics(self):
        """Test getting training statistics."""
        env = GridWorld({"grid_size": 5, "num_agents": 2, "max_steps": 20})
        agents = [
            DQNAgent(0, env.observation_space.shape[0], env.action_space.n),
        ]
        
        trainer = MultiAgentTrainer(
            agents=agents,
            env=env,
            config={"num_episodes": 5}
        )
        
        # Train a few episodes
        for _ in range(3):
            trainer.train_episode()
        
        stats = trainer.get_statistics()
        
        assert "total_episodes" in stats
        assert "mean_episode_reward" in stats
        assert stats["total_episodes"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

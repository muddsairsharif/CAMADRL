"""
Base Trainer implementation for CAMADRL framework.

Provides the abstract base class that all trainers should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import torch


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers in the CAMADRL framework.
    
    This class defines the common interface that all trainer implementations
    must follow, ensuring consistency across different training strategies.
    """
    
    def __init__(
        self,
        agents: List[Any],
        env: Any,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the base trainer.
        
        Args:
            agents: List of agents to train
            env: Environment for training
            config: Configuration dictionary for trainer parameters
        """
        self.agents = agents
        self.env = env
        self.config = config or {}
        
        self.num_episodes = self.config.get("num_episodes", 1000)
        self.max_steps = self.config.get("max_steps", 1000)
        self.eval_freq = self.config.get("eval_freq", 10)
        self.save_freq = self.config.get("save_freq", 50)
        self.log_freq = self.config.get("log_freq", 1)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {}
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.
        
        Returns:
            Dictionary containing training results and statistics
        """
        pass
    
    @abstractmethod
    def train_episode(self) -> Dict[str, Any]:
        """
        Train for one episode.
        
        Returns:
            Dictionary containing episode statistics
        """
        pass
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained agents.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select actions (no exploration during evaluation)
                actions = []
                for agent in self.agents:
                    action = agent.select_action(obs, explore=False)
                    actions.append(action)
                
                # Take action in environment
                obs, reward, terminated, truncated, _ = self.env.step(actions[0])
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_metrics": self.training_metrics,
        }
        
        # Save agent models
        for i, agent in enumerate(self.agents):
            agent.save(f"{filepath}_agent_{i}.pth")
        
        torch.save(checkpoint, f"{filepath}_trainer.pth")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(f"{filepath}_trainer.pth", map_location=self.device)
        
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        self.training_metrics = checkpoint.get("training_metrics", {})
        
        # Load agent models
        for i, agent in enumerate(self.agents):
            agent.load(f"{filepath}_agent_{i}.pth")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            "total_episodes": len(self.episode_rewards),
            "mean_episode_reward": np.mean(self.episode_rewards[-100:]),
            "std_episode_reward": np.std(self.episode_rewards[-100:]),
            "mean_episode_length": np.mean(self.episode_lengths[-100:]),
            "best_episode_reward": max(self.episode_rewards),
        }

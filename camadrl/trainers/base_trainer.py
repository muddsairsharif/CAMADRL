"""
Base trainer class for CAMADRL framework.

This module provides the abstract base class for all trainers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

from camadrl.agents.base_agent import BaseAgent
from camadrl.environments.base_env import BaseEnv
from camadrl.utils.logger import Logger
from camadrl.utils.metrics import Metrics


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers in CAMADRL framework.
    
    This class defines the interface for training agents in environments,
    including training loops, evaluation, and logging.
    
    Attributes:
        env: Training environment
        agents: List of agents to train
        logger: Logger for tracking training progress
        metrics: Metrics calculator
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        env: BaseEnv,
        agents: List[BaseAgent],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base trainer.
        
        Args:
            env: Training environment
            agents: List of agents to train
            config: Configuration dictionary
            
        Raises:
            ValueError: If agents list is empty or incompatible with environment
        """
        if not agents:
            raise ValueError("agents list cannot be empty")
        
        if len(agents) != env.num_agents:
            raise ValueError(f"Number of agents ({len(agents)}) must match "
                           f"environment agents ({env.num_agents})")
        
        self.env = env
        self.agents = agents
        self.config = config or {}
        
        # Training parameters
        self.num_episodes = self.config.get("num_episodes", 1000)
        self.max_steps_per_episode = self.config.get("max_steps_per_episode", 1000)
        self.eval_frequency = self.config.get("eval_frequency", 100)
        self.save_frequency = self.config.get("save_frequency", 100)
        self.log_frequency = self.config.get("log_frequency", 10)
        
        # Logging and metrics
        self.logger = Logger(config=self.config)
        self.metrics = Metrics()
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
    @abstractmethod
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode.
        
        Returns:
            Dictionary of episode metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agents' performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Returns:
            Dictionary of final training statistics
        """
        print(f"Starting training for {self.num_episodes} episodes...")
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            self.current_episode = episode
            
            # Train one episode
            episode_metrics = self.train_episode()
            
            # Log metrics
            if episode % self.log_frequency == 0:
                self.logger.log(episode, episode_metrics)
                self._print_progress(episode, episode_metrics)
            
            # Evaluate periodically
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_metrics = self.evaluate()
                self.logger.log(episode, eval_metrics, prefix="eval")
                print(f"\nEvaluation at episode {episode}:")
                for key, value in eval_metrics.items():
                    print(f"  {key}: {value:.4f}")
            
            # Save checkpoints
            if episode % self.save_frequency == 0 and episode > 0:
                self.save_checkpoint(f"checkpoint_ep{episode}.pt")
        
        print("\nTraining completed!")
        
        # Final evaluation
        final_eval = self.evaluate()
        print("\nFinal Evaluation:")
        for key, value in final_eval.items():
            print(f"  {key}: {value:.4f}")
        
        return {
            "total_episodes": self.num_episodes,
            "total_steps": self.total_steps,
            "final_metrics": final_eval,
        }
    
    def _print_progress(self, episode: int, metrics: Dict[str, float]) -> None:
        """
        Print training progress.
        
        Args:
            episode: Current episode number
            metrics: Episode metrics
        """
        info_str = f"Episode {episode}"
        for key, value in metrics.items():
            if isinstance(value, float):
                info_str += f" | {key}: {value:.4f}"
        print(f"\r{info_str}", end="")
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "episode": self.current_episode,
            "total_steps": self.total_steps,
            "best_reward": self.best_reward,
            "config": self.config,
        }
        
        # Save each agent
        for i, agent in enumerate(self.agents):
            agent_path = path.replace(".pt", f"_agent{i}.pt")
            agent.save(agent_path)
        
        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        
        self.current_episode = checkpoint["episode"]
        self.total_steps = checkpoint["total_steps"]
        self.best_reward = checkpoint["best_reward"]
        
        # Load each agent
        for i, agent in enumerate(self.agents):
            agent_path = path.replace(".pt", f"_agent{i}.pt")
            agent.load(agent_path)
        
        print(f"Checkpoint loaded from {path}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get trainer information.
        
        Returns:
            Dictionary containing trainer info
        """
        return {
            "current_episode": self.current_episode,
            "total_steps": self.total_steps,
            "num_agents": len(self.agents),
            "num_episodes": self.num_episodes,
            "best_reward": self.best_reward,
        }

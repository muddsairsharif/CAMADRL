"""
Distributed Trainer implementation.

Implements distributed training across multiple processes or machines
for scalable multi-agent reinforcement learning.
"""

from typing import Any, Dict, List
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from camadrl.trainers.base_trainer import BaseTrainer


class DistributedTrainer(BaseTrainer):
    """
    Distributed Trainer for scalable multi-agent learning.
    
    Supports distributed training across multiple processes/GPUs
    for efficient large-scale multi-agent scenarios.
    """
    
    def __init__(
        self,
        agents: List[Any],
        env: Any,
        config: Dict[str, Any] = None
    ):
        """
        Initialize Distributed Trainer.
        
        Args:
            agents: List of agents to train
            env: Environment for training
            config: Configuration dictionary with keys:
                - num_workers: Number of parallel workers
                - num_episodes: Number of training episodes
                - backend: Distributed backend (default: 'gloo')
                - world_size: Total number of processes
        """
        super().__init__(agents, env, config)
        
        self.num_workers = self.config.get("num_workers", 4)
        self.backend = self.config.get("backend", "gloo")
        self.world_size = self.config.get("world_size", 1)
        
        self.is_distributed = self.world_size > 1
        
    def train(self) -> Dict[str, Any]:
        """Execute the training loop."""
        if self.is_distributed:
            return self._train_distributed()
        else:
            return self._train_single()
    
    def _train_single(self) -> Dict[str, Any]:
        """Train on a single process."""
        print(f"Starting single-process training for {self.num_episodes} episodes...")
        
        for episode in range(self.num_episodes):
            episode_stats = self.train_episode()
            
            if episode % self.log_freq == 0:
                print(f"Episode {episode}: "
                      f"Reward={episode_stats['episode_reward']:.2f}")
            
            if episode % self.eval_freq == 0 and episode > 0:
                eval_stats = self.evaluate(num_episodes=5)
                print(f"Evaluation: Mean Reward={eval_stats['mean_reward']:.2f}")
        
        return self.get_statistics()
    
    def _train_distributed(self) -> Dict[str, Any]:
        """Train using distributed processes."""
        print(f"Starting distributed training with {self.world_size} processes...")
        
        # Use multiprocessing to spawn workers
        mp.spawn(
            self._train_worker,
            args=(self.world_size,),
            nprocs=self.world_size,
            join=True
        )
        
        return self.get_statistics()
    
    def _train_worker(self, rank: int, world_size: int) -> None:
        """
        Training worker for distributed training.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
        """
        # Initialize process group
        torch.distributed.init_process_group(
            backend=self.backend,
            init_method='tcp://localhost:23456',
            world_size=world_size,
            rank=rank
        )
        
        # Set device for this worker
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
        
        print(f"Worker {rank} starting on device {device}")
        
        # Wrap agents with DDP
        ddp_agents = []
        for agent in self.agents:
            # Move agent to device
            if hasattr(agent, 'network'):
                agent.network = agent.network.to(device)
                if world_size > 1:
                    agent.network = DDP(agent.network, device_ids=[rank] if torch.cuda.is_available() else None)
            ddp_agents.append(agent)
        
        # Training loop for this worker
        episodes_per_worker = self.num_episodes // world_size
        
        for episode in range(episodes_per_worker):
            episode_stats = self.train_episode_worker(rank, ddp_agents)
            
            if episode % self.log_freq == 0 and rank == 0:
                print(f"Worker {rank} - Episode {episode}: "
                      f"Reward={episode_stats['episode_reward']:.2f}")
        
        # Cleanup
        torch.distributed.destroy_process_group()
    
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode (single process)."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        
        while not done:
            # Select actions
            actions = []
            for agent in self.agents:
                action = agent.select_action(obs, explore=True)
                actions.append(action)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = self.env.step(actions[0])
            done = terminated or truncated
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length
        }
    
    def train_episode_worker(self, rank: int, agents: List[Any]) -> Dict[str, Any]:
        """
        Train for one episode on a specific worker.
        
        Args:
            rank: Worker rank
            agents: List of agents (DDP wrapped)
            
        Returns:
            Episode statistics
        """
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        
        while not done:
            # Select actions
            actions = []
            for agent in agents:
                action = agent.select_action(obs, explore=True)
                actions.append(action)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = self.env.step(actions[0])
            done = terminated or truncated
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "worker_rank": rank
        }


class AsyncDistributedTrainer(BaseTrainer):
    """
    Asynchronous Distributed Trainer.
    
    Implements asynchronous training where workers independently
    collect experience and update shared parameters.
    """
    
    def __init__(
        self,
        agents: List[Any],
        env: Any,
        config: Dict[str, Any] = None
    ):
        """
        Initialize Async Distributed Trainer.
        
        Args:
            agents: List of agents to train
            env: Environment for training
            config: Configuration dictionary
        """
        super().__init__(agents, env, config)
        
        self.num_workers = self.config.get("num_workers", 4)
        self.update_freq = self.config.get("update_freq", 10)
        
    def train(self) -> Dict[str, Any]:
        """Execute asynchronous training."""
        print(f"Starting async distributed training with {self.num_workers} workers...")
        
        # Create shared parameters
        for agent in self.agents:
            if hasattr(agent, 'network'):
                agent.network.share_memory()
        
        # Spawn workers
        processes = []
        for rank in range(self.num_workers):
            p = mp.Process(target=self._async_worker, args=(rank,))
            p.start()
            processes.append(p)
        
        # Wait for all workers to complete
        for p in processes:
            p.join()
        
        return self.get_statistics()
    
    def _async_worker(self, rank: int) -> None:
        """
        Asynchronous training worker.
        
        Args:
            rank: Worker rank
        """
        episodes_per_worker = self.num_episodes // self.num_workers
        
        for episode in range(episodes_per_worker):
            # Collect trajectory
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                actions = []
                for agent in self.agents:
                    action = agent.select_action(obs, explore=True)
                    actions.append(action)
                
                next_obs, reward, terminated, truncated, _ = self.env.step(actions[0])
                done = terminated or truncated
                
                obs = next_obs
                episode_reward += reward
            
            if rank == 0 and episode % self.log_freq == 0:
                print(f"Worker {rank} - Episode {episode}: Reward={episode_reward:.2f}")

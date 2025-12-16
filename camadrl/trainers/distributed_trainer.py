"""
Distributed trainer for CAMADRL framework.

This module implements distributed training using PyTorch's distributed
data parallel (DDP) for large-scale training.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from camadrl.trainers.multi_agent_trainer import MultiAgentTrainer
from camadrl.agents.base_agent import BaseAgent
from camadrl.environments.base_env import BaseEnv


class DistributedTrainer(MultiAgentTrainer):
    """
    Distributed trainer for large-scale multi-agent training.
    
    Implements distributed training across multiple GPUs or machines
    using PyTorch's DistributedDataParallel.
    
    Attributes:
        world_size: Total number of processes
        rank: Rank of current process
        backend: Distributed backend ('nccl', 'gloo', etc.)
    """
    
    def __init__(
        self,
        env: BaseEnv,
        agents: List[BaseAgent],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize distributed trainer.
        
        Args:
            env: Training environment
            agents: List of agents to train
            config: Configuration dictionary
        """
        super().__init__(env, agents, config)
        
        # Distributed training parameters
        self.world_size = self.config.get("world_size", 1)
        self.rank = self.config.get("rank", 0)
        self.backend = self.config.get("backend", "nccl")
        self.distributed = self.world_size > 1
        
        # Initialize distributed training if needed
        if self.distributed and not dist.is_initialized():
            self._init_distributed()
    
    def _init_distributed(self) -> None:
        """Initialize distributed training environment."""
        dist.init_process_group(
            backend=self.backend,
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
            self.device = torch.device(f"cuda:{self.rank}")
        else:
            self.device = torch.device("cpu")
        
        print(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run distributed training loop.
        
        Returns:
            Dictionary of final training statistics
        """
        if self.distributed:
            return self._train_distributed()
        else:
            return super().train()
    
    def _train_distributed(self) -> Dict[str, Any]:
        """
        Run the distributed training loop.
        
        Returns:
            Dictionary of final training statistics
        """
        if self.rank == 0:
            print(f"Starting distributed training on {self.world_size} processes...")
        
        # Wrap agent networks with DDP
        self._wrap_agents_ddp()
        
        for episode in range(self.num_episodes):
            self.current_episode = episode
            
            # Train one episode
            episode_metrics = self.train_episode()
            
            # Synchronize metrics across processes
            if self.distributed:
                episode_metrics = self._sync_metrics(episode_metrics)
            
            # Only log and evaluate on rank 0
            if self.rank == 0:
                if episode % self.log_frequency == 0:
                    self.logger.log(episode, episode_metrics)
                    self._print_progress(episode, episode_metrics)
                
                if episode % self.eval_frequency == 0 and episode > 0:
                    eval_metrics = self.evaluate()
                    self.logger.log(episode, eval_metrics, prefix="eval")
                    print(f"\nEvaluation at episode {episode}:")
                    for key, value in eval_metrics.items():
                        print(f"  {key}: {value:.4f}")
                
                if episode % self.save_frequency == 0 and episode > 0:
                    self.save_checkpoint(f"checkpoint_ep{episode}.pt")
        
        # Cleanup
        if self.distributed:
            dist.destroy_process_group()
        
        if self.rank == 0:
            print("\nDistributed training completed!")
            final_eval = self.evaluate()
            print("\nFinal Evaluation:")
            for key, value in final_eval.items():
                print(f"  {key}: {value:.4f}")
            
            return {
                "total_episodes": self.num_episodes,
                "total_steps": self.total_steps,
                "world_size": self.world_size,
                "final_metrics": final_eval,
            }
        
        return {}
    
    def _wrap_agents_ddp(self) -> None:
        """Wrap agent networks with DistributedDataParallel."""
        if not self.distributed:
            return
        
        for agent in self.agents:
            # Wrap networks that have parameters
            if hasattr(agent, 'q_network'):
                agent.q_network = DDP(
                    agent.q_network,
                    device_ids=[self.rank] if torch.cuda.is_available() else None
                )
            
            if hasattr(agent, 'target_network'):
                agent.target_network = DDP(
                    agent.target_network,
                    device_ids=[self.rank] if torch.cuda.is_available() else None
                )
            
            if hasattr(agent, 'policy_network'):
                agent.policy_network = DDP(
                    agent.policy_network,
                    device_ids=[self.rank] if torch.cuda.is_available() else None
                )
            
            if hasattr(agent, 'actor_critic'):
                agent.actor_critic = DDP(
                    agent.actor_critic,
                    device_ids=[self.rank] if torch.cuda.is_available() else None
                )
    
    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Synchronize metrics across all processes.
        
        Args:
            metrics: Local metrics dictionary
            
        Returns:
            Synchronized metrics (averaged across processes)
        """
        synced_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, dtype=torch.float32).to(self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                synced_metrics[key] = (tensor / self.world_size).item()
            else:
                synced_metrics[key] = value
        
        return synced_metrics
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode (distributed version).
        
        Returns:
            Dictionary of episode metrics
        """
        # Use parent's train_episode implementation
        return super().train_episode()
    
    @staticmethod
    def spawn_distributed_training(
        train_fn,
        world_size: int,
        env_fn,
        agents_fn,
        config: Dict[str, Any]
    ) -> None:
        """
        Spawn distributed training processes.
        
        Args:
            train_fn: Training function to run
            world_size: Number of processes
            env_fn: Function to create environment
            agents_fn: Function to create agents
            config: Configuration dictionary
        """
        mp.spawn(
            train_fn,
            args=(world_size, env_fn, agents_fn, config),
            nprocs=world_size,
            join=True
        )


def distributed_train_worker(
    rank: int,
    world_size: int,
    env_fn,
    agents_fn,
    config: Dict[str, Any]
) -> None:
    """
    Worker function for distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        env_fn: Function to create environment
        agents_fn: Function to create agents
        config: Configuration dictionary
    """
    # Update config with distributed parameters
    config["rank"] = rank
    config["world_size"] = world_size
    
    # Create environment and agents
    env = env_fn()
    agents = agents_fn()
    
    # Create trainer
    trainer = DistributedTrainer(env, agents, config)
    
    # Train
    trainer.train()


class AsyncDistributedTrainer(DistributedTrainer):
    """
    Asynchronous distributed trainer using parameter server architecture.
    
    Implements asynchronous training where workers independently update
    a shared parameter server.
    """
    
    def __init__(
        self,
        env: BaseEnv,
        agents: List[BaseAgent],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize asynchronous distributed trainer.
        
        Args:
            env: Training environment
            agents: List of agents to train
            config: Configuration dictionary
        """
        super().__init__(env, agents, config)
        
        self.update_frequency = self.config.get("async_update_frequency", 10)
        self.is_parameter_server = self.rank == 0
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode with asynchronous updates.
        
        Returns:
            Dictionary of episode metrics
        """
        # Pull parameters from parameter server
        if not self.is_parameter_server:
            self._pull_parameters()
        
        # Train episode locally
        metrics = super().train_episode()
        
        # Push gradients to parameter server
        if not self.is_parameter_server and self.current_episode % self.update_frequency == 0:
            self._push_gradients()
        
        return metrics
    
    def _pull_parameters(self) -> None:
        """Pull parameters from parameter server."""
        # TODO: Implement parameter pulling logic
        pass
    
    def _push_gradients(self) -> None:
        """Push gradients to parameter server."""
        # TODO: Implement gradient pushing logic
        pass

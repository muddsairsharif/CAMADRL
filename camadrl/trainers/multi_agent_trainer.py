"""
Multi-Agent Trainer implementation.

Implements coordinated training for multiple agents with
shared or independent learning strategies.
"""

from typing import Any, Dict, List
import numpy as np
import torch
from tqdm import tqdm

from camadrl.trainers.base_trainer import BaseTrainer
from camadrl.utils import ReplayBuffer


class MultiAgentTrainer(BaseTrainer):
    """
    Multi-Agent Trainer for coordinated learning.
    
    Supports both independent learning and centralized training with
    decentralized execution (CTDE) paradigms.
    """
    
    def __init__(
        self,
        agents: List[Any],
        env: Any,
        config: Dict[str, Any] = None
    ):
        """
        Initialize Multi-Agent Trainer.
        
        Args:
            agents: List of agents to train
            env: Environment for training
            config: Configuration dictionary with keys:
                - num_episodes: Number of training episodes
                - batch_size: Batch size for training
                - buffer_size: Size of replay buffer
                - learning_starts: Steps before learning starts
                - update_freq: Frequency of network updates
                - ctde: Use centralized training (default: False)
        """
        super().__init__(agents, env, config)
        
        self.batch_size = self.config.get("batch_size", 64)
        self.buffer_size = self.config.get("buffer_size", 100000)
        self.learning_starts = self.config.get("learning_starts", 1000)
        self.update_freq = self.config.get("update_freq", 1)
        self.ctde = self.config.get("ctde", False)
        
        # Create replay buffers for each agent
        self.replay_buffers = [
            ReplayBuffer(
                self.buffer_size,
                self.env.observation_space.shape[0],
                1  # Discrete actions
            )
            for _ in self.agents
        ]
        
        self.total_steps = 0
        
    def train(self) -> Dict[str, Any]:
        """Execute the training loop."""
        print(f"Starting training for {self.num_episodes} episodes...")
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            episode_stats = self.train_episode()
            
            # Log episode statistics
            if episode % self.log_freq == 0:
                print(f"\nEpisode {episode}: "
                      f"Reward={episode_stats['episode_reward']:.2f}, "
                      f"Length={episode_stats['episode_length']}")
            
            # Evaluate periodically
            if episode % self.eval_freq == 0 and episode > 0:
                eval_stats = self.evaluate(num_episodes=5)
                print(f"\nEvaluation: Mean Reward={eval_stats['mean_reward']:.2f}")
            
            # Save checkpoint
            if episode % self.save_freq == 0 and episode > 0:
                self.save_checkpoint(f"checkpoint_episode_{episode}")
        
        return self.get_statistics()
    
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        done = False
        
        while not done:
            # Select actions for all agents
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(obs, explore=True)
                actions.append(action)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(actions[0])
            done = terminated or truncated
            
            # Store transitions in replay buffers
            agent_idx = info.get("agent_idx", 0)
            self.replay_buffers[agent_idx].add(
                obs,
                actions[agent_idx],
                reward,
                next_obs,
                float(done)
            )
            
            # Update agents
            if self.total_steps > self.learning_starts and self.total_steps % self.update_freq == 0:
                for i, agent in enumerate(self.agents):
                    if len(self.replay_buffers[i]) >= self.batch_size:
                        batch = self.replay_buffers[i].sample(self.batch_size)
                        
                        # Convert to tensors
                        batch_tensors = {
                            "states": torch.FloatTensor(batch["observations"]),
                            "actions": torch.LongTensor(batch["actions"]),
                            "rewards": torch.FloatTensor(batch["rewards"]),
                            "next_states": torch.FloatTensor(batch["next_observations"]),
                            "dones": torch.FloatTensor(batch["dones"])
                        }
                        
                        # Update agent
                        metrics = agent.update(batch_tensors)
                        episode_losses.append(metrics.get("loss", 0))
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "mean_loss": np.mean(episode_losses) if episode_losses else 0,
            "total_steps": self.total_steps
        }
    
    def train_ctde(self) -> Dict[str, Any]:
        """
        Train using Centralized Training Decentralized Execution.
        
        Returns:
            Dictionary containing training results
        """
        print(f"Starting CTDE training for {self.num_episodes} episodes...")
        
        for episode in tqdm(range(self.num_episodes), desc="CTDE Training"):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Collect trajectory
            states = []
            actions = []
            rewards = []
            
            done = False
            while not done:
                # Collect states from all agents
                agent_states = [obs for _ in self.agents]
                states.append(agent_states)
                
                # Select actions
                agent_actions = []
                for agent in self.agents:
                    action = agent.select_action(obs, explore=True)
                    agent_actions.append(action)
                actions.append(agent_actions)
                
                # Take action
                next_obs, reward, terminated, truncated, _ = self.env.step(agent_actions[0])
                done = terminated or truncated
                rewards.append(reward)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
            
            # Centralized update using collected trajectory
            if len(states) > 0:
                self._update_ctde(states, actions, rewards)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if episode % self.log_freq == 0:
                print(f"\nEpisode {episode}: Reward={episode_reward:.2f}")
        
        return self.get_statistics()
    
    def _update_ctde(
        self,
        states: List[List[np.ndarray]],
        actions: List[List[int]],
        rewards: List[float]
    ) -> None:
        """
        Perform centralized update with decentralized execution.
        
        Args:
            states: List of state observations for all agents at each timestep
            actions: List of actions for all agents at each timestep
            rewards: List of rewards at each timestep
        """
        # Convert to tensors
        # This is a simplified version - in practice, you'd implement
        # a more sophisticated centralized critic
        for i, agent in enumerate(self.agents):
            # Create training batch for each agent
            agent_states = [s[i] for s in states[:-1]]
            agent_actions = [a[i] for a in actions]
            next_states = [s[i] for s in states[1:]]
            
            if len(agent_states) >= self.batch_size:
                batch_tensors = {
                    "states": torch.FloatTensor(agent_states[:self.batch_size]),
                    "actions": torch.LongTensor(agent_actions[:self.batch_size]),
                    "rewards": torch.FloatTensor(rewards[:self.batch_size]),
                    "next_states": torch.FloatTensor(next_states[:self.batch_size]),
                    "dones": torch.zeros(self.batch_size)
                }
                
                agent.update(batch_tensors)

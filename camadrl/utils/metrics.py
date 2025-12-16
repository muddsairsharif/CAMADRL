"""
Performance metrics calculator for reinforcement learning.

This module implements various metrics for evaluating agent performance
and training progress.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from collections import deque


class Metrics:
    """
    Calculator for reinforcement learning metrics.
    
    Provides methods to calculate and track various performance metrics
    such as returns, success rates, and convergence measures.
    
    Attributes:
        window_size: Size of moving average window
        episode_rewards: History of episode rewards
        episode_lengths: History of episode lengths
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics calculator.
        
        Args:
            window_size: Size of window for moving averages
        """
        self.window_size = window_size
        
        # History storage
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_successes = deque(maxlen=window_size)
        
        # Cumulative statistics
        self.total_episodes = 0
        self.total_steps = 0
    
    def update(
        self,
        episode_reward: float,
        episode_length: int,
        success: bool = False
    ) -> None:
        """
        Update metrics with episode data.
        
        Args:
            episode_reward: Total reward for the episode
            episode_length: Length of the episode
            success: Whether the episode was successful
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_successes.append(float(success))
        
        self.total_episodes += 1
        self.total_steps += episode_length
    
    def get_mean_reward(self) -> float:
        """
        Get mean episode reward over recent episodes.
        
        Returns:
            Mean reward
        """
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards)
    
    def get_std_reward(self) -> float:
        """
        Get standard deviation of episode rewards.
        
        Returns:
            Standard deviation
        """
        if len(self.episode_rewards) < 2:
            return 0.0
        return np.std(self.episode_rewards)
    
    def get_mean_length(self) -> float:
        """
        Get mean episode length.
        
        Returns:
            Mean episode length
        """
        if not self.episode_lengths:
            return 0.0
        return np.mean(self.episode_lengths)
    
    def get_success_rate(self) -> float:
        """
        Get success rate over recent episodes.
        
        Returns:
            Success rate (0 to 1)
        """
        if not self.episode_successes:
            return 0.0
        return np.mean(self.episode_successes)
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all available metrics.
        
        Returns:
            Dictionary of all metrics
        """
        return {
            "mean_reward": self.get_mean_reward(),
            "std_reward": self.get_std_reward(),
            "mean_length": self.get_mean_length(),
            "success_rate": self.get_success_rate(),
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_successes.clear()
        self.total_episodes = 0
        self.total_steps = 0


class MultiAgentMetrics:
    """
    Metrics calculator for multi-agent scenarios.
    
    Extends basic metrics with multi-agent specific measurements
    like coordination and fairness.
    """
    
    def __init__(self, num_agents: int, window_size: int = 100):
        """
        Initialize multi-agent metrics calculator.
        
        Args:
            num_agents: Number of agents
            window_size: Size of window for moving averages
        """
        self.num_agents = num_agents
        self.window_size = window_size
        
        # Per-agent metrics
        self.agent_metrics = [Metrics(window_size) for _ in range(num_agents)]
        
        # Joint metrics
        self.joint_rewards = deque(maxlen=window_size)
        self.coordination_scores = deque(maxlen=window_size)
    
    def update(
        self,
        agent_rewards: List[float],
        agent_lengths: List[int],
        agent_successes: Optional[List[bool]] = None
    ) -> None:
        """
        Update metrics with multi-agent episode data.
        
        Args:
            agent_rewards: List of rewards for each agent
            agent_lengths: List of episode lengths for each agent
            agent_successes: Optional list of success flags
        """
        if agent_successes is None:
            agent_successes = [False] * self.num_agents
        
        # Update per-agent metrics
        for i in range(self.num_agents):
            self.agent_metrics[i].update(
                agent_rewards[i],
                agent_lengths[i],
                agent_successes[i]
            )
        
        # Update joint metrics
        joint_reward = np.sum(agent_rewards)
        self.joint_rewards.append(joint_reward)
        
        # Calculate coordination score (inverse of reward variance)
        reward_variance = np.var(agent_rewards)
        coordination_score = 1.0 / (1.0 + reward_variance)
        self.coordination_scores.append(coordination_score)
    
    def get_coordination_score(self) -> float:
        """
        Get mean coordination score.
        
        Returns:
            Coordination score
        """
        if not self.coordination_scores:
            return 0.0
        return np.mean(self.coordination_scores)
    
    def get_fairness_score(self) -> float:
        """
        Get fairness score based on reward distribution.
        
        Returns:
            Fairness score (higher is more fair)
        """
        agent_mean_rewards = [m.get_mean_reward() for m in self.agent_metrics]
        
        if not agent_mean_rewards:
            return 1.0
        
        # Gini coefficient for fairness
        sorted_rewards = sorted(agent_mean_rewards)
        n = len(sorted_rewards)
        
        if n == 0 or np.sum(sorted_rewards) == 0:
            return 1.0
        
        cumsum = np.cumsum(sorted_rewards)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_rewards)) / (n * np.sum(sorted_rewards)) - (n + 1) / n
        
        # Convert to fairness score (1 - gini)
        fairness = 1.0 - gini
        
        return fairness
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all multi-agent metrics.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            "joint_reward_mean": np.mean(self.joint_rewards) if self.joint_rewards else 0.0,
            "coordination_score": self.get_coordination_score(),
            "fairness_score": self.get_fairness_score(),
        }
        
        # Add per-agent metrics
        for i in range(self.num_agents):
            agent_metrics = self.agent_metrics[i].get_all_metrics()
            for key, value in agent_metrics.items():
                metrics[f"agent_{i}_{key}"] = value
        
        return metrics


def calculate_discounted_return(
    rewards: List[float],
    gamma: float = 0.99
) -> float:
    """
    Calculate discounted return from reward sequence.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        Discounted return
    """
    discounted_return = 0.0
    for i, reward in enumerate(rewards):
        discounted_return += (gamma ** i) * reward
    
    return discounted_return


def calculate_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> np.ndarray:
    """
    Calculate Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Value estimate for next state
        gamma: Discount factor
        lambda_: GAE parameter
        
    Returns:
        Array of advantage estimates
    """
    advantages = np.zeros(len(rewards))
    last_advantage = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = delta + gamma * lambda_ * last_advantage
        last_advantage = advantages[t]
    
    return advantages


def calculate_success_rate(
    rewards: List[float],
    threshold: float = 0.0
) -> float:
    """
    Calculate success rate based on reward threshold.
    
    Args:
        rewards: List of episode rewards
        threshold: Reward threshold for success
        
    Returns:
        Success rate (0 to 1)
    """
    if not rewards:
        return 0.0
    
    successes = sum(1 for r in rewards if r >= threshold)
    return successes / len(rewards)

"""
Metrics implementation for evaluation and analysis.

Provides various metrics for evaluating agent performance
and coordination in multi-agent scenarios.
"""

from typing import List, Dict, Any
import numpy as np


class Metrics:
    """
    Metrics calculator for multi-agent reinforcement learning.
    
    Computes various performance metrics for agents and coordination.
    """
    
    @staticmethod
    def compute_episode_metrics(
        rewards: List[float],
        episode_length: int
    ) -> Dict[str, float]:
        """
        Compute metrics for a single episode.
        
        Args:
            rewards: List of rewards received during episode
            episode_length: Length of the episode
            
        Returns:
            Dictionary of episode metrics
        """
        return {
            "episode_reward": sum(rewards),
            "episode_length": episode_length,
            "average_reward": np.mean(rewards) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "min_reward": min(rewards) if rewards else 0,
        }
    
    @staticmethod
    def compute_training_metrics(
        episode_rewards: List[float],
        window_size: int = 100
    ) -> Dict[str, float]:
        """
        Compute training metrics over multiple episodes.
        
        Args:
            episode_rewards: List of total rewards per episode
            window_size: Window size for moving average
            
        Returns:
            Dictionary of training metrics
        """
        if len(episode_rewards) == 0:
            return {}
        
        recent_rewards = episode_rewards[-window_size:]
        
        return {
            "mean_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "min_reward": np.min(recent_rewards),
            "max_reward": np.max(recent_rewards),
            "median_reward": np.median(recent_rewards),
        }
    
    @staticmethod
    def compute_coordination_metrics(
        agent_rewards: List[List[float]],
        joint_reward: float
    ) -> Dict[str, float]:
        """
        Compute coordination metrics for multi-agent systems.
        
        Args:
            agent_rewards: List of reward lists for each agent
            joint_reward: Joint reward for the team
            
        Returns:
            Dictionary of coordination metrics
        """
        num_agents = len(agent_rewards)
        individual_rewards = [sum(rewards) for rewards in agent_rewards]
        
        # Coordination score: how well agents coordinate vs individual performance
        avg_individual = np.mean(individual_rewards)
        coordination_score = joint_reward / (avg_individual * num_agents) if avg_individual > 0 else 0
        
        # Fairness: variance in individual rewards (lower is more fair)
        fairness = 1.0 / (1.0 + np.std(individual_rewards))
        
        return {
            "coordination_score": coordination_score,
            "fairness": fairness,
            "individual_reward_mean": avg_individual,
            "individual_reward_std": np.std(individual_rewards),
            "joint_reward": joint_reward,
        }
    
    @staticmethod
    def compute_success_rate(
        successes: List[bool],
        window_size: int = 100
    ) -> float:
        """
        Compute success rate over recent episodes.
        
        Args:
            successes: List of success indicators
            window_size: Window size for calculation
            
        Returns:
            Success rate (0 to 1)
        """
        if len(successes) == 0:
            return 0.0
        
        recent = successes[-window_size:]
        return sum(recent) / len(recent)
    
    @staticmethod
    def compute_efficiency_metrics(
        episode_lengths: List[int],
        episode_rewards: List[float]
    ) -> Dict[str, float]:
        """
        Compute efficiency metrics.
        
        Args:
            episode_lengths: List of episode lengths
            episode_rewards: List of episode rewards
            
        Returns:
            Dictionary of efficiency metrics
        """
        if len(episode_lengths) == 0 or len(episode_rewards) == 0:
            return {}
        
        # Reward per step
        rewards_per_step = [
            r / l if l > 0 else 0
            for r, l in zip(episode_rewards, episode_lengths)
        ]
        
        return {
            "mean_episode_length": np.mean(episode_lengths),
            "mean_reward_per_step": np.mean(rewards_per_step),
            "efficiency_score": np.mean(rewards_per_step) / np.mean(episode_lengths) if np.mean(episode_lengths) > 0 else 0,
        }
    
    @staticmethod
    def compute_convergence_metrics(
        episode_rewards: List[float],
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Compute convergence metrics.
        
        Args:
            episode_rewards: List of episode rewards over time
            threshold: Reward threshold for convergence (if None, uses 90% of max)
            
        Returns:
            Dictionary of convergence metrics
        """
        if len(episode_rewards) == 0:
            return {}
        
        if threshold is None:
            threshold = 0.9 * max(episode_rewards)
        
        # Find first episode where performance exceeds threshold
        convergence_episode = None
        for i, reward in enumerate(episode_rewards):
            if reward >= threshold:
                convergence_episode = i
                break
        
        # Compute moving average for smoothing
        window_size = min(50, len(episode_rewards) // 10)
        if window_size > 0:
            moving_avg = np.convolve(
                episode_rewards,
                np.ones(window_size) / window_size,
                mode='valid'
            )
        else:
            moving_avg = episode_rewards
        
        return {
            "convergence_episode": convergence_episode,
            "converged": convergence_episode is not None,
            "threshold": threshold,
            "final_performance": np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards),
        }
    
    @staticmethod
    def compute_stability_metrics(
        episode_rewards: List[float],
        window_size: int = 50
    ) -> Dict[str, float]:
        """
        Compute stability metrics.
        
        Args:
            episode_rewards: List of episode rewards
            window_size: Window size for stability calculation
            
        Returns:
            Dictionary of stability metrics
        """
        if len(episode_rewards) < window_size:
            return {}
        
        # Compute rolling standard deviation
        rolling_stds = []
        for i in range(len(episode_rewards) - window_size + 1):
            window = episode_rewards[i:i + window_size]
            rolling_stds.append(np.std(window))
        
        return {
            "mean_rolling_std": np.mean(rolling_stds),
            "stability_score": 1.0 / (1.0 + np.mean(rolling_stds)),
            "final_stability": 1.0 / (1.0 + rolling_stds[-1]) if rolling_stds else 0,
        }


class ChargingMetrics:
    """
    Specialized metrics for EV charging coordination scenarios.
    """
    
    @staticmethod
    def compute_charging_efficiency(
        battery_levels: List[List[float]],
        charging_costs: List[float]
    ) -> Dict[str, float]:
        """
        Compute charging efficiency metrics.
        
        Args:
            battery_levels: Battery levels over time for each agent
            charging_costs: Costs incurred for charging
            
        Returns:
            Dictionary of charging efficiency metrics
        """
        # Average final battery level
        final_battery_levels = [levels[-1] for levels in battery_levels if levels]
        avg_final_battery = np.mean(final_battery_levels) if final_battery_levels else 0
        
        # Total charging cost
        total_cost = sum(charging_costs)
        
        # Cost per unit charge
        total_charge = sum([
            levels[-1] - levels[0] for levels in battery_levels if len(levels) > 1
        ])
        cost_per_charge = total_cost / total_charge if total_charge > 0 else float('inf')
        
        return {
            "avg_final_battery": avg_final_battery,
            "total_charging_cost": total_cost,
            "cost_per_charge": cost_per_charge,
            "charging_efficiency": avg_final_battery / total_cost if total_cost > 0 else 0,
        }
    
    @staticmethod
    def compute_grid_impact(
        charging_loads: List[float],
        peak_threshold: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute grid impact metrics.
        
        Args:
            charging_loads: Charging load at each timestep
            peak_threshold: Threshold for peak load
            
        Returns:
            Dictionary of grid impact metrics
        """
        peak_load = max(charging_loads) if charging_loads else 0
        avg_load = np.mean(charging_loads) if charging_loads else 0
        
        # Peak-to-average ratio
        peak_to_avg = peak_load / avg_load if avg_load > 0 else 0
        
        # Number of peak violations
        peak_violations = sum(1 for load in charging_loads if load > peak_threshold)
        
        return {
            "peak_load": peak_load,
            "avg_load": avg_load,
            "peak_to_avg_ratio": peak_to_avg,
            "peak_violations": peak_violations,
            "grid_stress": peak_to_avg * (1 + peak_violations / len(charging_loads)) if charging_loads else 0,
        }

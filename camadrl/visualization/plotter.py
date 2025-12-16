"""
Plotter for training metrics and performance visualization.

Provides utilities for creating plots of training progress,
rewards, losses, and other metrics.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    """
    Plotter for training metrics and visualization.
    
    Creates various plots for analyzing agent performance
    and training progress.
    """
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize Plotter.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        
        sns.set_palette("husl")
        self.figures = []
    
    def plot_rewards(
        self,
        episode_rewards: List[float],
        window_size: int = 100,
        title: str = "Episode Rewards",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot episode rewards with moving average.
        
        Args:
            episode_rewards: List of episode rewards
            window_size: Window size for moving average
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(episode_rewards))
        ax.plot(episodes, episode_rewards, alpha=0.3, label='Raw')
        
        # Moving average
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(
                episode_rewards,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax.plot(
                range(window_size - 1, len(episode_rewards)),
                moving_avg,
                label=f'Moving Avg (window={window_size})',
                linewidth=2
            )
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_multi_agent_rewards(
        self,
        agent_rewards: Dict[int, List[float]],
        title: str = "Multi-Agent Rewards",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot rewards for multiple agents.
        
        Args:
            agent_rewards: Dictionary mapping agent IDs to reward lists
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for agent_id, rewards in agent_rewards.items():
            episodes = range(len(rewards))
            ax.plot(episodes, rewards, label=f'Agent {agent_id}', alpha=0.7)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_losses(
        self,
        losses: List[float],
        title: str = "Training Loss",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training losses.
        
        Args:
            losses: List of loss values
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = range(len(losses))
        ax.plot(steps, losses, alpha=0.7)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Metrics Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot multiple metrics for comparison.
        
        Args:
            metrics: Dictionary mapping metric names to value lists
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            steps = range(len(values))
            ax.plot(steps, values)
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over Time')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, y=1.0)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_convergence(
        self,
        episode_rewards: List[float],
        threshold: float = None,
        title: str = "Convergence Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot convergence analysis.
        
        Args:
            episode_rewards: List of episode rewards
            threshold: Convergence threshold
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(episode_rewards))
        ax.plot(episodes, episode_rewards, alpha=0.5, label='Episode Rewards')
        
        # Moving average
        window_size = min(50, len(episode_rewards) // 10)
        if window_size > 0:
            moving_avg = np.convolve(
                episode_rewards,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax.plot(
                range(window_size - 1, len(episode_rewards)),
                moving_avg,
                label='Moving Average',
                linewidth=2
            )
        
        # Threshold line
        if threshold is None:
            threshold = 0.9 * max(episode_rewards)
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_distribution(
        self,
        values: List[float],
        title: str = "Value Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of values.
        
        Args:
            values: List of values
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(values, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Histogram')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(values, vert=True)
        ax2.set_ylabel('Value')
        ax2.set_title('Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def close_all(self) -> None:
        """Close all open figures."""
        plt.close('all')
        self.figures = []

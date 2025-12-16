"""
General plotting utilities for CAMADRL framework.

This module provides various plotting functions for visualizing
training progress, metrics, and results.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    """
    General plotter for training metrics and results.
    
    Provides various plotting functions for visualizing agent performance,
    training progress, and analysis results.
    """
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize plotter.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default style if specified style not available
        
        self.colors = sns.color_palette("husl", 10)
    
    def plot_training_curve(
        self,
        episodes: List[int],
        rewards: List[float],
        title: str = "Training Curve",
        xlabel: str = "Episode",
        ylabel: str = "Reward",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training curve showing reward over episodes.
        
        Args:
            episodes: List of episode numbers
            rewards: List of rewards
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(episodes, rewards, linewidth=2, alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)
    
    def plot_multi_agent_comparison(
        self,
        episodes: List[int],
        agent_rewards: Dict[str, List[float]],
        title: str = "Multi-Agent Performance",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of multiple agents' performance.
        
        Args:
            episodes: List of episode numbers
            agent_rewards: Dictionary mapping agent names to reward lists
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (agent_name, rewards) in enumerate(agent_rewards.items()):
            ax.plot(episodes, rewards, label=agent_name, 
                   color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)
    
    def plot_metrics_dashboard(
        self,
        metrics: Dict[str, List[Tuple[int, float]]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot dashboard with multiple metrics.
        
        Args:
            metrics: Dictionary mapping metric names to (episode, value) tuples
            save_path: Path to save figure
        """
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            episodes, vals = zip(*values)
            
            ax.plot(episodes, vals, linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)

"""Trajectory visualization for multi-agent systems."""
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryVisualizer:
    """Visualize agent trajectories in environments."""
    
    def __init__(self):
        """Initialize trajectory visualizer."""
        self.trajectories = []
    
    def plot_trajectories(
        self,
        trajectories: List[np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Plot agent trajectories."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i, traj in enumerate(trajectories):
            ax.plot(traj[:, 0], traj[:, 1], label=f'Agent {i}', linewidth=2)
            ax.scatter(traj[0, 0], traj[0, 1], marker='o', s=100, label=f'Start {i}')
            ax.scatter(traj[-1, 0], traj[-1, 1], marker='*', s=200, label=f'End {i}')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Agent Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)

"""
Trajectory Visualizer for agent paths and movements.

Visualizes agent trajectories, positions, and movements
in environments for analysis and debugging.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import seaborn as sns


class TrajectoryVisualizer:
    """
    Visualizer for agent trajectories and movements.
    
    Creates visualizations of agent paths, positions, and
    interactions in the environment.
    """
    
    def __init__(self):
        """Initialize Trajectory Visualizer."""
        sns.set_style("whitegrid")
        self.figures = []
    
    def plot_trajectory(
        self,
        positions: List[Tuple[float, float]],
        agent_id: int = 0,
        title: str = "Agent Trajectory",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot a single agent's trajectory.
        
        Args:
            positions: List of (x, y) positions
            agent_id: Agent identifier
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        positions = np.array(positions)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, linewidth=2)
        ax.scatter(positions[:, 0], positions[:, 1], c=range(len(positions)),
                  cmap='viridis', s=50, alpha=0.6, label='Positions')
        
        # Mark start and end
        ax.scatter(positions[0, 0], positions[0, 1], c='green', s=200,
                  marker='o', edgecolors='black', linewidths=2, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200,
                  marker='X', edgecolors='black', linewidths=2, label='End')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'{title} - Agent {agent_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_multi_agent_trajectories(
        self,
        trajectories: Dict[int, List[Tuple[float, float]]],
        title: str = "Multi-Agent Trajectories",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot trajectories for multiple agents.
        
        Args:
            trajectories: Dictionary mapping agent IDs to position lists
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
        
        for (agent_id, positions), color in zip(trajectories.items(), colors):
            positions = np.array(positions)
            
            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], '-', 
                   color=color, alpha=0.5, linewidth=2,
                   label=f'Agent {agent_id}')
            
            # Mark start
            ax.scatter(positions[0, 0], positions[0, 1],
                      c=[color], s=150, marker='o',
                      edgecolors='black', linewidths=2)
            
            # Mark end
            ax.scatter(positions[-1, 0], positions[-1, 1],
                      c=[color], s=150, marker='X',
                      edgecolors='black', linewidths=2)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def plot_trajectory_with_environment(
        self,
        positions: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float]] = None,
        charging_stations: List[Tuple[float, float]] = None,
        grid_size: Tuple[float, float] = (10, 10),
        title: str = "Trajectory with Environment",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot trajectory with environment features.
        
        Args:
            positions: List of (x, y) positions
            obstacles: List of (x, y, radius) for circular obstacles
            charging_stations: List of (x, y) for charging stations
            grid_size: (width, height) of environment
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Draw obstacles
        if obstacles:
            for x, y, radius in obstacles:
                circle = Circle((x, y), radius, color='gray', alpha=0.5)
                ax.add_patch(circle)
        
        # Draw charging stations
        if charging_stations:
            stations = np.array(charging_stations)
            ax.scatter(stations[:, 0], stations[:, 1],
                      c='orange', s=300, marker='s',
                      edgecolors='black', linewidths=2,
                      label='Charging Stations', zorder=5)
        
        # Draw trajectory
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1],
               'b-', alpha=0.6, linewidth=2)
        ax.scatter(positions[:, 0], positions[:, 1],
                  c=range(len(positions)), cmap='viridis',
                  s=50, alpha=0.6)
        
        # Mark start and end
        ax.scatter(positions[0, 0], positions[0, 1],
                  c='green', s=200, marker='o',
                  edgecolors='black', linewidths=2,
                  label='Start', zorder=10)
        ax.scatter(positions[-1, 0], positions[-1, 1],
                  c='red', s=200, marker='X',
                  edgecolors='black', linewidths=2,
                  label='End', zorder=10)
        
        ax.set_xlim(0, grid_size[0])
        ax.set_ylim(0, grid_size[1])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def create_animation(
        self,
        trajectories: Dict[int, List[Tuple[float, float]]],
        grid_size: Tuple[float, float] = (10, 10),
        interval: int = 100,
        save_path: Optional[str] = None
    ) -> animation.FuncAnimation:
        """
        Create animation of agent movements.
        
        Args:
            trajectories: Dictionary mapping agent IDs to position lists
            grid_size: (width, height) of environment
            interval: Milliseconds between frames
            save_path: Path to save animation (e.g., 'animation.gif')
            
        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Find maximum trajectory length
        max_length = max(len(traj) for traj in trajectories.values())
        
        # Initialize plots
        colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
        lines = []
        points = []
        
        for (agent_id, _), color in zip(trajectories.items(), colors):
            line, = ax.plot([], [], '-', color=color, alpha=0.5, linewidth=2,
                          label=f'Agent {agent_id}')
            point, = ax.plot([], [], 'o', color=color, markersize=10)
            lines.append(line)
            points.append(point)
        
        ax.set_xlim(0, grid_size[0])
        ax.set_ylim(0, grid_size[1])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Agent Movement Animation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            return lines + points
        
        def animate(frame):
            for i, (agent_id, positions) in enumerate(trajectories.items()):
                if frame < len(positions):
                    # Update line
                    pos_array = np.array(positions[:frame+1])
                    lines[i].set_data(pos_array[:, 0], pos_array[:, 1])
                    
                    # Update point
                    points[i].set_data([positions[frame][0]], [positions[frame][1]])
            
            return lines + points
        
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=max_length, interval=interval,
            blit=True, repeat=True
        )
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=10)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=10)
        
        return anim
    
    def close_all(self) -> None:
        """Close all open figures."""
        plt.close('all')
        self.figures = []

"""
Heatmap Generator for spatial analysis.

Generates heatmaps for analyzing agent behavior, resource utilization,
and spatial patterns in multi-agent environments.
"""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """
    Generator for heatmaps and spatial visualizations.
    
    Creates various heatmaps for analyzing spatial patterns,
    resource utilization, and agent density.
    """
    
    def __init__(self):
        """Initialize Heatmap Generator."""
        self.figures = []
    
    def generate_position_heatmap(
        self,
        positions: List[Tuple[float, float]],
        grid_size: Tuple[int, int] = (50, 50),
        bounds: Tuple[float, float, float, float] = (0, 10, 0, 10),
        title: str = "Position Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate heatmap of agent positions.
        
        Args:
            positions: List of (x, y) positions
            grid_size: (rows, cols) for heatmap grid
            bounds: (xmin, xmax, ymin, ymax) for the environment
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create 2D histogram
        xmin, xmax, ymin, ymax = bounds
        positions = np.array(positions)
        
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=grid_size,
            range=[[xmin, xmax], [ymin, ymax]]
        )
        
        # Plot heatmap
        im = ax.imshow(
            heatmap.T,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            cmap='hot',
            interpolation='bilinear',
            aspect='auto'
        )
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visit Count')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def generate_density_heatmap(
        self,
        positions: List[Tuple[float, float]],
        sigma: float = 1.0,
        grid_size: Tuple[int, int] = (100, 100),
        bounds: Tuple[float, float, float, float] = (0, 10, 0, 10),
        title: str = "Density Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate smoothed density heatmap.
        
        Args:
            positions: List of (x, y) positions
            sigma: Gaussian smoothing parameter
            grid_size: (rows, cols) for heatmap grid
            bounds: (xmin, xmax, ymin, ymax) for the environment
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create 2D histogram
        xmin, xmax, ymin, ymax = bounds
        positions = np.array(positions)
        
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=grid_size,
            range=[[xmin, xmax], [ymin, ymax]]
        )
        
        # Apply Gaussian smoothing
        heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)
        
        # Plot heatmap
        im = ax.imshow(
            heatmap_smooth.T,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            cmap='YlOrRd',
            interpolation='bilinear',
            aspect='auto'
        )
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def generate_resource_utilization_heatmap(
        self,
        resource_positions: List[Tuple[float, float]],
        utilization: List[float],
        title: str = "Resource Utilization",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate heatmap of resource utilization.
        
        Args:
            resource_positions: List of (x, y) positions of resources
            utilization: List of utilization values for each resource
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        positions = np.array(resource_positions)
        utilization = np.array(utilization)
        
        # Scatter plot with size/color based on utilization
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1],
            c=utilization,
            s=utilization * 500,
            cmap='RdYlGn',
            alpha=0.6,
            edgecolors='black',
            linewidths=2
        )
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Utilization')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def generate_multi_agent_density_heatmap(
        self,
        agent_positions: dict,
        grid_size: Tuple[int, int] = (50, 50),
        bounds: Tuple[float, float, float, float] = (0, 10, 0, 10),
        title: str = "Multi-Agent Density",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate combined density heatmap for multiple agents.
        
        Args:
            agent_positions: Dictionary mapping agent IDs to position lists
            grid_size: (rows, cols) for heatmap grid
            bounds: (xmin, xmax, ymin, ymax) for the environment
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_agents = len(agent_positions)
        fig, axes = plt.subplots(1, num_agents + 1, figsize=(5 * (num_agents + 1), 4))
        
        if num_agents == 1:
            axes = [axes]
        
        xmin, xmax, ymin, ymax = bounds
        combined_heatmap = np.zeros(grid_size)
        
        # Generate individual heatmaps
        for idx, (agent_id, positions) in enumerate(agent_positions.items()):
            positions = np.array(positions)
            
            heatmap, _, _ = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=grid_size,
                range=[[xmin, xmax], [ymin, ymax]]
            )
            
            combined_heatmap += heatmap
            
            # Plot individual agent heatmap
            im = axes[idx].imshow(
                heatmap.T,
                origin='lower',
                extent=[xmin, xmax, ymin, ymax],
                cmap='hot',
                interpolation='bilinear',
                aspect='auto'
            )
            axes[idx].set_xlabel('X Position')
            axes[idx].set_ylabel('Y Position')
            axes[idx].set_title(f'Agent {agent_id}')
            plt.colorbar(im, ax=axes[idx])
        
        # Plot combined heatmap
        im = axes[-1].imshow(
            combined_heatmap.T,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            cmap='hot',
            interpolation='bilinear',
            aspect='auto'
        )
        axes[-1].set_xlabel('X Position')
        axes[-1].set_ylabel('Y Position')
        axes[-1].set_title('Combined')
        plt.colorbar(im, ax=axes[-1])
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def generate_correlation_matrix(
        self,
        data: dict,
        title: str = "Correlation Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate correlation matrix heatmap.
        
        Args:
            data: Dictionary of metric names to value arrays
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to DataFrame-like structure
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def close_all(self) -> None:
        """Close all open figures."""
        plt.close('all')
        self.figures = []

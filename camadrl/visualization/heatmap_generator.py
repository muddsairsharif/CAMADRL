"""Heatmap generation for analysis and visualization."""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HeatmapGenerator:
    """Generate heatmaps for analysis."""
    
    def __init__(self):
        """Initialize heatmap generator."""
        pass
    
    def generate_value_heatmap(
        self,
        values: np.ndarray,
        title: str = "Value Heatmap",
        save_path: Optional[str] = None
    ) -> None:
        """Generate heatmap of values."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(values, annot=False, cmap='viridis', ax=ax)
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)
    
    def generate_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Generate attention weights heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(attention_weights, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Attention Weights')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)

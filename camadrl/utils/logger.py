"""
Logger for tracking training progress and metrics.

This module implements logging functionality for tracking training progress,
metrics, and creating visualizations.
"""

from typing import Any, Dict, List, Optional
import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger for training progress and metrics.
    
    Provides functionality for logging metrics, saving checkpoints,
    and creating visualizations using TensorBoard.
    
    Attributes:
        log_dir: Directory for saving logs
        writer: TensorBoard SummaryWriter
        metrics_history: Dictionary storing metric histories
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for saving logs
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.use_tensorboard = self.config.get("use_tensorboard", True)
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
        
        # Store metrics history
        self.metrics_history = {}
        
        # Save config
        self._save_config()
        
        print(f"Logger initialized. Logs will be saved to: {self.log_dir}")
    
    def log(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = ""
    ) -> None:
        """
        Log metrics at a specific step.
        
        Args:
            step: Training step or episode number
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for metric names
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                full_key = f"{prefix}/{key}" if prefix else key
                
                # Store in history
                if full_key not in self.metrics_history:
                    self.metrics_history[full_key] = []
                self.metrics_history[full_key].append((step, value))
                
                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar(full_key, value, step)
    
    def log_histogram(
        self,
        step: int,
        name: str,
        values: np.ndarray
    ) -> None:
        """
        Log histogram of values.
        
        Args:
            step: Training step
            name: Name of the histogram
            values: Array of values
        """
        if self.writer is not None:
            self.writer.add_histogram(name, values, step)
    
    def log_image(
        self,
        step: int,
        name: str,
        image: np.ndarray
    ) -> None:
        """
        Log an image.
        
        Args:
            step: Training step
            name: Name of the image
            image: Image array (H, W, C) or (C, H, W)
        """
        if self.writer is not None:
            self.writer.add_image(name, image, step, dataformats='HWC')
    
    def log_text(
        self,
        step: int,
        name: str,
        text: str
    ) -> None:
        """
        Log text.
        
        Args:
            step: Training step
            name: Name of the text entry
            text: Text content
        """
        if self.writer is not None:
            self.writer.add_text(name, text, step)
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """
        Save metrics history to JSON file.
        
        Args:
            filename: Name of the output file
        """
        filepath = os.path.join(self.log_dir, filename)
        
        # Convert to serializable format
        serializable_history = {}
        for key, values in self.metrics_history.items():
            serializable_history[key] = [
                {"step": int(step), "value": float(value)}
                for step, value in values
            ]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")
    
    def plot_metrics(
        self,
        metrics: Optional[List[str]] = None,
        save: bool = True,
        show: bool = False
    ) -> None:
        """
        Plot metrics over time.
        
        Args:
            metrics: List of metric names to plot (plots all if None)
            save: Whether to save the plot
            show: Whether to display the plot
        """
        if metrics is None:
            metrics = list(self.metrics_history.keys())
        
        # Filter metrics that exist
        metrics = [m for m in metrics if m in self.metrics_history]
        
        if not metrics:
            print("No metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, metric_name in enumerate(metrics):
            ax = axes[idx]
            history = self.metrics_history[metric_name]
            steps, values = zip(*history)
            
            ax.plot(steps, values)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.log_dir, "metrics_plot.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Metrics plot saved to: {filepath}")
        
        if show:
            plt.show()
        
        plt.close(fig)
    
    def _save_config(self) -> None:
        """Save configuration to JSON file."""
        if self.config:
            filepath = os.path.join(self.log_dir, "config.json")
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def close(self) -> None:
        """Close the logger and save all data."""
        self.save_metrics()
        
        if self.writer is not None:
            self.writer.close()
        
        print(f"Logger closed. All data saved to: {self.log_dir}")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()


class ConsoleLogger:
    """
    Simple console logger for basic logging needs.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console logger.
        
        Args:
            verbose: Whether to print logs
        """
        self.verbose = verbose
        self.logs = []
    
    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level ('INFO', 'WARNING', 'ERROR')
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        self.logs.append(log_entry)
        
        if self.verbose:
            print(log_entry)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.log(message, "ERROR")
    
    def save(self, filepath: str) -> None:
        """
        Save logs to file.
        
        Args:
            filepath: Path to save logs
        """
        with open(filepath, 'w') as f:
            f.write('\n'.join(self.logs))

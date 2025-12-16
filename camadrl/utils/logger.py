"""
Logger implementation for training and evaluation.

Provides logging capabilities for tracking experiments,
metrics, and visualizations.
"""

import os
import json
from typing import Any, Dict, List
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger for tracking training metrics and experiments.
    
    Supports logging to console, files, and TensorBoard for
    comprehensive experiment tracking.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = None,
        use_tensorboard: bool = True
    ):
        """
        Initialize Logger.
        
        Args:
            log_dir: Directory for storing logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.log_dir = log_dir
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.experiment_dir)
        else:
            self.writer = None
        
        # Initialize metrics storage
        self.metrics = {}
        self.episode_logs = []
        
        # Log file
        self.log_file = os.path.join(self.experiment_dir, "training.log")
        
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value.
        
        Args:
            tag: Name of the metric
            value: Value to log
            step: Step number (episode or iteration)
        """
        if tag not in self.metrics:
            self.metrics[tag] = []
        
        self.metrics[tag].append({"step": step, "value": value})
        
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        """
        Log multiple scalar values.
        
        Args:
            tag: Group name for the metrics
            values: Dictionary of metric names and values
            step: Step number
        """
        if self.writer:
            self.writer.add_scalars(tag, values, step)
        
        for key, value in values.items():
            full_tag = f"{tag}/{key}"
            self.log_scalar(full_tag, value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """
        Log a histogram of values.
        
        Args:
            tag: Name of the histogram
            values: Array of values
            step: Step number
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]) -> None:
        """
        Log episode information.
        
        Args:
            episode: Episode number
            metrics: Dictionary of episode metrics
        """
        log_entry = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.episode_logs.append(log_entry)
        
        # Log to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log scalars to TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"episode/{key}", value, episode)
    
    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """
        Log text information.
        
        Args:
            tag: Tag for the text
            text: Text content
            step: Step number
        """
        if self.writer:
            self.writer.add_text(tag, text, step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        # Log as text to TensorBoard
        config_text = json.dumps(config, indent=2)
        self.log_text("config", config_text)
    
    def get_metrics(self, tag: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get logged metrics.
        
        Args:
            tag: Specific metric tag (if None, returns all metrics)
            
        Returns:
            Dictionary of metrics
        """
        if tag:
            return {tag: self.metrics.get(tag, [])}
        return self.metrics
    
    def get_episode_logs(self) -> List[Dict[str, Any]]:
        """
        Get episode logs.
        
        Returns:
            List of episode log entries
        """
        return self.episode_logs
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """
        Save metrics to file.
        
        Args:
            filename: Name of the file
        """
        filepath = os.path.join(self.experiment_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=4)
    
    def close(self) -> None:
        """Close the logger and clean up resources."""
        if self.writer:
            self.writer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ConsoleLogger:
    """
    Simple console logger for quick debugging.
    """
    
    def __init__(self, prefix: str = ""):
        """
        Initialize console logger.
        
        Args:
            prefix: Prefix for log messages
        """
        self.prefix = prefix
    
    def info(self, message: str) -> None:
        """Log info message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.prefix}INFO: {message}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.prefix}DEBUG: {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.prefix}WARNING: {message}")
    
    def error(self, message: str) -> None:
        """Log error message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.prefix}ERROR: {message}")

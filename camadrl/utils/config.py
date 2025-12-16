"""
Configuration management for CAMADRL framework.

This module provides utilities for loading, saving, and managing
configuration files.
"""

from typing import Any, Dict, Optional
import os
import json

# Import optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Config:
    """
    Configuration manager for CAMADRL framework.
    
    Provides methods to load, save, and access configuration parameters
    from various file formats (JSON, YAML).
    
    Attributes:
        config: Dictionary storing configuration parameters
        filepath: Path to configuration file
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config: Initial configuration dictionary
        """
        self.config = config or {}
        self.filepath = None
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file (.json or .yaml)
            
        Returns:
            Config instance
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower()
        
        with open(filepath, 'r') as f:
            if ext == '.json':
                config_dict = json.load(f)
            elif ext in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required for YAML file support. Install it with: pip install pyyaml")
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        
        instance = cls(config_dict)
        instance.filepath = filepath
        
        return instance
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration (uses loaded filepath if None)
            
        Raises:
            ValueError: If no filepath is provided and none was loaded
        """
        if filepath is None:
            if self.filepath is None:
                raise ValueError("No filepath provided and no file was loaded")
            filepath = self.filepath
        
        ext = os.path.splitext(filepath)[1].lower()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            if ext == '.json':
                json.dump(self.config, f, indent=2)
            elif ext in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required for YAML file support. Install it with: pip install pyyaml")
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        
        print(f"Configuration saved to: {filepath}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary.
        
        Args:
            config_dict: Dictionary to update configuration with
        """
        self._deep_update(self.config, config_dict)
    
    def _deep_update(self, base: Dict, update: Dict) -> None:
        """
        Recursively update nested dictionaries.
        
        Args:
            base: Base dictionary to update
            update: Dictionary with updates
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Get item using bracket notation."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set item using bracket notation."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
    
    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self.config, indent=2)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for CAMADRL framework.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "environment": {
            "name": "GridWorld",
            "num_agents": 4,
            "max_steps": 1000,
        },
        "agent": {
            "type": "CADRLAgent",
            "state_dim": 10,
            "action_dim": 5,
            "hidden_dim": 256,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
        },
        "training": {
            "num_episodes": 1000,
            "max_steps_per_episode": 1000,
            "batch_size": 64,
            "buffer_size": 100000,
            "eval_frequency": 100,
            "save_frequency": 100,
            "log_frequency": 10,
        },
        "multi_agent": {
            "communication_enabled": True,
            "coordination_bonus": 0.1,
            "shared_reward": False,
        },
        "logging": {
            "log_dir": "./logs",
            "use_tensorboard": True,
        },
        "device": "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu",
    }

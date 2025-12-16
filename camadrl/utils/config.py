"""
Configuration management for CAMADRL framework.

Provides utilities for loading, saving, and managing
configuration parameters for experiments.
"""

import os
import json
import yaml
from typing import Any, Dict
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    """
    Configuration class for CAMADRL experiments.
    
    Stores all hyperparameters and settings for agents, environments,
    and training procedures.
    """
    
    # Environment settings
    env_name: str = "GridWorld"
    num_agents: int = 3
    max_steps: int = 1000
    
    # Agent settings
    agent_type: str = "DQN"
    state_dim: int = 10
    action_dim: int = 5
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # Network settings
    hidden_dims: list = field(default_factory=lambda: [256, 256, 128])
    dueling: bool = False
    use_attention: bool = False
    
    # Training settings
    num_episodes: int = 1000
    batch_size: int = 64
    buffer_size: int = 100000
    learning_starts: int = 1000
    update_freq: int = 1
    target_update_freq: int = 100
    
    # Multi-agent settings
    ctde: bool = False  # Centralized Training Decentralized Execution
    communication: bool = False
    shared_reward: bool = False
    
    # Logging settings
    log_dir: str = "logs"
    experiment_name: str = "camadrl_experiment"
    log_freq: int = 1
    eval_freq: int = 10
    save_freq: int = 50
    use_tensorboard: bool = True
    
    # Distributed training settings
    distributed: bool = False
    num_workers: int = 4
    backend: str = "gloo"
    
    # Seed
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create config from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            Config object
        """
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration (supports .json and .yaml)
        """
        config_dict = self.to_dict()
        
        ext = os.path.splitext(filepath)[1]
        
        if ext == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
        elif ext in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .json or .yaml")
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file (.json or .yaml)
            
        Returns:
            Config object
        """
        ext = os.path.splitext(filepath)[1]
        
        if ext == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .json or .yaml")
        
        return cls.from_dict(config_dict)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of parameters to update
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: '{key}' is not a valid configuration parameter")
    
    def __str__(self) -> str:
        """String representation of config."""
        lines = ["Configuration:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class ConfigManager:
    """
    Manager for handling multiple configurations.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize Config Manager.
        
        Args:
            config_dir: Directory for storing configurations
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, config: Config, name: str) -> str:
        """
        Save a configuration with a name.
        
        Args:
            config: Config object to save
            name: Name for the configuration
            
        Returns:
            Path to saved configuration
        """
        filepath = os.path.join(self.config_dir, f"{name}.yaml")
        config.save(filepath)
        return filepath
    
    def load_config(self, name: str) -> Config:
        """
        Load a configuration by name.
        
        Args:
            name: Name of the configuration
            
        Returns:
            Config object
        """
        # Try yaml first, then json
        yaml_path = os.path.join(self.config_dir, f"{name}.yaml")
        json_path = os.path.join(self.config_dir, f"{name}.json")
        
        if os.path.exists(yaml_path):
            return Config.load(yaml_path)
        elif os.path.exists(json_path):
            return Config.load(json_path)
        else:
            raise FileNotFoundError(f"Configuration '{name}' not found in {self.config_dir}")
    
    def list_configs(self) -> list:
        """
        List all available configurations.
        
        Returns:
            List of configuration names
        """
        configs = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith(('.yaml', '.yml', '.json')):
                name = os.path.splitext(filename)[0]
                configs.append(name)
        return configs
    
    def delete_config(self, name: str) -> None:
        """
        Delete a configuration.
        
        Args:
            name: Name of the configuration to delete
        """
        yaml_path = os.path.join(self.config_dir, f"{name}.yaml")
        json_path = os.path.join(self.config_dir, f"{name}.json")
        
        if os.path.exists(yaml_path):
            os.remove(yaml_path)
        elif os.path.exists(json_path):
            os.remove(json_path)
        else:
            raise FileNotFoundError(f"Configuration '{name}' not found")


def create_default_configs() -> Dict[str, Config]:
    """
    Create default configurations for common scenarios.
    
    Returns:
        Dictionary of default configurations
    """
    configs = {}
    
    # Basic DQN configuration
    configs['dqn_basic'] = Config(
        agent_type="DQN",
        env_name="GridWorld",
        num_episodes=500,
        batch_size=64
    )
    
    # CADRL configuration
    configs['cadrl'] = Config(
        agent_type="CADRL",
        env_name="TrafficSim",
        num_agents=5,
        num_episodes=1000,
        use_attention=True
    )
    
    # Multi-agent with CTDE
    configs['marl_ctde'] = Config(
        agent_type="PolicyGradient",
        env_name="GridWorld",
        num_agents=3,
        ctde=True,
        communication=True,
        num_episodes=1000
    )
    
    # Distributed training
    configs['distributed'] = Config(
        agent_type="DQN",
        env_name="TrafficSim",
        num_agents=10,
        distributed=True,
        num_workers=4,
        num_episodes=2000
    )
    
    return configs

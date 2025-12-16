"""
CAMADRL: Context-Aware Multi-Agent Deep Reinforcement Learning Framework

This package provides a comprehensive framework for multi-agent reinforcement learning
with context-aware coordination capabilities.
"""

__version__ = "0.1.0"
__author__ = "Mudds Air Sharif"
__email__ = "muddsair.sharif@hft-stuttgart.de"

from camadrl.agents import BaseAgent, CADRLAgent, DQNAgent, PolicyGradientAgent
from camadrl.environments import BaseEnv, GridWorld, TrafficSim, CustomEnv
from camadrl.networks import BaseNetwork, DQNNetwork, ActorCriticNetwork, CommunicationNetwork
from camadrl.trainers import BaseTrainer, MultiAgentTrainer, DistributedTrainer
from camadrl.utils import ReplayBuffer, Logger, Metrics, Config

__all__ = [
    # Agents
    "BaseAgent",
    "CADRLAgent",
    "DQNAgent",
    "PolicyGradientAgent",
    # Environments
    "BaseEnv",
    "GridWorld",
    "TrafficSim",
    "CustomEnv",
    # Networks
    "BaseNetwork",
    "DQNNetwork",
    "ActorCriticNetwork",
    "CommunicationNetwork",
    # Trainers
    "BaseTrainer",
    "MultiAgentTrainer",
    "DistributedTrainer",
    # Utils
    "ReplayBuffer",
    "Logger",
    "Metrics",
    "Config",
]

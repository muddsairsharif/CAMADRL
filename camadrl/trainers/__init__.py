"""
Training loops and utilities for CAMADRL framework.

This module contains various trainer implementations:
- BaseTrainer: Abstract base class for all trainers
- MultiAgentTrainer: Trainer for multi-agent scenarios
- DistributedTrainer: Trainer for distributed training
"""

from camadrl.trainers.base_trainer import BaseTrainer
from camadrl.trainers.multi_agent_trainer import MultiAgentTrainer
from camadrl.trainers.distributed_trainer import DistributedTrainer

__all__ = [
    "BaseTrainer",
    "MultiAgentTrainer",
    "DistributedTrainer",
]

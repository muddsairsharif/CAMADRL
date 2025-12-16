# CAMADRL Architecture

## Overview

CAMADRL (Context-Aware Multi-Agent Deep Reinforcement Learning) is a modular framework designed for multi-agent reinforcement learning research with emphasis on context awareness and agent coordination.

## Core Components

### 1. Agents (`camadrl/agents/`)

The agent module provides various RL agent implementations:

- **BaseAgent**: Abstract base class defining the agent interface
- **DQNAgent**: Deep Q-Network agent with experience replay
- **CADRLAgent**: Context-Aware DRL agent with communication capabilities
- **PolicyGradientAgent**: Policy gradient agent using REINFORCE

#### Key Features:
- Modular design for easy extension
- Support for both value-based and policy-based methods
- Context-aware decision making
- Built-in exploration strategies

### 2. Environments (`camadrl/environments/`)

Environment implementations for training and evaluation:

- **BaseEnv**: Abstract base class extending gymnasium.Env
- **GridWorld**: Grid-based navigation environment
- **TrafficSim**: Traffic simulation for vehicle coordination
- **CustomEnv**: Template for custom environments

#### Key Features:
- Multi-agent support
- Configurable reward structures
- Visualization capabilities
- Gymnasium-compatible interface

### 3. Networks (`camadrl/networks/`)

Neural network architectures:

- **BaseNetwork**: Abstract base class for networks
- **DQNNetwork**: Deep Q-Network architecture with optional dueling
- **ActorCriticNetwork**: Shared feature extraction with separate heads
- **CommunicationNetwork**: Context processing with attention mechanisms

#### Key Features:
- PyTorch-based implementations
- Support for recurrent architectures
- Multi-head attention for communication
- Graph neural networks for structured communication

### 4. Trainers (`camadrl/trainers/`)

Training loop implementations:

- **BaseTrainer**: Abstract trainer with logging and checkpointing
- **MultiAgentTrainer**: Training for multi-agent scenarios
- **DistributedTrainer**: Distributed training across multiple GPUs

#### Key Features:
- Automatic logging and checkpointing
- Periodic evaluation
- Support for distributed training
- Coordination metrics tracking

### 5. Utils (`camadrl/utils/`)

Utility modules:

- **ReplayBuffer**: Experience replay with prioritization support
- **Logger**: TensorBoard integration and metric tracking
- **Metrics**: Performance metrics calculator
- **Config**: Configuration management

### 6. Visualization (`camadrl/visualization/`)

Visualization tools:

- **Plotter**: General plotting utilities
- **TrajectoryVisualizer**: Agent trajectory visualization
- **HeatmapGenerator**: Analysis heatmap generation

## Design Principles

1. **Modularity**: Each component is self-contained and can be used independently
2. **Extensibility**: Easy to extend with new agents, environments, or networks
3. **Configurability**: Extensive configuration options through dictionaries
4. **Reproducibility**: Built-in seeding and logging for reproducible experiments

## Data Flow

```
Environment → Agents → Networks → Trainers → Metrics → Visualization
```

1. Environment provides observations to agents
2. Agents use networks to select actions
3. Trainers coordinate the training loop
4. Metrics track performance
5. Visualization tools analyze results

## Multi-Agent Coordination

CAMADRL supports multiple coordination mechanisms:

1. **Context Sharing**: Agents share state information
2. **Communication Networks**: Explicit message passing
3. **Attention Mechanisms**: Learn which agents to pay attention to
4. **Shared Rewards**: Cooperative reward structures

## Distributed Training

The framework supports distributed training:

- Data parallelism using PyTorch DDP
- Synchronous and asynchronous training
- Parameter server architecture
- Multi-GPU and multi-node support

# CAMADRL Architecture

## Overview

CAMADRL (Context-Aware Multi-Agent Deep Reinforcement Learning) is a comprehensive framework for intelligent EV charging coordination using multi-agent deep reinforcement learning with context awareness.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CAMADRL Framework                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Agents  │  │   Envs   │  │ Networks │  │ Trainers │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       │             │              │              │          │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Utils & Visualization                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agents Module (`camadrl/agents/`)

Implements various agent types for multi-agent reinforcement learning:

#### BaseAgent
- Abstract base class for all agents
- Defines common interface: `select_action()`, `update()`, `save()`, `load()`
- Maintains training statistics

#### DQNAgent
- Deep Q-Network implementation
- Features:
  - Experience replay
  - Target network for stability
  - Double DQN for reduced overestimation
  - Epsilon-greedy exploration

#### CADRLAgent
- Context-Aware Deep RL agent
- Features:
  - Context integration in decision making
  - Grid state awareness
  - Time-dependent policies
  - Multi-stakeholder optimization

#### PolicyGradientAgent
- Actor-Critic implementation
- Features:
  - Policy gradient methods
  - Continuous/discrete action spaces
  - Entropy regularization
  - GAE (Generalized Advantage Estimation)

### 2. Environments Module (`camadrl/environments/`)

Provides simulation environments for multi-agent scenarios:

#### BaseEnv
- Abstract base class following OpenAI Gym interface
- Methods: `reset()`, `step()`, `render()`, `close()`

#### GridWorld
- 2D grid-based environment
- Features:
  - Agent navigation
  - Charging station placement
  - Battery management
  - Collision avoidance

#### TrafficSim
- Urban traffic simulation
- Features:
  - Road network modeling
  - Traffic flow dynamics
  - Charging station queueing
  - Real-time optimization

#### CustomEnv
- Flexible custom environment
- Features:
  - Custom reward functions
  - Custom observation functions
  - Dynamic environment modification

### 3. Networks Module (`camadrl/networks/`)

Neural network architectures for deep RL:

#### BaseNetwork
- Abstract base for all networks
- Common utilities: parameter counting, freezing

#### DQNNetwork
- Q-network with optional dueling architecture
- Features:
  - Standard DQN
  - Dueling DQN (separate value and advantage streams)
  - Noisy layers for exploration

#### ActorCriticNetwork
- Separate actor and critic networks
- Features:
  - Shared feature extraction
  - Policy and value heads
  - Support for continuous/discrete actions

#### CommunicationNetwork
- Agent communication mechanisms
- Features:
  - Message passing
  - Graph neural networks
  - Attention mechanisms
  - Dynamic communication graphs

### 4. Trainers Module (`camadrl/trainers/`)

Training algorithms and coordination strategies:

#### BaseTrainer
- Abstract trainer interface
- Methods: `train()`, `train_episode()`, `evaluate()`

#### MultiAgentTrainer
- Coordinated multi-agent training
- Features:
  - Independent learning
  - Centralized Training Decentralized Execution (CTDE)
  - Experience sharing
  - Coordinated exploration

#### DistributedTrainer
- Scalable distributed training
- Features:
  - Multi-process training
  - GPU parallelization
  - Asynchronous updates
  - Gradient aggregation

### 5. Utils Module (`camadrl/utils/`)

Utility functions and helpers:

#### ReplayBuffer
- Experience replay implementation
- Variants:
  - Standard replay buffer
  - Prioritized experience replay
  - Multi-agent replay buffer

#### Logger
- Experiment tracking
- Features:
  - TensorBoard integration
  - Metric logging
  - Configuration management

#### Metrics
- Performance evaluation
- Metrics:
  - Episode rewards
  - Coordination scores
  - Convergence analysis
  - Charging efficiency

#### Config
- Configuration management
- Features:
  - YAML/JSON support
  - Parameter validation
  - Default configurations

### 6. Visualization Module (`camadrl/visualization/`)

Visualization and analysis tools:

#### Plotter
- Training progress plots
- Reward curves
- Loss curves
- Multi-agent comparisons

#### TrajectoryVisualizer
- Agent trajectory visualization
- Features:
  - Path plotting
  - Animation generation
  - Multi-agent trajectories

#### HeatmapGenerator
- Spatial analysis
- Features:
  - Position density heatmaps
  - Resource utilization maps
  - Correlation matrices

## Data Flow

### Training Loop

```
1. Environment Reset
   └─> Initial observations

2. Agent Action Selection
   └─> State + Context → Policy → Actions

3. Environment Step
   └─> Actions → Next State, Rewards

4. Experience Storage
   └─> Transitions → Replay Buffer

5. Agent Update
   └─> Batch Sampling → Network Update

6. Logging & Evaluation
   └─> Metrics → TensorBoard/Files
```

### Multi-Agent Coordination

```
Agent 1 ──┐
          ├─> Communication Network ─> Coordinated Actions
Agent 2 ──┤
          │
Agent N ──┘
```

## Key Design Patterns

### 1. Strategy Pattern
- Different agent types (DQN, CADRL, PG)
- Different training strategies (independent, CTDE)

### 2. Observer Pattern
- Logging and monitoring
- Event-driven metrics collection

### 3. Factory Pattern
- Agent creation
- Environment instantiation
- Network building

### 4. Template Method Pattern
- Base classes define algorithm structure
- Subclasses implement specific steps

## Performance Considerations

### Memory Management
- Replay buffer size limits
- Efficient numpy arrays
- Gradient checkpointing for large networks

### Computational Efficiency
- Batch processing
- GPU acceleration
- Parallel environment execution

### Scalability
- Distributed training support
- Multi-GPU training
- Asynchronous updates

## Extension Points

### Adding New Agents
1. Inherit from `BaseAgent`
2. Implement required methods
3. Add to `agents/__init__.py`

### Adding New Environments
1. Inherit from `BaseEnv`
2. Implement Gym interface
3. Add to `environments/__init__.py`

### Adding New Networks
1. Inherit from `BaseNetwork`
2. Implement `forward()` method
3. Add to `networks/__init__.py`

## Best Practices

1. **Configuration Management**
   - Use Config class for all parameters
   - Save configurations with experiments

2. **Experiment Tracking**
   - Use Logger for all experiments
   - Enable TensorBoard for visualization

3. **Code Organization**
   - Keep agents, envs, networks modular
   - Use clear naming conventions

4. **Testing**
   - Write unit tests for new components
   - Use pytest for test execution

5. **Documentation**
   - Document all public APIs
   - Provide usage examples
   - Keep README updated

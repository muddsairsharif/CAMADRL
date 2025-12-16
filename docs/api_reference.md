# API Reference

## Agents

### BaseAgent

Abstract base class for all agents.

```python
class BaseAgent(agent_id, state_dim, action_dim, config=None)
```

**Parameters:**
- `agent_id` (int): Unique identifier for the agent
- `state_dim` (int): Dimension of the state space
- `action_dim` (int): Dimension of the action space
- `config` (dict, optional): Configuration dictionary

**Methods:**
- `select_action(state, explore=True)`: Select an action
- `update(batch)`: Update agent with experience batch
- `save(filepath)`: Save agent model
- `load(filepath)`: Load agent model
- `reset()`: Reset agent for new episode
- `get_info()`: Get agent statistics

### DQNAgent

Deep Q-Network agent implementation.

```python
class DQNAgent(agent_id, state_dim, action_dim, config=None)
```

**Additional Configuration:**
- `learning_rate` (float): Learning rate (default: 0.001)
- `gamma` (float): Discount factor (default: 0.99)
- `epsilon` (float): Exploration rate (default: 1.0)
- `epsilon_decay` (float): Epsilon decay rate (default: 0.995)
- `target_update_freq` (int): Target network update frequency (default: 100)

### CADRLAgent

Context-Aware Deep RL agent.

```python
class CADRLAgent(agent_id, state_dim, action_dim, context_dim=10, config=None)
```

**Parameters:**
- `context_dim` (int): Dimension of context information

**Methods:**
- `select_action(state, context=None, explore=True)`: Select action with context

### PolicyGradientAgent

Actor-Critic policy gradient agent.

```python
class PolicyGradientAgent(agent_id, state_dim, action_dim, config=None)
```

**Methods:**
- `store_reward(reward)`: Store reward for current step
- `update(batch=None)`: Update using episode data or batch

## Environments

### BaseEnv

Abstract base environment class.

```python
class BaseEnv(config=None)
```

**Methods:**
- `reset(seed=None)`: Reset environment
- `step(action)`: Execute one timestep
- `render()`: Render environment
- `close()`: Clean up resources
- `get_state()`: Get environment state

### GridWorld

2D grid-based environment for agent navigation.

```python
class GridWorld(config=None)
```

**Configuration:**
- `grid_size` (int): Size of the grid (default: 10)
- `num_agents` (int): Number of agents (default: 3)
- `num_charging_stations` (int): Number of charging stations (default: 5)

**Observation Space:**
- Shape: (6,)
- [agent_x, agent_y, battery_level, station_x, station_y, availability]

**Action Space:**
- Discrete(5): [up, down, left, right, charge]

### TrafficSim

Urban traffic simulation environment.

```python
class TrafficSim(config=None)
```

**Configuration:**
- `network_size` (int): Size of road network (default: 20)
- `num_agents` (int): Number of EVs (default: 5)
- `traffic_density` (float): Traffic density factor (default: 0.3)

**Observation Space:**
- Shape: (10,)
- Includes position, velocity, battery, destination, traffic info

**Action Space:**
- Discrete(6): [accelerate, decelerate, turn_left, turn_right, change_lane, charge]

### CustomEnv

Flexible custom environment.

```python
class CustomEnv(config=None)
```

**Configuration:**
- `state_dim` (int): State space dimension
- `action_dim` (int): Action space dimension
- `reward_fn` (callable): Custom reward function
- `observation_fn` (callable): Custom observation function

**Methods:**
- `set_reward_function(reward_fn)`: Set custom reward
- `set_observation_function(obs_fn)`: Set custom observation

## Networks

### DQNNetwork

Deep Q-Network architecture.

```python
class DQNNetwork(input_dim, output_dim, hidden_dims=None, dueling=False, config=None)
```

**Parameters:**
- `input_dim` (int): Input dimension
- `output_dim` (int): Output dimension (number of actions)
- `hidden_dims` (list): Hidden layer dimensions
- `dueling` (bool): Use dueling architecture

**Methods:**
- `forward(x)`: Forward pass
- `get_action(state, epsilon=0.0)`: Get action with exploration

### ActorCriticNetwork

Actor-Critic network.

```python
class ActorCriticNetwork(input_dim, action_dim, hidden_dims=None, continuous=False, config=None)
```

**Parameters:**
- `continuous` (bool): Continuous action space

**Methods:**
- `forward(x)`: Return (action_params, value)
- `get_action(state, deterministic=False)`: Get action
- `evaluate_actions(state, action)`: Evaluate actions

### CommunicationNetwork

Multi-agent communication network.

```python
class CommunicationNetwork(state_dim, message_dim, hidden_dim=128, num_agents=3, config=None)
```

**Parameters:**
- `message_dim` (int): Dimension of messages
- `num_agents` (int): Number of agents

**Methods:**
- `forward(states, adjacency=None)`: Message passing
- `encode_message(state)`: Encode state to message

## Trainers

### MultiAgentTrainer

Multi-agent training coordinator.

```python
class MultiAgentTrainer(agents, env, config=None)
```

**Configuration:**
- `num_episodes` (int): Number of training episodes
- `batch_size` (int): Batch size for updates
- `learning_starts` (int): Steps before learning starts
- `ctde` (bool): Use centralized training

**Methods:**
- `train()`: Execute training loop
- `train_episode()`: Train single episode
- `evaluate(num_episodes=10)`: Evaluate agents

### DistributedTrainer

Distributed training across multiple processes.

```python
class DistributedTrainer(agents, env, config=None)
```

**Configuration:**
- `num_workers` (int): Number of parallel workers
- `backend` (str): Distributed backend (default: 'gloo')

## Utils

### ReplayBuffer

Experience replay buffer.

```python
class ReplayBuffer(capacity, state_dim, action_dim, seed=None)
```

**Methods:**
- `add(observation, action, reward, next_observation, done)`: Add transition
- `sample(batch_size)`: Sample batch
- `__len__()`: Get buffer size

### Logger

Experiment logging and tracking.

```python
class Logger(log_dir="logs", experiment_name=None, use_tensorboard=True)
```

**Methods:**
- `log_scalar(tag, value, step)`: Log scalar metric
- `log_episode(episode, metrics)`: Log episode data
- `log_config(config)`: Log configuration
- `save_metrics(filename)`: Save metrics to file

### Metrics

Performance metrics calculator.

```python
class Metrics
```

**Static Methods:**
- `compute_episode_metrics(rewards, length)`: Episode metrics
- `compute_training_metrics(episode_rewards, window_size=100)`: Training metrics
- `compute_coordination_metrics(agent_rewards, joint_reward)`: Coordination metrics
- `compute_convergence_metrics(episode_rewards, threshold=None)`: Convergence analysis

### Config

Configuration management.

```python
class Config(**kwargs)
```

**Methods:**
- `to_dict()`: Convert to dictionary
- `from_dict(config_dict)`: Create from dictionary
- `save(filepath)`: Save to file (JSON/YAML)
- `load(filepath)`: Load from file
- `update(updates)`: Update parameters

## Visualization

### Plotter

Training metrics plotting.

```python
class Plotter(style="seaborn-v0_8")
```

**Methods:**
- `plot_rewards(episode_rewards, window_size=100, ...)`: Plot rewards
- `plot_multi_agent_rewards(agent_rewards, ...)`: Multi-agent rewards
- `plot_convergence(episode_rewards, ...)`: Convergence analysis

### TrajectoryVisualizer

Agent trajectory visualization.

```python
class TrajectoryVisualizer()
```

**Methods:**
- `plot_trajectory(positions, ...)`: Plot single trajectory
- `plot_multi_agent_trajectories(trajectories, ...)`: Multi-agent trajectories
- `create_animation(trajectories, ...)`: Create animation

### HeatmapGenerator

Spatial analysis heatmaps.

```python
class HeatmapGenerator()
```

**Methods:**
- `generate_position_heatmap(positions, ...)`: Position heatmap
- `generate_density_heatmap(positions, ...)`: Density heatmap
- `generate_resource_utilization_heatmap(...)`: Resource utilization

## Usage Examples

### Basic Training

```python
from camadrl import DQNAgent, GridWorld, Config, Logger

# Setup
config = Config(num_episodes=100)
env = GridWorld()
agent = DQNAgent(0, env.observation_space.shape[0], env.action_space.n)
logger = Logger(experiment_name="basic_training")

# Train
for episode in range(config.num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs = next_obs
    
    logger.log_episode(episode, {"reward": total_reward})
```

### Multi-Agent Training

```python
from camadrl import DQNAgent, GridWorld, MultiAgentTrainer

# Setup
env = GridWorld({"num_agents": 3})
agents = [DQNAgent(i, env.observation_space.shape[0], env.action_space.n) 
          for i in range(3)]

# Train
trainer = MultiAgentTrainer(agents, env, {"num_episodes": 200})
stats = trainer.train()
```

### Context-Aware Learning

```python
from camadrl import CADRLAgent, TrafficSim

env = TrafficSim()
agent = CADRLAgent(0, env.observation_space.shape[0], 
                   env.action_space.n, context_dim=10)

obs, _ = env.reset()
context = get_context()  # Get environmental context
action = agent.select_action(obs, context=context)
```

# CAMADRL Examples

This document provides comprehensive examples for using the CAMADRL framework.

## Table of Contents

1. [Basic Single-Agent Training](#basic-single-agent-training)
2. [Multi-Agent Coordination](#multi-agent-coordination)
3. [Context-Aware Learning](#context-aware-learning)
4. [Custom Environments](#custom-environments)
5. [Advanced Topics](#advanced-topics)

## Basic Single-Agent Training

### Simple DQN Training

```python
from camadrl.agents import DQNAgent
from camadrl.environments import GridWorld
from camadrl.utils import Config, Logger

# Configuration
config = Config(
    num_episodes=100,
    learning_rate=0.001,
    batch_size=64
)

# Environment
env = GridWorld({"grid_size": 10})

# Agent
agent = DQNAgent(
    agent_id=0,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=config.to_dict()
)

# Training loop
for episode in range(config.num_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(obs, explore=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        obs = next_obs
    
    print(f"Episode {episode}: Reward = {episode_reward}")
```

### With Experience Replay

```python
from camadrl.utils import ReplayBuffer

# Create replay buffer
buffer = ReplayBuffer(
    capacity=10000,
    state_dim=env.observation_space.shape[0],
    action_dim=1
)

# Training with replay
for episode in range(100):
    obs, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        buffer.add(obs, action, reward, next_obs, float(done))
        
        # Update agent
        if len(buffer) > 64:
            batch = buffer.sample(64)
            batch_tensors = {
                "states": torch.FloatTensor(batch["observations"]),
                "actions": torch.LongTensor(batch["actions"]),
                "rewards": torch.FloatTensor(batch["rewards"]),
                "next_states": torch.FloatTensor(batch["next_observations"]),
                "dones": torch.FloatTensor(batch["dones"])
            }
            agent.update(batch_tensors)
        
        obs = next_obs
```

## Multi-Agent Coordination

### Independent Learning

```python
from camadrl.agents import DQNAgent
from camadrl.environments import GridWorld
from camadrl.trainers import MultiAgentTrainer

# Environment with multiple agents
env = GridWorld({
    "grid_size": 15,
    "num_agents": 3,
    "num_charging_stations": 5
})

# Create multiple agents
agents = []
for i in range(3):
    agent = DQNAgent(
        agent_id=i,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agents.append(agent)

# Multi-agent trainer
trainer = MultiAgentTrainer(
    agents=agents,
    env=env,
    config={
        "num_episodes": 200,
        "batch_size": 64,
        "learning_starts": 500
    }
)

# Train
results = trainer.train()
print(f"Training completed: {results}")
```

### Centralized Training Decentralized Execution (CTDE)

```python
# Enable CTDE
trainer = MultiAgentTrainer(
    agents=agents,
    env=env,
    config={
        "num_episodes": 200,
        "ctde": True,  # Enable CTDE
        "communication": True  # Enable communication
    }
)

results = trainer.train_ctde()
```

### With Communication

```python
from camadrl.networks import CommunicationNetwork

# Create communication network
comm_network = CommunicationNetwork(
    state_dim=10,
    message_dim=8,
    num_agents=3
)

# Use in training
def train_with_communication():
    for episode in range(100):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Collect states from all agents
            states = torch.stack([torch.FloatTensor(obs) for _ in range(3)])
            states = states.unsqueeze(0)  # Add batch dimension
            
            # Message passing
            updated_states, messages = comm_network(states)
            
            # Select actions based on updated states
            actions = [agent.select_action(updated_states[0, i].numpy()) 
                      for i, agent in enumerate(agents)]
            
            # Environment step
            obs, reward, terminated, truncated, _ = env.step(actions[0])
            done = terminated or truncated
```

## Context-Aware Learning

### Basic Context-Aware Training

```python
from camadrl.agents import CADRLAgent
from camadrl.environments import TrafficSim
import numpy as np

# Environment
env = TrafficSim({
    "network_size": 20,
    "num_agents": 5,
    "traffic_density": 0.3
})

# CADRL Agent
agent = CADRLAgent(
    agent_id=0,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    context_dim=10
)

# Training with context
for episode in range(100):
    obs, _ = env.reset()
    done = False
    
    # Generate context (e.g., time of day, grid load, weather)
    context = np.array([
        np.sin(2 * np.pi * episode / 100),  # Time of day
        np.random.rand(),  # Grid load
        np.random.rand(),  # Weather
        *np.random.rand(7)  # Other context features
    ])
    
    while not done:
        action = agent.select_action(obs, context=context, explore=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs
```

### Dynamic Context

```python
def get_dynamic_context(env, time_step):
    """Generate dynamic context based on environment state."""
    context = np.zeros(10)
    
    # Time of day (cyclical)
    hour = (time_step % 24) / 24.0
    context[0] = np.sin(2 * np.pi * hour)
    context[1] = np.cos(2 * np.pi * hour)
    
    # Grid load
    if hasattr(env, 'charging_stations'):
        context[2] = len([s for s in env.station_queues if len(s) > 0]) / len(env.charging_stations)
    
    # Traffic density
    if hasattr(env, 'traffic_grid'):
        context[3] = env.traffic_grid.mean()
    
    # Battery levels
    if hasattr(env, 'agent_batteries'):
        context[4] = env.agent_batteries.mean()
        context[5] = env.agent_batteries.min()
    
    return context

# Use in training
for episode in range(100):
    obs, _ = env.reset()
    step = 0
    done = False
    
    while not done:
        context = get_dynamic_context(env, step)
        action = agent.select_action(obs, context=context)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
```

## Custom Environments

### Creating Custom Environment

```python
from camadrl.environments import CustomEnv
import numpy as np

def custom_reward_fn(state, agent_states, action, agent_idx):
    """Custom reward for EV charging optimization."""
    # Penalize high state values (energy consumption)
    energy_penalty = -np.sum(state ** 2) * 0.01
    
    # Reward coordination
    if len(agent_states) > 1:
        coordination_reward = -np.var(agent_states, axis=0).mean() * 0.1
    else:
        coordination_reward = 0
    
    # Peak load penalty
    total_load = sum([s[0] for s in agent_states])
    peak_penalty = -max(0, total_load - 5.0) * 0.5
    
    return energy_penalty + coordination_reward + peak_penalty

def custom_obs_fn(state, agent_states, agent_idx):
    """Custom observation including neighbor information."""
    own_state = agent_states[agent_idx]
    neighbor_mean = np.mean([s for i, s in enumerate(agent_states) if i != agent_idx], axis=0)
    
    # Combine global state, own state, and neighbor info
    obs = np.concatenate([state * 0.5, own_state * 0.3, neighbor_mean * 0.2])
    return obs.astype(np.float32)[:len(state)]

# Create custom environment
env = CustomEnv({
    "state_dim": 10,
    "action_dim": 5,
    "num_agents": 3,
    "reward_fn": custom_reward_fn,
    "observation_fn": custom_obs_fn
})

# Train with custom environment
agent = PolicyGradientAgent(0, 10, 5)
for episode in range(100):
    obs, _ = env.reset()
    # ... training loop
```

## Advanced Topics

### Distributed Training

```python
from camadrl.trainers import DistributedTrainer

trainer = DistributedTrainer(
    agents=agents,
    env=env,
    config={
        "num_episodes": 500,
        "num_workers": 4,
        "backend": "gloo"
    }
)

results = trainer.train()
```

### Prioritized Experience Replay

```python
from camadrl.utils import PrioritizedReplayBuffer

buffer = PrioritizedReplayBuffer(
    capacity=100000,
    state_dim=10,
    action_dim=1,
    alpha=0.6,
    beta=0.4
)

# Sample with priorities
batch, weights, indices = buffer.sample(64)

# Update priorities after learning
td_errors = compute_td_errors(batch)  # Your TD error computation
buffer.update_priorities(indices, td_errors)
```

### Custom Logging and Visualization

```python
from camadrl.utils import Logger
from camadrl.visualization import Plotter, TrajectoryVisualizer

# Setup logger
logger = Logger(
    log_dir="experiments",
    experiment_name="custom_experiment",
    use_tensorboard=True
)

# Log training progress
for episode in range(100):
    # ... training ...
    
    logger.log_episode(episode, {
        "reward": episode_reward,
        "loss": loss,
        "epsilon": epsilon
    })
    
    logger.log_scalar("custom_metric", custom_value, episode)

# Visualize results
plotter = Plotter()
plotter.plot_rewards(episode_rewards, save_path="results/rewards.png")

viz = TrajectoryVisualizer()
viz.plot_trajectory(positions, save_path="results/trajectory.png")
```

### Checkpointing and Resume

```python
# Save checkpoint
trainer.save_checkpoint("checkpoints/episode_100")

# Load checkpoint
trainer.load_checkpoint("checkpoints/episode_100")

# Continue training
results = trainer.train()
```

### Hyperparameter Tuning

```python
from camadrl.utils import Config, ConfigManager

# Create multiple configurations
configs = []
for lr in [0.001, 0.0005, 0.0001]:
    for gamma in [0.95, 0.99]:
        config = Config(
            learning_rate=lr,
            gamma=gamma,
            experiment_name=f"lr_{lr}_gamma_{gamma}"
        )
        configs.append(config)

# Train with each configuration
results = []
for config in configs:
    agent = DQNAgent(0, state_dim, action_dim, config=config.to_dict())
    # ... train agent ...
    results.append({
        "config": config,
        "performance": final_reward
    })

# Find best configuration
best_config = max(results, key=lambda x: x["performance"])
print(f"Best config: {best_config}")
```

## Running the Examples

All example scripts are available in the `examples/` directory:

```bash
# Basic training
python examples/basic_training.py

# Multi-agent coordination
python examples/multi_agent_coordination.py

# Context-aware learning
python examples/context_aware_learning.py

# Custom environment
python examples/custom_environment.py
```

## Next Steps

- Read the [Architecture Documentation](architecture.md)
- Check the [API Reference](api_reference.md)
- See [Troubleshooting Guide](troubleshooting.md)

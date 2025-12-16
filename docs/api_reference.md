# API Reference

## Agents

### BaseAgent

```python
class BaseAgent(agent_id, state_dim, action_dim, config=None, device="cpu")
```

Base class for all agents.

**Methods:**
- `select_action(state, explore=True)`: Select action given state
- `update(state, action, reward, next_state, done)`: Update policy
- `save(path)`: Save agent parameters
- `load(path)`: Load agent parameters
- `reset()`: Reset agent for new episode

### DQNAgent

```python
class DQNAgent(agent_id, state_dim, action_dim, config=None, device="cpu")
```

Deep Q-Network agent with experience replay.

**Additional Config:**
- `batch_size`: Batch size for training (default: 64)
- `buffer_size`: Replay buffer size (default: 100000)
- `target_update_freq`: Target network update frequency (default: 100)

### CADRLAgent

```python
class CADRLAgent(agent_id, state_dim, action_dim, context_dim=64, config=None, device="cpu")
```

Context-aware agent with communication.

**Methods:**
- `select_action(state, context=None, explore=True)`: Select action with context

## Environments

### BaseEnv

```python
class BaseEnv(num_agents, state_dim, action_dim, config=None)
```

Base environment class.

**Methods:**
- `reset(seed=None, options=None)`: Reset environment
- `step(actions)`: Execute actions
- `render(mode="human")`: Render environment
- `get_state(agent_id)`: Get state for specific agent

### GridWorld

```python
class GridWorld(num_agents=4, grid_size=10, config=None)
```

Grid-based navigation environment.

**Config Options:**
- `num_obstacles`: Number of obstacles (default: grid_size)
- `reward_goal`: Reward for reaching goal (default: 10.0)
- `reward_collision`: Penalty for collision (default: -5.0)

## Networks

### DQNNetwork

```python
class DQNNetwork(state_dim, action_dim, hidden_dim=256, num_layers=3, config=None)
```

Deep Q-Network architecture.

**Config Options:**
- `use_dueling`: Use dueling architecture (default: False)
- `dropout_rate`: Dropout rate (default: 0.0)

### ActorCriticNetwork

```python
class ActorCriticNetwork(state_dim, action_dim, hidden_dim=256, num_layers=2, config=None)
```

Actor-critic network with shared features.

**Returns:**
- Tuple of (action_probabilities, state_value)

## Trainers

### MultiAgentTrainer

```python
class MultiAgentTrainer(env, agents, config=None)
```

Trainer for multi-agent scenarios.

**Config Options:**
- `num_episodes`: Number of training episodes (default: 1000)
- `eval_frequency`: Evaluation frequency (default: 100)
- `communication_enabled`: Enable agent communication (default: True)

**Methods:**
- `train()`: Run full training loop
- `evaluate(num_episodes=10)`: Evaluate agents
- `save_checkpoint(path)`: Save checkpoint

## Utils

### ReplayBuffer

```python
class ReplayBuffer(capacity, state_dim, device="cpu")
```

Experience replay buffer.

**Methods:**
- `add(state, action, reward, next_state, done)`: Add experience
- `sample(batch_size)`: Sample batch
- `__len__()`: Get buffer size

### Logger

```python
class Logger(log_dir="./logs", config=None)
```

Training logger with TensorBoard.

**Methods:**
- `log(step, metrics, prefix="")`: Log metrics
- `save_metrics(filename)`: Save metrics to JSON
- `plot_metrics(metrics, save, show)`: Plot metrics

## Configuration

Default configuration structure:

```python
config = {
    "environment": {
        "name": "GridWorld",
        "num_agents": 4,
        "max_steps": 1000,
    },
    "agent": {
        "type": "CADRLAgent",
        "learning_rate": 0.001,
        "gamma": 0.99,
    },
    "training": {
        "num_episodes": 1000,
        "batch_size": 64,
    }
}
```

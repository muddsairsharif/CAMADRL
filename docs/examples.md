# Examples

## Basic Training

Simple example training DQN agents in GridWorld:

```python
from camadrl.environments import GridWorld
from camadrl.agents import DQNAgent
from camadrl.trainers import MultiAgentTrainer

# Create environment
env = GridWorld(num_agents=2, grid_size=10)

# Create agents
agents = [
    DQNAgent(0, env.state_dim, env.action_dim),
    DQNAgent(1, env.state_dim, env.action_dim)
]

# Train
trainer = MultiAgentTrainer(env, agents, {"num_episodes": 100})
trainer.train()
```

## Context-Aware Learning

Using CADRL agents with communication:

```python
from camadrl.environments import TrafficSim
from camadrl.agents import CADRLAgent
from camadrl.trainers import MultiAgentTrainer

# Create environment
env = TrafficSim(num_agents=5)

# Create CADRL agents
agents = [
    CADRLAgent(i, env.state_dim, env.action_dim, context_dim=64)
    for i in range(5)
]

# Train with communication
config = {
    "num_episodes": 200,
    "communication_enabled": True,
    "coordination_bonus": 0.2
}
trainer = MultiAgentTrainer(env, agents, config)
trainer.train()
```

## Custom Environment

Creating a custom environment:

```python
from camadrl.environments import BaseEnv
import numpy as np

class MyEnv(BaseEnv):
    def reset(self, seed=None, options=None):
        # Initialize environment
        states = [np.random.randn(self.state_dim) for _ in range(self.num_agents)]
        return states, {}
    
    def step(self, actions):
        # Execute actions and return results
        next_states = [...]
        rewards = [...]
        dones = [...]
        return next_states, rewards, dones, {}
```

## Distributed Training

Running distributed training:

```python
from camadrl.trainers import DistributedTrainer

config = {
    "world_size": 4,
    "backend": "nccl",
    "num_episodes": 1000
}

trainer = DistributedTrainer(env, agents, config)
trainer.train()
```

## Visualization

Visualizing training results:

```python
from camadrl.visualization import Plotter

plotter = Plotter()
plotter.plot_training_curve(episodes, rewards, save_path="training.png")
plotter.plot_multi_agent_comparison(episodes, agent_rewards)
```

## Saving and Loading

Saving and loading agents:

```python
# Save
for i, agent in enumerate(agents):
    agent.save(f"agent_{i}.pt")

# Load
for i, agent in enumerate(agents):
    agent.load(f"agent_{i}.pt")
```

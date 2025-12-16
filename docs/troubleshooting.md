# Troubleshooting Guide

This guide helps you resolve common issues when using the CAMADRL framework.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Training Problems](#training-problems)
3. [Performance Issues](#performance-issues)
4. [Environment Issues](#environment-issues)
5. [Visualization Issues](#visualization-issues)
6. [Common Errors](#common-errors)

## Installation Issues

### Problem: Package Import Errors

**Symptom:**
```python
ModuleNotFoundError: No module named 'camadrl'
```

**Solutions:**

1. Install the package in development mode:
```bash
cd CAMADRL
pip install -e .
```

2. Add the package to your Python path:
```python
import sys
sys.path.insert(0, '/path/to/CAMADRL')
```

3. Verify installation:
```bash
python -c "import camadrl; print(camadrl.__version__)"
```

### Problem: Missing Dependencies

**Symptom:**
```
ImportError: cannot import name 'xxx' from 'yyy'
```

**Solution:**
```bash
pip install -r requirements.txt
```

For specific packages:
```bash
pip install torch>=2.0.0
pip install gymnasium>=0.28.0
pip install matplotlib seaborn plotly
```

### Problem: CUDA/GPU Issues

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce batch size:
```python
config = Config(batch_size=32)  # Instead of 64 or 128
```

2. Use CPU instead:
```python
# In agent initialization
self.device = torch.device("cpu")
```

3. Clear GPU cache:
```python
import torch
torch.cuda.empty_cache()
```

## Training Problems

### Problem: Agent Not Learning

**Symptom:** Rewards remain constant or don't improve over episodes.

**Diagnostic Steps:**

1. Check if agent is actually updating:
```python
# Add logging in training loop
metrics = agent.update(batch)
print(f"Loss: {metrics.get('loss', 'N/A')}")
```

2. Verify learning has started:
```python
if step < learning_starts:
    continue  # Not learning yet
```

**Solutions:**

1. Adjust learning rate:
```python
config = Config(learning_rate=0.001)  # Try different values: 0.01, 0.0001
```

2. Check epsilon decay:
```python
# Ensure exploration is happening
print(f"Epsilon: {agent.epsilon}")
```

3. Verify reward signals:
```python
# Check if rewards are too sparse or too small
print(f"Reward range: {min(rewards)} to {max(rewards)}")
```

4. Adjust network architecture:
```python
config = Config(hidden_dims=[256, 256, 128])  # Try larger/smaller networks
```

### Problem: Training Unstable

**Symptom:** Loss spikes or oscillates wildly.

**Solutions:**

1. Enable gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1.0)
```

2. Reduce learning rate:
```python
config = Config(learning_rate=0.0001)
```

3. Increase target network update frequency:
```python
config = Config(target_update_freq=1000)
```

4. Use double DQN:
```python
# Already implemented in DQNAgent
# Ensure you're using latest version
```

### Problem: Convergence Too Slow

**Symptom:** Training takes too long to reach good performance.

**Solutions:**

1. Increase learning rate (carefully):
```python
config = Config(learning_rate=0.003)
```

2. Reduce epsilon decay:
```python
config = Config(epsilon_decay=0.99)  # Faster exploration
```

3. Use larger batch size:
```python
config = Config(batch_size=128)
```

4. Warm start with demonstration:
```python
# Pre-fill replay buffer with good examples
for _ in range(1000):
    # Add expert demonstrations
    buffer.add(...)
```

## Performance Issues

### Problem: Training Too Slow

**Symptom:** Each episode takes a long time.

**Solutions:**

1. Use GPU if available:
```python
# Verify GPU is being used
print(f"Using device: {agent.device}")
```

2. Reduce environment complexity:
```python
env = GridWorld({"grid_size": 10})  # Instead of 20
```

3. Vectorize operations:
```python
# Use batch operations instead of loops
states = torch.FloatTensor(batch_states)
```

4. Enable distributed training:
```python
from camadrl.trainers import DistributedTrainer
trainer = DistributedTrainer(agents, env, config={"num_workers": 4})
```

### Problem: High Memory Usage

**Symptom:** Out of memory errors or system slowdown.

**Solutions:**

1. Reduce buffer size:
```python
buffer = ReplayBuffer(capacity=10000)  # Instead of 100000
```

2. Use smaller batch sizes:
```python
config = Config(batch_size=32)
```

3. Clear memory periodically:
```python
import gc
gc.collect()
torch.cuda.empty_cache()
```

## Environment Issues

### Problem: Environment Errors

**Symptom:**
```
AttributeError: 'GridWorld' object has no attribute 'agent_positions'
```

**Solutions:**

1. Always reset before using:
```python
obs, info = env.reset()
```

2. Check environment state:
```python
state = env.get_state()
print(f"Environment state: {state}")
```

### Problem: Invalid Actions

**Symptom:** Actions outside valid range.

**Solutions:**

1. Clip actions:
```python
action = np.clip(action, 0, env.action_space.n - 1)
```

2. Verify action space:
```python
print(f"Action space: {env.action_space}")
print(f"Valid actions: 0 to {env.action_space.n - 1}")
```

### Problem: Episode Never Terminates

**Symptom:** Episode runs indefinitely.

**Solutions:**

1. Set max_steps:
```python
env = GridWorld({"max_steps": 1000})
```

2. Check termination conditions:
```python
if step >= max_steps:
    break
```

3. Use truncation flag:
```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

## Visualization Issues

### Problem: Plots Not Showing

**Symptom:** Plotting functions don't display figures.

**Solutions:**

1. Use explicit show:
```python
import matplotlib.pyplot as plt
plotter.plot_rewards(rewards)
plt.show()
```

2. Save to file instead:
```python
plotter.plot_rewards(rewards, save_path="rewards.png")
```

3. Check backend:
```python
import matplotlib
print(matplotlib.get_backend())
matplotlib.use('TkAgg')  # or 'Agg' for file output only
```

### Problem: TensorBoard Not Working

**Symptom:** TensorBoard doesn't show data.

**Solutions:**

1. Verify log directory:
```bash
ls logs/experiment_name/
```

2. Start TensorBoard correctly:
```bash
tensorboard --logdir=logs --port=6006
```

3. Flush logger:
```python
logger.writer.flush()
```

4. Check TensorBoard version:
```bash
pip install --upgrade tensorboard
```

## Common Errors

### Error: "Shapes not compatible"

**Problem:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Solutions:**

1. Check input dimensions:
```python
print(f"State shape: {obs.shape}")
print(f"Expected: {agent.state_dim}")
```

2. Ensure correct reshaping:
```python
state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
```

### Error: "Expected tensor on device X but got Y"

**Problem:**
```
RuntimeError: Expected all tensors to be on the same device
```

**Solutions:**

1. Move tensors to correct device:
```python
state = state.to(agent.device)
```

2. Verify network device:
```python
print(f"Network device: {next(agent.network.parameters()).device}")
```

### Error: "Buffer is empty"

**Problem:**
```
ValueError: Cannot sample from empty buffer
```

**Solutions:**

1. Check buffer size:
```python
if len(buffer) >= batch_size:
    batch = buffer.sample(batch_size)
```

2. Increase learning_starts:
```python
config = Config(learning_starts=1000)
```

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
trainer.train()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Verify Gradients

```python
# Check if gradients are flowing
for name, param in agent.network.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

### Monitor Values

```python
# Add checkpoints to monitor values
print(f"State min/max: {state.min()}/{state.max()}")
print(f"Q-values min/max: {q_values.min()}/{q_values.max()}")
print(f"Loss: {loss.item()}")
```

## Getting Help

If you're still experiencing issues:

1. **Check Documentation:** Review the [API Reference](api_reference.md) and [Architecture](architecture.md) docs.

2. **Search Issues:** Look through existing GitHub issues for similar problems.

3. **Create Issue:** Open a new issue with:
   - Clear description of the problem
   - Minimal reproducible example
   - Error messages and stack traces
   - Environment information (Python version, OS, package versions)

4. **Community:** Join discussions in GitHub Discussions.

## Best Practices

To avoid common issues:

1. **Always use Config:** Define all hyperparameters in Config objects.

2. **Log Everything:** Use Logger to track experiments.

3. **Start Simple:** Begin with small environments and few agents.

4. **Validate Early:** Check shapes and values at each step.

5. **Save Checkpoints:** Save regularly to avoid losing progress.

6. **Monitor Training:** Watch TensorBoard during training.

7. **Test Components:** Use unit tests for custom components.

```bash
pytest tests/
```

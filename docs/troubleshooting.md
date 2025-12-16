# Troubleshooting

## Common Issues

### Installation Issues

**Problem**: PyTorch installation fails

**Solution**: Install PyTorch separately first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: Gymnasium not found

**Solution**: Install gymnasium:
```bash
pip install gymnasium
```

### Training Issues

**Problem**: Agents not learning

**Solutions**:
1. Check learning rate - try values between 1e-5 and 1e-2
2. Verify reward scaling - rewards should be roughly in [-10, 10]
3. Increase exploration (epsilon) early in training
4. Check network architecture - may need larger hidden dimensions

**Problem**: Training is slow

**Solutions**:
1. Use GPU if available: `device="cuda"`
2. Increase batch size
3. Reduce network size
4. Use vectorized environments

**Problem**: Out of memory

**Solutions**:
1. Reduce batch size
2. Reduce buffer size
3. Reduce network hidden dimensions
4. Use smaller number of agents

### Environment Issues

**Problem**: Environment not rendering

**Solution**: Install matplotlib:
```bash
pip install matplotlib
```

**Problem**: Custom environment crashes

**Solutions**:
1. Ensure reset() returns correct format: (states, info)
2. Ensure step() returns: (states, rewards, dones, info)
3. All arrays should be numpy arrays
4. Check dimensions match state_dim and action_dim

### Multi-Agent Issues

**Problem**: Agents not coordinating

**Solutions**:
1. Enable communication: `communication_enabled=True`
2. Increase coordination bonus
3. Try shared rewards: `shared_reward=True`
4. Increase context dimension

**Problem**: One agent dominates

**Solutions**:
1. Use fairness-based rewards
2. Reduce coordination bonus
3. Train agents separately first
4. Check individual agent rewards

## Performance Tips

### Training Speed

1. **Use GPU**: Set `device="cuda"` for GPU acceleration
2. **Batch Processing**: Increase batch size for better GPU utilization
3. **Parallel Environments**: Use vectorized environments
4. **Reduce Logging**: Decrease log_frequency

### Memory Optimization

1. **Smaller Networks**: Reduce hidden_dim
2. **Smaller Buffer**: Reduce buffer_size
3. **Gradient Checkpointing**: For very deep networks
4. **Mixed Precision**: Use torch.cuda.amp

### Convergence

1. **Learning Rate Schedule**: Use learning rate decay
2. **Reward Shaping**: Design informative rewards
3. **Curriculum Learning**: Start with easier tasks
4. **Warm Start**: Pre-train in simpler environments

## Debugging

### Enable Verbose Logging

```python
config = {
    "log_frequency": 1,
    "use_tensorboard": True
}
```

### Check Gradients

```python
for name, param in agent.q_network.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### Monitor Network Outputs

```python
with torch.no_grad():
    q_values = agent.q_network(state)
    print(f"Q-values: {q_values}")
```

## Getting Help

If you encounter issues not covered here:

1. Check GitHub Issues
2. Review documentation
3. Enable verbose logging
4. Create minimal reproducible example
5. Report issue with full error trace

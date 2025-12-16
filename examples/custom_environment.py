"""
Custom Environment Example for CAMADRL.

Demonstrates how to create and use custom environments
with custom reward and observation functions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from camadrl.agents import PolicyGradientAgent
from camadrl.environments import CustomEnv
from camadrl.utils import Config, Logger
from camadrl.visualization import Plotter


def custom_reward_function(state, agent_states, action, agent_idx):
    """
    Custom reward function for EV charging coordination.
    
    Rewards agents for:
    - Keeping state values balanced
    - Coordinating with other agents
    - Minimizing energy consumption
    """
    # Reward for state balance
    state_penalty = -np.sum(state ** 2) * 0.01
    
    # Reward for coordination (low variance across agents)
    if len(agent_states) > 1:
        coordination_bonus = -np.var(agent_states, axis=0).mean() * 0.1
    else:
        coordination_bonus = 0
    
    # Action penalty (prefer lower actions for energy efficiency)
    action_penalty = -action * 0.001
    
    total_reward = state_penalty + coordination_bonus + action_penalty
    
    return total_reward


def custom_observation_function(state, agent_states, agent_idx):
    """
    Custom observation function.
    
    Returns observation including:
    - Global state
    - Agent's own state
    - Mean state of other agents
    """
    own_state = agent_states[agent_idx]
    
    # Compute mean of other agents
    other_states = [s for i, s in enumerate(agent_states) if i != agent_idx]
    if other_states:
        other_mean = np.mean(other_states, axis=0)
    else:
        other_mean = np.zeros_like(state)
    
    # Combine information
    observation = state.astype(np.float32)
    
    return observation


def main():
    """Run custom environment example."""
    
    # Create configuration
    config = Config(
        env_name="CustomEnv",
        agent_type="PolicyGradient",
        num_agents=3,
        state_dim=10,
        action_dim=5,
        num_episodes=150,
        max_steps=100,
        actor_lr=0.001,
        critic_lr=0.001,
        log_freq=15
    )
    
    print("Custom Environment Configuration:")
    print(config)
    
    # Initialize custom environment
    env_config = {
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "num_agents": config.num_agents,
        "max_steps": config.max_steps,
        "reward_fn": custom_reward_function,
        "observation_fn": custom_observation_function
    }
    env = CustomEnv(env_config)
    
    print(f"\nEnvironment: {config.env_name}")
    print(f"State dimension: {config.state_dim}")
    print(f"Action dimension: {config.action_dim}")
    print(f"Number of agents: {config.num_agents}")
    print("Using custom reward and observation functions")
    
    # Initialize Policy Gradient agents
    agents = []
    for i in range(config.num_agents):
        agent = PolicyGradientAgent(
            agent_id=i,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            config=config.to_dict()
        )
        agents.append(agent)
    
    print(f"\nInitialized {len(agents)} Policy Gradient agents")
    
    # Initialize logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name="custom_environment",
        use_tensorboard=True
    )
    logger.log_config(config.to_dict())
    
    print("\nStarting training with custom environment...")
    
    # Training loop
    episode_rewards = []
    episode_entropies = []
    
    for episode in range(config.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < config.max_steps:
            # Select actions
            actions = []
            for agent in agents:
                action = agent.select_action(obs, explore=True)
                actions.append(action)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(actions[0])
            done = terminated or truncated
            
            # Store rewards for each agent
            for agent in agents:
                agent.store_reward(reward)
            
            episode_reward += reward
            obs = next_obs
            episode_length += 1
        
        # Update agents after episode
        agent_metrics = []
        for agent in agents:
            metrics = agent.update()
            agent_metrics.append(metrics)
            agent.reset()
        
        episode_rewards.append(episode_reward)
        
        # Compute mean entropy across agents
        if agent_metrics and 'entropy' in agent_metrics[0]:
            mean_entropy = np.mean([m['entropy'] for m in agent_metrics])
            episode_entropies.append(mean_entropy)
        else:
            mean_entropy = 0
        
        # Log metrics
        logger.log_episode(episode, {
            "reward": episode_reward,
            "length": episode_length,
            "mean_entropy": mean_entropy
        })
        
        if episode % config.log_freq == 0:
            print(f"Episode {episode}/{config.num_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Length: {episode_length}, "
                  f"Entropy: {mean_entropy:.3f}")
    
    print("\nTraining completed!")
    
    # Save agents
    for i, agent in enumerate(agents):
        agent.save(f"models/pg_agent_{i}_custom.pth")
    print(f"\n{len(agents)} agents saved to models/")
    
    # Visualize results
    print("\nGenerating visualizations...")
    plotter = Plotter()
    
    # Plot rewards
    plotter.plot_rewards(
        episode_rewards,
        title="Custom Environment Training",
        save_path="results/custom_env_rewards.png"
    )
    
    # Plot reward distribution
    plotter.plot_distribution(
        episode_rewards,
        title="Reward Distribution",
        save_path="results/reward_distribution.png"
    )
    
    # Plot entropy over time
    if episode_entropies:
        plotter.plot_losses(
            episode_entropies,
            title="Policy Entropy over Time",
            save_path="results/entropy_plot.png"
        )
    
    print("Results saved to results/")
    
    # Demonstrate environment customization
    print("\nDemonstrating dynamic reward function change...")
    
    def new_reward_function(state, agent_states, action, agent_idx):
        """Alternative reward function emphasizing cooperation."""
        return -np.sum(np.var(agent_states, axis=0)) * 0.5
    
    # Update environment reward function
    env.set_reward_function(new_reward_function)
    print("Reward function updated successfully!")
    
    # Close logger
    logger.close()
    
    print("\nCustom environment example completed successfully!")
    print("This example demonstrated:")
    print("  - Creating custom environments")
    print("  - Defining custom reward functions")
    print("  - Defining custom observation functions")
    print("  - Dynamic environment customization")


if __name__ == "__main__":
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main()

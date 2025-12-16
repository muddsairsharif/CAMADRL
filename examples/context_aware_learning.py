"""
Context-Aware Learning Example for CAMADRL.

Demonstrates context-aware deep reinforcement learning using
CADRL agents that consider environmental context in decision making.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from camadrl.agents import CADRLAgent
from camadrl.environments import TrafficSim
from camadrl.trainers import MultiAgentTrainer
from camadrl.utils import Config, Logger, Metrics
from camadrl.visualization import Plotter, HeatmapGenerator


def main():
    """Run context-aware learning example."""
    
    # Create configuration
    config = Config(
        env_name="TrafficSim",
        agent_type="CADRL",
        num_agents=5,
        num_episodes=300,
        max_steps=200,
        learning_rate=0.0005,
        batch_size=64,
        buffer_size=50000,
        learning_starts=1000,
        log_freq=25,
        eval_freq=50,
        use_attention=True
    )
    
    print("Context-Aware Learning Configuration:")
    print(config)
    
    # Initialize environment
    env_config = {
        "network_size": 20,
        "num_agents": config.num_agents,
        "num_charging_stations": 8,
        "traffic_density": 0.3,
        "max_steps": config.max_steps
    }
    env = TrafficSim(env_config)
    
    print(f"\nEnvironment: {config.env_name}")
    print(f"Network size: {env_config['network_size']}")
    print(f"Charging stations: {env_config['num_charging_stations']}")
    print(f"Traffic density: {env_config['traffic_density']}")
    
    # Initialize CADRL agents with context awareness
    agents = []
    context_dim = 10  # Dimension of context information
    
    for i in range(config.num_agents):
        agent = CADRLAgent(
            agent_id=i,
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            context_dim=context_dim,
            config=config.to_dict()
        )
        agents.append(agent)
    
    print(f"\nInitialized {len(agents)} CADRL agents with context dimension {context_dim}")
    
    # Initialize logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name="context_aware_learning",
        use_tensorboard=True
    )
    logger.log_config(config.to_dict())
    
    print("\nStarting context-aware training...")
    
    # Custom training loop to incorporate context
    episode_rewards = []
    episode_contexts = []
    
    for episode in range(config.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Generate context information (e.g., traffic conditions, time of day, grid load)
        context = np.random.rand(context_dim)  # Simplified context
        episode_contexts.append(context)
        
        while not done and episode_length < config.max_steps:
            # Select actions with context awareness
            actions = []
            for agent in agents:
                action = agent.select_action(obs, context=context, explore=True)
                actions.append(action)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(actions[0])
            done = terminated or truncated
            
            episode_reward += reward
            obs = next_obs
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        
        # Log metrics
        logger.log_episode(episode, {
            "reward": episode_reward,
            "length": episode_length,
            "context_mean": context.mean(),
            "context_std": context.std()
        })
        
        if episode % config.log_freq == 0:
            print(f"Episode {episode}/{config.num_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Length: {episode_length}, "
                  f"Context: [{context[0]:.2f}, ...]")
    
    print("\nTraining completed!")
    
    # Compute metrics
    metrics = Metrics()
    training_metrics = metrics.compute_training_metrics(episode_rewards)
    
    print("\nTraining Metrics:")
    for key, value in training_metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # Save agents
    for i, agent in enumerate(agents):
        agent.save(f"models/cadrl_agent_{i}.pth")
    print(f"\n{len(agents)} CADRL agents saved to models/")
    
    # Visualize results
    print("\nGenerating visualizations...")
    plotter = Plotter()
    
    # Plot training progress
    plotter.plot_rewards(
        episode_rewards,
        window_size=50,
        title="Context-Aware Learning Progress",
        save_path="results/cadrl_training_rewards.png"
    )
    
    # Plot convergence analysis
    plotter.plot_convergence(
        episode_rewards,
        title="CADRL Convergence Analysis",
        save_path="results/cadrl_convergence.png"
    )
    
    # Analyze context influence
    context_means = [ctx.mean() for ctx in episode_contexts]
    metrics_dict = {
        "Rewards": episode_rewards,
        "Context Mean": context_means
    }
    plotter.plot_metrics_comparison(
        metrics_dict,
        title="Reward vs Context Analysis",
        save_path="results/context_analysis.png"
    )
    
    print("Results saved to results/")
    
    # Generate heatmaps for spatial analysis
    print("\nGenerating spatial analysis...")
    heatmap_gen = HeatmapGenerator()
    
    # Simulate episode to collect position data
    obs, _ = env.reset()
    positions = []
    done = False
    step = 0
    
    while not done and step < 100:
        context = np.random.rand(context_dim)
        action = agents[0].select_action(obs, context=context, explore=False)
        next_obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if hasattr(env, 'agent_positions') and env.agent_positions is not None:
            positions.append(tuple(env.agent_positions[0]))
        
        obs = next_obs
        step += 1
    
    if positions:
        heatmap_gen.generate_position_heatmap(
            positions,
            bounds=(0, env_config['network_size'], 0, env_config['network_size']),
            title="Agent Position Distribution",
            save_path="results/position_heatmap.png"
        )
    
    # Close logger
    logger.close()
    
    print("\nContext-aware learning example completed successfully!")


if __name__ == "__main__":
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main()

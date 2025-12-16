"""
Multi-Agent Coordination Example for CAMADRL.

Demonstrates coordinated multi-agent training using
multiple DQN agents in a shared environment.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camadrl.agents import DQNAgent
from camadrl.environments import GridWorld
from camadrl.trainers import MultiAgentTrainer
from camadrl.utils import Config, Logger
from camadrl.visualization import Plotter, TrajectoryVisualizer


def main():
    """Run multi-agent coordination example."""
    
    # Create configuration
    config = Config(
        env_name="GridWorld",
        agent_type="DQN",
        num_agents=3,
        num_episodes=200,
        max_steps=150,
        learning_rate=0.001,
        batch_size=64,
        buffer_size=10000,
        learning_starts=500,
        log_freq=20,
        eval_freq=50
    )
    
    print("Multi-Agent Configuration:")
    print(config)
    
    # Initialize environment
    env_config = {
        "grid_size": 10,
        "num_agents": config.num_agents,
        "num_charging_stations": 5,
        "max_steps": config.max_steps
    }
    env = GridWorld(env_config)
    
    # Initialize agents
    agents = []
    for i in range(config.num_agents):
        agent = DQNAgent(
            agent_id=i,
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=config.to_dict()
        )
        agents.append(agent)
    
    print(f"\nInitialized {len(agents)} agents")
    
    # Initialize logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name="multi_agent_coordination",
        use_tensorboard=True
    )
    logger.log_config(config.to_dict())
    
    # Initialize trainer
    trainer = MultiAgentTrainer(
        agents=agents,
        env=env,
        config=config.to_dict()
    )
    
    print("\nStarting multi-agent training...")
    
    # Train agents
    training_stats = trainer.train()
    
    print("\nTraining completed!")
    print(f"Total episodes: {training_stats['total_episodes']}")
    print(f"Mean reward (last 100): {training_stats['mean_episode_reward']:.2f}")
    print(f"Best episode reward: {training_stats['best_episode_reward']:.2f}")
    
    # Save agents
    for i, agent in enumerate(agents):
        agent.save(f"models/agent_{i}_coordination.pth")
    print(f"\n{len(agents)} agents saved to models/")
    
    # Visualize results
    plotter = Plotter()
    
    # Plot combined rewards
    plotter.plot_rewards(
        trainer.episode_rewards,
        title="Multi-Agent Training Progress",
        save_path="results/multi_agent_rewards.png"
    )
    
    # Plot individual agent rewards
    agent_rewards = {i: [] for i in range(config.num_agents)}
    # Note: In a real implementation, you'd track individual rewards
    # For this example, we'll use the combined rewards
    for i in range(config.num_agents):
        agent_rewards[i] = trainer.episode_rewards
    
    plotter.plot_multi_agent_rewards(
        agent_rewards,
        title="Individual Agent Rewards",
        save_path="results/individual_agent_rewards.png"
    )
    
    print("Results saved to results/")
    
    # Demonstrate coordination visualization
    print("\nGenerating trajectory visualization...")
    viz = TrajectoryVisualizer()
    
    # Simulate a test episode to get trajectories
    obs, _ = env.reset()
    trajectories = {i: [] for i in range(config.num_agents)}
    done = False
    step = 0
    
    while not done and step < 50:
        for i, agent in enumerate(agents):
            action = agent.select_action(obs, explore=False)
            trajectories[i].append((
                env.agent_positions[i][0] if hasattr(env, 'agent_positions') and env.agent_positions is not None else step,
                env.agent_positions[i][1] if hasattr(env, 'agent_positions') and env.agent_positions is not None else step
            ))
        
        obs, _, terminated, truncated, _ = env.step(agents[0].select_action(obs, explore=False))
        done = terminated or truncated
        step += 1
    
    viz.plot_multi_agent_trajectories(
        trajectories,
        title="Agent Trajectories",
        save_path="results/agent_trajectories.png"
    )
    
    # Close logger
    logger.close()
    
    print("\nMulti-agent coordination example completed successfully!")


if __name__ == "__main__":
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main()

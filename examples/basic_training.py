"""
Basic Training Example for CAMADRL.

Demonstrates basic single-agent training using DQN
in a GridWorld environment.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camadrl.agents import DQNAgent
from camadrl.environments import GridWorld
from camadrl.trainers import BaseTrainer
from camadrl.utils import Config, Logger
from camadrl.visualization import Plotter


def main():
    """Run basic training example."""
    
    # Create configuration
    config = Config(
        env_name="GridWorld",
        agent_type="DQN",
        num_episodes=100,
        max_steps=100,
        learning_rate=0.001,
        batch_size=32,
        epsilon=1.0,
        epsilon_decay=0.995,
        log_freq=10
    )
    
    print("Configuration:")
    print(config)
    
    # Initialize environment
    env_config = {
        "grid_size": 10,
        "num_agents": 1,
        "num_charging_stations": 3,
        "max_steps": config.max_steps
    }
    env = GridWorld(env_config)
    
    # Initialize agent
    agent = DQNAgent(
        agent_id=0,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config.to_dict()
    )
    
    # Initialize logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name="basic_training",
        use_tensorboard=True
    )
    logger.log_config(config.to_dict())
    
    print("\nStarting training...")
    print(f"Environment: {config.env_name}")
    print(f"Agent: {config.agent_type}")
    print(f"Episodes: {config.num_episodes}")
    
    # Training loop
    episode_rewards = []
    
    for episode in range(config.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < config.max_steps:
            # Select action
            action = agent.select_action(obs, explore=True)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            obs = next_obs
            step += 1
        
        episode_rewards.append(episode_reward)
        agent.reset()
        
        # Log metrics
        logger.log_episode(episode, {
            "reward": episode_reward,
            "length": step,
            "epsilon": agent.epsilon
        })
        
        if episode % config.log_freq == 0:
            print(f"Episode {episode}/{config.num_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Steps: {step}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("\nTraining completed!")
    
    # Save agent
    agent.save("models/basic_agent.pth")
    print("Agent saved to models/basic_agent.pth")
    
    # Visualize results
    plotter = Plotter()
    plotter.plot_rewards(
        episode_rewards,
        title="Training Progress",
        save_path="results/basic_training_rewards.png"
    )
    print("Results saved to results/basic_training_rewards.png")
    
    # Close logger
    logger.close()
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main()

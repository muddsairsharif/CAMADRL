"""Basic training example for single and multi-agent scenarios."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camadrl.environments import GridWorld
from camadrl.agents import DQNAgent
from camadrl.trainers import MultiAgentTrainer

def main():
    """Run basic training example."""
    # Create environment
    num_agents = 2
    env = GridWorld(num_agents=num_agents, grid_size=10)
    
    # Create agents
    agents = []
    for i in range(num_agents):
        agent = DQNAgent(
            agent_id=i,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            config={"hidden_dim": 128, "learning_rate": 0.001}
        )
        agents.append(agent)
    
    # Create trainer
    config = {
        "num_episodes": 100,
        "eval_frequency": 20,
        "log_frequency": 5
    }
    trainer = MultiAgentTrainer(env, agents, config)
    
    # Train
    print("Starting basic training...")
    results = trainer.train()
    print(f"Training completed! Final metrics: {results}")

if __name__ == "__main__":
    main()

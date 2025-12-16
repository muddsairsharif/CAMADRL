"""Multi-agent coordination example with communication."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camadrl.environments import TrafficSim
from camadrl.agents import CADRLAgent
from camadrl.trainers import MultiAgentTrainer

def main():
    """Run multi-agent coordination example."""
    # Create environment
    num_agents = 5
    env = TrafficSim(num_agents=num_agents)
    
    # Create CADRL agents with communication
    agents = []
    for i in range(num_agents):
        agent = CADRLAgent(
            agent_id=i,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            context_dim=64,
            config={"hidden_dim": 256, "learning_rate": 0.0005}
        )
        agents.append(agent)
    
    # Create trainer with communication enabled
    config = {
        "num_episodes": 200,
        "communication_enabled": True,
        "coordination_bonus": 0.2,
        "eval_frequency": 25,
    }
    trainer = MultiAgentTrainer(env, agents, config)
    
    # Train
    print("Starting multi-agent coordination training...")
    results = trainer.train()
    print(f"Training completed! Final metrics: {results}")

if __name__ == "__main__":
    main()

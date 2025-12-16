"""Context-aware learning example."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camadrl.environments import GridWorld
from camadrl.agents import CADRLAgent
from camadrl.trainers import MultiAgentTrainer

def main():
    """Run context-aware learning example."""
    num_agents = 4
    env = GridWorld(num_agents=num_agents, grid_size=15)
    
    agents = []
    for i in range(num_agents):
        agent = CADRLAgent(
            agent_id=i,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            context_dim=128,
            config={"hidden_dim": 256}
        )
        agents.append(agent)
    
    config = {
        "num_episodes": 150,
        "communication_enabled": True,
        "shared_reward": False,
    }
    trainer = MultiAgentTrainer(env, agents, config)
    
    print("Starting context-aware learning...")
    trainer.train()

if __name__ == "__main__":
    main()

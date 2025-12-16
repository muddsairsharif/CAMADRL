"""Custom environment example."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camadrl.environments import CustomEnv
from camadrl.agents import PolicyGradientAgent
from camadrl.trainers import MultiAgentTrainer

def main():
    """Run custom environment example."""
    num_agents = 3
    state_dim = 8
    action_dim = 4
    
    env = CustomEnv(num_agents=num_agents, state_dim=state_dim, action_dim=action_dim)
    
    agents = []
    for i in range(num_agents):
        agent = PolicyGradientAgent(
            agent_id=i,
            state_dim=state_dim,
            action_dim=action_dim,
            config={"hidden_dim": 128}
        )
        agents.append(agent)
    
    config = {"num_episodes": 100}
    trainer = MultiAgentTrainer(env, agents, config)
    
    print("Starting custom environment training...")
    trainer.train()

if __name__ == "__main__":
    main()

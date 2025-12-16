"""
Multi-agent trainer for CAMADRL framework.

This module implements training loops for multi-agent scenarios with
coordination and communication.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch

from camadrl.trainers.base_trainer import BaseTrainer
from camadrl.agents.base_agent import BaseAgent
from camadrl.environments.base_env import BaseEnv


class MultiAgentTrainer(BaseTrainer):
    """
    Trainer for multi-agent scenarios.
    
    Implements training loops for multiple agents that can coordinate
    and share information to achieve common or individual goals.
    
    Attributes:
        communication_enabled: Whether agents can communicate
        coordination_bonus: Bonus reward for coordination
    """
    
    def __init__(
        self,
        env: BaseEnv,
        agents: List[BaseAgent],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-agent trainer.
        
        Args:
            env: Training environment
            agents: List of agents to train
            config: Configuration dictionary
        """
        super().__init__(env, agents, config)
        
        # Multi-agent specific parameters
        self.communication_enabled = self.config.get("communication_enabled", True)
        self.coordination_bonus = self.config.get("coordination_bonus", 0.1)
        self.shared_reward = self.config.get("shared_reward", False)
        
        # Context sharing
        self.context_history = []
        
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode with multiple agents.
        
        Returns:
            Dictionary of episode metrics
        """
        # Reset environment
        states, _ = self.env.reset()
        
        # Reset agents
        for agent in self.agents:
            agent.reset()
        
        episode_rewards = [0.0] * len(self.agents)
        episode_steps = 0
        done = False
        
        while not done and episode_steps < self.max_steps_per_episode:
            # Collect contexts from all agents if communication is enabled
            contexts = None
            if self.communication_enabled:
                contexts = self._collect_contexts(states)
            
            # Select actions for all agents
            actions = []
            for i, agent in enumerate(self.agents):
                if self.communication_enabled and hasattr(agent, 'select_action'):
                    # Check if agent supports context
                    try:
                        action, _ = agent.select_action(states[i], context=contexts[i])
                    except TypeError:
                        action = agent.select_action(states[i])
                else:
                    action = agent.select_action(states[i])
                actions.append(action)
            
            # Execute actions in environment
            next_states, rewards, dones, info = self.env.step(actions)
            
            # Apply coordination bonus if enabled
            if self.coordination_bonus > 0:
                coordination_reward = self._calculate_coordination_bonus(actions, states)
                rewards = [r + coordination_reward for r in rewards]
            
            # Apply shared reward if enabled
            if self.shared_reward:
                avg_reward = np.mean(rewards)
                rewards = [avg_reward] * len(rewards)
            
            # Update agents
            for i, agent in enumerate(self.agents):
                context = contexts[i] if contexts is not None else None
                
                # Handle different agent update signatures
                if self.communication_enabled and hasattr(agent, 'update'):
                    try:
                        agent.update(
                            states[i],
                            actions[i],
                            rewards[i],
                            next_states[i],
                            dones[i],
                            context=context
                        )
                    except TypeError:
                        agent.update(
                            states[i],
                            actions[i],
                            rewards[i],
                            next_states[i],
                            dones[i]
                        )
                else:
                    agent.update(
                        states[i],
                        actions[i],
                        rewards[i],
                        next_states[i],
                        dones[i]
                    )
                
                episode_rewards[i] += rewards[i]
            
            # Update state
            states = next_states
            episode_steps += 1
            self.total_steps += 1
            
            # Check if all agents are done
            done = all(dones)
        
        # Calculate episode metrics
        metrics = {
            "episode_reward_mean": np.mean(episode_rewards),
            "episode_reward_sum": np.sum(episode_rewards),
            "episode_reward_max": np.max(episode_rewards),
            "episode_reward_min": np.min(episode_rewards),
            "episode_steps": episode_steps,
            "coordination_score": self._calculate_coordination_score(episode_rewards),
        }
        
        # Add per-agent rewards
        for i, reward in enumerate(episode_rewards):
            metrics[f"agent_{i}_reward"] = reward
        
        return metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agents' performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Set agents to evaluation mode
        for agent in self.agents:
            agent.set_training_mode(False)
        
        all_rewards = []
        all_steps = []
        
        for _ in range(num_episodes):
            states, _ = self.env.reset()
            
            episode_rewards = [0.0] * len(self.agents)
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.max_steps_per_episode:
                # Collect contexts
                contexts = None
                if self.communication_enabled:
                    contexts = self._collect_contexts(states)
                
                # Select actions (no exploration)
                actions = []
                for i, agent in enumerate(self.agents):
                    if self.communication_enabled and hasattr(agent, 'select_action'):
                        try:
                            action, _ = agent.select_action(states[i], context=contexts[i], explore=False)
                        except TypeError:
                            action = agent.select_action(states[i], explore=False)
                    else:
                        action = agent.select_action(states[i], explore=False)
                    actions.append(action)
                
                # Execute actions
                next_states, rewards, dones, _ = self.env.step(actions)
                
                for i in range(len(self.agents)):
                    episode_rewards[i] += rewards[i]
                
                states = next_states
                episode_steps += 1
                done = all(dones)
            
            all_rewards.append(episode_rewards)
            all_steps.append(episode_steps)
        
        # Set agents back to training mode
        for agent in self.agents:
            agent.set_training_mode(True)
        
        # Calculate metrics
        all_rewards = np.array(all_rewards)
        metrics = {
            "eval_reward_mean": np.mean(all_rewards),
            "eval_reward_std": np.std(all_rewards),
            "eval_steps_mean": np.mean(all_steps),
            "eval_success_rate": self._calculate_success_rate(all_rewards),
        }
        
        return metrics
    
    def _collect_contexts(self, states: List[np.ndarray]) -> List[np.ndarray]:
        """
        Collect context information from all agents.
        
        Args:
            states: List of agent states
            
        Returns:
            List of context arrays for each agent
        """
        # Simple context: concatenate other agents' states
        contexts = []
        for i in range(len(states)):
            other_states = [states[j] for j in range(len(states)) if j != i]
            if other_states:
                context = np.concatenate(other_states)
            else:
                context = np.zeros_like(states[i])
            contexts.append(context)
        
        return contexts
    
    def _calculate_coordination_bonus(
        self,
        actions: List[int],
        states: List[np.ndarray]
    ) -> float:
        """
        Calculate coordination bonus based on agent actions.
        
        Args:
            actions: List of actions taken by agents
            states: List of agent states
            
        Returns:
            Coordination bonus value
        """
        # Simple coordination metric: reward similar actions
        if len(actions) <= 1:
            return 0.0
        
        action_variance = np.var(actions)
        bonus = self.coordination_bonus * np.exp(-action_variance)
        
        return bonus
    
    def _calculate_coordination_score(self, rewards: List[float]) -> float:
        """
        Calculate coordination score based on reward distribution.
        
        Args:
            rewards: List of agent rewards
            
        Returns:
            Coordination score
        """
        if len(rewards) <= 1:
            return 1.0
        
        # Coordination score: inverse of reward variance
        reward_std = np.std(rewards)
        score = 1.0 / (1.0 + reward_std)
        
        return score
    
    def _calculate_success_rate(self, rewards: np.ndarray) -> float:
        """
        Calculate success rate based on rewards.
        
        Args:
            rewards: Array of episode rewards
            
        Returns:
            Success rate
        """
        # Consider episode successful if mean reward is positive
        success_threshold = self.config.get("success_threshold", 0.0)
        successes = np.mean(rewards, axis=1) > success_threshold
        
        return np.mean(successes)

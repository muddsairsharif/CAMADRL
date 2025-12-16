"""
Grid World environment for multi-agent navigation.

This module implements a grid-based navigation environment where multiple
agents must navigate to their goals while avoiding collisions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from camadrl.environments.base_env import BaseEnv


class GridWorld(BaseEnv):
    """
    Grid World environment for multi-agent navigation.
    
    Agents navigate on a 2D grid to reach their goal positions while
    avoiding obstacles and collisions with other agents.
    
    Attributes:
        grid_size: Size of the grid (grid_size x grid_size)
        agent_positions: Current positions of all agents
        goal_positions: Goal positions for all agents
        obstacles: Set of obstacle positions
    """
    
    def __init__(
        self,
        num_agents: int = 4,
        grid_size: int = 10,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Grid World environment.
        
        Args:
            num_agents: Number of agents
            grid_size: Size of the grid
            config: Configuration dictionary
        """
        # State: [agent_x, agent_y, goal_x, goal_y, other_agents_info]
        state_dim = 4 + (num_agents - 1) * 2  # Own pos, goal, other agents' positions
        action_dim = 5  # Up, Down, Left, Right, Stay
        
        super().__init__(num_agents, state_dim, action_dim, config)
        
        self.grid_size = grid_size
        self.agent_positions = np.zeros((num_agents, 2), dtype=np.int32)
        self.goal_positions = np.zeros((num_agents, 2), dtype=np.int32)
        self.obstacles = set()
        
        # Reward parameters
        self.reward_goal = self.config.get("reward_goal", 10.0)
        self.reward_step = self.config.get("reward_step", -0.1)
        self.reward_collision = self.config.get("reward_collision", -5.0)
        
        # Action mapping
        self.action_map = {
            0: np.array([0, 1]),   # Up
            1: np.array([0, -1]),  # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0]),   # Right
            4: np.array([0, 0]),   # Stay
        }
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (initial states, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # Randomly place agents
        occupied = set()
        for i in range(self.num_agents):
            while True:
                pos = np.random.randint(0, self.grid_size, size=2)
                pos_tuple = tuple(pos)
                if pos_tuple not in occupied and pos_tuple not in self.obstacles:
                    self.agent_positions[i] = pos
                    occupied.add(pos_tuple)
                    break
        
        # Randomly place goals
        for i in range(self.num_agents):
            while True:
                pos = np.random.randint(0, self.grid_size, size=2)
                pos_tuple = tuple(pos)
                if pos_tuple not in occupied and pos_tuple not in self.obstacles:
                    self.goal_positions[i] = pos
                    occupied.add(pos_tuple)
                    break
        
        # Generate obstacles
        num_obstacles = self.config.get("num_obstacles", self.grid_size)
        self.obstacles.clear()
        for _ in range(num_obstacles):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if pos not in occupied:
                    self.obstacles.add(pos)
                    break
        
        states = [self._get_agent_state(i) for i in range(self.num_agents)]
        info = self.get_info()
        
        return states, info
    
    def step(
        self,
        actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            Tuple of (states, rewards, dones, info)
        """
        self.current_step += 1
        
        # Calculate new positions
        new_positions = []
        for i, action in enumerate(actions):
            if action not in self.action_map:
                action = 4  # Stay if invalid action
            
            new_pos = self.agent_positions[i] + self.action_map[action]
            # Clip to grid boundaries
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            new_positions.append(new_pos)
        
        # Check for collisions and update positions
        rewards = [self.reward_step] * self.num_agents
        dones = [False] * self.num_agents
        
        for i in range(self.num_agents):
            new_pos = tuple(new_positions[i])
            
            # Check obstacle collision
            if new_pos in self.obstacles:
                rewards[i] += self.reward_collision
                continue  # Don't move
            
            # Check agent collision
            collision = False
            for j in range(self.num_agents):
                if i != j and np.array_equal(new_positions[i], new_positions[j]):
                    collision = True
                    rewards[i] += self.reward_collision
                    break
            
            if not collision:
                self.agent_positions[i] = new_positions[i]
                
                # Check goal reached
                if np.array_equal(self.agent_positions[i], self.goal_positions[i]):
                    rewards[i] += self.reward_goal
                    dones[i] = True
        
        # Check episode termination
        all_done = all(dones) or self.current_step >= self.max_steps
        if all_done:
            dones = [True] * self.num_agents
        
        states = [self._get_agent_state(i) for i in range(self.num_agents)]
        info = self.get_info()
        
        return states, rewards, dones, info
    
    def _get_agent_state(self, agent_id: int) -> np.ndarray:
        """
        Get state observation for specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            State observation array
        """
        # Own position and goal
        own_pos = self.agent_positions[agent_id].astype(np.float32) / self.grid_size
        goal_pos = self.goal_positions[agent_id].astype(np.float32) / self.grid_size
        
        # Other agents' positions (relative)
        other_positions = []
        for i in range(self.num_agents):
            if i != agent_id:
                rel_pos = (self.agent_positions[i] - self.agent_positions[agent_id]).astype(np.float32) / self.grid_size
                other_positions.extend(rel_pos)
        
        state = np.concatenate([own_pos, goal_pos, other_positions])
        return state
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame if mode is 'rgb_array'
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Draw obstacles
        for obs in self.obstacles:
            ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color='black'))
        
        # Draw goals
        for i, goal in enumerate(self.goal_positions):
            ax.plot(goal[0], goal[1], 'g*', markersize=20, label=f'Goal {i}' if i == 0 else '')
        
        # Draw agents
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_agents))
        for i, pos in enumerate(self.agent_positions):
            ax.plot(pos[0], pos[1], 'o', color=colors[i], markersize=15, label=f'Agent {i}')
        
        ax.legend()
        ax.set_title(f'Grid World - Step {self.current_step}')
        
        if mode == "rgb_array":
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        else:
            plt.show()
            plt.close(fig)
            return None
    
    def get_state(self, agent_id: int) -> np.ndarray:
        """Get state for specific agent."""
        return self._get_agent_state(agent_id)
    
    def get_global_state(self) -> np.ndarray:
        """Get global state."""
        return np.concatenate([
            self.agent_positions.flatten(),
            self.goal_positions.flatten()
        ]).astype(np.float32) / self.grid_size

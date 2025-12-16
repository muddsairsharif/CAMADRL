"""
Grid World Environment for EV charging coordination.

A 2D grid-based environment where agents navigate and coordinate
charging activities at different locations.
"""

from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from camadrl.environments.base_env import BaseEnv


class GridWorld(BaseEnv):
    """
    Grid World environment for multi-agent EV charging coordination.
    
    Agents navigate a 2D grid with charging stations and must coordinate
    to optimize charging schedules while avoiding conflicts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Grid World environment.
        
        Args:
            config: Configuration dictionary with keys:
                - grid_size: Size of the grid (default: 10)
                - num_agents: Number of agents (default: 3)
                - num_charging_stations: Number of charging stations (default: 5)
                - max_steps: Maximum steps per episode (default: 100)
        """
        super().__init__(config)
        
        self.grid_size = self.config.get("grid_size", 10)
        self.num_agents = self.config.get("num_agents", 3)
        self.num_charging_stations = self.config.get("num_charging_stations", 5)
        
        # Define spaces
        # Observation: [agent_x, agent_y, battery_level, nearest_station_x, nearest_station_y, station_availability]
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(6,),
            dtype=np.float32
        )
        
        # Action: [up, down, left, right, charge]
        self.action_space = spaces.Discrete(5)
        
        # Environment state
        self.agent_positions = None
        self.agent_batteries = None
        self.charging_stations = None
        self.station_availability = None
        
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize agent positions randomly
        self.agent_positions = np.random.randint(0, self.grid_size, size=(self.num_agents, 2))
        
        # Initialize battery levels (0.2 to 1.0)
        self.agent_batteries = np.random.uniform(0.2, 1.0, size=self.num_agents)
        
        # Place charging stations
        self.charging_stations = np.random.randint(
            0, self.grid_size, size=(self.num_charging_stations, 2)
        )
        self.station_availability = np.ones(self.num_charging_stations, dtype=bool)
        
        obs = self._get_observation(0)
        info = self.get_state()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        agent_idx = self.current_step % self.num_agents
        
        reward = 0.0
        pos = self.agent_positions[agent_idx]
        
        # Execute action
        if action == 0:  # Up
            pos[1] = min(pos[1] + 1, self.grid_size - 1)
            reward -= 0.01  # Movement cost
        elif action == 1:  # Down
            pos[1] = max(pos[1] - 1, 0)
            reward -= 0.01
        elif action == 2:  # Left
            pos[0] = max(pos[0] - 1, 0)
            reward -= 0.01
        elif action == 3:  # Right
            pos[0] = min(pos[0] + 1, self.grid_size - 1)
            reward -= 0.01
        elif action == 4:  # Charge
            # Check if at charging station
            for i, station_pos in enumerate(self.charging_stations):
                if np.array_equal(pos, station_pos) and self.station_availability[i]:
                    charge_amount = min(0.2, 1.0 - self.agent_batteries[agent_idx])
                    self.agent_batteries[agent_idx] += charge_amount
                    reward += charge_amount * 10  # Reward for charging
                    break
            else:
                reward -= 0.5  # Penalty for trying to charge at wrong location
        
        # Battery drain
        self.agent_batteries[agent_idx] -= 0.01
        
        # Check if battery depleted
        if self.agent_batteries[agent_idx] <= 0:
            reward -= 10  # Large penalty for battery depletion
            terminated = True
        else:
            terminated = False
        
        # Reward for maintaining high battery
        reward += self.agent_batteries[agent_idx] * 0.1
        
        super().step(action)
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_observation(agent_idx)
        info = self.get_state()
        info["agent_idx"] = agent_idx
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for a specific agent."""
        pos = self.agent_positions[agent_idx]
        battery = self.agent_batteries[agent_idx]
        
        # Find nearest charging station
        distances = np.linalg.norm(self.charging_stations - pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_station = self.charging_stations[nearest_idx]
        station_available = float(self.station_availability[nearest_idx])
        
        obs = np.array([
            pos[0] / self.grid_size,
            pos[1] / self.grid_size,
            battery,
            nearest_station[0] / self.grid_size,
            nearest_station[1] / self.grid_size,
            station_available
        ], dtype=np.float32)
        
        return obs
    
    def render(self) -> np.ndarray:
        """Render the grid world as a 2D array."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        
        # Place charging stations
        for station_pos in self.charging_stations:
            grid[station_pos[1], station_pos[0]] = 'C'
        
        # Place agents
        for i, pos in enumerate(self.agent_positions):
            grid[pos[1], pos[0]] = str(i)
        
        return grid
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        state = super().get_state()
        state.update({
            "agent_positions": self.agent_positions.tolist() if self.agent_positions is not None else None,
            "agent_batteries": self.agent_batteries.tolist() if self.agent_batteries is not None else None,
            "charging_stations": self.charging_stations.tolist() if self.charging_stations is not None else None,
        })
        return state

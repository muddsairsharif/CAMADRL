"""
Traffic Simulation Environment for EV charging coordination.

Simulates urban traffic with EVs that need to coordinate charging
while navigating through traffic networks.
"""

from typing import Any, Dict, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from camadrl.environments.base_env import BaseEnv


class TrafficSim(BaseEnv):
    """
    Traffic simulation environment for EV charging coordination.
    
    Models a traffic network with roads, intersections, and charging stations.
    EVs must navigate traffic while optimizing charging schedules.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Traffic Simulation environment.
        
        Args:
            config: Configuration dictionary with keys:
                - network_size: Size of the road network (default: 20)
                - num_agents: Number of EV agents (default: 5)
                - num_charging_stations: Number of charging stations (default: 8)
                - traffic_density: Traffic density factor (default: 0.3)
                - max_steps: Maximum steps per episode (default: 200)
        """
        super().__init__(config)
        
        self.network_size = self.config.get("network_size", 20)
        self.num_agents = self.config.get("num_agents", 5)
        self.num_charging_stations = self.config.get("num_charging_stations", 8)
        self.traffic_density = self.config.get("traffic_density", 0.3)
        
        # Define spaces
        # Observation: [position_x, position_y, velocity, battery_level, destination_x, destination_y,
        #               nearest_station_x, nearest_station_y, station_queue_length, traffic_density]
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(10,),
            dtype=np.float32
        )
        
        # Action: [accelerate, decelerate, turn_left, turn_right, change_lane, go_to_station]
        self.action_space = spaces.Discrete(6)
        
        # Environment state
        self.agent_positions = None
        self.agent_velocities = None
        self.agent_batteries = None
        self.agent_destinations = None
        self.charging_stations = None
        self.station_queues = None
        self.traffic_grid = None
        
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize agent positions on roads
        self.agent_positions = np.random.rand(self.num_agents, 2) * self.network_size
        self.agent_velocities = np.random.uniform(0.5, 1.0, size=self.num_agents)
        self.agent_batteries = np.random.uniform(0.3, 1.0, size=self.num_agents)
        self.agent_destinations = np.random.rand(self.num_agents, 2) * self.network_size
        
        # Place charging stations at strategic locations
        self.charging_stations = np.random.rand(self.num_charging_stations, 2) * self.network_size
        self.station_queues = [[] for _ in range(self.num_charging_stations)]
        
        # Initialize traffic grid
        self.traffic_grid = np.random.rand(self.network_size, self.network_size) * self.traffic_density
        
        obs = self._get_observation(0)
        info = self.get_state()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        agent_idx = self.current_step % self.num_agents
        
        reward = 0.0
        pos = self.agent_positions[agent_idx]
        vel = self.agent_velocities[agent_idx]
        
        # Execute action
        if action == 0:  # Accelerate
            self.agent_velocities[agent_idx] = min(vel + 0.1, 1.0)
            reward -= 0.02  # Higher energy consumption
        elif action == 1:  # Decelerate
            self.agent_velocities[agent_idx] = max(vel - 0.1, 0.1)
            reward += 0.01  # Energy saved
        elif action == 2:  # Turn left
            angle = np.random.uniform(-np.pi/4, 0)
            self._move_agent(agent_idx, angle)
        elif action == 3:  # Turn right
            angle = np.random.uniform(0, np.pi/4)
            self._move_agent(agent_idx, angle)
        elif action == 4:  # Change lane
            self._move_agent(agent_idx, 0)
            reward -= 0.01
        elif action == 5:  # Go to nearest charging station
            nearest_idx = self._find_nearest_station(pos)
            direction = self.charging_stations[nearest_idx] - pos
            if np.linalg.norm(direction) < 0.5:
                # At charging station
                self._charge_agent(agent_idx, nearest_idx)
                reward += 5.0
            else:
                # Move towards station
                angle = np.arctan2(direction[1], direction[0])
                self._move_agent(agent_idx, angle)
        
        # Move agent based on velocity
        self._move_agent(agent_idx, 0)
        
        # Battery drain based on velocity
        battery_drain = 0.005 * self.agent_velocities[agent_idx]
        self.agent_batteries[agent_idx] -= battery_drain
        
        # Check if reached destination
        dest_distance = np.linalg.norm(self.agent_destinations[agent_idx] - pos)
        if dest_distance < 1.0:
            reward += 20.0
            # Set new destination
            self.agent_destinations[agent_idx] = np.random.rand(2) * self.network_size
        
        # Reward shaping
        reward += 0.1 / (dest_distance + 1)  # Closer to destination
        reward += self.agent_batteries[agent_idx] * 0.05  # Maintain battery
        
        # Check termination conditions
        if self.agent_batteries[agent_idx] <= 0:
            reward -= 50  # Severe penalty for battery depletion
            terminated = True
        else:
            terminated = False
        
        # Traffic penalties
        traffic_level = self._get_traffic_at_position(pos)
        reward -= traffic_level * 0.1
        
        super().step(action)
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_observation(agent_idx)
        info = self.get_state()
        info["agent_idx"] = agent_idx
        
        return obs, reward, terminated, truncated, info
    
    def _move_agent(self, agent_idx: int, angle: float) -> None:
        """Move agent in specified direction."""
        vel = self.agent_velocities[agent_idx]
        dx = vel * np.cos(angle)
        dy = vel * np.sin(angle)
        
        new_pos = self.agent_positions[agent_idx] + np.array([dx, dy])
        
        # Keep within bounds
        new_pos = np.clip(new_pos, 0, self.network_size - 0.1)
        self.agent_positions[agent_idx] = new_pos
    
    def _find_nearest_station(self, pos: np.ndarray) -> int:
        """Find the nearest charging station."""
        distances = np.linalg.norm(self.charging_stations - pos, axis=1)
        return np.argmin(distances)
    
    def _charge_agent(self, agent_idx: int, station_idx: int) -> None:
        """Charge agent at a station."""
        charge_amount = min(0.3, 1.0 - self.agent_batteries[agent_idx])
        self.agent_batteries[agent_idx] += charge_amount
    
    def _get_traffic_at_position(self, pos: np.ndarray) -> float:
        """Get traffic density at position."""
        grid_x = int(pos[0])
        grid_y = int(pos[1])
        grid_x = min(grid_x, self.network_size - 1)
        grid_y = min(grid_y, self.network_size - 1)
        return self.traffic_grid[grid_y, grid_x]
    
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for a specific agent."""
        pos = self.agent_positions[agent_idx]
        vel = self.agent_velocities[agent_idx]
        battery = self.agent_batteries[agent_idx]
        dest = self.agent_destinations[agent_idx]
        
        # Find nearest charging station
        nearest_idx = self._find_nearest_station(pos)
        nearest_station = self.charging_stations[nearest_idx]
        queue_length = len(self.station_queues[nearest_idx])
        
        # Get local traffic
        traffic = self._get_traffic_at_position(pos)
        
        obs = np.array([
            pos[0] / self.network_size,
            pos[1] / self.network_size,
            vel,
            battery,
            dest[0] / self.network_size,
            dest[1] / self.network_size,
            nearest_station[0] / self.network_size,
            nearest_station[1] / self.network_size,
            queue_length / self.num_agents,
            traffic
        ], dtype=np.float32)
        
        return obs
    
    def render(self) -> Dict[str, Any]:
        """Render the traffic simulation state."""
        return {
            "agent_positions": self.agent_positions.tolist(),
            "charging_stations": self.charging_stations.tolist(),
            "traffic_grid": self.traffic_grid.tolist(),
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        state = super().get_state()
        state.update({
            "agent_positions": self.agent_positions.tolist() if self.agent_positions is not None else None,
            "agent_velocities": self.agent_velocities.tolist() if self.agent_velocities is not None else None,
            "agent_batteries": self.agent_batteries.tolist() if self.agent_batteries is not None else None,
        })
        return state

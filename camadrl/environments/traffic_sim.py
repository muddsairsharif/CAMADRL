"""
Traffic simulation environment for multi-agent vehicle coordination.

This module implements a traffic simulation where vehicles (agents) must
coordinate to optimize traffic flow and minimize congestion.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from camadrl.environments.base_env import BaseEnv


class TrafficSim(BaseEnv):
    """
    Traffic simulation environment for vehicle coordination.
    
    Simulates traffic flow where vehicles must coordinate their speeds
    and lane changes to optimize overall traffic efficiency.
    
    Attributes:
        num_lanes: Number of lanes in the road
        road_length: Length of the road segment
        vehicle_positions: Current positions of vehicles
        vehicle_speeds: Current speeds of vehicles
    """
    
    def __init__(
        self,
        num_agents: int = 10,
        num_lanes: int = 3,
        road_length: float = 1000.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Traffic Simulation environment.
        
        Args:
            num_agents: Number of vehicles (agents)
            num_lanes: Number of lanes
            road_length: Length of road segment
            config: Configuration dictionary
        """
        # State: [position, speed, lane, leader_distance, leader_speed, neighbors_info]
        state_dim = 5 + 4  # Own state + nearest neighbor info
        action_dim = 4  # Accelerate, Decelerate, Change left, Change right
        
        super().__init__(num_agents, state_dim, action_dim, config)
        
        self.num_lanes = num_lanes
        self.road_length = road_length
        
        # Vehicle dynamics
        self.max_speed = self.config.get("max_speed", 30.0)  # m/s
        self.min_speed = self.config.get("min_speed", 0.0)
        self.max_accel = self.config.get("max_accel", 3.0)  # m/s^2
        self.dt = self.config.get("dt", 0.1)  # Time step
        
        # Safety parameters
        self.safe_distance = self.config.get("safe_distance", 10.0)  # meters
        self.vehicle_length = self.config.get("vehicle_length", 5.0)
        
        # State variables
        self.vehicle_positions = np.zeros(num_agents)
        self.vehicle_speeds = np.zeros(num_agents)
        self.vehicle_lanes = np.zeros(num_agents, dtype=np.int32)
        
        # Reward parameters
        self.reward_speed = self.config.get("reward_speed", 0.1)
        self.reward_safety = self.config.get("reward_safety", 1.0)
        self.reward_collision = self.config.get("reward_collision", -10.0)
        
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
        
        # Initialize vehicle positions (spread along road)
        self.vehicle_positions = np.sort(
            np.random.uniform(0, self.road_length * 0.8, self.num_agents)
        )
        
        # Initialize speeds
        self.vehicle_speeds = np.random.uniform(
            self.max_speed * 0.5,
            self.max_speed * 0.8,
            self.num_agents
        )
        
        # Initialize lanes
        self.vehicle_lanes = np.random.randint(0, self.num_lanes, self.num_agents)
        
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
        
        rewards = np.zeros(self.num_agents)
        dones = [False] * self.num_agents
        
        # Apply actions
        for i, action in enumerate(actions):
            if action == 0:  # Accelerate
                self.vehicle_speeds[i] = min(
                    self.vehicle_speeds[i] + self.max_accel * self.dt,
                    self.max_speed
                )
            elif action == 1:  # Decelerate
                self.vehicle_speeds[i] = max(
                    self.vehicle_speeds[i] - self.max_accel * self.dt,
                    self.min_speed
                )
            elif action == 2:  # Change left
                if self.vehicle_lanes[i] > 0:
                    if self._is_lane_change_safe(i, self.vehicle_lanes[i] - 1):
                        self.vehicle_lanes[i] -= 1
            elif action == 3:  # Change right
                if self.vehicle_lanes[i] < self.num_lanes - 1:
                    if self._is_lane_change_safe(i, self.vehicle_lanes[i] + 1):
                        self.vehicle_lanes[i] += 1
        
        # Update positions
        self.vehicle_positions += self.vehicle_speeds * self.dt
        
        # Calculate rewards
        for i in range(self.num_agents):
            # Speed reward (encourage maintaining high speed)
            speed_reward = self.reward_speed * (self.vehicle_speeds[i] / self.max_speed)
            
            # Safety reward (maintain safe distance)
            leader_idx, leader_dist = self._get_leader(i)
            if leader_idx is not None:
                if leader_dist < self.safe_distance:
                    safety_reward = self.reward_collision
                else:
                    safety_reward = self.reward_safety
            else:
                safety_reward = self.reward_safety
            
            rewards[i] = speed_reward + safety_reward
            
            # Check if vehicle finished the road
            if self.vehicle_positions[i] >= self.road_length:
                dones[i] = True
                rewards[i] += 10.0  # Bonus for completing
        
        # Episode termination
        all_done = all(dones) or self.current_step >= self.max_steps
        if all_done:
            dones = [True] * self.num_agents
        
        states = [self._get_agent_state(i) for i in range(self.num_agents)]
        info = self.get_info()
        info["average_speed"] = np.mean(self.vehicle_speeds)
        
        return states, list(rewards), dones, info
    
    def _get_leader(self, agent_id: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the leading vehicle in the same lane.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Tuple of (leader_id, distance) or (None, None)
        """
        same_lane = self.vehicle_lanes == self.vehicle_lanes[agent_id]
        ahead = self.vehicle_positions > self.vehicle_positions[agent_id]
        candidates = same_lane & ahead
        
        if not np.any(candidates):
            return None, None
        
        distances = self.vehicle_positions[candidates] - self.vehicle_positions[agent_id]
        leader_idx = np.where(candidates)[0][np.argmin(distances)]
        leader_dist = distances.min()
        
        return leader_idx, leader_dist
    
    def _is_lane_change_safe(self, agent_id: int, target_lane: int) -> bool:
        """
        Check if lane change is safe.
        
        Args:
            agent_id: ID of the agent
            target_lane: Target lane number
            
        Returns:
            True if safe, False otherwise
        """
        # Check vehicles in target lane
        in_target_lane = self.vehicle_lanes == target_lane
        
        for i in np.where(in_target_lane)[0]:
            distance = abs(self.vehicle_positions[i] - self.vehicle_positions[agent_id])
            if distance < self.safe_distance:
                return False
        
        return True
    
    def _get_agent_state(self, agent_id: int) -> np.ndarray:
        """
        Get state observation for specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            State observation array
        """
        # Normalize position and speed
        position = self.vehicle_positions[agent_id] / self.road_length
        speed = self.vehicle_speeds[agent_id] / self.max_speed
        lane = self.vehicle_lanes[agent_id] / (self.num_lanes - 1) if self.num_lanes > 1 else 0
        
        # Leader information
        leader_idx, leader_dist = self._get_leader(agent_id)
        if leader_idx is not None:
            leader_distance = leader_dist / self.road_length
            leader_speed = self.vehicle_speeds[leader_idx] / self.max_speed
        else:
            leader_distance = 1.0  # Far away
            leader_speed = 1.0
        
        # Neighbor information (simplified)
        left_clear = 1.0 if self.vehicle_lanes[agent_id] == 0 else float(
            self._is_lane_change_safe(agent_id, self.vehicle_lanes[agent_id] - 1)
        )
        right_clear = 1.0 if self.vehicle_lanes[agent_id] == self.num_lanes - 1 else float(
            self._is_lane_change_safe(agent_id, self.vehicle_lanes[agent_id] + 1)
        )
        
        state = np.array([
            position,
            speed,
            lane,
            leader_distance,
            leader_speed,
            left_clear,
            right_clear,
            0.0,  # Reserved for future use
            0.0,  # Reserved for future use
        ], dtype=np.float32)
        
        return state
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame if mode is 'rgb_array'
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Draw road
        for lane in range(self.num_lanes + 1):
            y = lane
            ax.plot([0, self.road_length], [y, y], 'k--', alpha=0.3)
        
        # Draw vehicles
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_agents))
        for i in range(self.num_agents):
            x = self.vehicle_positions[i]
            y = self.vehicle_lanes[i] + 0.5
            ax.plot(x, y, 'o', color=colors[i], markersize=10, label=f'V{i}')
        
        ax.set_xlim(0, self.road_length)
        ax.set_ylim(-0.5, self.num_lanes + 0.5)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Lane')
        ax.set_title(f'Traffic Simulation - Step {self.current_step}')
        ax.legend(loc='upper right', ncol=5)
        
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
            self.vehicle_positions / self.road_length,
            self.vehicle_speeds / self.max_speed,
            self.vehicle_lanes.astype(np.float32) / (self.num_lanes - 1) if self.num_lanes > 1 else np.zeros(self.num_agents)
        ])

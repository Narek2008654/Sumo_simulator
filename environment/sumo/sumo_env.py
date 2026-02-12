"""
Mini Sumo Environment for Reinforcement Learning.

This environment simulates mini sumo robot competitions following
official rules from competitions like Robotex and RoboChallenge.

The environment is compatible with Gymnasium (OpenAI Gym) interface.
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

from .boards.dohyo import Dohyo, DohyoConfig
from .physics import (
    RobotState, RobotPhysics, 
    integrate_robot_state, 
    check_robot_collision, 
    resolve_collision,
    get_robot_corners
)
from .sensors import SensorSuite, read_all_sensors


@dataclass
class SumoEnvConfig:
    """Configuration for the sumo environment."""
    # Simulation settings
    dt: float = 0.02  # Time step (50 Hz)
    max_episode_steps: int = 1000  # ~20 seconds at 50Hz
    
    # Physics
    collision_restitution: float = 0.3
    
    # Rendering
    render_mode: Optional[str] = None  # "human", "rgb_array", or None
    window_width: int = 800  # pixels (wider for side panels)
    window_height: int = 600  # pixels
    fps: int = 50
    
    # Reward shaping
    win_reward: float = 100.0
    lose_reward: float = -100.0
    draw_penalty: float = -50.0  # Penalty for timeout/draw (encourages decisive action)
    push_reward_scale: float = 5.0  # Reward for pushing opponent toward edge (aggressive)
    edge_penalty_scale: float = 0.5  # Penalty for being near edge
    time_penalty: float = -0.1  # Heavy penalty per step to discourage time wasting
    
    # Randomization for training
    random_start: bool = True
    
    # Dohyo configuration
    dohyo_config: Optional[DohyoConfig] = None


class SumoEnv(gym.Env):
    """
    Mini Sumo Robot Competition Environment.
    
    Two robots compete to push each other out of a circular ring (dohyo).
    This environment supports single-agent training (opponent can be scripted
    or another trained agent) or multi-agent training.
    
    Observation Space (for each robot):
        - Edge sensor readings (4 values): 0=black surface, 1=white border
        - Opponent sensor readings (5 values): 0=not detected, 1=very close
        - Own velocity (2 values): vx, vy normalized
        - Own angular velocity (1 value): omega normalized
        - Relative position to center (2 values): normalized
        - Relative angle to center (1 value): normalized
        Total: 15 values
    
    Action Space (for each robot):
        - Continuous: [left_motor, right_motor] in range [-1, 1]
          These are normalized force values.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        config: Optional[SumoEnvConfig] = None,
        robot1_physics: Optional[RobotPhysics] = None,
        robot2_physics: Optional[RobotPhysics] = None,
        robot1_sensors: Optional[SensorSuite] = None,
        robot2_sensors: Optional[SensorSuite] = None,
    ):
        self.config = config or SumoEnvConfig()
        
        # Initialize dohyo
        self.dohyo = Dohyo(self.config.dohyo_config)
        
        # Initialize robot physics
        self.robot1_physics = robot1_physics or RobotPhysics.from_config({})
        self.robot2_physics = robot2_physics or RobotPhysics.from_config({})
        
        # Initialize sensors
        self.robot1_sensors = robot1_sensors or SensorSuite.default_mini_sumo(
            self.robot1_physics.width, self.robot1_physics.length
        )
        self.robot2_sensors = robot2_sensors or SensorSuite.default_mini_sumo(
            self.robot2_physics.width, self.robot2_physics.length
        )
        
        # Random number generator
        self.rng = np.random.default_rng()
        
        # State variables (initialized in reset)
        self.robot1_state: Optional[RobotState] = None
        self.robot2_state: Optional[RobotState] = None
        self.step_count = 0
        self.robot1_out = False
        self.robot2_out = False
        
        # Score tracking (persists across resets)
        self.robot1_score = 0
        self.robot2_score = 0
        self.match_count = 0
        
        # Robot labels (can be set externally)
        self.robot1_name = "Agent1"
        self.robot2_name = "Agent2"
        
        # Rendering
        self.window = None
        self.clock = None
        self.render_mode = self.config.render_mode
        
        # Define spaces for gym compatibility
        self._define_spaces()
    
    def _define_spaces(self):
        """Define observation and action spaces."""
        # Observation: sensors + velocity + angular vel + relative pos/angle
        obs_dim = (
            self.robot1_sensors.total_sensors +  # sensor readings
            2 +  # velocity (vx, vy)
            1 +  # angular velocity
            2 +  # relative position to center
            1    # relative angle to center
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action: [left_motor, right_motor] normalized
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            (observation, info) tuple
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Reset episode state
        self.step_count = 0
        self.robot1_out = False
        self.robot2_out = False
        
        # Get starting positions
        robot_length = max(self.robot1_physics.length, self.robot2_physics.length)
        if self.config.random_start:
            positions = self.dohyo.get_random_starting_positions(
                min_separation=robot_length * 2,
                rng=self.rng
            )
        else:
            positions = self.dohyo.get_starting_positions(robot_length=robot_length)
        
        # Initialize robot states
        pos1, theta1 = positions[0]
        pos2, theta2 = positions[1]
        
        self.robot1_state = RobotState(
            x=pos1[0], y=pos1[1], theta=theta1,
            vx=0.0, vy=0.0, omega=0.0
        )
        self.robot2_state = RobotState(
            x=pos2[0], y=pos2[1], theta=theta2,
            vx=0.0, vy=0.0, omega=0.0
        )
        
        # Get initial observation
        obs = self._get_observation(self.robot1_state, self.robot2_state, self.robot1_sensors)
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, info
    
    def step(
        self, 
        action: np.ndarray,
        opponent_action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action for robot 1 [left_motor, right_motor]
            opponent_action: Action for robot 2 (if None, uses simple scripted agent)
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Get opponent action if not provided
        if opponent_action is None:
            opponent_action = self._get_opponent_action()
        
        # Store previous states for reward calculation
        prev_robot1_dist = np.linalg.norm(self.robot1_state.position)
        prev_robot2_dist = np.linalg.norm(self.robot2_state.position)
        
        # Apply actions (convert normalized to actual forces)
        action = np.clip(action, -1, 1)
        opponent_action = np.clip(opponent_action, -1, 1)
        
        left1 = action[0] * self.robot1_physics.max_force
        right1 = action[1] * self.robot1_physics.max_force
        left2 = opponent_action[0] * self.robot2_physics.max_force
        right2 = opponent_action[1] * self.robot2_physics.max_force
        
        # Integrate physics for both robots
        new_state1 = integrate_robot_state(
            self.robot1_state, self.robot1_physics,
            left1, right1, self.config.dt
        )
        new_state2 = integrate_robot_state(
            self.robot2_state, self.robot2_physics,
            left2, right2, self.config.dt
        )
        
        # Check and resolve collisions between robots
        colliding, normal, penetration = check_robot_collision(
            new_state1, self.robot1_physics,
            new_state2, self.robot2_physics
        )
        
        if colliding:
            # resolve_collision returns new states with corrected positions and velocities
            new_state1, new_state2 = resolve_collision(
                new_state1, self.robot1_physics,
                new_state2, self.robot2_physics,
                normal, penetration,
                self.config.collision_restitution
            )
        
        # Update states
        self.robot1_state = new_state1
        self.robot2_state = new_state2
        
        # Check for out-of-bounds
        self.robot1_out = self.dohyo.is_robot_out(self.robot1_state, self.robot1_physics)
        self.robot2_out = self.dohyo.is_robot_out(self.robot2_state, self.robot2_physics)
        
        # Calculate reward
        reward = self._calculate_reward(
            prev_robot1_dist, prev_robot2_dist, colliding
        )
        
        # Check termination
        terminated = self.robot1_out or self.robot2_out
        truncated = self.step_count >= self.config.max_episode_steps
        
        # Penalize draws (timeout without winner)
        if truncated and not terminated:
            reward += self.config.draw_penalty
        
        # Update scores when match ends
        if terminated:
            self.match_count += 1
            if self.robot2_out and not self.robot1_out:
                self.robot1_score += 1  # Robot 1 wins
            elif self.robot1_out and not self.robot2_out:
                self.robot2_score += 1  # Robot 2 wins
            # Both out = no score change (draw)
        
        # Get observation
        obs = self._get_observation(self.robot1_state, self.robot2_state, self.robot1_sensors)
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(
        self,
        robot_state: RobotState,
        opponent_state: RobotState,
        sensors: SensorSuite,
        robot_physics: Optional[RobotPhysics] = None,
        opponent_physics: Optional[RobotPhysics] = None
    ) -> np.ndarray:
        """Get observation for a robot."""
        if robot_physics is None:
            robot_physics = self.robot1_physics
        if opponent_physics is None:
            opponent_physics = self.robot2_physics

        # Read sensors
        sensor_readings = read_all_sensors(
            robot_state, robot_physics, sensors,
            opponent_state, opponent_physics,
            self.dohyo, add_noise=True, rng=self.rng
        )

        # Normalize velocity
        max_speed = robot_physics.max_speed
        norm_vx = robot_state.vx / max_speed
        norm_vy = robot_state.vy / max_speed
        
        # Normalize angular velocity (assume max ~10 rad/s)
        norm_omega = robot_state.omega / 10.0
        
        # Relative position to center (normalized by dohyo radius)
        rel_pos = robot_state.position / self.dohyo.radius
        
        # Relative angle to center
        angle_to_center = np.arctan2(-robot_state.y, -robot_state.x)
        rel_angle = (angle_to_center - robot_state.theta) / np.pi
        
        obs = np.concatenate([
            sensor_readings,
            [norm_vx, norm_vy],
            [norm_omega],
            rel_pos,
            [rel_angle]
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(
        self, 
        prev_robot1_dist: float, 
        prev_robot2_dist: float,
        collision_occurred: bool
    ) -> float:
        """Calculate reward for robot 1."""
        reward = self.config.time_penalty
        
        # Win/lose rewards
        if self.robot2_out and not self.robot1_out:
            reward += self.config.win_reward
        elif self.robot1_out and not self.robot2_out:
            reward += self.config.lose_reward
        elif self.robot1_out and self.robot2_out:
            # Both out simultaneously - small negative (shouldn't happen often)
            reward += -10.0
        
        # Reward for pushing opponent toward edge
        curr_robot2_dist = np.linalg.norm(self.robot2_state.position)
        push_reward = (curr_robot2_dist - prev_robot2_dist) * self.config.push_reward_scale
        reward += push_reward
        
        # Penalty for being near edge
        curr_robot1_dist = np.linalg.norm(self.robot1_state.position)
        edge_proximity = curr_robot1_dist / self.dohyo.inner_radius
        if edge_proximity > 0.7:
            reward -= (edge_proximity - 0.7) * self.config.edge_penalty_scale
        
        # Reward for collision (aggression)
        if collision_occurred:
            reward += 1.0
        
        return reward
    
    def _get_opponent_action(self) -> np.ndarray:
        """Simple scripted opponent for training."""
        # Simple seeking behavior - turn toward opponent and charge
        opp_state = self.robot2_state
        target_state = self.robot1_state
        
        # Vector to target
        to_target = target_state.position - opp_state.position
        dist = np.linalg.norm(to_target)
        
        if dist < 1e-6:
            return np.array([1.0, 1.0])  # Charge forward
        
        # Angle to target
        angle_to_target = np.arctan2(to_target[1], to_target[0])
        angle_diff = angle_to_target - opp_state.theta
        
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Check if near edge (use distance from center instead of edge sensors)
        dist_from_center = np.linalg.norm(opp_state.position)
        if dist_from_center > self.dohyo.inner_radius * 0.8:
            # Near edge - turn toward center
            angle_to_center = np.arctan2(-opp_state.y, -opp_state.x)
            center_diff = angle_to_center - opp_state.theta
            while center_diff > np.pi:
                center_diff -= 2 * np.pi
            while center_diff < -np.pi:
                center_diff += 2 * np.pi
            
            if center_diff > 0:
                return np.array([0.3, 0.8])  # Turn left toward center
            else:
                return np.array([0.8, 0.3])  # Turn right toward center
        
        # Turn toward target, then charge
        if abs(angle_diff) > 0.3:  # Need to turn
            if angle_diff > 0:
                return np.array([0.5, 1.0])  # Turn left
            else:
                return np.array([1.0, 0.5])  # Turn right
        else:
            return np.array([1.0, 1.0])  # Charge forward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        return {
            "step": self.step_count,
            "robot1_position": self.robot1_state.position.copy(),
            "robot2_position": self.robot2_state.position.copy(),
            "robot1_out": self.robot1_out,
            "robot2_out": self.robot2_out,
            "robot1_dist_to_edge": self.dohyo.inner_radius - np.linalg.norm(self.robot1_state.position),
            "robot2_dist_to_edge": self.dohyo.inner_radius - np.linalg.norm(self.robot2_state.position),
            "robot1_score": self.robot1_score,
            "robot2_score": self.robot2_score,
            "match_count": self.match_count,
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
        return None
    
    def _render_frame(self) -> Optional[np.ndarray]:
        """Render a single frame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.config.window_width, self.config.window_height)
            )
            pygame.display.set_caption("Mini Sumo Simulator")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create surface
        canvas = pygame.Surface((self.config.window_width, self.config.window_height))
        
        # Calculate scale and offset - center the dohyo in the window
        padding = 50
        available_size = self.config.window_height - 2 * padding  # Use height for scaling
        scale = available_size / (self.dohyo.config.diameter * 1.2)
        offset = (self.config.window_width // 2, self.config.window_height // 2)  # Center
        
        # Render dohyo
        self.dohyo.render(canvas, scale, offset)
        
        # Render sensor beams for both robots
        self._render_sensor_beams(canvas, self.robot1_state, self.robot1_physics,
                                  self.robot1_sensors, self.robot2_state,
                                  self.robot2_physics,
                                  scale, offset, (0, 150, 255, 100))  # Blue beams
        self._render_sensor_beams(canvas, self.robot2_state, self.robot2_physics,
                                  self.robot2_sensors, self.robot1_state,
                                  self.robot1_physics,
                                  scale, offset, (255, 150, 0, 100))  # Orange beams
        
        # Render robots
        self._render_robot(canvas, self.robot1_state, self.robot1_physics, 
                          scale, offset, (0, 100, 255))  # Blue
        self._render_robot(canvas, self.robot2_state, self.robot2_physics, 
                          scale, offset, (255, 100, 0))  # Orange
        
        # Render score table
        self._render_score_table(canvas)
        
        # Render sensor readings panel
        self._render_sensor_panel(canvas)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.config.fps)
        
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), 
                axes=(1, 0, 2)
            )
        
        return None
    
    def _render_robot(
        self, 
        surface: pygame.Surface, 
        state: RobotState, 
        physics: RobotPhysics,
        scale: float,
        offset: Tuple[int, int],
        color: Tuple[int, int, int]
    ):
        """Render a robot on the surface."""
        corners = get_robot_corners(state, physics)
        
        # Transform to pixel coordinates
        corners_px = []
        for corner in corners:
            px = int(offset[0] + corner[0] * scale)
            py = int(offset[1] - corner[1] * scale)  # Flip y
            corners_px.append((px, py))
        
        # Draw robot body
        pygame.draw.polygon(surface, color, corners_px)
        pygame.draw.polygon(surface, (0, 0, 0), corners_px, 2)  # Border
        
        # Draw direction indicator (front of robot)
        front_center = (corners_px[0][0] + corners_px[1][0]) // 2, \
                       (corners_px[0][1] + corners_px[1][1]) // 2
        center = (int(offset[0] + state.x * scale), 
                  int(offset[1] - state.y * scale))
        pygame.draw.line(surface, (255, 255, 0), center, front_center, 3)
    
    def _render_score_table(self, surface: pygame.Surface):
        """Render the score table on the surface."""
        # Colors
        bg_color = (40, 40, 50)
        border_color = (100, 100, 120)
        text_color = (255, 255, 255)
        robot1_color = (0, 100, 255)   # Blue for robot 1
        robot2_color = (255, 100, 0)   # Orange for robot 2
        
        # Fonts
        title_font = pygame.font.Font(None, 28)
        score_font = pygame.font.Font(None, 48)
        label_font = pygame.font.Font(None, 26)
        info_font = pygame.font.Font(None, 24)
        
        # Score table dimensions
        table_width = 180
        table_height = 120
        table_x = 10
        table_y = 10
        
        # Draw table background
        table_rect = pygame.Rect(table_x, table_y, table_width, table_height)
        pygame.draw.rect(surface, bg_color, table_rect, border_radius=8)
        pygame.draw.rect(surface, border_color, table_rect, 2, border_radius=8)
        
        # Title
        title = title_font.render("SCORE", True, text_color)
        title_rect = title.get_rect(centerx=table_x + table_width // 2, top=table_y + 8)
        surface.blit(title, title_rect)
        
        # Score display
        score_y = table_y + 40
        
        # Robot 1 score (left)
        r1_score = score_font.render(str(self.robot1_score), True, robot1_color)
        r1_rect = r1_score.get_rect(centerx=table_x + 45, centery=score_y + 15)
        surface.blit(r1_score, r1_rect)
        
        # VS separator
        vs_text = title_font.render("-", True, text_color)
        vs_rect = vs_text.get_rect(centerx=table_x + table_width // 2, centery=score_y + 15)
        surface.blit(vs_text, vs_rect)
        
        # Robot 2 score (right)
        r2_score = score_font.render(str(self.robot2_score), True, robot2_color)
        r2_rect = r2_score.get_rect(centerx=table_x + table_width - 45, centery=score_y + 15)
        surface.blit(r2_score, r2_rect)
        
        # Robot labels (use configured names)
        r1_label = label_font.render(self.robot1_name, True, robot1_color)
        r2_label = label_font.render(self.robot2_name, True, robot2_color)
        r1_label_rect = r1_label.get_rect(centerx=table_x + 45, top=score_y + 40)
        r2_label_rect = r2_label.get_rect(centerx=table_x + table_width - 45, top=score_y + 40)
        surface.blit(r1_label, r1_label_rect)
        surface.blit(r2_label, r2_label_rect)
        
        # Match info (top right of screen)
        info_x = self.config.window_width - 150
        info_y = 10
        
        # Time/Step display
        time_seconds = self.step_count * self.config.dt
        time_text = info_font.render(f"Time: {time_seconds:.1f}s", True, text_color)
        surface.blit(time_text, (info_x, info_y))
        
        match_text = info_font.render(f"Match: {self.match_count + 1}", True, text_color)
        surface.blit(match_text, (info_x, info_y + 25))
    
    def _render_sensor_beams(
        self,
        surface: pygame.Surface,
        robot_state: RobotState,
        robot_physics: RobotPhysics,
        sensors: SensorSuite,
        opponent_state: RobotState,
        opponent_physics: RobotPhysics,
        scale: float,
        offset: Tuple[int, int],
        beam_color: Tuple[int, int, int, int]
    ):
        """Render sensor detection beams emanating from the robot."""
        from .sensors import read_opponent_sensor, transform_to_world

        for i, sensor in enumerate(sensors.opponent_sensors):
            # Get sensor position and direction in world frame
            sensor_pos, sensor_dir = transform_to_world(
                robot_state, sensor.position, sensor.direction
            )

            # Read sensor value
            reading = read_opponent_sensor(
                sensor, robot_state, opponent_state,
                opponent_physics, add_noise=False
            )

            # Calculate beam end point
            end_pos = sensor_pos + sensor_dir * sensor.max_range
            
            # Convert to pixel coordinates
            start_px = (
                int(offset[0] + sensor_pos[0] * scale),
                int(offset[1] - sensor_pos[1] * scale)
            )
            end_px = (
                int(offset[0] + end_pos[0] * scale),
                int(offset[1] - end_pos[1] * scale)
            )
            
            # Draw beam line (brighter when detecting) - now 0=close, 1=far
            if reading < 0.9:
                # Detected - draw bright beam (lower reading = closer = brighter)
                intensity = int(100 + 155 * (1 - reading))
                color = (beam_color[0], intensity, beam_color[2])
                width = 3
            else:
                # Not detected - dim beam
                color = (beam_color[0] // 3, beam_color[1] // 3, beam_color[2] // 3)
                width = 1
            
            pygame.draw.line(surface, color, start_px, end_px, width)
            
            # Draw sensor FOV arc when detecting
            if reading < 0.9:
                # Draw a small circle at detection point (reading = normalized distance)
                detect_dist = reading * sensor.max_range
                detect_pos = sensor_pos + sensor_dir * detect_dist
                detect_px = (
                    int(offset[0] + detect_pos[0] * scale),
                    int(offset[1] - detect_pos[1] * scale)
                )
                pygame.draw.circle(surface, (255, 255, 0), detect_px, 5)
    
    def _render_sensor_panel(self, surface: pygame.Surface):
        """Render two sensor panels on left and right sides with vertical bars side by side."""
        # Get sensor readings (original order: F, FR, FL, R, L)
        raw_readings1 = read_all_sensors(
            self.robot1_state, self.robot1_physics, self.robot1_sensors,
            self.robot2_state, self.robot2_physics, None, add_noise=False
        )
        raw_readings2 = read_all_sensors(
            self.robot2_state, self.robot2_physics, self.robot2_sensors,
            self.robot1_state, self.robot1_physics, None, add_noise=False
        )
        
        # Reorder to natural layout: L, FL, F, FR, R (left to right as seen from above)
        # Original indices: F=0, FR=1, FL=2, R=3, L=4
        reorder = [4, 2, 0, 1, 3]  # L, FL, F, FR, R
        readings1 = raw_readings1[reorder]
        readings2 = raw_readings2[reorder]
        
        # Sensor names in natural order (left to right)
        sensor_labels = ["L", "FL", "F", "FR", "R"]
        
        # Render left panel (Robot 1 - Blue)
        self._render_single_sensor_panel(
            surface, readings1, sensor_labels,
            panel_x=8, 
            panel_y=self.config.window_height // 2 - 60,
            robot_name=self.robot1_name,
            robot_color=(0, 150, 255)
        )
        
        # Render right panel (Robot 2 - Orange)
        panel_width = 22 * len(sensor_labels) + 20
        self._render_single_sensor_panel(
            surface, readings2, sensor_labels,
            panel_x=self.config.window_width - panel_width - 8,
            panel_y=self.config.window_height // 2 - 60,
            robot_name=self.robot2_name,
            robot_color=(255, 150, 0)
        )
    
    def _render_single_sensor_panel(
        self,
        surface: pygame.Surface,
        readings: np.ndarray,
        sensor_labels: List[str],
        panel_x: int,
        panel_y: int,
        robot_name: str,
        robot_color: Tuple[int, int, int]
    ):
        """Render sensor panel with vertical bars arranged side by side."""
        num_sensors = len(readings)
        
        # Panel settings - wide panel with bars side by side
        bar_width = 14
        bar_spacing = 22  # More space between bars
        bar_max_height = 70
        panel_width = num_sensors * bar_spacing + 20
        panel_height = 110
        
        # Colors
        border_color = (80, 80, 100)
        text_color = (255, 255, 255)
        bar_bg = (50, 50, 60)
        
        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(surface, (30, 30, 40), panel_rect, border_radius=8)
        pygame.draw.rect(surface, border_color, panel_rect, 2, border_radius=8)
        
        # Fonts
        font = pygame.font.Font(None, 16)
        tiny_font = pygame.font.Font(None, 12)
        
        # Robot name at top
        name_text = font.render(robot_name, True, robot_color)
        name_rect = name_text.get_rect(centerx=panel_x + panel_width // 2, top=panel_y + 4)
        surface.blit(name_text, name_rect)
        
        # Draw vertical bars side by side (centered)
        total_bars_width = (num_sensors - 1) * bar_spacing + bar_width
        start_x = panel_x + (panel_width - total_bars_width) // 2
        bar_bottom_y = panel_y + panel_height - 18
        
        for i, (label, reading) in enumerate(zip(sensor_labels, readings)):
            x = start_x + i * bar_spacing
            
            # Background bar (vertical)
            bar_rect = pygame.Rect(x, bar_bottom_y - bar_max_height, bar_width, bar_max_height)
            pygame.draw.rect(surface, bar_bg, bar_rect, border_radius=2)
            
            # Fill bar (from bottom up) - 0=close, 1=far (shows actual distance)
            fill_height = int(bar_max_height * min(reading, 1.0))
            if fill_height > 0:
                fill_rect = pygame.Rect(
                    x, 
                    bar_bottom_y - fill_height,
                    bar_width, 
                    fill_height
                )
                pygame.draw.rect(surface, robot_color, fill_rect, border_radius=2)
            
            # Sensor label below bar
            label_text = tiny_font.render(label, True, text_color)
            label_rect = label_text.get_rect(centerx=x + bar_width // 2, top=bar_bottom_y + 2)
            surface.blit(label_text, label_rect)
    
    def reset_scores(self):
        """Reset the score table (call between sessions)."""
        self.robot1_score = 0
        self.robot2_score = 0
        self.match_count = 0
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# Multi-agent wrapper for self-play training
class SumoEnvMultiAgent(SumoEnv):
    """
    Multi-agent version of SumoEnv for self-play training.
    
    Returns observations and accepts actions for both robots.
    """
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset and return observations for both agents."""
        obs, info = super().reset(seed, options)
        
        obs2 = self._get_observation(
            self.robot2_state, self.robot1_state, self.robot2_sensors,
            self.robot2_physics, self.robot1_physics
        )

        return {"robot1": obs, "robot2": obs2}, info

    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """Take a step with actions for both agents."""
        action1 = actions.get("robot1", np.zeros(2))
        action2 = actions.get("robot2", np.zeros(2))

        obs, reward, terminated, truncated, info = super().step(action1, action2)

        obs2 = self._get_observation(
            self.robot2_state, self.robot1_state, self.robot2_sensors,
            self.robot2_physics, self.robot1_physics
        )
        
        # Calculate reward for robot 2 (inverse of robot 1)
        reward2 = -reward
        if self.robot1_out and not self.robot2_out:
            reward2 = self.config.win_reward
        elif self.robot2_out and not self.robot1_out:
            reward2 = self.config.lose_reward
        
        observations = {"robot1": obs, "robot2": obs2}
        rewards = {"robot1": reward, "robot2": reward2}
        terminateds = {"robot1": terminated, "robot2": terminated, "__all__": terminated}
        truncateds = {"robot1": truncated, "robot2": truncated, "__all__": truncated}
        
        return observations, rewards, terminateds, truncateds, info

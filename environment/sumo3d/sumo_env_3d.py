"""
3D Mini Sumo Environment for Reinforcement Learning using PyBullet.

Drop-in replacement for the 2D SumoEnv with identical observation (11 dims)
and action (2 dims) spaces so existing agents work unchanged.
"""

import time
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .physics3d import (
    BulletWorld, DohyoBody, RobotBody,
    Dohyo3DConfig, Robot3DConfig,
)
from .sensors3d import (
    SensorSuite3D, cast_opponent_rays, cast_edge_rays,
)


@dataclass
class SumoEnv3DConfig:
    """Configuration for the 3D sumo environment."""
    # Simulation timing
    dt: float = 0.02  # env step period (50 Hz)
    bullet_time_step: float = 1.0 / 240  # PyBullet internal step
    max_episode_steps: int = 1000  # ~20 seconds at 50 Hz

    # Rendering
    render_mode: Optional[str] = None  # "human" or None

    # Reward shaping (same defaults as 2D)
    win_reward: float = 100.0
    lose_reward: float = -100.0
    draw_penalty: float = -50.0
    push_reward_scale: float = 5.0
    edge_penalty_scale: float = 0.5
    time_penalty: float = -0.1

    # Randomization
    random_start: bool = True

    # Fall detection
    fall_z_threshold: float = 0.03  # how far below surface_z counts as fallen

    # Physics settling
    settle_steps: int = 10

    # Configs
    dohyo_config: Optional[Dohyo3DConfig] = None
    robot1_config: Optional[Robot3DConfig] = None
    robot2_config: Optional[Robot3DConfig] = None

    @property
    def sub_steps(self) -> int:
        """Number of PyBullet steps per env step."""
        return max(1, int(round(self.dt / self.bullet_time_step)))


class SumoEnv3D(gym.Env):
    """
    3D Mini Sumo Robot Competition Environment (PyBullet).

    Observation Space (11 dims, identical to 2D):
        [0:5]  - 5 opponent sensor readings (0=close, 1=far)
        [5:7]  - velocity (vx, vy) normalized by max_linear_speed
        [7]    - angular velocity (omega_z) normalized
        [8:10] - relative position to dohyo center / radius
        [10]   - relative angle to center / pi

    Action Space:
        [left_motor, right_motor] in [-1, 1]
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config: Optional[SumoEnv3DConfig] = None):
        self.config = config or SumoEnv3DConfig()

        # Will be created on first reset
        self.world: Optional[BulletWorld] = None
        self.dohyo: Optional[DohyoBody] = None
        self.robot1: Optional[RobotBody] = None
        self.robot2: Optional[RobotBody] = None
        self.sensors1: Optional[SensorSuite3D] = None
        self.sensors2: Optional[SensorSuite3D] = None

        self.rng = np.random.default_rng()

        # Episode state
        self.step_count = 0
        self.robot1_out = False
        self.robot2_out = False

        # Score tracking
        self.robot1_score = 0
        self.robot2_score = 0
        self.match_count = 0
        self.robot1_name = "Agent1"
        self.robot2_name = "Agent2"

        self.render_mode = self.config.render_mode

        # GUI debug text IDs
        self._score_text_id = -1
        self._time_text_id = -1

        # Timing for real-time pacing
        self._last_step_time = 0.0

        # Resolve configs
        self._dohyo_cfg = self.config.dohyo_config or Dohyo3DConfig()
        self._robot1_cfg = self.config.robot1_config or Robot3DConfig(
            body_color=(0.0, 0.4, 1.0, 1.0)  # blue
        )
        self._robot2_cfg = self.config.robot2_config or Robot3DConfig(
            body_color=(1.0, 0.4, 0.0, 1.0)  # orange
        )

        # Spaces (11-dim obs, 2-dim action)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _create_world(self):
        """Create PyBullet world, dohyo, robots."""
        gui = self.render_mode == "human"
        self.world = BulletWorld(gui=gui, time_step=self.config.bullet_time_step)
        self.dohyo = self.world.create_dohyo(self._dohyo_cfg)

        # Create sensor suites
        self.sensors1 = SensorSuite3D.default_mini_sumo(
            self._robot1_cfg.body_length,
            self._robot1_cfg.body_width,
            self._robot1_cfg.body_height,
        )
        self.sensors2 = SensorSuite3D.default_mini_sumo(
            self._robot2_cfg.body_length,
            self._robot2_cfg.body_width,
            self._robot2_cfg.body_height,
        )

    def _place_robots(self):
        """Place or reset robots at starting positions."""
        robot_length = max(self._robot1_cfg.body_length, self._robot2_cfg.body_length)

        if self.config.random_start:
            positions = self.dohyo.get_random_starting_positions(
                min_separation=robot_length * 2, rng=self.rng
            )
        else:
            positions = self.dohyo.get_starting_positions(robot_length=robot_length)

        pos1, yaw1 = positions[0]
        pos2, yaw2 = positions[1]
        surface_z = self.dohyo.surface_z

        if self.robot1 is None:
            self.robot1 = self.world.create_robot(
                self._robot1_cfg, pos1, yaw1, surface_z
            )
            self.robot2 = self.world.create_robot(
                self._robot2_cfg, pos2, yaw2, surface_z
            )
        else:
            self.robot1.reset(pos1, yaw1, surface_z)
            self.robot2.reset(pos2, yaw2, surface_z)

        # Let physics settle
        for _ in range(self.config.settle_steps):
            self.world.step()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.robot1_out = False
        self.robot2_out = False

        if self.world is None:
            self._create_world()

        self._place_robots()
        self._last_step_time = time.time()

        obs = self._get_observation(
            self.robot1, self.robot2, self.sensors1, self._robot1_cfg
        )
        info = self._get_info()

        if self.render_mode == "human":
            self._update_gui_overlay()

        return obs, info

    def step(
        self,
        action: np.ndarray,
        opponent_action: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        if opponent_action is None:
            opponent_action = self._get_default_opponent_action()

        action = np.clip(action, -1, 1)
        opponent_action = np.clip(opponent_action, -1, 1)

        # Store previous distances for reward
        prev_r1_dist = np.linalg.norm(self.robot1.get_xy_position())
        prev_r2_dist = np.linalg.norm(self.robot2.get_xy_position())

        # Apply motor commands
        self.robot1.set_motor_speeds(action[0], action[1])
        self.robot2.set_motor_speeds(opponent_action[0], opponent_action[1])

        # Sub-step physics
        for _ in range(self.config.sub_steps):
            self.world.step()

        # Detect collision between robots
        contacts = self.world.get_contact_points(
            self.robot1.body_id, self.robot2.body_id
        )
        collision_occurred = len(contacts) > 0

        # Out-of-bounds detection: robot fell off dohyo
        surface_z = self.dohyo.surface_z
        threshold = self.config.fall_z_threshold
        r1_z = self.robot1.get_z_position()
        r2_z = self.robot2.get_z_position()
        self.robot1_out = r1_z < (surface_z - threshold)
        self.robot2_out = r2_z < (surface_z - threshold)

        # Reward
        reward = self._calculate_reward(
            prev_r1_dist, prev_r2_dist, collision_occurred
        )

        # Termination
        terminated = self.robot1_out or self.robot2_out
        truncated = self.step_count >= self.config.max_episode_steps

        if truncated and not terminated:
            reward += self.config.draw_penalty

        # Update scores
        if terminated:
            self.match_count += 1
            if self.robot2_out and not self.robot1_out:
                self.robot1_score += 1
            elif self.robot1_out and not self.robot2_out:
                self.robot2_score += 1

        obs = self._get_observation(
            self.robot1, self.robot2, self.sensors1, self._robot1_cfg
        )
        info = self._get_info()

        if self.render_mode == "human":
            self._update_gui_overlay()
            self._pace_realtime()

        return obs, reward, terminated, truncated, info

    def _get_observation(
        self,
        robot: RobotBody,
        opponent: RobotBody,
        sensors: SensorSuite3D,
        robot_cfg: Robot3DConfig,
    ) -> np.ndarray:
        """Build 11-dim observation identical to 2D."""
        robot_pos, robot_orn = robot.get_position_and_orientation()
        opp_pos, _ = opponent.get_position_and_orientation()

        # Opponent sensor readings (5 values)
        sensor_readings = cast_opponent_rays(
            self.world.client,
            robot.body_id,
            opponent.body_id,
            robot_pos,
            robot_orn,
            sensors,
            add_noise=True,
            rng=self.rng,
        )

        # Velocity in world frame
        lin_vel, ang_vel = robot.get_velocity()
        max_speed = robot_cfg.max_linear_speed
        norm_vx = lin_vel[0] / max_speed if max_speed > 0 else 0.0
        norm_vy = lin_vel[1] / max_speed if max_speed > 0 else 0.0

        # Angular velocity (z component)
        norm_omega = ang_vel[2] / 10.0

        # Relative position to center (normalized by dohyo radius)
        xy = robot_pos[:2]
        rel_pos = xy / self._dohyo_cfg.radius

        # Relative angle to center
        yaw = robot.get_yaw()
        angle_to_center = np.arctan2(-xy[1], -xy[0])
        rel_angle = (angle_to_center - yaw) / np.pi

        obs = np.concatenate([
            sensor_readings,        # [0:5]
            [norm_vx, norm_vy],     # [5:7]
            [norm_omega],           # [7]
            rel_pos,                # [8:10]
            [rel_angle],            # [10]
        ]).astype(np.float32)

        return obs

    def _calculate_reward(
        self,
        prev_r1_dist: float,
        prev_r2_dist: float,
        collision_occurred: bool,
    ) -> float:
        """Calculate reward for robot 1 (same structure as 2D)."""
        cfg = self.config
        reward = cfg.time_penalty

        # Win/lose
        if self.robot2_out and not self.robot1_out:
            reward += cfg.win_reward
        elif self.robot1_out and not self.robot2_out:
            reward += cfg.lose_reward
        elif self.robot1_out and self.robot2_out:
            reward += -10.0

        # Push reward
        curr_r2_dist = np.linalg.norm(self.robot2.get_xy_position())
        reward += (curr_r2_dist - prev_r2_dist) * cfg.push_reward_scale

        # Edge penalty
        curr_r1_dist = np.linalg.norm(self.robot1.get_xy_position())
        edge_proximity = curr_r1_dist / self._dohyo_cfg.inner_radius
        if edge_proximity > 0.7:
            reward -= (edge_proximity - 0.7) * cfg.edge_penalty_scale

        # Collision aggression reward
        if collision_occurred:
            reward += 1.0

        return reward

    def _get_default_opponent_action(self) -> np.ndarray:
        """Simple seeking opponent (same logic as 2D)."""
        opp_xy = self.robot2.get_xy_position()
        target_xy = self.robot1.get_xy_position()
        opp_yaw = self.robot2.get_yaw()

        to_target = target_xy - opp_xy
        dist = np.linalg.norm(to_target)
        if dist < 1e-6:
            return np.array([1.0, 1.0])

        angle_to_target = np.arctan2(to_target[1], to_target[0])
        angle_diff = angle_to_target - opp_yaw
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Edge avoidance
        dist_from_center = np.linalg.norm(opp_xy)
        if dist_from_center > self._dohyo_cfg.inner_radius * 0.8:
            angle_to_center = np.arctan2(-opp_xy[1], -opp_xy[0])
            center_diff = angle_to_center - opp_yaw
            center_diff = (center_diff + np.pi) % (2 * np.pi) - np.pi
            if center_diff > 0:
                return np.array([0.3, 0.8])
            else:
                return np.array([0.8, 0.3])

        if abs(angle_diff) > 0.3:
            if angle_diff > 0:
                return np.array([0.5, 1.0])
            else:
                return np.array([1.0, 0.5])

        return np.array([1.0, 1.0])

    def _get_info(self) -> Dict[str, Any]:
        r1_xy = self.robot1.get_xy_position()
        r2_xy = self.robot2.get_xy_position()
        return {
            "step": self.step_count,
            "robot1_position": r1_xy.copy(),
            "robot2_position": r2_xy.copy(),
            "robot1_out": self.robot1_out,
            "robot2_out": self.robot2_out,
            "robot1_dist_to_edge": self._dohyo_cfg.inner_radius - np.linalg.norm(r1_xy),
            "robot2_dist_to_edge": self._dohyo_cfg.inner_radius - np.linalg.norm(r2_xy),
            "robot1_score": self.robot1_score,
            "robot2_score": self.robot2_score,
            "match_count": self.match_count,
        }

    def _get_hud_position(self, x_frac: float, y_frac: float,
                          depth: float = 0.5) -> list:
        """
        Compute a 3D world position that appears at (x_frac, y_frac) in
        the camera viewport.  (0,0)=top-left, (1,1)=bottom-right.
        """
        cam = p.getDebugVisualizerCamera(physicsClientId=self.world.client)
        width, height = cam[0], cam[1]
        view_mat = np.array(cam[2]).reshape(4, 4, order="F")
        proj_mat = np.array(cam[3]).reshape(4, 4, order="F")

        # Normalized device coords from screen fractions
        ndc_x = 2.0 * x_frac - 1.0
        ndc_y = 1.0 - 2.0 * y_frac  # flip vertical
        # Depth in NDC (near plane)
        ndc_z = -1.0 + 2.0 * depth

        ndc = np.array([ndc_x, ndc_y, ndc_z, 1.0])

        # Unproject: clip -> eye -> world
        inv_proj = np.linalg.inv(proj_mat)
        eye = inv_proj @ ndc
        eye = eye / eye[3]

        inv_view = np.linalg.inv(view_mat)
        world = inv_view @ eye
        world = world / world[3]

        return [float(world[0]), float(world[1]), float(world[2])]

    def _update_gui_overlay(self):
        """Update debug text pinned to the camera viewport."""
        if self.world is None or not self.world.gui:
            return

        t = self.step_count * self.config.dt
        score_str = (
            f"{self.robot1_name} {self.robot1_score} - "
            f"{self.robot2_score} {self.robot2_name}"
        )
        time_str = f"Time: {t:.1f}s  Match: {self.match_count + 1}"

        # Pin text to lower-left of viewport, yellow color
        pos_score = self._get_hud_position(0.02, 0.88)
        pos_time = self._get_hud_position(0.02, 0.93)

        self._score_text_id = self.world.add_debug_text(
            score_str, pos_score,
            color=[1, 1, 0], size=2.0,
            replace_id=self._score_text_id,
        )
        self._time_text_id = self.world.add_debug_text(
            time_str, pos_time,
            color=[1, 1, 0], size=1.5,
            replace_id=self._time_text_id,
        )

    def _pace_realtime(self):
        """Pace simulation to real-time in GUI mode."""
        now = time.time()
        elapsed = now - self._last_step_time
        target = self.config.dt
        if elapsed < target:
            time.sleep(target - elapsed)
        self._last_step_time = time.time()

    def render(self):
        """Rendering is handled automatically in step() for GUI mode."""
        pass

    def reset_scores(self):
        self.robot1_score = 0
        self.robot2_score = 0
        self.match_count = 0

    def close(self):
        if self.world is not None:
            self.world.close()
            self.world = None
            self.dohyo = None
            self.robot1 = None
            self.robot2 = None


class SumoEnv3DMultiAgent(SumoEnv3D):
    """
    Multi-agent version of SumoEnv3D for self-play training.
    Returns observations and accepts actions for both robots.
    """

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs1, info = super().reset(seed, options)
        obs2 = self._get_observation(
            self.robot2, self.robot1, self.sensors2, self._robot2_cfg
        )
        return {"robot1": obs1, "robot2": obs2}, info

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        action1 = actions.get("robot1", np.zeros(2))
        action2 = actions.get("robot2", np.zeros(2))

        obs1, reward1, terminated, truncated, info = super().step(action1, action2)
        obs2 = self._get_observation(
            self.robot2, self.robot1, self.sensors2, self._robot2_cfg
        )

        # Reward for robot2 (inverse perspective)
        reward2 = -reward1
        if self.robot1_out and not self.robot2_out:
            reward2 = self.config.win_reward
        elif self.robot2_out and not self.robot1_out:
            reward2 = self.config.lose_reward

        observations = {"robot1": obs1, "robot2": obs2}
        rewards = {"robot1": reward1, "robot2": reward2}
        terminateds = {"robot1": terminated, "robot2": terminated, "__all__": terminated}
        truncateds = {"robot1": truncated, "robot2": truncated, "__all__": truncated}

        return observations, rewards, terminateds, truncateds, info

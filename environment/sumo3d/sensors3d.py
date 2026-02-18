"""
3D Raycast-based sensors for mini sumo robots using PyBullet.

Provides:
- SensorSuite3D: configurable set of opponent and edge sensors
- Batched raycasting via p.rayTestBatch for efficiency
"""

import numpy as np
import pybullet as p
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class OpponentRayConfig:
    """Configuration for a single opponent-detection ray."""
    position_body: np.ndarray  # origin in body frame (x, y, z)
    direction_body: np.ndarray  # direction in body frame (x, y, z)
    max_range: float = 0.5  # meters

    def __post_init__(self):
        self.position_body = np.array(self.position_body, dtype=float)
        self.direction_body = np.array(self.direction_body, dtype=float)
        norm = np.linalg.norm(self.direction_body)
        if norm > 1e-8:
            self.direction_body = self.direction_body / norm


@dataclass
class EdgeRayConfig:
    """Configuration for a single downward edge-detection ray."""
    position_body: np.ndarray  # origin in body frame (x, y, z)
    max_range: float = 0.15  # meters downward

    def __post_init__(self):
        self.position_body = np.array(self.position_body, dtype=float)


@dataclass
class SensorSuite3D:
    """Complete 3D sensor configuration for a sumo robot."""
    opponent_rays: List[OpponentRayConfig] = field(default_factory=list)
    edge_rays: List[EdgeRayConfig] = field(default_factory=list)
    sensor_names: List[str] = field(default_factory=list)

    @classmethod
    def default_mini_sumo(
        cls,
        body_length: float = 0.098,
        body_width: float = 0.098,
        body_height: float = 0.04,
    ) -> "SensorSuite3D":
        """
        Create default sensor layout matching the 2D SensorSuite.

        5 opponent rays: Front, Front-Right, Front-Left, Right, Left
        4 edge rays: downward from the 4 corners
        """
        hl = body_length / 2 - 0.005
        hw = body_width / 2 - 0.005
        ray_z = 0.0  # at body center height

        opponent_rays = [
            # Front center
            OpponentRayConfig(
                position_body=[hl, 0, ray_z],
                direction_body=[1, 0, 0],
                max_range=0.5,
            ),
            # Front-right (~30 deg)
            OpponentRayConfig(
                position_body=[hl * 0.8, -hw * 0.8, ray_z],
                direction_body=[np.cos(-np.pi / 6), np.sin(-np.pi / 6), 0],
                max_range=0.4,
            ),
            # Front-left (~30 deg)
            OpponentRayConfig(
                position_body=[hl * 0.8, hw * 0.8, ray_z],
                direction_body=[np.cos(np.pi / 6), np.sin(np.pi / 6), 0],
                max_range=0.4,
            ),
            # Right side
            OpponentRayConfig(
                position_body=[0, -hw, ray_z],
                direction_body=[0, -1, 0],
                max_range=0.35,
            ),
            # Left side
            OpponentRayConfig(
                position_body=[0, hw, ray_z],
                direction_body=[0, 1, 0],
                max_range=0.35,
            ),
        ]

        # Edge rays: downward from corners
        corner_z = -body_height / 2  # bottom of body
        edge_rays = [
            EdgeRayConfig(position_body=[hl, hw, corner_z], max_range=0.15),
            EdgeRayConfig(position_body=[hl, -hw, corner_z], max_range=0.15),
            EdgeRayConfig(position_body=[-hl, hw, corner_z], max_range=0.15),
            EdgeRayConfig(position_body=[-hl, -hw, corner_z], max_range=0.15),
        ]

        sensor_names = ["Front", "F-Right", "F-Left", "Right", "Left"]

        return cls(
            opponent_rays=opponent_rays,
            edge_rays=edge_rays,
            sensor_names=sensor_names,
        )

    @property
    def num_opponent_rays(self) -> int:
        return len(self.opponent_rays)


def _transform_point(pos_body: np.ndarray, robot_pos: np.ndarray,
                     rot_matrix: np.ndarray) -> np.ndarray:
    """Transform a point from body frame to world frame."""
    return robot_pos + rot_matrix @ pos_body


def _transform_direction(dir_body: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
    """Transform a direction from body frame to world frame."""
    return rot_matrix @ dir_body


def _get_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Get 3x3 rotation matrix from quaternion [x,y,z,w]."""
    mat = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    return mat


def cast_opponent_rays(
    client: int,
    robot_body_id: int,
    opponent_body_id: int,
    robot_pos: np.ndarray,
    robot_orn: np.ndarray,
    sensor_suite: SensorSuite3D,
    add_noise: bool = True,
    noise_std: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Cast opponent detection rays and return normalized readings.

    Returns:
        Array of shape (num_opponent_rays,) with values in [0, 1].
        0 = very close, 1 = far / not detected.
    """
    if rng is None:
        rng = np.random.default_rng()

    rot = _get_rotation_matrix(robot_orn)
    rays = sensor_suite.opponent_rays
    n = len(rays)

    if n == 0:
        return np.array([], dtype=np.float32)

    ray_froms = []
    ray_tos = []

    for ray in rays:
        origin_world = _transform_point(ray.position_body, robot_pos, rot)
        dir_world = _transform_direction(ray.direction_body, rot)
        end_world = origin_world + dir_world * ray.max_range
        ray_froms.append(origin_world.tolist())
        ray_tos.append(end_world.tolist())

    results = p.rayTestBatch(
        ray_froms, ray_tos,
        parentObjectUniqueId=robot_body_id,
        parentLinkIndex=-1,
        physicsClientId=client,
    )

    readings = np.ones(n, dtype=np.float32)
    for i, result in enumerate(results):
        hit_object_id = result[0]
        hit_fraction = result[2]

        if hit_object_id == opponent_body_id:
            reading = hit_fraction  # 0=close, 1=at max range
            if add_noise:
                reading += rng.normal(0, noise_std)
                reading = np.clip(reading, 0.0, 1.0)
            readings[i] = float(reading)
        # else: remains 1.0 (no detection)

    return readings


def cast_edge_rays(
    client: int,
    robot_body_id: int,
    dohyo_body_id: int,
    robot_pos: np.ndarray,
    robot_orn: np.ndarray,
    sensor_suite: SensorSuite3D,
) -> np.ndarray:
    """
    Cast downward edge-detection rays.

    Returns:
        Array of shape (num_edge_rays,).
        0.0 = ray hits dohyo (safe), 1.0 = ray misses (over edge).
    """
    rot = _get_rotation_matrix(robot_orn)
    rays = sensor_suite.edge_rays
    n = len(rays)

    if n == 0:
        return np.array([], dtype=np.float32)

    ray_froms = []
    ray_tos = []

    for ray in rays:
        origin_world = _transform_point(ray.position_body, robot_pos, rot)
        end_world = origin_world + np.array([0, 0, -ray.max_range])
        ray_froms.append(origin_world.tolist())
        ray_tos.append(end_world.tolist())

    results = p.rayTestBatch(
        ray_froms, ray_tos,
        parentObjectUniqueId=robot_body_id,
        parentLinkIndex=-1,
        physicsClientId=client,
    )

    readings = np.ones(n, dtype=np.float32)
    for i, result in enumerate(results):
        hit_object_id = result[0]
        if hit_object_id == dohyo_body_id:
            readings[i] = 0.0  # safe — over dohyo

    return readings

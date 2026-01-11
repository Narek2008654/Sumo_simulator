"""
Sensor models for mini sumo robots.

Typical mini sumo robots have:
- Edge/Line sensors: Detect white border to avoid falling out
- Opponent sensors: IR proximity or distance sensors to detect opponent
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from .physics import RobotState, RobotPhysics


@dataclass
class EdgeSensorConfig:
    """Configuration for an edge detection sensor."""
    position: np.ndarray  # Position relative to robot center (body frame)
    detection_threshold: float = 0.02  # Distance at which white is detected
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)


@dataclass  
class OpponentSensorConfig:
    """Configuration for an opponent detection sensor (IR/ultrasonic)."""
    position: np.ndarray  # Position relative to robot center (body frame)
    direction: np.ndarray  # Direction the sensor points (body frame)
    max_range: float = 0.5  # Maximum detection range (meters)
    fov: float = np.pi / 6  # Field of view (radians, ~30 degrees)
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.direction = np.array(self.direction, dtype=float)
        # Normalize direction
        self.direction = self.direction / np.linalg.norm(self.direction)


@dataclass
class SensorSuite:
    """Complete sensor configuration for a sumo robot."""
    opponent_sensors: List[OpponentSensorConfig] = field(default_factory=list)
    
    # Sensor names for display
    sensor_names: List[str] = field(default_factory=list)
    
    @classmethod
    def default_mini_sumo(cls, robot_width: float = 0.1, robot_length: float = 0.1) -> 'SensorSuite':
        """
        Create a default sensor configuration for mini sumo.
        
        Configuration: 5 opponent sensors covering all directions
        - Front: straight ahead
        - Front-Right, Front-Left: angled forward
        - Right, Left: side detection
        """
        hw = robot_width / 2 - 0.005  # Slightly inside edges
        hl = robot_length / 2 - 0.005
        
        opponent_sensors = [
            # Front center - main attack direction
            OpponentSensorConfig(
                position=np.array([hl, 0]),
                direction=np.array([1, 0]),
                max_range=0.5,
                fov=np.pi/4  # 45 degree FOV
            ),
            # Front-right (angled 30 degrees)
            OpponentSensorConfig(
                position=np.array([hl * 0.8, -hw * 0.8]),
                direction=np.array([np.cos(-np.pi/6), np.sin(-np.pi/6)]),
                max_range=0.4,
                fov=np.pi/4
            ),
            # Front-left (angled 30 degrees)
            OpponentSensorConfig(
                position=np.array([hl * 0.8, hw * 0.8]),
                direction=np.array([np.cos(np.pi/6), np.sin(np.pi/6)]),
                max_range=0.4,
                fov=np.pi/4
            ),
            # Right side (90 degrees)
            OpponentSensorConfig(
                position=np.array([0, -hw]),
                direction=np.array([0, -1]),
                max_range=0.35,
                fov=np.pi/3  # 60 degree FOV
            ),
            # Left side (90 degrees)
            OpponentSensorConfig(
                position=np.array([0, hw]),
                direction=np.array([0, 1]),
                max_range=0.35,
                fov=np.pi/3
            ),
        ]
        
        sensor_names = ["Front", "F-Right", "F-Left", "Right", "Left"]
        
        return cls(opponent_sensors=opponent_sensors, sensor_names=sensor_names)
    
    @property
    def num_opponent_sensors(self) -> int:
        return len(self.opponent_sensors)
    
    @property
    def total_sensors(self) -> int:
        return self.num_opponent_sensors


def transform_to_world(
    robot_state: RobotState,
    position_body: np.ndarray,
    direction_body: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Transform position and direction from body frame to world frame."""
    c, s = np.cos(robot_state.theta), np.sin(robot_state.theta)
    R = np.array([[c, -s], [s, c]])
    
    pos_world = robot_state.position + R @ position_body
    dir_world = R @ direction_body if direction_body is not None else None
    
    return pos_world, dir_world


def read_opponent_sensor(
    sensor_config: OpponentSensorConfig,
    robot_state: RobotState,
    opponent_state: RobotState,
    opponent_physics: RobotPhysics,
    add_noise: bool = True,
    noise_std: float = 0.02,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Read opponent sensor value.
    
    Args:
        sensor_config: Sensor configuration
        robot_state: State of the robot with the sensor
        opponent_state: State of the opponent robot
        opponent_physics: Physical properties of opponent
        add_noise: Whether to add sensor noise
        noise_std: Standard deviation of noise (meters)
        rng: Random number generator
    
    Returns:
        Normalized distance reading (0 = touching/close, 1 = at max range or not detected)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Transform sensor to world frame
    sensor_pos, sensor_dir = transform_to_world(
        robot_state, 
        sensor_config.position, 
        sensor_config.direction
    )
    
    # Vector to opponent center
    to_opponent = opponent_state.position - sensor_pos
    dist_to_opponent = np.linalg.norm(to_opponent)
    
    if dist_to_opponent < 1e-6:
        return 0.0  # Touching
    
    to_opponent_normalized = to_opponent / dist_to_opponent
    
    # Check if opponent is within field of view
    cos_angle = np.dot(sensor_dir, to_opponent_normalized)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    if angle > sensor_config.fov / 2:
        return 1.0  # Not in FOV (far/not detected)
    
    # Account for opponent size (approximate as circle)
    opponent_radius = max(opponent_physics.width, opponent_physics.length) / 2
    effective_dist = max(0, dist_to_opponent - opponent_radius)
    
    # Add noise
    if add_noise:
        effective_dist += rng.normal(0, noise_std)
        effective_dist = max(0, effective_dist)
    
    # Check if within range
    if effective_dist > sensor_config.max_range:
        return 1.0  # Beyond range (far)
    
    # Return normalized reading (0 = close, 1 = far)
    return effective_dist / sensor_config.max_range


def read_all_sensors(
    robot_state: RobotState,
    robot_physics: RobotPhysics,
    sensor_suite: SensorSuite,
    opponent_state: RobotState,
    opponent_physics: RobotPhysics,
    dohyo=None,  # Not used anymore, kept for compatibility
    add_noise: bool = True,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Read all opponent sensors and return observation vector.
    
    Returns:
        Array of opponent sensor readings (0=very close, 1=far/not detected)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    readings = []
    
    # Read opponent sensors only
    for opp_sensor in sensor_suite.opponent_sensors:
        reading = read_opponent_sensor(
            opp_sensor,
            robot_state,
            opponent_state,
            opponent_physics,
            add_noise=add_noise,
            rng=rng
        )
        readings.append(reading)
    
    return np.array(readings, dtype=np.float32)


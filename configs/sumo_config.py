"""
Configuration for Mini Sumo Robot Simulation.

Based on official Mini Sumo specifications:
- Robot: Max 10cm x 10cm, Max 500g
- Dohyo: 77cm diameter, 2.5cm white border
"""

from dataclasses import dataclass
from typing import Dict, Any

# Import canonical config classes from their source modules
from environment.sumo.sumo_env import SumoEnvConfig
from environment.sumo.boards.dohyo import DohyoConfig


# ============================================================================
# Robot Physics Configuration
# ============================================================================

@dataclass
class RobotPhysicsConfig:
    """Physical properties of a mini sumo robot."""
    # Dimensions (meters) - Max 10cm x 10cm
    width: float = 0.098  # 9.8cm
    length: float = 0.098  # 9.8cm
    
    # Mass (kg) - Max 500g
    mass: float = 0.48  # 480g
    
    # Motor characteristics
    # Realistic for small DC motors with gearbox
    # F = ma, for 0.5kg at 2m/s² acceleration = 1N
    max_speed: float = 1.2  # m/s (typical mini sumo speed)
    max_force: float = 1.0  # N (total motor force, each side 0.5N)
    
    # Surface interaction
    friction_coeff: float = 0.6  # Rubber on painted steel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'width': self.width,
            'length': self.length,
            'mass': self.mass,
            'max_speed': self.max_speed,
            'max_force': self.max_force,
            'friction_coeff': self.friction_coeff,
        }


# ============================================================================
# Preset Configurations
# ============================================================================

# Standard competition robot
STANDARD_ROBOT = RobotPhysicsConfig()

# Fast robot (lighter, faster motors)
FAST_ROBOT = RobotPhysicsConfig(
    mass=0.35,
    max_speed=1.5,
    max_force=0.8,
)

# Heavy robot (maximum weight, high torque)
HEAVY_ROBOT = RobotPhysicsConfig(
    mass=0.50,
    max_speed=1.0,
    max_force=1.2,
    friction_coeff=0.7,
)

# Small robot (uses minimum space)
COMPACT_ROBOT = RobotPhysicsConfig(
    width=0.08,
    length=0.08,
    mass=0.30,
    max_speed=1.4,
    max_force=0.7,
)


# ============================================================================
# Sensor Configuration
# ============================================================================

@dataclass
class SensorConfig:
    """Sensor configuration for a robot."""
    # Edge sensors (line detection)
    num_edge_sensors: int = 4
    edge_sensor_positions: str = "corners"  # "corners" or "custom"
    edge_detection_threshold: float = 0.02  # 2cm detection range
    
    # Opponent sensors (IR/ultrasonic)
    num_opponent_sensors: int = 5
    opponent_sensor_layout: str = "standard"  # "standard" or "custom"
    max_detection_range: float = 0.6  # 60cm
    sensor_fov: float = 0.52  # ~30 degrees
    
    # Noise
    add_noise: bool = True
    edge_noise_std: float = 0.05
    opponent_noise_std: float = 0.02


# Default sensor configuration
DEFAULT_SENSORS = SensorConfig()

# High precision sensors
PRECISION_SENSORS = SensorConfig(
    num_opponent_sensors=8,
    max_detection_range=0.8,
    edge_noise_std=0.02,
    opponent_noise_std=0.01,
)


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    # Environment
    num_envs: int = 8  # Parallel environments
    max_episode_steps: int = 1000
    
    # Learning
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95
    
    # Network architecture
    policy_hidden_dims: tuple = (256, 256)
    value_hidden_dims: tuple = (256, 256)
    
    # Exploration
    ent_coef: float = 0.01  # Entropy coefficient
    clip_range: float = 0.2  # PPO clip range
    
    # Evaluation
    eval_freq: int = 10000
    n_eval_episodes: int = 20
    
    # Logging
    log_dir: str = "./logs/sumo"
    save_freq: int = 50000


# ============================================================================
# Complete Configuration Builder
# ============================================================================

def create_standard_config() -> Dict[str, Any]:
    """Create a standard configuration for mini sumo simulation."""
    return {
        'env': SumoEnvConfig(),
        'dohyo': DohyoConfig(),
        'robot1': STANDARD_ROBOT,
        'robot2': STANDARD_ROBOT,
        'sensors': DEFAULT_SENSORS,
    }


def create_training_config() -> Dict[str, Any]:
    """Create configuration optimized for RL training."""
    return {
        'env': SumoEnvConfig(
            render_mode=None,  # No rendering during training
            random_start=True,
            max_episode_steps=500,  # Shorter episodes for faster training
        ),
        'dohyo': DohyoConfig(),
        'robot1': STANDARD_ROBOT,
        'robot2': STANDARD_ROBOT,
        'sensors': DEFAULT_SENSORS,
        'training': TrainingConfig(),
    }


def create_visualization_config() -> Dict[str, Any]:
    """Create configuration for visualization/demo."""
    return {
        'env': SumoEnvConfig(
            render_mode="human",
            random_start=False,  # Start from standard positions
            fps=30,  # Slower for visibility
        ),
        'dohyo': DohyoConfig(),
        'robot1': STANDARD_ROBOT,
        'robot2': STANDARD_ROBOT,
        'sensors': DEFAULT_SENSORS,
    }

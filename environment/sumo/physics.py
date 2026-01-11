"""
Physics module for mini sumo robot simulation.
Handles robot dynamics, collision detection, and force calculations.

Based on real mini sumo specifications:
- Max mass: 500g (0.5 kg)
- Max dimensions: 10cm x 10cm
- Typical speed: 1-2 m/s
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RobotPhysics:
    """Physical properties of a sumo robot."""
    mass: float  # kg (max 0.5 kg for mini sumo)
    width: float  # meters (max 0.1m for mini sumo)
    length: float  # meters (max 0.1m for mini sumo)
    max_speed: float  # m/s
    max_force: float  # N (motor force)
    friction_coeff: float  # coefficient of friction with dohyo surface
    moment_of_inertia: float  # kg*m^2 (for rotation)
    
    @classmethod
    def from_config(cls, config: dict) -> 'RobotPhysics':
        """Create RobotPhysics from configuration dictionary."""
        mass = config.get('mass', 0.5)  # 500g default
        width = config.get('width', 0.1)  # 10cm
        length = config.get('length', 0.1)  # 10cm
        
        # Moment of inertia for rectangular body rotating about center
        moment_of_inertia = (mass / 12) * (width**2 + length**2)
        
        return cls(
            mass=mass,
            width=width,
            length=length,
            max_speed=config.get('max_speed', 1.5),  # 1.5 m/s typical
            max_force=config.get('max_force', 1.0),  # 1N - realistic for small motors
            friction_coeff=config.get('friction_coeff', 0.6),  # rubber on steel
            moment_of_inertia=moment_of_inertia
        )


@dataclass
class RobotState:
    """Kinematic state of a sumo robot."""
    x: float  # position x (meters)
    y: float  # position y (meters)
    theta: float  # heading angle (radians, 0 = facing right/positive x)
    vx: float  # velocity x (m/s)
    vy: float  # velocity y (m/s)
    omega: float  # angular velocity (rad/s)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.vx, self.vy, self.omega])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'RobotState':
        return cls(x=arr[0], y=arr[1], theta=arr[2], vx=arr[3], vy=arr[4], omega=arr[5])
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])
    
    @property
    def speed(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2)
    
    @property
    def heading_vector(self) -> np.ndarray:
        """Unit vector in the direction the robot is facing."""
        return np.array([np.cos(self.theta), np.sin(self.theta)])


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def integrate_robot_state(
    state: RobotState,
    physics: RobotPhysics,
    left_force: float,
    right_force: float,
    dt: float,
    wheel_base: Optional[float] = None,
    external_force: Optional[np.ndarray] = None
) -> RobotState:
    """
    Integrate robot state forward in time using differential drive model.
    
    Args:
        state: Current robot state
        physics: Robot physical properties
        left_force: Force from left motor (N)
        right_force: Force from right motor (N)
        dt: Time step (seconds)
        wheel_base: Distance between wheels (defaults to robot width * 0.8)
        external_force: External force vector [fx, fy] (N) - not used here, handled separately
    
    Returns:
        New robot state after integration
    """
    if wheel_base is None:
        wheel_base = physics.width * 0.8
    
    # Clamp motor forces
    max_f = physics.max_force
    left_force = np.clip(left_force, -max_f, max_f)
    right_force = np.clip(right_force, -max_f, max_f)
    
    # Total forward force and torque
    total_force = left_force + right_force
    # Torque from differential force (positive = turn left/counterclockwise)
    torque = (right_force - left_force) * wheel_base / 2
    
    # Current velocity in body frame
    c, s = np.cos(state.theta), np.sin(state.theta)
    v_body_x = c * state.vx + s * state.vy  # forward velocity
    
    # Simplified friction model for wheeled robots
    # Rolling resistance is much lower than sliding friction
    # Use velocity-dependent drag: F_drag = -k * v
    drag_coeff = physics.friction_coeff * physics.mass * 9.81 / physics.max_speed
    
    if abs(v_body_x) > 0.001:
        # Kinetic friction / drag
        drag_force = -drag_coeff * v_body_x
    else:
        # At rest - no drag
        drag_force = 0
    
    net_force = total_force + drag_force
    
    # Linear acceleration (F = ma)
    accel = net_force / physics.mass
    
    # Angular dynamics
    # Rotational drag proportional to angular velocity
    # Calibrated so max torque gives reasonable max angular velocity
    max_torque = max_f * wheel_base  # Maximum possible torque
    max_angular_vel = 8.0  # rad/s target max rotation (~460 deg/s)
    rot_drag_coeff = max_torque / max_angular_vel
    rot_drag = -rot_drag_coeff * state.omega
    
    angular_accel = (torque + rot_drag) / physics.moment_of_inertia
    
    # Clamp angular acceleration for stability
    max_angular_accel = 50.0  # rad/s^2
    angular_accel = np.clip(angular_accel, -max_angular_accel, max_angular_accel)
    
    # Integrate velocities
    new_v_body_x = v_body_x + accel * dt
    new_omega = state.omega + angular_accel * dt
    
    # Clamp to max speed
    new_v_body_x = np.clip(new_v_body_x, -physics.max_speed, physics.max_speed)
    max_omega = 10.0  # rad/s max rotation (~573 deg/s)
    new_omega = np.clip(new_omega, -max_omega, max_omega)
    
    # Integrate angle
    avg_omega = (state.omega + new_omega) / 2
    new_theta = normalize_angle(state.theta + avg_omega * dt)
    
    # Convert velocity back to world frame
    new_c, new_s = np.cos(new_theta), np.sin(new_theta)
    new_vx = new_c * new_v_body_x
    new_vy = new_s * new_v_body_x
    
    # Integrate position (using average velocity for better accuracy)
    avg_vx = (state.vx + new_vx) / 2
    avg_vy = (state.vy + new_vy) / 2
    new_x = state.x + avg_vx * dt
    new_y = state.y + avg_vy * dt
    
    return RobotState(
        x=new_x,
        y=new_y,
        theta=new_theta,
        vx=new_vx,
        vy=new_vy,
        omega=new_omega
    )


def get_robot_corners(state: RobotState, physics: RobotPhysics) -> np.ndarray:
    """
    Get the four corners of the robot in world coordinates.
    
    Returns:
        Array of shape (4, 2) with corner positions
    """
    half_w = physics.width / 2
    half_l = physics.length / 2
    
    # Corners in body frame (front-right, front-left, back-left, back-right)
    corners_body = np.array([
        [half_l, -half_w],
        [half_l, half_w],
        [-half_l, half_w],
        [-half_l, -half_w]
    ])
    
    # Rotation matrix
    c, s = np.cos(state.theta), np.sin(state.theta)
    R = np.array([[c, -s], [s, c]])
    
    # Transform to world frame
    corners_world = (R @ corners_body.T).T + state.position
    
    return corners_world


def check_robot_collision(
    state1: RobotState, physics1: RobotPhysics,
    state2: RobotState, physics2: RobotPhysics
) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
    """
    Check collision between two robots using SAT (Separating Axis Theorem).
    
    Returns:
        (is_colliding, collision_normal, penetration_depth)
        collision_normal points from robot1 to robot2
    """
    corners1 = get_robot_corners(state1, physics1)
    corners2 = get_robot_corners(state2, physics2)
    
    def get_axes(corners):
        axes = []
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            length = np.linalg.norm(edge)
            if length > 1e-6:
                normal = np.array([-edge[1], edge[0]]) / length
                axes.append(normal)
        return axes
    
    axes = get_axes(corners1) + get_axes(corners2)
    
    min_overlap = float('inf')
    collision_axis = None
    
    for axis in axes:
        proj1 = corners1 @ axis
        proj2 = corners2 @ axis
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2 or max2 < min1:
            return False, None, None
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            collision_axis = axis
    
    # Ensure normal points from robot1 to robot2
    center_diff = state2.position - state1.position
    if np.dot(collision_axis, center_diff) < 0:
        collision_axis = -collision_axis
    
    return True, collision_axis, min_overlap


def resolve_collision(
    state1: RobotState, physics1: RobotPhysics,
    state2: RobotState, physics2: RobotPhysics,
    normal: np.ndarray,
    penetration: float,
    restitution: float = 0.2
) -> Tuple[RobotState, RobotState]:
    """
    Resolve collision between two robots using impulse-based method.
    
    This properly handles:
    1. Position correction (separate overlapping robots)
    2. Velocity change (momentum conservation with restitution)
    
    Args:
        state1, state2: Current states of both robots
        physics1, physics2: Physical properties
        normal: Collision normal (pointing from robot1 to robot2)
        penetration: Overlap depth
        restitution: Coefficient of restitution (0 = inelastic, 1 = elastic)
    
    Returns:
        (new_state1, new_state2) - Updated states after collision resolution
    """
    m1, m2 = physics1.mass, physics2.mass
    total_mass = m1 + m2
    
    # 1. Position correction - separate the robots
    # Move each robot proportionally to inverse mass (lighter robot moves more)
    correction = penetration + 0.002  # Add small buffer to prevent re-collision
    pos1_correction = -normal * correction * (m2 / total_mass)
    pos2_correction = normal * correction * (m1 / total_mass)
    
    new_x1 = state1.x + pos1_correction[0]
    new_y1 = state1.y + pos1_correction[1]
    new_x2 = state2.x + pos2_correction[0]
    new_y2 = state2.y + pos2_correction[1]
    
    # 2. Velocity resolution using impulse method
    # Relative velocity along collision normal (v1 - v2 projected onto normal)
    # Positive = approaching, Negative = separating
    rel_vel = state1.velocity - state2.velocity
    rel_vel_normal = np.dot(rel_vel, normal)
    
    # If objects are already separating, only do position correction
    if rel_vel_normal < 0:
        return (
            RobotState(new_x1, new_y1, state1.theta, state1.vx, state1.vy, state1.omega),
            RobotState(new_x2, new_y2, state2.theta, state2.vx, state2.vy, state2.omega)
        )
    
    # Calculate impulse magnitude (conservation of momentum + restitution)
    # j = -(1 + e) * v_rel_n / (1/m1 + 1/m2)
    j = -(1 + restitution) * rel_vel_normal / (1/m1 + 1/m2)
    
    # Apply impulse to velocities
    # Δv1 = j * n / m1 (robot1 pushed in +normal direction)
    # Δv2 = -j * n / m2 (robot2 pushed in -normal direction)
    impulse = j * normal
    
    new_vx1 = state1.vx + impulse[0] / m1
    new_vy1 = state1.vy + impulse[1] / m1
    new_vx2 = state2.vx - impulse[0] / m2
    new_vy2 = state2.vy - impulse[1] / m2
    
    # Clamp velocities to reasonable bounds
    max_v = max(physics1.max_speed, physics2.max_speed) * 1.5
    
    speed1 = np.sqrt(new_vx1**2 + new_vy1**2)
    if speed1 > max_v:
        scale = max_v / speed1
        new_vx1 *= scale
        new_vy1 *= scale
    
    speed2 = np.sqrt(new_vx2**2 + new_vy2**2)
    if speed2 > max_v:
        scale = max_v / speed2
        new_vx2 *= scale
        new_vy2 *= scale
    
    # Create new states
    new_state1 = RobotState(
        x=new_x1, y=new_y1, theta=state1.theta,
        vx=new_vx1, vy=new_vy1, omega=state1.omega * 0.9  # Dampen rotation on collision
    )
    new_state2 = RobotState(
        x=new_x2, y=new_y2, theta=state2.theta,
        vx=new_vx2, vy=new_vy2, omega=state2.omega * 0.9
    )
    
    return new_state1, new_state2

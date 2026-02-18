"""
3D Physics module using PyBullet for mini sumo robot simulation.

Provides:
- BulletWorld: manages PyBullet client, gravity, ground plane
- DohyoBody: raised static cylinder dohyo platform
- RobotBody: differential-drive robot with box body + cylinder wheels
- Config dataclasses for robots and dohyo
"""

import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class Dohyo3DConfig:
    """Configuration for the 3D dohyo."""
    diameter: float = 0.77  # meters
    height: float = 0.05  # platform thickness
    border_width: float = 0.025  # tawara width
    surface_friction: float = 0.6  # painted steel

    # Visual colors (RGBA normalized)
    surface_color: Tuple[float, ...] = (0.12, 0.12, 0.12, 1.0)  # matte black
    border_color: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)  # white tawara

    # Starting line params (same as 2D)
    starting_line_separation: float = 0.10

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def inner_radius(self) -> float:
        return self.radius - self.border_width


@dataclass
class Robot3DConfig:
    """Configuration for a 3D sumo robot."""
    # Body dimensions
    body_length: float = 0.098  # 9.8 cm (x)
    body_width: float = 0.098   # 9.8 cm (y)
    body_height: float = 0.04   # 4 cm (z)

    # Wheel dimensions
    wheel_radius: float = 0.018  # 1.8 cm
    wheel_width: float = 0.012   # 1.2 cm

    # Mass
    body_mass: float = 0.44  # kg
    wheel_mass: float = 0.02  # kg each

    # Motor
    max_motor_velocity: float = 50.0  # rad/s (~0.9 m/s linear at r=0.018)
    max_motor_force: float = 1.5  # Nm torque per wheel

    # Friction
    lateral_friction: float = 0.8  # rubber on steel
    spinning_friction: float = 0.01
    rolling_friction: float = 0.001

    # Color (RGBA normalized)
    body_color: Tuple[float, ...] = (0.0, 0.4, 1.0, 1.0)  # blue
    wheel_color: Tuple[float, ...] = (0.2, 0.2, 0.2, 1.0)  # dark grey

    @property
    def total_mass(self) -> float:
        return self.body_mass + 2 * self.wheel_mass

    @property
    def max_linear_speed(self) -> float:
        """Approximate max linear speed from wheel angular velocity."""
        return self.max_motor_velocity * self.wheel_radius

    @property
    def wheel_base(self) -> float:
        """Distance between wheel centers (y-axis)."""
        return self.body_width + self.wheel_width


class DohyoBody:
    """Raised static cylinder dohyo platform in PyBullet."""

    def __init__(self, client: int, config: Dohyo3DConfig):
        self.client = client
        self.config = config
        self.body_id = -1
        self.border_id = -1
        self._create()

    def _create(self):
        """Create dohyo as two stacked cylinders: white border + black inner."""
        c = self.config
        half_height = c.height / 2

        # Outer cylinder (white border) — full radius
        border_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=c.radius,
            height=c.height,
            physicsClientId=self.client,
        )
        border_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=c.radius,
            length=c.height,
            rgbaColor=c.border_color,
            physicsClientId=self.client,
        )
        self.body_id = p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=border_col,
            baseVisualShapeIndex=border_vis,
            basePosition=[0, 0, half_height],
            physicsClientId=self.client,
        )

        # Inner disc visual (black surface) — sits slightly above border
        inner_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=c.inner_radius,
            length=0.001,  # thin disc
            rgbaColor=c.surface_color,
            physicsClientId=self.client,
        )
        self.border_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # no collision — visual only
            baseVisualShapeIndex=inner_vis,
            basePosition=[0, 0, c.height + 0.0005],
            physicsClientId=self.client,
        )

        # Set dohyo surface friction
        p.changeDynamics(
            self.body_id, -1,
            lateralFriction=c.surface_friction,
            spinningFriction=0.01,
            rollingFriction=0.001,
            physicsClientId=self.client,
        )

    @property
    def surface_z(self) -> float:
        """Z coordinate of the top surface."""
        return self.config.height

    def get_starting_positions(
        self, robot_length: float = 0.1
    ) -> List[Tuple[np.ndarray, float]]:
        """Same logic as 2D Dohyo.get_starting_positions."""
        half_sep = self.config.starting_line_separation / 2
        half_len = robot_length / 2
        pos1 = np.array([-(half_sep + half_len), 0.0])
        pos2 = np.array([(half_sep + half_len), 0.0])
        return [(pos1, 0.0), (pos2, np.pi)]

    def get_random_starting_positions(
        self,
        min_separation: float = 0.15,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Tuple[np.ndarray, float]]:
        """Same logic as 2D Dohyo.get_random_starting_positions."""
        if rng is None:
            rng = np.random.default_rng()

        safe_radius = self.config.inner_radius - 0.08

        r1 = rng.uniform(0, safe_radius)
        a1 = rng.uniform(0, 2 * np.pi)
        pos1 = np.array([r1 * np.cos(a1), r1 * np.sin(a1)])

        for _ in range(100):
            r2 = rng.uniform(0, safe_radius)
            a2 = rng.uniform(0, 2 * np.pi)
            pos2 = np.array([r2 * np.cos(a2), r2 * np.sin(a2)])
            if np.linalg.norm(pos2 - pos1) >= min_separation:
                break

        d = pos2 - pos1
        base1 = np.arctan2(d[1], d[0])
        base2 = base1 + np.pi
        theta1 = base1 + rng.uniform(-np.pi / 4, np.pi / 4)
        theta2 = base2 + rng.uniform(-np.pi / 4, np.pi / 4)

        return [(pos1, theta1), (pos2, theta2)]


class RobotBody:
    """
    Differential-drive robot built with p.createMultiBody.

    Structure:
        base link  — box body
        link 0     — left wheel (revolute joint, Y-axis)
        link 1     — right wheel (revolute joint, Y-axis)
    """

    LEFT_WHEEL = 0
    RIGHT_WHEEL = 1

    def __init__(self, client: int, config: Robot3DConfig, pos_xy: np.ndarray,
                 yaw: float, surface_z: float):
        self.client = client
        self.config = config
        self.body_id = -1
        self._create(pos_xy, yaw, surface_z)

    def _create(self, pos_xy: np.ndarray, yaw: float, surface_z: float):
        c = self.config

        # Half-extents for body box
        hx = c.body_length / 2
        hy = c.body_width / 2
        hz = c.body_height / 2

        # Body center z: wheels sit on the surface, body center is above
        body_z = surface_z + c.wheel_radius + hz

        # --- Collision / visual shapes ---
        body_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            physicsClientId=self.client,
        )
        body_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=c.body_color,
            physicsClientId=self.client,
        )

        wheel_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=c.wheel_radius,
            height=c.wheel_width,
            physicsClientId=self.client,
        )
        wheel_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=c.wheel_radius,
            length=c.wheel_width,
            rgbaColor=c.wheel_color,
            physicsClientId=self.client,
        )

        # Wheel attachment points relative to body center
        # Wheels at the back (-X) and raised slightly (like real mini sumo bots)
        wheel_x_offset = -hx * 0.65  # rear of body
        wheel_z_offset = -hz * 0.4   # slightly below center, not at very bottom
        left_pos = [wheel_x_offset, hy + c.wheel_width / 2, wheel_z_offset]
        right_pos = [wheel_x_offset, -(hy + c.wheel_width / 2), wheel_z_offset]

        # Wheel orientation: rotate 90 deg about X so cylinder axis aligns with Y
        wheel_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])

        self.body_id = p.createMultiBody(
            baseMass=c.body_mass,
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=[pos_xy[0], pos_xy[1], body_z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
            linkMasses=[c.wheel_mass, c.wheel_mass],
            linkCollisionShapeIndices=[wheel_col, wheel_col],
            linkVisualShapeIndices=[wheel_vis, wheel_vis],
            linkPositions=[left_pos, right_pos],
            linkOrientations=[wheel_orn, wheel_orn],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, -1], [0, 0, -1]],  # child -Z = parent -Y; positive vel = forward
            physicsClientId=self.client,
        )

        # Set friction on body and wheels
        p.changeDynamics(
            self.body_id, -1,
            lateralFriction=c.lateral_friction * 0.3,  # body slides more
            spinningFriction=c.spinning_friction,
            rollingFriction=c.rolling_friction,
            physicsClientId=self.client,
        )
        for link in [self.LEFT_WHEEL, self.RIGHT_WHEEL]:
            p.changeDynamics(
                self.body_id, link,
                lateralFriction=c.lateral_friction,
                spinningFriction=c.spinning_friction,
                rollingFriction=c.rolling_friction,
                physicsClientId=self.client,
            )

        # Disable default motor damping so velocity control works properly
        for joint in [self.LEFT_WHEEL, self.RIGHT_WHEEL]:
            p.setJointMotorControl2(
                self.body_id, joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
                physicsClientId=self.client,
            )

    def set_motor_speeds(self, left: float, right: float):
        """
        Apply motor commands.

        Args:
            left: normalized [-1, 1] left wheel command
            right: normalized [-1, 1] right wheel command
        """
        c = self.config
        left = float(np.clip(left, -1, 1))
        right = float(np.clip(right, -1, 1))

        # Map to target angular velocity (joint axis is -Y, so positive vel = forward)
        left_vel = left * c.max_motor_velocity
        right_vel = right * c.max_motor_velocity

        p.setJointMotorControl2(
            self.body_id, self.LEFT_WHEEL,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_vel,
            force=c.max_motor_force,
            physicsClientId=self.client,
        )
        p.setJointMotorControl2(
            self.body_id, self.RIGHT_WHEEL,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_vel,
            force=c.max_motor_force,
            physicsClientId=self.client,
        )

    def get_position_and_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position[3], quaternion[4]) of body base."""
        pos, orn = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client
        )
        return np.array(pos), np.array(orn)

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (linear_vel[3], angular_vel[3])."""
        lin, ang = p.getBaseVelocity(
            self.body_id, physicsClientId=self.client
        )
        return np.array(lin), np.array(ang)

    def get_yaw(self) -> float:
        """Get heading angle (yaw) in radians."""
        _, orn = self.get_position_and_orientation()
        euler = p.getEulerFromQuaternion(orn)
        return euler[2]  # yaw

    def get_z_position(self) -> float:
        """Get z coordinate of base center."""
        pos, _ = self.get_position_and_orientation()
        return pos[2]

    def get_xy_position(self) -> np.ndarray:
        """Get [x, y] position of base center."""
        pos, _ = self.get_position_and_orientation()
        return pos[:2]

    def reset(self, pos_xy: np.ndarray, yaw: float, surface_z: float):
        """Reset robot to given pose with zero velocity."""
        c = self.config
        hx = c.body_length / 2
        hy = c.body_width / 2
        hz = c.body_height / 2
        body_z = surface_z + c.wheel_radius + hz

        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(
            self.body_id,
            [pos_xy[0], pos_xy[1], body_z],
            orn,
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.client,
        )
        # Reset wheel joints
        for joint in [self.LEFT_WHEEL, self.RIGHT_WHEEL]:
            p.resetJointState(
                self.body_id, joint,
                targetValue=0,
                targetVelocity=0,
                physicsClientId=self.client,
            )


class BulletWorld:
    """Manages the PyBullet physics client."""

    def __init__(self, gui: bool = False, time_step: float = 1.0 / 240):
        self.gui = gui
        self.time_step = time_step

        if gui:
            self.client = p.connect(p.GUI)
            # Configure camera
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0.02],
                physicsClientId=self.client,
            )
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)

        # Load ground plane well below dohyo (catch fallen robots)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadURDF(
            "plane.urdf",
            basePosition=[0, 0, -0.5],
            physicsClientId=self.client,
        )

    def step(self):
        """Advance simulation by one time step."""
        p.stepSimulation(physicsClientId=self.client)

    def create_dohyo(self, config: Optional[Dohyo3DConfig] = None) -> DohyoBody:
        config = config or Dohyo3DConfig()
        return DohyoBody(self.client, config)

    def create_robot(
        self,
        config: Optional[Robot3DConfig] = None,
        pos_xy: Optional[np.ndarray] = None,
        yaw: float = 0.0,
        surface_z: float = 0.05,
    ) -> RobotBody:
        config = config or Robot3DConfig()
        if pos_xy is None:
            pos_xy = np.array([0.0, 0.0])
        return RobotBody(self.client, config, pos_xy, yaw, surface_z)

    def get_contact_points(self, body_a: int, body_b: int):
        """Get contact points between two bodies."""
        return p.getContactPoints(
            bodyA=body_a, bodyB=body_b,
            physicsClientId=self.client,
        )

    def get_keyboard_events(self):
        """Get keyboard events (GUI mode only)."""
        return p.getKeyboardEvents(physicsClientId=self.client)

    def add_debug_text(self, text: str, position: List[float],
                       color: List[float] = None,
                       size: float = 1.5,
                       replace_id: int = -1) -> int:
        """Add/update debug text in GUI."""
        if color is None:
            color = [1, 1, 1]
        return p.addUserDebugText(
            text, position,
            textColorRGB=color,
            textSize=size,
            replaceItemUniqueId=replace_id,
            physicsClientId=self.client,
        )

    def close(self):
        """Disconnect from physics server."""
        try:
            p.disconnect(physicsClientId=self.client)
        except p.error:
            pass

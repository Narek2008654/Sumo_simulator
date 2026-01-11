"""
Dohyo (Mini Sumo Ring) implementation.

Official Mini Sumo Dohyo Specifications:
- Diameter: 77 cm (0.77 m)
- Surface: Matte black
- Border (Tawara): 2.5 cm white ring around the edge
- Starting lines (Shikiri): Two brown/red lines, 10 cm apart, centered
- Starting line dimensions: 10 cm long, 1 cm wide
"""

import numpy as np
import pygame
from typing import Tuple, List, Optional
from dataclasses import dataclass

from ..physics import RobotState, RobotPhysics, get_robot_corners


@dataclass
class DohyoConfig:
    """Configuration for the dohyo ring."""
    diameter: float = 0.77  # meters
    border_width: float = 0.025  # meters (tawara)
    starting_line_length: float = 0.10  # meters
    starting_line_width: float = 0.01  # meters
    starting_line_separation: float = 0.10  # meters (distance between lines)
    
    # Colors (RGB)
    surface_color: Tuple[int, int, int] = (30, 30, 30)  # matte black
    border_color: Tuple[int, int, int] = (255, 255, 255)  # white tawara
    starting_line_color: Tuple[int, int, int] = (139, 69, 19)  # brown shikiri
    background_color: Tuple[int, int, int] = (80, 80, 80)  # outside ring
    
    @property
    def radius(self) -> float:
        return self.diameter / 2
    
    @property
    def inner_radius(self) -> float:
        """Radius of the playable area (inside white border)."""
        return self.radius - self.border_width


class Dohyo:
    """
    Mini Sumo Dohyo (ring) implementation.
    
    The dohyo is a circular ring where sumo robots compete.
    A robot loses when any part of it touches the white border (tawara).
    """
    
    def __init__(self, config: Optional[DohyoConfig] = None):
        self.config = config or DohyoConfig()
        self.center = np.array([0.0, 0.0])  # Dohyo centered at origin
        
    @property
    def radius(self) -> float:
        return self.config.radius
    
    @property
    def inner_radius(self) -> float:
        return self.config.inner_radius
    
    def is_point_in_ring(self, point: np.ndarray) -> bool:
        """Check if a point is inside the ring (not on border)."""
        dist = np.linalg.norm(point - self.center)
        return dist < self.inner_radius
    
    def is_point_on_border(self, point: np.ndarray) -> bool:
        """Check if a point is on the white border."""
        dist = np.linalg.norm(point - self.center)
        return self.inner_radius <= dist <= self.radius
    
    def is_point_outside(self, point: np.ndarray) -> bool:
        """Check if a point is completely outside the ring."""
        dist = np.linalg.norm(point - self.center)
        return dist > self.radius
    
    def is_robot_out(self, state: RobotState, physics: RobotPhysics) -> bool:
        """
        Check if any part of the robot has crossed outside the ring.
        
        Official rules: Robot loses when any part touches the ground
        OUTSIDE the white border (tawara). The white border itself
        is part of the ring and is safe.
        
        - Black surface: safe
        - White border: safe (still on the ring)
        - Outside ring: OUT (loses)
        """
        corners = get_robot_corners(state, physics)
        for corner in corners:
            if self.is_point_outside(corner):
                return True
        # Also check center
        if self.is_point_outside(state.position):
            return True
        return False
    
    def get_edge_sensor_reading(
        self, 
        state: RobotState, 
        sensor_offset: np.ndarray,
        max_detection_dist: float = 0.03
    ) -> float:
        """
        Get reading from an edge detection sensor (line sensor).
        
        Args:
            state: Robot state
            sensor_offset: Position of sensor relative to robot center (body frame)
            max_detection_dist: Maximum distance at which white line can be detected
        
        Returns:
            0.0 if on black surface, 1.0 if on white border
        """
        # Transform sensor position to world frame
        c, s = np.cos(state.theta), np.sin(state.theta)
        R = np.array([[c, -s], [s, c]])
        sensor_world = state.position + R @ sensor_offset
        
        # Check distance to border
        dist_from_center = np.linalg.norm(sensor_world - self.center)
        dist_to_border = self.inner_radius - dist_from_center
        
        if dist_to_border <= 0:
            return 1.0  # On or past border
        elif dist_to_border < max_detection_dist:
            return 1.0 - (dist_to_border / max_detection_dist)  # Gradient
        else:
            return 0.0  # On black surface
    
    def get_starting_positions(self, robot_length: float = 0.1) -> List[Tuple[np.ndarray, float]]:
        """
        Get the two starting positions behind the shikiri lines.
        
        In official rules, robots start with their front edge just behind
        the starting line (shikiri). They cannot cross it until match starts.
        
        Args:
            robot_length: Length of the robot (to position center correctly)
        
        Returns:
            List of (position, heading) tuples for each starting position.
            Robots face each other from opposite sides.
        """
        half_sep = self.config.starting_line_separation / 2
        robot_half_len = robot_length / 2
        
        # Robot centers are positioned so front edge is at the shikiri line
        # Robot 1: on the left, facing right (front at x = -half_sep)
        pos1 = np.array([-(half_sep + robot_half_len), 0.0])
        theta1 = 0  # facing positive x (toward opponent)
        
        # Robot 2: on the right, facing left (front at x = +half_sep)
        pos2 = np.array([half_sep + robot_half_len, 0.0])
        theta2 = np.pi  # facing negative x (toward opponent)
        
        return [(pos1, theta1), (pos2, theta2)]
    
    def get_random_starting_positions(
        self, 
        min_separation: float = 0.15,
        rng: Optional[np.random.Generator] = None
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Get randomized starting positions for training variety.
        
        Args:
            min_separation: Minimum distance between robots
            rng: Random number generator
        
        Returns:
            List of (position, heading) tuples
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Safe radius (keeping robots away from edge)
        safe_radius = self.inner_radius - 0.08  # robot half-diagonal
        
        # Generate first position
        r1 = rng.uniform(0, safe_radius)
        theta1_pos = rng.uniform(0, 2 * np.pi)
        pos1 = np.array([r1 * np.cos(theta1_pos), r1 * np.sin(theta1_pos)])
        
        # Generate second position with minimum separation
        for _ in range(100):  # Max attempts
            r2 = rng.uniform(0, safe_radius)
            theta2_pos = rng.uniform(0, 2 * np.pi)
            pos2 = np.array([r2 * np.cos(theta2_pos), r2 * np.sin(theta2_pos)])
            
            if np.linalg.norm(pos2 - pos1) >= min_separation:
                break
        
        # Random headings, biased toward facing each other
        dir_to_opponent = pos2 - pos1
        base_angle1 = np.arctan2(dir_to_opponent[1], dir_to_opponent[0])
        base_angle2 = base_angle1 + np.pi
        
        theta1 = base_angle1 + rng.uniform(-np.pi/4, np.pi/4)
        theta2 = base_angle2 + rng.uniform(-np.pi/4, np.pi/4)
        
        return [(pos1, theta1), (pos2, theta2)]
    
    def render(
        self, 
        surface: pygame.Surface, 
        scale: float, 
        offset: Tuple[int, int]
    ) -> None:
        """
        Render the dohyo on a pygame surface.
        
        Args:
            surface: Pygame surface to draw on
            scale: Pixels per meter
            offset: (x, y) offset to center of screen
        """
        cx, cy = offset
        
        # Draw background
        surface.fill(self.config.background_color)
        
        # Draw white border (tawara)
        outer_radius_px = int(self.radius * scale)
        pygame.draw.circle(surface, self.config.border_color, (cx, cy), outer_radius_px)
        
        # Draw black surface
        inner_radius_px = int(self.inner_radius * scale)
        pygame.draw.circle(surface, self.config.surface_color, (cx, cy), inner_radius_px)
        
        # Draw starting lines (shikiri)
        line_half_len = self.config.starting_line_length / 2 * scale
        line_half_width = self.config.starting_line_width / 2 * scale
        half_sep = self.config.starting_line_separation / 2 * scale
        
        for x_offset in [-half_sep, half_sep]:
            rect = pygame.Rect(
                cx + x_offset - line_half_width,
                cy - line_half_len,
                line_half_width * 2,
                line_half_len * 2
            )
            pygame.draw.rect(surface, self.config.starting_line_color, rect)
        
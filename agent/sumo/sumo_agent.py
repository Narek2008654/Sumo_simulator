"""
Sumo Robot Agent implementations.

Provides different agent types for mini sumo:
- Manual control agent
- Scripted behavior agents
"""

import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class SumoAgent(ABC):
    """Base class for sumo robot agents."""
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Choose an action based on observation.
        
        Args:
            observation: Sensor readings and state information
        
        Returns:
            Action array [left_motor, right_motor] in range [-1, 1]
        """
        pass
    
    def reset(self):
        """Reset agent state for new episode."""
        pass


class ManualAgent(SumoAgent):
    """Agent controlled by keyboard input."""
    
    def __init__(self):
        self.left_motor = 0.0
        self.right_motor = 0.0
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.array([self.left_motor, self.right_motor], dtype=np.float32)
    
    def set_controls(self, left: float, right: float):
        """Set motor values directly."""
        self.left_motor = np.clip(left, -1, 1)
        self.right_motor = np.clip(right, -1, 1)
    
    def handle_keys(self, keys) -> None:
        """
        Handle pygame key states.
        
        Controls:
        - W: Forward
        - S: Backward
        - A: Turn left
        - D: Turn right
        - Q: Spin left
        - E: Spin right
        """
        import pygame
        
        left = 0.0
        right = 0.0
        
        if keys[pygame.K_w]:  # Forward
            left += 1.0
            right += 1.0
        if keys[pygame.K_s]:  # Backward
            left -= 1.0
            right -= 1.0
        if keys[pygame.K_a]:  # Turn left
            left -= 0.5
            right += 0.5
        if keys[pygame.K_d]:  # Turn right
            left += 0.5
            right -= 0.5
        if keys[pygame.K_q]:  # Spin left
            left = -1.0
            right = 1.0
        if keys[pygame.K_e]:  # Spin right
            left = 1.0
            right = -1.0
        
        self.left_motor = np.clip(left, -1, 1)
        self.right_motor = np.clip(right, -1, 1)


class AggressiveAgent(SumoAgent):
    """
    Aggressive scripted agent that seeks and charges the opponent.
    """
    
    def __init__(self, aggression: float = 1.0):
        """
        Args:
            aggression: How aggressively to pursue (0-1)
        """
        self.aggression = aggression
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Action based on sensor readings.
        
        Observation format (11 values):
        - [0:5] Opponent sensors (front, front-right, front-left, right, left)
        - [5:7] Velocity (vx, vy normalized)
        - [7] Angular velocity (normalized)
        - [8:10] Relative position to center
        - [10] Relative angle to center
        """
        # Parse opponent sensors (0=close, 1=far)
        front = observation[0]
        front_right = observation[1]
        front_left = observation[2]
        right = observation[3]
        left = observation[4]
        
        # Get position relative to center (for edge avoidance)
        rel_pos = observation[8:10]
        dist_from_center = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        
        # If near edge (dist > 0.8 of normalized radius), be careful
        if dist_from_center > 0.8:
            # Turn toward center
            return np.array([0.3, 0.7], dtype=np.float32)
        
        # If opponent detected in front (low value = close), charge!
        if front < 0.7:
            return np.array([1.0, 1.0], dtype=np.float32) * self.aggression
        
        # Turn toward opponent (low value = detected)
        if front_right < 0.8 or right < 0.8:
            return np.array([0.9, 0.4], dtype=np.float32) * self.aggression
        
        if front_left < 0.8 or left < 0.8:
            return np.array([0.4, 0.9], dtype=np.float32) * self.aggression
        
        # No opponent detected (all sensors = 1.0) - search (spin slowly)
        return np.array([0.5, 0.8], dtype=np.float32) * self.aggression


class DefensiveAgent(SumoAgent):
    """
    Defensive agent that avoids edges and waits for opponent.
    """
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Observation format (11 values):
        - [0:5] Opponent sensors (front, front-right, front-left, right, left)
        - [5:7] Velocity (vx, vy normalized)
        - [7] Angular velocity (normalized)
        - [8:10] Relative position to center
        - [10] Relative angle to center
        """
        # Parse sensors
        front = observation[0]
        front_right = observation[1]
        front_left = observation[2]
        right = observation[3]
        left = observation[4]
        rel_pos = observation[8:10]
        rel_angle = observation[10]
        
        # Priority 1: Stay near center
        dist_from_center = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        if dist_from_center > 0.6:  # Getting far from center
            # Turn toward center and move
            turn = rel_angle * 0.6
            return np.array([0.6 - turn, 0.6 + turn], dtype=np.float32)
        
        # Priority 2: Face and counter opponent (0=close, 1=far)
        if front < 0.6:
            # Opponent approaching from front - counter charge!
            return np.array([1.0, 1.0], dtype=np.float32)
        
        if front_right < 0.8 or right < 0.85:
            return np.array([0.6, 0.3], dtype=np.float32)
        if front_left < 0.8 or left < 0.85:
            return np.array([0.3, 0.6], dtype=np.float32)
        
        # Priority 3: Stay centered and search
        if dist_from_center > 0.3:
            turn = rel_angle * 0.3
            return np.array([0.4 - turn, 0.4 + turn], dtype=np.float32)
        
        # Default: Slow rotation to search while staying put
        return np.array([0.3, 0.5], dtype=np.float32)


class RandomAgent(SumoAgent):
    """Random action agent for baseline comparison."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        # Bias toward forward motion
        base = self.rng.uniform(0.3, 1.0)
        diff = self.rng.uniform(-0.5, 0.5)
        return np.array([base + diff, base - diff], dtype=np.float32)


class PPOGameAgent(SumoAgent):
    """Wrapper for the trained PPO agent to be used in the game UI/scripts."""
    
    def __init__(self, model_dir: str = "tmp/ppo_continuous"):
        """
        Load a trained PPO model.
        
        Args:
            model_dir: Directory containing 'actor.pt' and 'critic.pt'
        """
        import sys
        import os
        
        # Ensure project root is in sys.path for the following imports
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        try:
            from agent.sumo.PPO.torch.ppo_torch_continuous import PPOAgent, PPOConfig
        except ImportError as e:
            print(f"Failed to import PPOAgent. Make sure project structure is correct. Error: {e}")
            raise

        # Initialize agent with same default config as training
        self.cfg = PPOConfig(
            chkpt_dir=model_dir,
            obs_dim=11,      # Must match training
            action_dim=2,
            hidden1=16,
            hidden2=16,
            use_tanh=True
        )
        
        self.agent = PPOAgent(self.cfg)
        
        # Load the trained weights
        try:
            self.agent.load_models()
            print(f"✅ Successfully loaded PPO models from {model_dir}")
        except FileNotFoundError:
            print(f"❌ Error: Could not find model files in {model_dir}")
            raise
            
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Get action from the trained PPO agent."""
        # PPOAgent.choose_action returns (action_env, action_raw, log_prob, val)
        # action_env is what the environment expects (normalized [-1, 1])
        action_env, _, _, _ = self.agent.choose_action(observation)
        return action_env


class PPODiscreteGameAgent(SumoAgent):
    """Wrapper for the trained discrete-action PPO agent."""
    
    def __init__(self, model_dir: str = "tmp/ppo_discrete"):
        """
        Load a trained discrete PPO model.
        
        Args:
            model_dir: Directory containing 'actor_discrete.pt' and 'critic_discrete.pt'
        """
        import sys
        import os
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Add the discrete PPO module directory
        discrete_dir = os.path.join(project_root, 'agent', 'sumo', 'PPO', 'torch_discrete')
        if discrete_dir not in sys.path:
            sys.path.insert(0, discrete_dir)
            
        try:
            from ppo_torch_discrete import PPODiscreteAgent, PPODiscreteConfig
        except ImportError as e:
            print(f"Failed to import PPODiscreteAgent. Error: {e}")
            raise

        self.cfg = PPODiscreteConfig(
            chkpt_dir=model_dir,
            obs_dim=11,
            num_actions=5,
            hidden1=64,
            hidden2=32,
        )
        
        self.agent = PPODiscreteAgent(self.cfg)
        
        try:
            self.agent.load_models()
            print(f"✅ Successfully loaded discrete PPO models from {model_dir}")
        except FileNotFoundError:
            print(f"❌ Error: Could not find model files in {model_dir}")
            raise
            
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Get action from the trained discrete PPO agent."""
        # choose_action returns (action_env, action_idx, log_prob, val)
        action_env, _, _, _ = self.agent.choose_action(observation)
        return action_env

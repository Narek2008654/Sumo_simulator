# Mini Sumo Robot Simulator

A physics-based simulation environment for mini sumo robot competitions. Based on official rules from Robotex and RoboChallenge competitions.

## 🤖 Mini Sumo

The mini sumo simulation follows official competition rules:
- **Robot**: 10cm × 10cm footprint, max 500g
- **Dohyo (Ring)**: 77cm diameter circular black surface with 2.5cm white border
- **Objective**: Push opponent out of the ring

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Play against opponent (you control blue robot)
python sumo_game.py --mode play

# Watch agent vs agent demo
python sumo_game.py --mode demo
```

### 3D Simulation (PyBullet)

```bash
# Play in 3D (you control blue robot)
python sumo_game_3d.py --mode play

# Watch agent vs agent in 3D
python sumo_game_3d.py --mode demo

# Options
python sumo_game_3d.py --mode play --opponent defensive   # defensive opponent
python sumo_game_3d.py --mode demo --episodes 20          # 20 demo episodes
python sumo_game_3d.py --mode play --random-start          # random starting positions
```

## Controls

### 2D (Pygame)

| Key | Action |
|-----|--------|
| W | Forward |
| S | Backward |
| A | Turn Left |
| D | Turn Right |
| Q | Spin Left |
| E | Spin Right |
| R | Reset Match |
| C | Clear Scores |
| ESC | Quit |

### 3D (PyBullet)

| Key | Action |
|-----|--------|
| Arrow Up | Forward |
| Arrow Down | Backward |
| Arrow Left | Turn Left |
| Arrow Right | Turn Right |
| Z | Spin Left |
| X | Spin Right |
| R | Reset Match |
| ESC | Quit |

## 📁 Project Structure

```
Sumo_simulator/
├── agent/
│   └── sumo/
│       └── sumo_agent.py  # Sumo robot agents (Manual, Aggressive, etc.)
├── environment/
│   ├── sumo/
│   │   ├── sumo_env.py    # 2D Gymnasium-compatible environment
│   │   ├── physics.py     # 2D robot physics and collision detection
│   │   ├── sensors.py     # Edge and opponent sensors
│   │   └── boards/
│   │       └── dohyo.py   # Dohyo ring implementation
│   └── sumo3d/
│       ├── sumo_env_3d.py # 3D PyBullet environment
│       ├── physics3d.py   # 3D physics (PyBullet)
│       └── sensors3d.py   # 3D sensor implementation
├── configs/
│   └── sumo_config.py     # Configuration presets
├── sumo_game.py           # 2D play/demo game script (Pygame)
├── sumo_game_3d.py        # 3D play/demo game script (PyBullet)
└── requirements.txt
```

## ⚙️ Configuration

### Robot Physics

```python
from configs.sumo_config import RobotPhysicsConfig, FAST_ROBOT, HEAVY_ROBOT

# Custom robot
my_robot = RobotPhysicsConfig(
    width=0.095,
    length=0.095,
    mass=0.45,
    max_speed=1.8,
    max_force=2.2,
    friction_coeff=0.75,
)
```

### Environment Settings

```python
from environment.sumo.sumo_env import SumoEnvConfig

config = SumoEnvConfig(
    dt=0.02,                  # Simulation timestep (50Hz)
    max_episode_steps=1000,   # Episode length
    random_start=True,        # Random starting positions
    render_mode="human",      # or "rgb_array" or None
)
```

## 🎮 Available Agents

| Agent | Description |
|-------|-------------|
| `ManualAgent` | Keyboard-controlled |
| `AggressiveAgent` | Seeks and charges opponent |
| `DefensiveAgent` | Stays near center, counters attacks |
| `RandomAgent` | Random actions (baseline) |

## 🎯 Using the Environment

The simulator provides a Gymnasium-compatible environment that can be used for testing algorithms or as a base for reinforcement learning:

```python
from environment.sumo.sumo_env import SumoEnv, SumoEnvConfig

# Create environment
config = SumoEnvConfig(render_mode="human", random_start=True)
env = SumoEnv(config=config)

# Reset and run
obs, info = env.reset()

for _ in range(1000):
    # Your action here: [left_motor, right_motor] in range [-1, 1]
    action = np.array([0.5, 0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Environment Details

**Observation Space** (15 dimensions):
- Opponent sensors (5): Proximity to opponent (0=close, 1=far)
- Velocity (2): Normalized vx, vy
- Angular velocity (1): Normalized omega
- Position to center (2): Normalized relative position
- Angle to center (1): Normalized relative angle

**Action Space** (2 dimensions):
- `[left_motor, right_motor]` in range `[-1, 1]`
- Differential drive control

## Requirements

- Python 3.8+
- pygame >= 2.1.0
- numpy >= 1.21.0
- gymnasium >= 0.29.0
- pybullet >= 3.2.0 (for 3D simulation)

## License

MIT License

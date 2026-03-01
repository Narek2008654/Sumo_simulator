"""
Compact DQN (Q-Model) for Mini Sumo — designed to fit on Arduino Nano.

Network: 5 → 16 → 16 → 5
  Input : 3 front sensors + last action (left_motor, right_motor)
  Output: Q-values for 5 discrete actions

Actions (same as PPO discrete):
  0: [-1, -1]  backward
  1: [ 1,  1]  forward
  2: [ 0,  0]  stop
  3: [-1,  1]  spin left
  4: [ 1, -1]  spin right

Total parameters: 453  (~1.8 KB float32, ~0.9 KB int8)
"""

import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── action mapping (shared with PPO discrete) ─────────────────────────────
ACTION_MAP = np.array([
    [-1, -1],   # 0  backward
    [ 1,  1],   # 1  forward
    [ 0,  0],   # 2  stop
    [-1,  1],   # 3  spin left
    [ 1, -1],   # 4  spin right
], dtype=np.float32)

NUM_ACTIONS = len(ACTION_MAP)


def discrete_to_continuous(action_idx: int) -> np.ndarray:
    """Convert a discrete action index to [left_motor, right_motor]."""
    return ACTION_MAP[action_idx].copy()


# ── config ─────────────────────────────────────────────────────────────────
@dataclass
class QModelConfig:
    obs_dim: int = 5            # 3 front sensors + 2 last-action motors
    num_actions: int = NUM_ACTIONS
    hidden1: int = 16
    hidden2: int = 16
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000
    target_update_freq: int = 1000   # steps between target-net sync
    replay_capacity: int = 10_000
    batch_size: int = 64
    min_replay_size: int = 500       # start training after this many transitions
    chkpt_dir: str = "tmp/q_model"
    model_name: str = "q_network.pt"


# ── replay buffer ──────────────────────────────────────────────────────────
class ReplayBuffer:
    """Simple ring buffer for experience replay."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── Q-network ──────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """Tiny MLP: obs_dim → hidden1 → hidden2 → num_actions."""

    def __init__(self, cfg: QModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden1),
            nn.ReLU(),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Linear(cfg.hidden2, cfg.num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── DQN agent ──────────────────────────────────────────────────────────────
class DQNAgent:
    """Compact DQN agent with target network and epsilon-greedy exploration."""

    def __init__(self, cfg: QModelConfig):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(cfg).to(self.device)
        self.target_net = QNetwork(cfg).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_capacity)

        self.epsilon = cfg.epsilon_start
        self.step_count = 0

        os.makedirs(cfg.chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(cfg.chkpt_dir, cfg.model_name)

    # ── epsilon schedule ───────────────────────────────────────────────
    def _update_epsilon(self):
        frac = min(1.0, self.step_count / self.cfg.epsilon_decay_steps)
        self.epsilon = self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    # ── action selection ───────────────────────────────────────────────
    def choose_action(self, obs_5d: np.ndarray):
        """
        Epsilon-greedy action selection.

        Args:
            obs_5d: 5-dim observation [front, f-right, f-left, last_L, last_R]

        Returns:
            action_env: np.array [left_motor, right_motor]
            action_idx: int  (0-4)
        """
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.cfg.num_actions)
        else:
            state_t = torch.tensor(obs_5d, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            action_idx = q_values.argmax(dim=1).item()

        return discrete_to_continuous(action_idx), action_idx

    def choose_action_greedy(self, obs_5d: np.ndarray):
        """Pure greedy (no exploration) — use at inference / play time."""
        state_t = torch.tensor(obs_5d, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        action_idx = q_values.argmax(dim=1).item()
        return discrete_to_continuous(action_idx), action_idx

    # ── memory ─────────────────────────────────────────────────────────
    def store(self, state, action_idx, reward, next_state, done):
        self.replay.push(state, action_idx, reward, next_state, done)

    # ── learning step ──────────────────────────────────────────────────
    def learn(self):
        """One gradient step of standard DQN."""
        if len(self.replay) < self.cfg.min_replay_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.cfg.batch_size
        )

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Current Q-values for chosen actions
        q_current = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values (no gradient)
        with torch.no_grad():
            q_next_max = self.target_net(next_states_t).max(dim=1).values
            q_target = rewards_t + self.cfg.gamma * q_next_max * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        self._update_epsilon()

        # Sync target network
        if self.step_count % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ── checkpointing ──────────────────────────────────────────────────
    def save_models(self):
        torch.save(self.q_net.state_dict(), self.checkpoint_file)
        print(f"  -> Q-network saved to {self.checkpoint_file}")

    def load_models(self):
        state_dict = torch.load(
            self.checkpoint_file,
            map_location=self.device,
            weights_only=True,
        )
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.q_net.eval()
        self.target_net.eval()
        print(f"  -> Q-network loaded from {self.checkpoint_file}")

    # ── utility ────────────────────────────────────────────────────────
    def param_count(self) -> int:
        return sum(p.numel() for p in self.q_net.parameters())

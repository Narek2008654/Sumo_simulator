"""
Discrete-action PPO agent for Mini Sumo.

Actions are mapped to 5 discrete motor commands:
  0: [-1, -1]  (backward)
  1: [ 1,  1]  (forward)
  2: [ 0,  0]  (stop)
  3: [-1,  1]  (spin left)
  4: [ 1, -1]  (spin right)

The actor network outputs logits → Softmax → Categorical distribution.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ── action table ────────────────────────────────────────────────────────────
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


# ── config ──────────────────────────────────────────────────────────────────
@dataclass
class PPODiscreteConfig:
    # dimensions
    obs_dim: int = 11
    num_actions: int = NUM_ACTIONS  # 5

    # network
    hidden1: int = 64
    hidden2: int = 32

    # training hyperparams
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    entropy_coef: float = 0.01     # helps exploration with discrete actions
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    # PPO loop
    batch_size: int = 64
    n_epochs: int = 10

    # checkpoints — saved in a SEPARATE folder
    chkpt_dir: str = "tmp/ppo_discrete"
    actor_name: str = "actor_discrete.pt"
    critic_name: str = "critic_discrete.pt"


# ── memory ──────────────────────────────────────────────────────────────────
class PPOMemory:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.vals: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def store(self, state, action, log_prob, val, reward, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.vals.append(float(val))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def clear(self):
        for attr in (self.states, self.actions, self.log_probs,
                     self.vals, self.rewards, self.dones):
            attr.clear()

    def generate_batches(self):
        n = len(self.states)
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)
        batch_starts = np.arange(0, n, self.batch_size, dtype=np.int64)
        batches = [indices[i:i + self.batch_size] for i in batch_starts]
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.vals, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.bool_),
            batches,
        )


# ── networks ────────────────────────────────────────────────────────────────
class ActorNetwork(nn.Module):
    """Outputs action probabilities via Softmax (discrete)."""

    def __init__(self, cfg: PPODiscreteConfig):
        super().__init__()
        os.makedirs(cfg.chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(cfg.chkpt_dir, cfg.actor_name)

        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden1),
            nn.ReLU(),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Linear(cfg.hidden2, cfg.num_actions),
            nn.Softmax(dim=-1),               # ← softmax last layer
        )

    def forward(self, state: T.Tensor) -> Categorical:
        probs = self.net(state)               # [B, num_actions]
        return Categorical(probs)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, device: T.device):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=device))


class CriticNetwork(nn.Module):
    def __init__(self, cfg: PPODiscreteConfig):
        super().__init__()
        os.makedirs(cfg.chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(cfg.chkpt_dir, cfg.critic_name)

        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden1),
            nn.ReLU(),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Linear(cfg.hidden2, 1),
        )

    def forward(self, state: T.Tensor) -> T.Tensor:
        return self.net(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, device: T.device):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=device))


# ── agent ───────────────────────────────────────────────────────────────────
class PPODiscreteAgent:
    def __init__(self, cfg: PPODiscreteConfig):
        self.cfg = cfg
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(cfg).to(self.device)
        self.critic = CriticNetwork(cfg).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.memory = PPOMemory(cfg.batch_size)

    # ── action selection ────────────────────────────────────────────────
    @T.no_grad()
    def choose_action(self, observation: np.ndarray) -> Tuple[np.ndarray, int, float, float]:
        """
        Returns
        -------
        action_env : np.ndarray  shape (2,)  [left_motor, right_motor]
        action_idx : int         discrete action index (0..4)
        log_prob   : float
        value      : float
        """
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        state = T.tensor(obs, dtype=T.float32, device=self.device)

        dist = self.actor(state)
        value = self.critic(state).squeeze(-1)

        action_idx = dist.sample()                          # [1]
        log_prob = dist.log_prob(action_idx)                # [1]

        idx = int(action_idx.item())
        action_env = discrete_to_continuous(idx)

        return action_env, idx, float(log_prob.item()), float(value.item())

    def remember(self, state, action_idx, log_prob, val, reward, done):
        self.memory.store(state, action_idx, log_prob, val, reward, done)

    # ── GAE ─────────────────────────────────────────────────────────────
    def _compute_gae(self, rewards, values, dones):
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - float(dones[t])
                nextvalues = values[t]
            else:
                nextnonterminal = 1.0 - float(dones[t])
                nextvalues = values[t + 1]
            delta = rewards[t] + self.cfg.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values
        return adv, returns

    # ── learning step ───────────────────────────────────────────────────
    def learn(self):
        (state_arr, action_arr, old_lp_arr, vals_arr,
         reward_arr, dones_arr, batches) = self.memory.generate_batches()

        advantages, returns = self._compute_gae(reward_arr, vals_arr, dones_arr)

        advantages_t = T.tensor(advantages, dtype=T.float32, device=self.device)
        returns_t = T.tensor(returns, dtype=T.float32, device=self.device)
        old_lps_t = T.tensor(old_lp_arr, dtype=T.float32, device=self.device)

        for _ in range(self.cfg.n_epochs):
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32, device=self.device)
                actions = T.tensor(action_arr[batch], dtype=T.long, device=self.device)
                adv_b = advantages_t[batch]
                ret_b = returns_t[batch]
                old_lp_b = old_lps_t[batch]

                dist = self.actor(states)
                new_lp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = T.clamp(ratio,
                                1.0 - self.cfg.policy_clip,
                                1.0 + self.cfg.policy_clip) * adv_b
                actor_loss = -T.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy

                values = self.critic(states).squeeze(-1)
                critic_loss = (ret_b - values).pow(2).mean()

                total_loss = actor_loss + self.cfg.value_coef * critic_loss

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                    T.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.cfg.max_grad_norm,
                    )
                self.actor_opt.step()
                self.critic_opt.step()

        self.memory.clear()

    # ── persistence ─────────────────────────────────────────────────────
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint(self.device)
        self.critic.load_checkpoint(self.device)

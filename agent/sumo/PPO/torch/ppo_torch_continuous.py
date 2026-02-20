import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


@dataclass
class PPOConfig:
    # dimensions
    obs_dim: int = 11
    action_dim: int = 2

    # network
    hidden1: int = 16
    hidden2: int = 16

    # training hyperparams
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    entropy_coef: float = 0.0        # set 0.001..0.01 if you need more exploration
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    # PPO loop
    batch_size: int = 64
    n_epochs: int = 10

    # action squashing
    use_tanh: bool = True
    # If you use tanh, we store *raw* actions for log_prob. Env receives tanh(raw).
    # This code does NOT apply the tanh Jacobian correction to log_prob (common simplification).

    # checkpoints
    chkpt_dir: str = "tmp/ppo_continuous"
    actor_name: str = "actor.pt"
    critic_name: str = "critic.pt"


class PPOMemory:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.states: List[np.ndarray] = []
        self.actions_raw: List[np.ndarray] = []  # raw actions (pre-tanh), shape (2,)
        self.log_probs: List[float] = []
        self.vals: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def store(self, state, action_raw, log_prob, val, reward, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions_raw.append(np.asarray(action_raw, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.vals.append(float(val))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def clear(self):
        self.states.clear()
        self.actions_raw.clear()
        self.log_probs.clear()
        self.vals.clear()
        self.rewards.clear()
        self.dones.clear()

    def generate_batches(self):
        n = len(self.states)
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)
        batch_starts = np.arange(0, n, self.batch_size, dtype=np.int64)
        batches = [indices[i:i+self.batch_size] for i in batch_starts]
        return (np.array(self.states, dtype=np.float32),
                np.array(self.actions_raw, dtype=np.float32),
                np.array(self.log_probs, dtype=np.float32),
                np.array(self.vals, dtype=np.float32),
                np.array(self.rewards, dtype=np.float32),
                np.array(self.dones, dtype=np.bool_),
                batches)


class ActorNetwork(nn.Module):
    def __init__(self, cfg: PPOConfig):
        super().__init__()
        os.makedirs(cfg.chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(cfg.chkpt_dir, cfg.actor_name)

        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden1),
            nn.ReLU(),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Linear(cfg.hidden2, cfg.action_dim),
        )
        # learnable log_std (one per action dim)
        self.log_std = nn.Parameter(T.zeros(cfg.action_dim))

    def forward(self, state: T.Tensor) -> Normal:
        mu = self.net(state)                             # [B,2]
        std = T.exp(self.log_std).expand_as(mu)          # [B,2]
        return Normal(mu, std)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, device: T.device):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=device))


class CriticNetwork(nn.Module):
    def __init__(self, cfg: PPOConfig):
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


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(cfg).to(self.device)
        self.critic = CriticNetwork(cfg).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.memory = PPOMemory(cfg.batch_size)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint(self.device)
        self.critic.load_checkpoint(self.device)

    @T.no_grad()
    def choose_action(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Returns:
          action_env: shape (2,) in [-1,1] if tanh enabled else raw
          action_raw: shape (2,) raw (pre-tanh)
          log_prob: scalar log pi(a_raw|s) summed over dims
          value: scalar V(s)
        """
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        state = T.tensor(obs, dtype=T.float32, device=self.device)

        dist = self.actor(state)
        value = self.critic(state).squeeze(-1)  # [1]

        action_raw = dist.sample()             # [1,2]
        log_prob = dist.log_prob(action_raw).sum(dim=-1)  # [1]

        if self.cfg.use_tanh:
            action_env = T.tanh(action_raw)
        else:
            action_env = action_raw

        return (action_env.squeeze(0).cpu().numpy(),
                action_raw.squeeze(0).cpu().numpy(),
                float(log_prob.item()),
                float(value.item()))

    def remember(self, state, action_raw, log_prob, val, reward, done):
        self.memory.store(state, action_raw, log_prob, val, reward, done)

    def _compute_gae(self, rewards, values, dones):
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - float(dones[t])
                nextvalues = values[t]
            else:
                nextnonterminal = 1.0 - float(dones[t])
                nextvalues = values[t+1]
            delta = rewards[t] + self.cfg.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values
        return adv, returns

    def learn(self):
        state_arr, action_raw_arr, old_logprob_arr, vals_arr, reward_arr, dones_arr, batches = \
            self.memory.generate_batches()

        advantages, returns = self._compute_gae(reward_arr, vals_arr, dones_arr)

        advantages = T.tensor(advantages, dtype=T.float32, device=self.device)
        returns = T.tensor(returns, dtype=T.float32, device=self.device)
        old_logprobs = T.tensor(old_logprob_arr, dtype=T.float32, device=self.device)

        for _ in range(self.cfg.n_epochs):
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32, device=self.device)
                actions_raw = T.tensor(action_raw_arr[batch], dtype=T.float32, device=self.device)
                adv_b = advantages[batch]
                ret_b = returns[batch]
                old_lp_b = old_logprobs[batch]

                dist = self.actor(states)
                new_lp = dist.log_prob(actions_raw).sum(dim=-1)  # [B]
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = T.clamp(ratio, 1.0 - self.cfg.policy_clip, 1.0 + self.cfg.policy_clip) * adv_b
                actor_loss = -T.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy

                values = self.critic(states).squeeze(-1)
                critic_loss = (ret_b - values).pow(2).mean()

                total_loss = actor_loss + self.cfg.value_coef * critic_loss

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                    T.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                               self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

        self.memory.clear()


def action_to_pwm(action_env: np.ndarray) -> np.ndarray:
    """
    action_env expected in [-1,1]. Returns PWM [0..255] for each motor.
    """
    a = np.asarray(action_env, dtype=np.float32)
    pwm = np.rint((a + 1.0) * 127.5).astype(np.int32)
    return np.clip(pwm, 0, 255)

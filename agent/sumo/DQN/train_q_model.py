"""
Train a compact DQN (Q-Model) agent for Mini Sumo.

The agent sees only 5 inputs:
  [0:3] - 3 front opponent sensors (front, front-right, front-left)
  [3:5] - last action motors (left, right)

Reward shaping matches PPO discrete / DQN-style:
  - Standing still       → -0.2
  - Backing away (close) → -0.5
  - Attacking  (close)   → +0.3
  - Win terminal         → +100
  - Lose terminal        → -100
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

import argparse
import numpy as np
from environment.sumo.sumo_env import SumoEnv, SumoEnvConfig
from q_model import DQNAgent, QModelConfig, ACTION_MAP


# ── helpers ────────────────────────────────────────────────────────────────

def extract_5d_obs(full_obs: np.ndarray, last_action: np.ndarray) -> np.ndarray:
    """
    Extract 5-dim state from the environment's 11-dim observation.

    full_obs layout:
      [0:5]  5 opponent sensors
      [5:7]  velocity
      [7]    angular velocity
      [8:10] relative position to center
      [10]   relative angle to center

    We keep only:
      [0:3]  3 front sensors (front, front-right, front-left)
      + last_action [left_motor, right_motor]
    """
    return np.concatenate([full_obs[0:3], last_action]).astype(np.float32)


def shape_reward(
    env_reward: float,
    action_idx: int,
    obs_5d: np.ndarray,
    terminated: bool,
    info: dict,
) -> float:
    """
    Refined DQN-style reward shaping (user-requested updates).
    """
    reward = 0.0

    # ── terminal rewards ────────────────────────────────────────────
    if terminated:
        if info.get("robot2_out", False) and not info.get("robot1_out", False):
            reward += 100.0   # win
        elif info.get("robot1_out", False) and not info.get("robot2_out", False):
            reward -= 100.0   # lose
        elif info.get("robot1_out", False) and info.get("robot2_out", False):
            reward -= 10.0    # both out
        return reward

    # ── per-step shaping ────────────────────────────────────────────
    action = ACTION_MAP[action_idx]
    front_min = min(obs_5d[0], obs_5d[1], obs_5d[2])   # closest front sensor

    # 1. Punish standing still MORE
    if action[0] == 0 and action[1] == 0:
        reward -= 1.0  # Increased from -0.2

    # 2. Punish backing away when enemy is close MORE
    if (action[0] < 0 or action[1] < 0) and front_min < 0.5:
        reward -= 2.0  # Increased from -0.5

    # 3. Reward "mushing" (attacking forward during collision)
    if info.get("collision_occurred", False) and (action[0] > 0 or action[1] > 0):
        reward += 1.0  # New mushing bonus

    # 4. MUCH higher reward for pushing from behind (outmaneuver)
    if info.get("behind_push_detected", False):
        reward += 15.0  # Massive bonus for superior positioning

    # 5. Base env rewards (push toward edge, etc.) -> scaled up
    reward += env_reward * 0.1

    return reward


# ── main training loop ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train compact Q-Model for sumo")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--save_every", type=int, default=5000, help="save every N steps")
    ap.add_argument("--render", action="store_true", help="render in GUI (slow)")
    ap.add_argument("--random-start", action="store_true", default=True)
    args = ap.parse_args()

    cfg = QModelConfig(
        obs_dim=5,
        num_actions=5,
        hidden1=16,
        hidden2=16,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100_000,
        target_update_freq=1000,
        replay_capacity=10_000,
        batch_size=64,
        min_replay_size=500,
    )

    env_cfg = SumoEnvConfig(
        render_mode="human" if args.render else None,
        random_start=args.random_start,
        max_episode_steps=1000,
    )

    agent = DQNAgent(cfg)
    env = SumoEnv(config=env_cfg)

    print(f"Q-Model parameters: {agent.param_count()}")
    print(f"Training for {args.steps} steps ...")

    obs_full, _ = env.reset()
    last_action = np.array([0.0, 0.0], dtype=np.float32)  # start still
    obs_5d = extract_5d_obs(obs_full, last_action)

    total_steps = 0
    ep_rewards = []
    ep_reward = 0.0
    ep_count = 0

    while total_steps < args.steps:
        # Choose action
        action_env, action_idx = agent.choose_action(obs_5d)

        # Step environment
        next_obs_full, env_reward, terminated, truncated, info = env.step(action_env)
        done = terminated or truncated

        # Build next 5D observation
        next_obs_5d = extract_5d_obs(next_obs_full, action_env)

        # Shape reward
        reward = shape_reward(
            env_reward, action_idx, obs_5d,
            terminated,
            info,
        )

        # Store transition
        agent.store(obs_5d, action_idx, reward, next_obs_5d, float(done))

        # Learn
        agent.learn()

        ep_reward += reward
        obs_5d = next_obs_5d
        last_action = action_env.copy()
        total_steps += 1

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            ep_count += 1

            obs_full, _ = env.reset()
            last_action = np.array([0.0, 0.0], dtype=np.float32)
            obs_5d = extract_5d_obs(obs_full, last_action)

        # Logging
        if total_steps % 2000 == 0:
            mean_r = np.mean(ep_rewards[-10:]) if ep_rewards else float("nan")
            print(
                f"step={total_steps:7d}  eps={agent.epsilon:.3f}  "
                f"episodes={ep_count:4d}  mean_reward(last10)={mean_r:.2f}"
            )

        # Save checkpoint
        if total_steps % args.save_every == 0:
            agent.save_models()

    # Final save
    agent.save_models()
    env.close()

    mean_r = np.mean(ep_rewards[-10:]) if ep_rewards else float("nan")
    print(
        f"\nTraining complete.  {total_steps} steps,  {ep_count} episodes.\n"
        f"Final mean reward (last 10 eps): {mean_r:.2f}\n"
        f"Model saved to {cfg.chkpt_dir}/{cfg.model_name}"
    )


if __name__ == "__main__":
    main()

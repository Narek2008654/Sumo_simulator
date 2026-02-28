"""
Train a discrete-action PPO agent for Mini Sumo.

Reward shaping follows the DQN-style logic:
  - Standing still  [0, 0]   → penalty  -0.2
  - Backing away when enemy close → penalty  -0.5
  - Attacking forward when enemy close → bonus  +0.3
  - Win terminal   → +100
  - Lose terminal  → -100
  - Push opponent toward edge → scaled reward
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))

import argparse
import numpy as np
from environment.sumo.sumo_env import SumoEnv, SumoEnvConfig
from ppo_torch_discrete import PPODiscreteAgent, PPODiscreteConfig, ACTION_MAP


# ── DQN-style reward shaping ───────────────────────────────────────────────
def shape_reward(
    env_reward: float,
    action_idx: int,
    obs: np.ndarray,
    terminated: bool,
    robot1_out: bool,
    robot2_out: bool,
) -> float:
    """
    Apply DQN-style reward shaping on top of the base environment reward.

    Observation layout (11 dims):
      [0:5]  - 5 opponent sensors (0=close, 1=far)
      [5:7]  - velocity (vx, vy)
      [7]    - angular velocity
      [8:10] - relative position to center
      [10]   - relative angle to center

    DQN reward rules (adapted to normalised sensors):
      - front_min = min(obs[0], obs[1], obs[2])
      - "enemy close" ↔ front_min < 0.5  (proxy for raw < 30)
    """
    reward = 0.0

    # ── terminal rewards ────────────────────────────────────────────────
    if terminated:
        if robot2_out and not robot1_out:
            reward += 100.0   # win
        elif robot1_out and not robot2_out:
            reward -= 100.0   # lose
        elif robot1_out and robot2_out:
            reward -= 10.0    # both out
        return reward         # skip shaping on terminal step

    # ── per-step reward shaping ─────────────────────────────────────────
    action = ACTION_MAP[action_idx]
    front_min = min(obs[0], obs[1], obs[2])   # closest front sensor

    # Punish standing still
    if action[0] == 0 and action[1] == 0:
        reward -= 0.2

    # Punish backing away when enemy is close
    if (action[0] == -1 or action[1] == -1) and front_min < 0.5:
        reward -= 0.5

    # Reward attacking forward when enemy is close
    if (action[0] == 1 or action[1] == 1) and front_min < 0.5:
        reward += 0.3

    # Push opponent toward edge (from env reward, scaled down)
    reward += env_reward * 0.01   # small env contribution for push signals

    return reward


# ── main training loop ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--rollout", type=int, default=2048, help="steps per PPO update")
    ap.add_argument("--save_every", type=int, default=20, help="save every N updates")
    ap.add_argument("--render", action="store_true", help="render in GUI (slow)")
    ap.add_argument("--random-start", action="store_true", default=True)
    args = ap.parse_args()

    cfg = PPODiscreteConfig(
        obs_dim=11,
        num_actions=5,
        hidden1=64,
        hidden2=32,
        batch_size=64,
        n_epochs=10,
        lr=3e-4,
        entropy_coef=0.01,
    )

    env_cfg = SumoEnvConfig(
        render_mode="human" if args.render else None,
        random_start=args.random_start,
        max_episode_steps=1000,
    )

    agent = PPODiscreteAgent(cfg)
    env = SumoEnv(config=env_cfg)

    obs, _ = env.reset()
    total_steps = 0
    update_i = 0
    ep_rewards = []
    ep_reward = 0.0

    while total_steps < args.steps:
        # ── collect one rollout ─────────────────────────────────────────
        for _ in range(args.rollout):
            action_env, action_idx, log_prob, val = agent.choose_action(obs)

            next_obs, env_reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            # Apply DQN-style reward shaping
            reward = shape_reward(
                env_reward, action_idx, obs,
                terminated,
                info.get("robot1_out", False),
                info.get("robot2_out", False),
            )

            agent.remember(obs, action_idx, log_prob, val, reward, done)

            ep_reward += reward
            obs = next_obs
            total_steps += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

            if total_steps >= args.steps:
                break

        # ── update ──────────────────────────────────────────────────────
        agent.learn()
        update_i += 1

        mean_r = np.mean(ep_rewards[-10:]) if ep_rewards else float("nan")
        print(
            f"update={update_i:4d}  steps={total_steps:7d}  "
            f"episodes={len(ep_rewards):4d}  mean_reward(last10)={mean_r:.2f}"
        )

        if update_i % args.save_every == 0:
            agent.save_models()
            print("  -> checkpoints saved")

    agent.save_models()
    env.close()
    print(
        f"\nTraining complete. Final mean reward (last 10 eps): "
        f"{np.mean(ep_rewards[-10:]) if ep_rewards else 'N/A':.2f}"
    )


if __name__ == "__main__":
    main()

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../..'))

import argparse
import numpy as np
from environment.sumo.sumo_env import SumoEnv, SumoEnvConfig
from ppo_torch_continuous import PPOAgent, PPOConfig, action_to_pwm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--rollout", type=int, default=2048, help="steps per PPO update")
    ap.add_argument("--save_every", type=int, default=20, help="save every N updates")
    ap.add_argument("--render", action="store_true", help="render in GUI (slow)")
    ap.add_argument("--random-start", action="store_true", default=True)
    args = ap.parse_args()

    cfg = PPOConfig(
        obs_dim=11,
        action_dim=2,
        hidden1=16,
        hidden2=16,
        batch_size=64,
        n_epochs=10,
        lr=3e-4,
        entropy_coef=0.005,   # small entropy bonus helps exploration in sumo
        use_tanh=True,
    )

    env_cfg = SumoEnvConfig(
        render_mode="human" if args.render else None,
        random_start=args.random_start,
        max_episode_steps=1000,   # ~20 s per match at 50 Hz
    )

    agent = PPOAgent(cfg)
    env = SumoEnv(config=env_cfg)

    # reset() returns (obs, info)
    obs, _ = env.reset()
    total_steps = 0
    update_i = 0
    ep_rewards = []
    ep_reward = 0.0

    while total_steps < args.steps:
        # ── collect one rollout ──────────────────────────────────────────────
        for _ in range(args.rollout):
            action_env, action_raw, log_prob, val = agent.choose_action(obs)

            # step() returns (obs, reward, terminated, truncated, info)
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            agent.remember(obs, action_raw, log_prob, val, reward, done)

            ep_reward += reward
            obs = next_obs
            total_steps += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

            if total_steps >= args.steps:
                break

        # ── update ───────────────────────────────────────────────────────────
        agent.learn()
        update_i += 1

        # rolling mean over last 10 episodes
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
    print(f"\nTraining complete. Final mean reward (last 10 eps): "
          f"{np.mean(ep_rewards[-10:]) if ep_rewards else 'N/A':.2f}")


if __name__ == "__main__":
    main()
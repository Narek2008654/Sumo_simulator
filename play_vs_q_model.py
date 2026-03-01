"""
Play against your trained Q-Model (DQN) bot!

Controls:
    W - Forward       Q - Spin Left
    S - Backward      E - Spin Right
    A - Turn Left     R - Reset match
    D - Turn Right    ESC - Quit
"""

import sys
import os
import numpy as np
import pygame

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.sumo.sumo_env import SumoEnv, SumoEnvConfig
from agent.sumo.sumo_agent import ManualAgent, QModelGameAgent


def main():
    print("\n" + "="*60)
    print("MINI SUMO SIMULATOR - PLAY VS Q-MODEL (DQN)")
    print("="*60)
    print("\nLoading trained Q-Model...")

    # Create environment
    config = SumoEnvConfig(
        render_mode="human",
        random_start=True,
        fps=50,
        max_episode_steps=2000,  # 40 seconds
    )

    env = SumoEnv(config=config)
    env.robot1_name = "YOU (Blue)"
    env.robot2_name = "Q-Bot (Orange)"

    # Create player (you)
    player = ManualAgent()

    # Load trained Q-Model
    try:
        opponent = QModelGameAgent(model_dir="tmp/q_model")
    except FileNotFoundError:
        print("\n❌ ERROR: Trained model not found!")
        print("   Did you train the model first?")
        print("   Run: python agent/sumo/DQN/train_q_model.py")
        return
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        return

    print("\n" + "="*60)
    print("CONTROLS:")
    print("  W - Forward       Q - Spin Left")
    print("  S - Backward      E - Spin Right")
    print("  A - Turn Left     R - Reset match")
    print("  D - Turn Right    C - Clear scores")
    print("                    ESC - Quit")
    print("\nYou are the BLUE robot. Push the Q-Model Bot out!")
    print("="*60 + "\n")

    running = True
    obs, info = env.reset()
    total_wins = 0
    total_losses = 0

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    opponent.reset()
                    print("🔄 Match reset!")
                elif event.key == pygame.K_c:
                    total_wins = 0
                    total_losses = 0
                    print("📊 Scores cleared!")

        # Get keyboard input
        keys = pygame.key.get_pressed()
        player.handle_keys(keys)

        # Get observations for both robots
        obs2 = env._get_observation(
            env.robot2_state, env.robot1_state, env.robot2_sensors,
            env.robot2_physics, env.robot1_physics
        )

        player_action = player.act(obs)
        opponent_action = opponent.act(obs2)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(player_action, opponent_action)

        # Check for match end
        if terminated:
            if info["robot2_out"] and not info["robot1_out"]:
                total_wins += 1
                print(f"🎉 YOU WIN! ({total_wins} - {total_losses})")
            elif info["robot1_out"] and not info["robot2_out"]:
                total_losses += 1
                print(f"😢 You lost! ({total_wins} - {total_losses})")
            else:
                print(f"🤝 Draw! ({total_wins} - {total_losses})")

            pygame.time.wait(1500)
            obs, info = env.reset()
            opponent.reset()

        elif truncated:
            print(f"⏰ Time's up! Draw! ({total_wins} - {total_losses})")
            pygame.time.wait(1500)
            obs, info = env.reset()
            opponent.reset()

    env.close()
    print(f"\n{'='*60}")
    print(f"FINAL SCORE: You {total_wins} - {total_losses} Q-Model Bot")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

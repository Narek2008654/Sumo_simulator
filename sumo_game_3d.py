"""
3D Mini Sumo Robot Simulation Game (PyBullet)

Run this script to play or test the 3D mini sumo simulation.

Usage:
    python sumo_game_3d.py                 # Play as blue robot vs opponent
    python sumo_game_3d.py --mode demo     # Watch agent vs agent
    python sumo_game_3d.py --help          # Show help

Controls (Player mode — PyBullet GUI window must be focused):
    Arrow UP    - Forward
    Arrow DOWN  - Backward
    Arrow LEFT  - Turn Left
    Arrow RIGHT - Turn Right
    Z - Spin Left
    X - Spin Right
    R - Reset match
    ESC - Quit
"""

import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.sumo3d.sumo_env_3d import SumoEnv3D, SumoEnv3DConfig
from agent.sumo.sumo_agent import AggressiveAgent, DefensiveAgent, RandomAgent


# PyBullet key codes — arrow keys avoid conflicting with PyBullet shortcuts (W/S/A/G)
KEY_UP = 65297
KEY_DOWN = 65298
KEY_LEFT = 65295
KEY_RIGHT = 65296
KEY_Z = ord('z')
KEY_X = ord('x')
KEY_R = ord('r')
KEY_ESC = 65307  # X11 ESC; Windows uses 27
KEY_ESC_WIN = 27


def parse_args():
    parser = argparse.ArgumentParser(description="3D Mini Sumo Robot Simulation (PyBullet)")
    parser.add_argument(
        "--mode",
        type=str,
        default="play",
        choices=["play", "demo"],
        help="Game mode: play (human vs opponent), demo (agent vs agent)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="aggressive",
        choices=["aggressive", "defensive", "random"],
        help="Opponent agent type",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for demo mode",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Use random starting positions",
    )
    return parser.parse_args()


def create_opponent(opponent_type: str):
    if opponent_type == "aggressive":
        return AggressiveAgent(aggression=0.9)
    elif opponent_type == "defensive":
        return DefensiveAgent()
    elif opponent_type == "random":
        return RandomAgent()
    return AggressiveAgent()


def get_keyboard_action(env: SumoEnv3D) -> tuple:
    """
    Read PyBullet keyboard events and return (action, should_reset, should_quit).
    """
    keys = env.world.get_keyboard_events()

    left = 0.0
    right = 0.0
    should_reset = False
    should_quit = False

    # PyBullet key states: 1=pressed down this frame, 3=held, 4=released
    active = {k for k, v in keys.items() if v & 3}  # pressed or held

    if KEY_UP in active:
        left += 1.0
        right += 1.0
    if KEY_DOWN in active:
        left -= 1.0
        right -= 1.0
    if KEY_LEFT in active:
        left -= 0.5
        right += 0.5
    if KEY_RIGHT in active:
        left += 0.5
        right -= 0.5
    if KEY_Z in active:
        left = -1.0
        right = 1.0
    if KEY_X in active:
        left = 1.0
        right = -1.0

    # Single-fire events (key down only)
    pressed = {k for k, v in keys.items() if v == 1}
    if KEY_R in pressed:
        should_reset = True
    if KEY_ESC in pressed or KEY_ESC_WIN in pressed:
        should_quit = True

    action = np.array([np.clip(left, -1, 1), np.clip(right, -1, 1)], dtype=np.float32)
    return action, should_reset, should_quit


def check_quit(env: SumoEnv3D) -> bool:
    """Check if ESC was pressed (for demo mode)."""
    keys = env.world.get_keyboard_events()
    pressed = {k for k, v in keys.items() if v == 1}
    return KEY_ESC in pressed or KEY_ESC_WIN in pressed


def run_play_mode(args):
    """Human player vs opponent agent in 3D."""
    print("\n" + "=" * 50)
    print("3D MINI SUMO SIMULATOR - PLAY MODE")
    print("=" * 50)
    print("\nControls (focus the PyBullet window):")
    print("  UP    - Forward       DOWN  - Backward")
    print("  LEFT  - Turn Left     RIGHT - Turn Right")
    print("  Z     - Spin Left     X     - Spin Right")
    print("  R     - Reset match   ESC   - Quit")
    print("\nYou are the BLUE robot. Push the opponent (orange) off the dohyo!")
    print("=" * 50 + "\n")

    config = SumoEnv3DConfig(
        render_mode="human",
        random_start=args.random_start,
        max_episode_steps=2000,  # 40 seconds
    )

    env = SumoEnv3D(config=config)
    env.robot1_name = "P1"
    env.robot2_name = "Opponent"

    opponent = create_opponent(args.opponent)

    obs, info = env.reset()
    running = True

    while running:
        player_action, should_reset, should_quit = get_keyboard_action(env)

        if should_quit:
            running = False
            break
        if should_reset:
            obs, info = env.reset()
            print("Match reset!")
            continue

        opponent_action = opponent.act(obs)
        obs, reward, terminated, truncated, info = env.step(player_action, opponent_action)

        if terminated:
            p_score = info["robot1_score"]
            o_score = info["robot2_score"]
            if info["robot2_out"] and not info["robot1_out"]:
                print(f"YOU WIN! Score: {p_score} - {o_score}")
            elif info["robot1_out"] and not info["robot2_out"]:
                print(f"You lose! Score: {p_score} - {o_score}")
            else:
                print(f"Draw! Score: {p_score} - {o_score}")
            time.sleep(1.0)
            obs, info = env.reset()

        if truncated:
            p_score = info["robot1_score"]
            o_score = info["robot2_score"]
            print(f"Time's up! Draw! Score: {p_score} - {o_score}")
            time.sleep(1.0)
            obs, info = env.reset()

    env.close()
    print(f"\nFinal Score: P1 {env.robot1_score} - {env.robot2_score} Opponent")


def run_demo_mode(args):
    """Watch agent vs agent matches in 3D."""
    print("\n" + "=" * 50)
    print("3D MINI SUMO SIMULATOR - DEMO MODE")
    print("=" * 50)
    print(f"Watching {args.episodes} episodes of agent vs agent")
    print("Press ESC in PyBullet window to quit")
    print("=" * 50 + "\n")

    config = SumoEnv3DConfig(
        render_mode="human",
        random_start=True,
        max_episode_steps=1000,
    )

    env = SumoEnv3D(config=config)
    env.robot1_name = "Agent1"
    env.robot2_name = "Agent2"

    agent1 = AggressiveAgent(aggression=0.95)
    if args.opponent == "defensive":
        agent2 = DefensiveAgent()
    else:
        agent2 = AggressiveAgent(aggression=0.8)

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        print(f"Episode {episode + 1}/{args.episodes}")

        while not done:
            if check_quit(env):
                env.close()
                return

            action1 = agent1.act(obs)
            action2 = agent2.act(obs)

            obs, reward, terminated, truncated, info = env.step(action1, action2)
            step += 1
            done = terminated or truncated

            if terminated:
                s1, s2 = info["robot1_score"], info["robot2_score"]
                if info["robot2_out"] and not info["robot1_out"]:
                    print(f"  -> Agent1 wins in {step} steps! [{s1} - {s2}]")
                elif info["robot1_out"] and not info["robot2_out"]:
                    print(f"  -> Agent2 wins in {step} steps! [{s1} - {s2}]")
                else:
                    print(f"  -> Draw (both out)! [{s1} - {s2}]")
            elif truncated:
                print(f"  -> Draw (timeout)! [{info['robot1_score']} - {info['robot2_score']}]")

        time.sleep(0.5)

    env.close()
    print(f"\nFinal Score: Agent1 {env.robot1_score} - {env.robot2_score} Agent2")


def main():
    args = parse_args()
    if args.mode == "play":
        run_play_mode(args)
    elif args.mode == "demo":
        run_demo_mode(args)


if __name__ == "__main__":
    main()

"""
Mini Sumo Robot Simulation Game

Run this script to play or test the mini sumo simulation.

Usage:
    python sumo_game.py                 # Play as blue robot vs opponent
    python sumo_game.py --mode demo     # Watch agent vs agent
    python sumo_game.py --help          # Show help

Controls (Player mode):
    W - Forward
    S - Backward  
    A - Turn Left
    D - Turn Right
    Q - Spin Left
    E - Spin Right
    R - Reset match
    ESC - Quit
"""

import argparse
import sys
import numpy as np

# Add project root to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from environment.sumo.sumo_env import SumoEnv, SumoEnvConfig
from agent.sumo.sumo_agent import ManualAgent, AggressiveAgent, DefensiveAgent, RandomAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Mini Sumo Robot Simulation")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="play",
        choices=["play", "demo"],
        help="Game mode: play (human vs opponent), demo (agent vs agent)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="aggressive",
        choices=["aggressive", "defensive", "random"],
        help="Opponent agent type"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for demo mode"
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Use random starting positions"
    )
    return parser.parse_args()


def create_opponent(opponent_type: str):
    """Create opponent agent based on type."""
    if opponent_type == "aggressive":
        return AggressiveAgent(aggression=0.9)
    elif opponent_type == "defensive":
        return DefensiveAgent()
    elif opponent_type == "random":
        return RandomAgent()
    else:
        return AggressiveAgent()


def run_play_mode(args):
    """Human player vs opponent agent."""
    print("\n" + "="*50)
    print("MINI SUMO SIMULATOR - PLAY MODE")
    print("="*50)
    print("\nControls:")
    print("  W - Forward       S - Backward")
    print("  A - Turn Left     D - Turn Right")
    print("  Q - Spin Left     E - Spin Right")
    print("  R - Reset match   C - Clear scores   ESC - Quit")
    print("\nYou are P1 (blue robot). Push the opponent (orange robot) out!")
    print("="*50 + "\n")
    
    # Create environment
    config = SumoEnvConfig(
        render_mode="human",
        random_start=args.random_start,
        fps=50,
        max_episode_steps=2000,  # 40 seconds
    )
    
    env = SumoEnv(config=config)
    env.robot1_name = "P1"   # Player 1 (human)
    env.robot2_name = "Opponent"   # Opponent agent
    
    player = ManualAgent()
    opponent = create_opponent(args.opponent)
    
    running = True
    obs, info = env.reset()
    
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
                    print("Match reset!")
                elif event.key == pygame.K_c:
                    # Clear scores
                    env.reset_scores()
                    print("Scores cleared!")
        
        # Get keyboard state and update player
        keys = pygame.key.get_pressed()
        player.handle_keys(keys)
        
        # Get actions
        player_action = player.act(obs)
        opponent_action = opponent.act(obs)  # Note: opponent sees same obs structure
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(player_action, opponent_action)
        
        # Check for match end
        if terminated:
            p_score = info["robot1_score"]
            o_score = info["robot2_score"]
            if info["robot2_out"] and not info["robot1_out"]:
                print(f"🎉 YOU WIN! Score: {p_score} - {o_score}")
            elif info["robot1_out"] and not info["robot2_out"]:
                print(f"💀 You lose! Score: {p_score} - {o_score}")
            else:
                print(f"🤝 Draw! Score: {p_score} - {o_score}")
            
            # Auto-reset after short delay
            pygame.time.wait(1000)
            obs, info = env.reset()
        
        if truncated:
            p_score = info["robot1_score"]
            o_score = info["robot2_score"]
            print(f"⏰ Time's up! Draw! Score: {p_score} - {o_score}")
            pygame.time.wait(1000)
            obs, info = env.reset()
    
    env.close()
    print(f"\nFinal Score: P1 {env.robot1_score} - {env.robot2_score} Opponent")


def run_demo_mode(args):
    """Watch agent vs agent matches."""
    print("\n" + "="*50)
    print("MINI SUMO SIMULATOR - DEMO MODE")
    print("="*50)
    print(f"Watching {args.episodes} episodes of agent vs agent")
    print("Press ESC to quit")
    print("="*50 + "\n")
    
    config = SumoEnvConfig(
        render_mode="human",
        random_start=True,
        fps=50,
        max_episode_steps=1000,
    )
    
    env = SumoEnv(config=config)
    env.robot1_name = "Agent1"
    env.robot2_name = "Agent2"
    
    agent1 = AggressiveAgent(aggression=0.95)
    agent2 = DefensiveAgent() if args.opponent == "defensive" else AggressiveAgent(aggression=0.8)
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        
        print(f"Episode {episode + 1}/{args.episodes}")
        
        while not done:
            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return
            
            # Get actions
            action1 = agent1.act(obs)
            action2 = agent2.act(obs)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action1, action2)
            step += 1
            done = terminated or truncated
            
            if terminated:
                score1, score2 = info["robot1_score"], info["robot2_score"]
                if info["robot2_out"] and not info["robot1_out"]:
                    print(f"  -> Agent1 wins in {step} steps! [{score1} - {score2}]")
                elif info["robot1_out"] and not info["robot2_out"]:
                    print(f"  -> Agent2 wins in {step} steps! [{score1} - {score2}]")
                else:
                    print(f"  -> Draw (both out)! [{score1} - {score2}]")
            elif truncated:
                print(f"  -> Draw (timeout)! [{info['robot1_score']} - {info['robot2_score']}]")
        
        pygame.time.wait(500)
    
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

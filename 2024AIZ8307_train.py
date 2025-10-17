#!/usr/bin/env python3
"""
Training Script for DQN Agent on PettingZoo Pong

This script trains a DQN agent using the Nature DQN architecture with:
- Experience replay
- Target network
- Frame stacking
- Epsilon-greedy exploration
- Double DQN
"""

import torch
import numpy as np
import random
import sys
import os

import matplotlib
matplotlib.use("Agg")  # <--- Add this line
import matplotlib.pyplot as plt
# import pandas as pd
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo_pong_wrapper import PettingZooPongWrapper
from gymnasium import spaces

# Import the student agent (rename according to your entry number)
# For example: from 2023CS12345_agent import StudentAgent
# Here we'll use a generic import that you should modify
try:
    from importlib import import_module
    # Get the entry number from directory or use default
    entry_number = "2024AIZ8307"  # CHANGE THIS to your entry number
    agent_module = import_module(f"{entry_number}_agent")
    StudentAgent = agent_module.StudentAgent
except ImportError:
    print("Error: Could not import StudentAgent. Make sure your agent file is named correctly.")
    print(f"Expected: {entry_number}_agent.py")
    sys.exit(1)


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def simple_opponent_action(obs):
    """
    Simple rule-based opponent for training.
    This provides a basic opponent for the agent to train against.
    """
    # Random action with bias toward staying in place and moving
    return random.choices([0, 1, 2, 3, 4, 5], weights=[0.3, 0.1, 0.2, 0.2, 0.1, 0.1])[0]


def train_dqn_agent(num_episodes=2000, save_path=None, seed=42):
    """
    Train a DQN agent on PettingZoo Pong.
    
    Args:
        num_episodes: Number of episodes to train for
        save_path: Path to save the trained model
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("Training DQN Agent on PettingZoo Pong")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    if save_path:
        print(f"Save path: {save_path}")
    print()
    
    # Set random seeds
    set_random_seeds(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Create environment
    env = PettingZooPongWrapper()
    action_space = spaces.Discrete(6)
    
    # Create agent
    print("Initializing DQN agent...")
    agent = StudentAgent(agent_id=0, action_space=action_space)
    agent.reset_epsilon()
    print(f"Agent initialized with epsilon: {agent.get_epsilon():.3f}")
    print()
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    epsilon_history = []
    win_history = []
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    print(f"{'Episode':<8} {'Reward':<10} {'Length':<8} {'Epsilon':<10} {'Wins':<8}")
    print("-" * 60)
    
    total_steps = 0
    
    # ---------------------------------------------------------
    # Hybrid Training Loop: Simple Opponent → Self-Play
    # ---------------------------------------------------------
    frozen_opponent = None
    switch_to_selfplay = 1000     # switch after 1/3rd training
    update_frozen_every = 200                   # update frozen copy every N episodes

    for episode in range(num_episodes):

        # Switch to self-play after threshold
        if episode == switch_to_selfplay:
            print("\n=== Switching to SELF-PLAY mode ===")
            frozen_opponent = StudentAgent(agent_id=1, action_space=action_space)
            frozen_opponent.policy_network.load_state_dict(agent.policy_network.state_dict())  # clone weights
            frozen_opponent.policy_network.eval()
        
        # Update frozen opponent periodically (for stability)
        if frozen_opponent and (episode + 1) % update_frozen_every == 0:
            print(f"\n[Update] Refreshing frozen opponent at episode {episode + 1}")
            frozen_opponent.policy_network.load_state_dict(agent.policy_network.state_dict())
            frozen_opponent.policy_network.eval()
        # Reset environment
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Reset agent's frame processor at start of episode
        agent.frame_processor.reset()
        
        # Episode loop
        while not done and episode_length < 10000:
            action0 = agent.act(obs[0])

            # Choose opponent policy
            if frozen_opponent is None:
                action1 = simple_opponent_action(obs[1])   # phase 1: rule-based
            else:
                with torch.no_grad():
                    action1 = frozen_opponent.act(obs[1])  # phase 2: self-play


            actions = [action0, action1]
            
            # Execute actions
            next_obs, rewards, done, _, _ = env.step(actions)
            
            # Update agent
            agent.update(obs[0], action0, rewards[0], next_obs[0], done)
            
            # Track statistics
            episode_reward += rewards[0]
            episode_length += 1
            total_steps += 1
            
            # Move to next state
            obs = next_obs
        
        # Decay epsilon after episode
        agent.update_epsilon()
        
        # Track episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        epsilon_history.append(agent.get_epsilon())
        
        # Check if agent won (positive reward)
        win = episode_reward > 0
        win_history.append(win)
        
        # Print progress
        if (episode + 1) % 100 == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            win_rate = np.mean(win_history[-100:]) if len(win_history) >= 100 else np.mean(win_history)
            
            print(f"{episode+1:<8} {episode_reward:<10.1f} {episode_length:<8} "
                  f"{agent.get_epsilon():<10.3f} {win_rate*100:<8.1f}%",flush=True)

        # ---------- SAVE MODEL PERIODICALLY ----------
        if (episode + 1) % 400 == 0:
            save_file = save_path if save_path else f"{entry_number}_model.pt"
            # Optionally, include episode number in filename to avoid overwriting
            save_file = save_file.replace(".pt", f"_ep{episode+1}.pt")
            agent.save_model(save_file)
            print(f"[Checkpoint] Model saved at episode {episode+1} to {save_file}")
        
        # Periodic detailed statistics
        if (episode + 1) % 100 == 0:
            print("-" * 60)
            print(f"Progress Update - Episode {episode + 1}/{num_episodes}")
            print(f"  Average Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"  Average Length (last 100): {np.mean(episode_lengths[-100:]):.1f}")
            print(f"  Win Rate (last 100): {np.mean(win_history[-100:])*100:.1f}%")
            print(f"  Current Epsilon: {agent.get_epsilon():.3f}")
            print(f"  Total Steps: {total_steps}")
            print("-" * 60)
    
    # Training completed
    print()
    print("=" * 60)
    print("Training Completed!")
    print("=" * 60)
    
    # Final statistics
    print("\nFinal Statistics:")
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Average Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Average Length (last 100): {np.mean(episode_lengths[-100:]):.1f}")
    print(f"  Win Rate (last 100): {np.mean(win_history[-100:])*100:.1f}%")
    print(f"  Final Epsilon: {agent.get_epsilon():.3f}")
    print()
    
    # Save model
    if save_path:
        agent.save_model(save_path)
        print(f"Model saved to: {save_path}")
    else:
        # Default save path using entry number
        default_path = f"{entry_number}_model.pt"
        agent.save_model(default_path)
        print(f"Model saved to: {default_path}")
    
    # Close environment
    env.close()

        # ---------------------------------------------------------
    # Save training metrics (plot + CSV)
    # ---------------------------------------------------------
    
    import csv 
    # Create directory if not provided
    output_dir = os.path.dirname(save_path) if save_path else "."
    os.makedirs(output_dir, exist_ok=True)

    # ---- Save rewards to CSV ----
    csv_path = os.path.join(output_dir, "training_rewards.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "episode_length", "epsilon", "win"])
        for i in range(len(episode_rewards)):
            writer.writerow([
                i + 1,
                episode_rewards[i],
                episode_lengths[i],
                epsilon_history[i],
                win_history[i]
            ])
    print(f"Training statistics saved to: {csv_path}")

    # ---- Plot rewards ----
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label="Episode Reward", alpha=0.6)

    # Compute moving average using numpy
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f"Moving Avg ({window})", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Progress")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, "training_rewards_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Reward plot saved to: {plot_path}")

    
    return agent, episode_rewards, episode_lengths, win_history


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent on PettingZoo Pong')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of episodes to train (default: 2000)')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save trained model (default: auto-generate from entry number)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Train agent
    agent, rewards, lengths, wins = train_dqn_agent(
        num_episodes=args.episodes,
        save_path=args.save_path,
        seed=args.seed
    )
    
    print("\nTraining script completed successfully!")
    print("You can now use the trained model for evaluation.")


if __name__ == "__main__":
    main()
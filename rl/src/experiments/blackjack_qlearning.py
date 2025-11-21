# ai use statement: github copilot assisted with experiment runner and learning curve plotting
"""q-learning on blackjack."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import QLearning, set_seeds, get_git_sha, make_filename
from experiments.config import BLACKJACK_CONFIG

try:
    import gymnasium as gym
except ImportError:
    print("error: gymnasium not installed. run: pip install gymnasium")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='q-learning on blackjack')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--episodes', type=int, default=BLACKJACK_CONFIG['num_episodes'], help='training episodes')
    parser.add_argument('--alpha', type=float, default=BLACKJACK_CONFIG['alpha'], help='learning rate')
    parser.add_argument('--gamma', type=float, default=BLACKJACK_CONFIG['gamma'], help='discount factor')
    parser.add_argument('--epsilon', type=float, default=BLACKJACK_CONFIG['epsilon'], help='exploration rate')
    args = parser.parse_args()
    
    # reproducibility
    set_seeds(args.seed)
    
    # create environment
    env = gym.make('Blackjack-v1')
    
    print(f"training q-learning on blackjack for {args.episodes} episodes...")
    agent = QLearning(env, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, 
                      num_episodes=args.episodes, seed=args.seed)
    results = agent.train()
    
    rewards = results['episode_rewards']
    print(f"training complete. final avg reward (last 100): {np.mean(rewards[-100:]):.3f}")
    
    # metadata
    sha = get_git_sha()
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    
    # create learning curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # raw rewards
    ax1.plot(rewards, alpha=0.3, color='forestgreen')
    window = 100
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(rewards)), smoothed, color='darkgreen', linewidth=2, label=f'{window}-episode avg')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('episode reward')
    ax1.set_title('q-learning learning curve (blackjack)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # cumulative average
    cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    ax2.plot(cumulative_avg, color='coral', linewidth=2)
    ax2.set_xlabel('episode')
    ax2.set_ylabel('cumulative average reward')
    ax2.set_title('cumulative average reward')
    ax2.grid(alpha=0.3)
    
    # save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    filename = make_filename('blackjack_qlearning', sha, 'png', timestamp, 'learning_curve')
    filepath = figures_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ saved: {filepath}")


if __name__ == '__main__':
    main()

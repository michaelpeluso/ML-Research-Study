# ai use statement: github copilot assisted with dqn runner structure (placeholder for extra credit)
"""dqn on cartpole - optional extra credit."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import DQN, set_seeds, get_git_sha, make_filename
from experiments.config import DQN_CONFIG

try:
    import gymnasium as gym
except ImportError:
    print("error: gymnasium not installed. run: pip install gymnasium")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='dqn on cartpole (extra credit)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--episodes', type=int, default=DQN_CONFIG['training']['num_episodes'], help='training episodes')
    args = parser.parse_args()
    
    # check pytorch
    try:
        import torch
    except ImportError:
        print("error: pytorch required for dqn. install with: pip install torch")
        sys.exit(1)
    
    # reproducibility
    set_seeds(args.seed)
    
    # create environment
    env = gym.make('CartPole-v1')
    
    print(f"training dqn on cartpole for {args.episodes} episodes...")
    print("note: this is a placeholder. implement dqn network and training loop for extra credit.")
    
    agent = DQN(env, 
                network_config=DQN_CONFIG['network'],
                replay_config=DQN_CONFIG['replay'],
                training_config=DQN_CONFIG['training'],
                seed=args.seed)
    
    results = agent.train()
    
    rewards = results['episode_rewards']
    print(f"training complete. final avg reward (last 100): {np.mean(rewards[-100:]):.1f}")
    
    # metadata
    sha = get_git_sha()
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    
    # create learning curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(rewards, alpha=0.3, color='purple')
    window = 100
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, color='darkviolet', linewidth=2, label=f'{window}-episode avg')
    ax.set_xlabel('episode')
    ax.set_ylabel('episode length')
    ax.set_title('dqn learning curve (cartpole)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    filename = make_filename('cartpole_dqn', sha, 'png', timestamp, 'learning_curve')
    filepath = figures_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ saved: {filepath}")
    print("\nto implement dqn properly:")
    print("  1. create neural network (torch.nn.Module)")
    print("  2. implement training loop with replay buffer")
    print("  3. add target network updates")
    print("  4. compare with tabular q-learning results")


if __name__ == '__main__':
    main()

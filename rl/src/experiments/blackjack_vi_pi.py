# ai use statement: github copilot assisted with experiment runner structure and plotting
"""compare value iteration vs policy iteration on blackjack."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ValueIteration, PolicyIteration, set_seeds, get_git_sha, make_filename
from experiments.config import BLACKJACK_CONFIG

try:
    import gymnasium as gym
except ImportError:
    print("error: gymnasium not installed. run: pip install gymnasium")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='compare vi vs pi on blackjack')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gamma', type=float, default=BLACKJACK_CONFIG['gamma'], help='discount factor')
    parser.add_argument('--theta', type=float, default=BLACKJACK_CONFIG['theta'], help='convergence threshold')
    args = parser.parse_args()
    
    # reproducibility
    set_seeds(args.seed)
    
    # create environment
    env = gym.make('Blackjack-v1')
    
    print(f"running value iteration on blackjack...")
    vi_agent = ValueIteration(env, gamma=args.gamma, theta=args.theta, seed=args.seed)
    vi_results = vi_agent.train()
    print(f"  converged in {vi_results['iterations']} iterations")
    
    print(f"running policy iteration on blackjack...")
    pi_agent = PolicyIteration(env, gamma=args.gamma, theta=args.theta, seed=args.seed)
    pi_results = pi_agent.train()
    print(f"  converged in {pi_results['iterations']} iterations")
    
    # metadata
    sha = get_git_sha()
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    
    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    algorithms = ['value iteration', 'policy iteration']
    iterations = [vi_results['iterations'], pi_results['iterations']]
    
    ax.bar(algorithms, iterations, color=['steelblue', 'coral'])
    ax.set_ylabel('iterations to convergence')
    ax.set_title('convergence comparison: vi vs pi (blackjack)')
    ax.grid(axis='y', alpha=0.3)
    
    # save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    filename = make_filename('blackjack_vi_pi', sha, 'png', timestamp, 'convergence')
    filepath = figures_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ saved: {filepath}")
    print(f"\nresults:")
    print(f"  vi iterations: {vi_results['iterations']}")
    print(f"  pi iterations: {pi_results['iterations']}")
    print(f"  faster: {'vi' if vi_results['iterations'] < pi_results['iterations'] else 'pi'}")


if __name__ == '__main__':
    main()

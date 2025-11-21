# ai use statement: github copilot assisted with discretization wrapper and experiment structure
"""compare value iteration vs policy iteration on cartpole with discretization."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import ValueIteration, PolicyIteration, StateDiscretizer, set_seeds, get_git_sha, make_filename
from experiments.config import CARTPOLE_CONFIG

try:
    import gymnasium as gym
except ImportError:
    print("error: gymnasium not installed. run: pip install gymnasium")
    sys.exit(1)


class DiscreteCartPoleWrapper(gym.Wrapper):
    """wrapper to discretize cartpole observations."""
    
    def __init__(self, env, discretizer):
        super().__init__(env)
        self.discretizer = discretizer
        # create discrete observation space
        self.observation_space = gym.spaces.Discrete(discretizer.get_num_states())
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        discrete_obs = self._state_to_index(self.discretizer.discretize(obs))
        return discrete_obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        discrete_obs = self._state_to_index(self.discretizer.discretize(obs))
        return discrete_obs, reward, terminated, truncated, info
    
    def _state_to_index(self, state_tuple):
        """convert discrete state tuple to single index."""
        bins = self.discretizer.bins
        index = 0
        multiplier = 1
        for i in reversed(range(len(state_tuple))):
            index += state_tuple[i] * multiplier
            multiplier *= bins[i]
        return index


def main():
    parser = argparse.ArgumentParser(description='compare vi vs pi on cartpole')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gamma', type=float, default=CARTPOLE_CONFIG['gamma'], help='discount factor')
    parser.add_argument('--theta', type=float, default=CARTPOLE_CONFIG['theta'], help='convergence threshold')
    args = parser.parse_args()
    
    # reproducibility
    set_seeds(args.seed)
    
    # create environment with discretization
    base_env = gym.make('CartPole-v1')
    discretizer = StateDiscretizer(base_env, CARTPOLE_CONFIG['discretization_bins'])
    env = DiscreteCartPoleWrapper(base_env, discretizer)
    
    print(f"cartpole state space discretized to {discretizer.get_num_states()} states")
    print(f"bins per dimension: {CARTPOLE_CONFIG['discretization_bins']}")
    
    print(f"\nrunning value iteration on cartpole...")
    vi_agent = ValueIteration(env, gamma=args.gamma, theta=args.theta, seed=args.seed)
    vi_results = vi_agent.train()
    print(f"  converged in {vi_results['iterations']} iterations")
    
    print(f"running policy iteration on cartpole...")
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
    ax.set_title(f'convergence comparison: vi vs pi (cartpole)\n{discretizer.get_num_states()} discrete states')
    ax.grid(axis='y', alpha=0.3)
    
    # save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    filename = make_filename('cartpole_vi_pi', sha, 'png', timestamp, 'convergence')
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

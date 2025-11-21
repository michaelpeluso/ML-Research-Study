# ai use statement: github copilot assisted with discretization and q-learning cartpole integration
"""q-learning on cartpole with discretization."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import QLearning, StateDiscretizer, set_seeds, get_git_sha, make_filename
from experiments.config import CARTPOLE_CONFIG

try:
    import gymnasium as gym
except ImportError:
    print("error: gymnasium not installed. run: pip install gymnasium")
    sys.exit(1)


class DiscreteQLearning(QLearning):
    """q-learning with state discretization for continuous spaces."""
    
    def __init__(self, env, discretizer, **kwargs):
        self.discretizer = discretizer
        super().__init__(env, **kwargs)
        # override state space size
        self.n_states = discretizer.get_num_states()
        self.discrete_states = True
        self.Q = np.zeros((self.n_states, self.n_actions))
    
    def _discretize_state(self, state):
        """convert continuous state to discrete index."""
        state_tuple = self.discretizer.discretize(state)
        bins = self.discretizer.bins
        index = 0
        multiplier = 1
        for i in reversed(range(len(state_tuple))):
            index += state_tuple[i] * multiplier
            multiplier *= bins[i]
        return index
    
    def get_state_key(self, state):
        """override to use discretized state."""
        return self._discretize_state(state)


def main():
    parser = argparse.ArgumentParser(description='q-learning on cartpole')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--episodes', type=int, default=CARTPOLE_CONFIG['num_episodes'], help='training episodes')
    parser.add_argument('--alpha', type=float, default=CARTPOLE_CONFIG['alpha'], help='learning rate')
    parser.add_argument('--gamma', type=float, default=CARTPOLE_CONFIG['gamma'], help='discount factor')
    parser.add_argument('--epsilon', type=float, default=CARTPOLE_CONFIG['epsilon'], help='exploration rate')
    args = parser.parse_args()
    
    # reproducibility
    set_seeds(args.seed)
    
    # create environment
    env = gym.make('CartPole-v1')
    discretizer = StateDiscretizer(env, CARTPOLE_CONFIG['discretization_bins'])
    
    print(f"training q-learning on cartpole for {args.episodes} episodes...")
    print(f"state space discretized to {discretizer.get_num_states()} states")
    
    agent = DiscreteQLearning(env, discretizer, alpha=args.alpha, gamma=args.gamma, 
                              epsilon=args.epsilon, num_episodes=args.episodes, seed=args.seed)
    results = agent.train()
    
    rewards = results['episode_rewards']
    print(f"training complete. final avg reward (last 100): {np.mean(rewards[-100:]):.1f}")
    
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
    ax1.set_ylabel('episode length')
    ax1.set_title('q-learning learning curve (cartpole)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # cumulative average
    cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    ax2.plot(cumulative_avg, color='coral', linewidth=2)
    ax2.set_xlabel('episode')
    ax2.set_ylabel('cumulative average length')
    ax2.set_title('cumulative average episode length')
    ax2.grid(alpha=0.3)
    
    # save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    filename = make_filename('cartpole_qlearning', sha, 'png', timestamp, 'learning_curve')
    filepath = figures_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ saved: {filepath}")


if __name__ == '__main__':
    main()

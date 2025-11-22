#!/usr/bin/env python3
# AI Use Statement: SARSA CartPole experiment runner created with GitHub Copilot assistance
"""sarsa on discretized cartpole"""
import sys
from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.sarsa import SARSA
from environments.cartpole_discretizer import StateDiscretizer
from utils.seeding import set_all_seeds
from utils.logging import ExperimentLogger
from utils.plotting import plot_learning_curve
import gymnasium as gym


def main():
    parser = argparse.ArgumentParser(description='train sarsa on cartpole')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--episodes', type=int, default=10000, help='number of episodes')
    parser.add_argument('--alpha', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='exploration rate')
    args = parser.parse_args()
    
    # set seeds for reproducibility
    set_all_seeds(args.seed)
    
    # create environment
    env = gym.make('CartPole-v1')
    env.reset(seed=args.seed)
    
    # discretize state space
    discretizer = StateDiscretizer(env=env, bins=[3, 3, 8, 12])
    print(f"discretized state space: {discretizer.get_num_states()} states")
    
    # wrap environment with discretizer
    class DiscretizedEnv:
        def __init__(self, env, discretizer):
            self.env = env
            self.discretizer = discretizer
            self.action_space = env.action_space
            self.observation_space = env.observation_space
        
        def reset(self, seed=None):
            state, info = self.env.reset(seed=seed)
            return self.discretizer.discretize(state), info
        
        def step(self, action):
            next_state, reward, terminated, truncated, info = self.env.step(action)
            return self.discretizer.discretize(next_state), reward, terminated, truncated, info
    
    wrapped_env = DiscretizedEnv(env, discretizer)
    
    # organize output by algorithm/environment
    algo_name = 'sarsa'
    env_name = 'cartpole'
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'raw' / algo_name / env_name
    metadata = {
        'algorithm': algo_name,
        'environment': 'CartPole-v1',
        'seed': args.seed,
        'episodes': args.episodes,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
    }

    # train agent
    print(f"training sarsa on cartpole for {args.episodes} episodes...")
    agent = SARSA(
        env=wrapped_env,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        num_episodes=args.episodes,
        seed=args.seed,
    )

    # train and log results using context-managed logger
    results = agent.train()

    with ExperimentLogger(output_dir=output_dir, experiment_name='sarsa_cartpole', seed=args.seed, metadata=metadata) as logger:
        # attach system/package info for the report
        logger.attach_system_info()

        # attach environment & discretizer info
        env_info = {
            'discretizer_bins': list(getattr(discretizer, 'bins', [])),
            'num_discrete_states': int(discretizer.get_num_states()),
            'action_space_n': int(getattr(wrapped_env.action_space, 'n', 0)),
            'observation_space_shape': list(getattr(env.observation_space, 'shape', [])),
            'observation_space_low': [float(x) for x in discretizer.low],
            'observation_space_high': [float(x) for x in discretizer.high],
            'command': ' '.join(sys.argv),
        }
        logger.update_metadata({'env': env_info})

        # log all useful per-episode metrics
        episode_returns = results.get('episode_returns', [])
        episode_steps = results.get('episode_steps', [])
        episode_td_errors = results.get('episode_td_errors', [])
        episode_q_changes = results.get('episode_q_changes', [])
        episode_explorations = results.get('episode_explorations', [])
        episode_q_table_sizes = results.get('episode_q_table_sizes', [])
        episode_q_table_nonzeros = results.get('episode_q_table_nonzeros', [])
        episode_max_q_values = results.get('episode_max_q_values', [])
        episode_mean_q_values = results.get('episode_mean_q_values', [])
        episode_action_entropies = results.get('episode_action_entropies', [])
        episode_td_error_stds = results.get('episode_td_error_stds', [])
        episode_q_change_stds = results.get('episode_q_change_stds', [])
        episode_unique_states = results.get('episode_unique_states', [])
        
        for episode in range(len(episode_returns)):
            metrics = {
                'episode_return': episode_returns[episode],
                'steps': episode_steps[episode] if episode < len(episode_steps) else None,
                'mean_abs_td_error': episode_td_errors[episode] if episode < len(episode_td_errors) else None,
                'mean_abs_q_change': episode_q_changes[episode] if episode < len(episode_q_changes) else None,
                'exploration_ratio': episode_explorations[episode] if episode < len(episode_explorations) else None,
                'q_table_size': episode_q_table_sizes[episode] if episode < len(episode_q_table_sizes) else None,
                'q_table_nonzero': episode_q_table_nonzeros[episode] if episode < len(episode_q_table_nonzeros) else None,
                'max_q_value': episode_max_q_values[episode] if episode < len(episode_max_q_values) else None,
                'mean_q_value': episode_mean_q_values[episode] if episode < len(episode_mean_q_values) else None,
                'action_entropy': episode_action_entropies[episode] if episode < len(episode_action_entropies) else None,
                'td_error_std': episode_td_error_stds[episode] if episode < len(episode_td_error_stds) else None,
                'q_change_std': episode_q_change_stds[episode] if episode < len(episode_q_change_stds) else None,
                'unique_states_visited': episode_unique_states[episode] if episode < len(episode_unique_states) else None,
            }
            logger.log_episode(episode, metrics)

        # compute summary statistics and attach to metadata for the report
        returns = np.array(results.get('episode_returns', []), dtype=float)
        if returns.size > 0:
            summary = {
                'episode_return_mean': float(np.mean(returns)),
                'episode_return_median': float(np.median(returns)),
                'episode_return_std': float(np.std(returns)),
                'episode_return_max': float(np.max(returns)),
                'episode_return_min': float(np.min(returns)),
                'last_100_mean': float(np.mean(returns[-100:])) if returns.size >= 100 else None,
            }
            logger.update_metadata({'summary': summary})

        print(f"results saved to {logger.get_log_path()}")
    
    # plot learning curve
    data = pd.read_csv(logger.get_log_path())
    figure_dir = Path(__file__).parent.parent.parent / 'results' / 'figures' / algo_name / env_name
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f'{algo_name}_{env_name}_seed{args.seed}.png'
    
    plot_learning_curve(
        data=data,
        x_col='episode',
        y_col='episode_return',
        output_path=figure_path,
        title=f'SARSA CartPole (seed={args.seed})',
        xlabel='Episode',
        ylabel='Episode Return'
    )
    
    print(f"figure saved to {figure_path}")


if __name__ == '__main__':
    main()

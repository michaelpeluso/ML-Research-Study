#!/usr/bin/env python3
"""unified experiment runner with individual algorithm dispatchers"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from typing import Dict, Any

# algorithm imports
from algorithms.sarsa import SARSA
from algorithms.q_learning import QLearning
from algorithms.value_iteration import ValueIteration
from algorithms.policy_iteration import PolicyIteration

# environment imports
from environments.blackjack_wrapper import BlackjackWrapper
from environments.blackjack_mdp import SimplifiedBlackjackMDP
from environments.cartpole_discretizer import StateDiscretizer, DiscretizedEnv

# utilities
from utils.seeding import set_all_seeds
from utils.logging import ExperimentLogger
from experiments.experiment_base import (
    get_hyperparams, get_env_config, print_experiment_header, log_results, save_plots
)


def run_sarsa(env_name: str, config: dict, seed: int):
    """run sarsa on blackjack or cartpole"""
    algo_name = 'sarsa'
    algo_class = SARSA
    
    # extract hyperparameters
    params = get_hyperparams(config, algo_name)
    env_config = get_env_config(config, env_name)
    
    # environment-specific setup
    if env_name == 'blackjack':
        natural = env_config.get('natural', False)
        sab = env_config.get('sab', False)
        display_params = {**params, 'natural': natural, 'sab': sab}
        
        print_experiment_header(algo_name, env_name, seed, display_params)
        set_all_seeds(seed)
        
        env = BlackjackWrapper(natural=natural, sab=sab)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        
        print(f"blackjack state space: ~{len(env.get_state_space())} states")
        
        metadata = {
            'algorithm': algo_name,
            'environment': 'Blackjack-v1',
            'seed': seed,
            **params,
            'natural': natural,
            'sab': sab,
        }
        
        env_info = {
            'action_space_n': getattr(env.action_space, 'n', None),
            'natural': natural,
            'sab': sab,
            'state_space_size': len(env.get_state_space()),
            'command': ' '.join(sys.argv),
        }
        
    elif env_name == 'cartpole':
        bins = env_config.get('bins', [3, 3, 8, 12])
        display_params = {**params, 'bins': bins}
        
        print_experiment_header(algo_name, env_name, seed, display_params)
        set_all_seeds(seed)
        
        base_env = gym.make('CartPole-v1')
        base_env.reset(seed=seed)
        base_env.action_space.seed(seed)
        base_env.observation_space.seed(seed)
        
        discretizer = StateDiscretizer(env=base_env, bins=bins)
        env = DiscretizedEnv(base_env, discretizer)
        
        print(f"discretized state space: {discretizer.get_num_states()} states")
        
        metadata = {
            'algorithm': algo_name,
            'environment': 'CartPole-v1',
            'seed': seed,
            **params,
            'bins': bins,
        }
        
        env_info = {
            'action_space_n': getattr(env.action_space, 'n', None),
            'bins': bins,
            'num_discrete_states': discretizer.get_num_states(),
            'command': ' '.join(sys.argv),
        }
    
    else:
        raise ValueError(f"unknown environment: {env_name}")
    
    # train agent
    print(f"training {algo_name} on {env_name} for {params['episodes']} episodes...")
    agent = algo_class(
        env=env,  # type: ignore
        alpha=params['alpha'],
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        num_episodes=params['episodes'],
        seed=seed,
    )
    
    results = agent.train()
    
    # log results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'raw' / algo_name / env_name
    with ExperimentLogger(output_dir, f'{algo_name}_{env_name}', seed, metadata) as logger:
        logger.update_metadata({'env': env_info})
        log_results(logger, results, env_info)
        print(f"results saved to {logger.log_file}")
        
        # generate plots
        save_plots(config, logger, algo_name, env_name, seed, f"{algo_name.upper()} on {env_name.capitalize()}")
    return 'success'


def run_qlearning(env_name: str, config: dict, seed: int):
    """run q-learning on blackjack or cartpole"""
    algo_name = 'qlearning'
    algo_class = QLearning
    
    # extract hyperparameters
    params = get_hyperparams(config, algo_name)
    env_config = get_env_config(config, env_name)
    
    # environment-specific setup
    if env_name == 'blackjack':
        natural = env_config.get('natural', False)
        sab = env_config.get('sab', False)
        display_params = {**params, 'natural': natural, 'sab': sab}
        
        print_experiment_header(algo_name, env_name, seed, display_params)
        set_all_seeds(seed)
        
        env = BlackjackWrapper(natural=natural, sab=sab)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        
        print(f"blackjack state space: ~{len(env.get_state_space())} states")
        
        metadata = {
            'algorithm': algo_name,
            'environment': 'Blackjack-v1',
            'seed': seed,
            **params,
            'natural': natural,
            'sab': sab,
        }
        
        env_info = {
            'action_space_n': getattr(env.action_space, 'n', None),
            'natural': natural,
            'sab': sab,
            'state_space_size': len(env.get_state_space()),
            'command': ' '.join(sys.argv),
        }
        
    elif env_name == 'cartpole':
        bins = env_config.get('bins', [3, 3, 8, 12])
        display_params = {**params, 'bins': bins}
        
        print_experiment_header(algo_name, env_name, seed, display_params)
        set_all_seeds(seed)
        
        base_env = gym.make('CartPole-v1')
        base_env.reset(seed=seed)
        base_env.action_space.seed(seed)
        base_env.observation_space.seed(seed)
        
        discretizer = StateDiscretizer(env=base_env, bins=bins)
        env = DiscretizedEnv(base_env, discretizer)
        
        print(f"discretized state space: {discretizer.get_num_states()} states")
        
        metadata = {
            'algorithm': algo_name,
            'environment': 'CartPole-v1',
            'seed': seed,
            **params,
            'bins': bins,
        }
        
        env_info = {
            'action_space_n': getattr(env.action_space, 'n', None),
            'bins': bins,
            'num_discrete_states': discretizer.get_num_states(),
            'command': ' '.join(sys.argv),
        }
    
    else:
        raise ValueError(f"unknown environment: {env_name}")
    
    # train agent
    print(f"training {algo_name} on {env_name} for {params['episodes']} episodes...")
    agent = algo_class(
        env=env,  # type: ignore
        alpha=params['alpha'],
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        num_episodes=params['episodes'],
        seed=seed,
    )
    
    results = agent.train()
    
    # log results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'raw' / algo_name / env_name
    with ExperimentLogger(output_dir, f'{algo_name}_{env_name}', seed, metadata) as logger:
        logger.update_metadata({'env': env_info})
        log_results(logger, results, env_info)
        print(f"results saved to {logger.log_file}")
        
        # generate plots
        save_plots(config, logger, algo_name, env_name, seed, f"{algo_name.upper()} on {env_name.capitalize()}")
    return 'success'


def run_vi(env_name: str, config: dict, seed: int):
    """run value iteration on blackjack (cartpole not supported)"""
    algo_name = 'vi'
    
    # only blackjack supported (cartpole vi skipped due to evaluation bugs)
    if env_name != 'blackjack':
        print(f"warning: {algo_name} on {env_name} not implemented (mdp evaluation issues)")
        return 'skipped'
    
    # extract hyperparameters
    env_config = get_env_config(config, env_name)
    natural = env_config.get('natural', False)
    sab = env_config.get('sab', False)
    
    # vi specific params
    gamma = config.get(algo_name, {}).get('gamma', config.get('hyperparameters', {}).get('gamma', 0.99))
    theta = config.get(algo_name, {}).get('theta', 0.0001)
    max_iterations = config.get(algo_name, {}).get('max_iterations', 10000)
    eval_episodes = config.get(algo_name, {}).get('eval_episodes', 100)
    
    display_params = {
        'gamma': gamma,
        'theta': theta,
        'max_iterations': max_iterations,
        'natural': natural,
        'sab': sab
    }
    
    print_experiment_header(algo_name, env_name, seed, display_params)
    set_all_seeds(seed)
    
    # create simplified mdp environment
    env = SimplifiedBlackjackMDP()
    print(f"blackjack state space: {env.num_states} states")
    
    # setup metadata
    metadata = {
        'algorithm': 'value_iteration',
        'environment': 'Blackjack-v1',
        'seed': seed,
        'gamma': gamma,
        'theta': theta,
        'max_iterations': max_iterations,
        'eval_episodes': eval_episodes,
        'natural': natural,
        'sab': sab,
    }
    
    # run algorithm
    print(f"running value iteration (gamma={gamma}, theta={theta})...")
    solver = ValueIteration(
        env=env,
        gamma=gamma,
        theta=theta,
        max_iterations=max_iterations,
        seed=seed,
    )
    results = solver.train()
    
    # evaluate learned policy
    print(f"evaluating policy over {eval_episodes} episodes...")
    eval_results = solver.evaluate_policy(num_episodes=eval_episodes)
    
    # log convergence results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'raw' / algo_name / env_name
    with ExperimentLogger(output_dir, f'{algo_name}_{env_name}', seed, metadata) as logger:
        env_info = {
            'state_space_size': results['metadata']['num_states'],
            'action_space_n': results['metadata']['num_actions'],
            'command': ' '.join(sys.argv),
        }
        logger.update_metadata({'env': env_info})
        
        # log convergence metrics per iteration
        for iteration in range(results['iterations']):
            metrics = {
                'iteration': iteration,
                'delta': results['deltas'][iteration],
                'wall_time': results['wall_times'][iteration],
            }
            logger.log_episode(iteration, metrics)
        
        # add summary with convergence and evaluation results
        summary = {
            'convergence': {
                'converged': results['converged'],
                'iterations': results['iterations'],
                'final_delta': results['metadata']['final_delta'],
                'total_time': results['metadata']['total_time'],
            },
            'policy_evaluation': {
                'mean_return': eval_results['mean_return'],
                'std_return': eval_results['std_return'],
                'mean_length': eval_results['mean_length'],
                'q1': float(eval_results['returns'][int(len(eval_results['returns']) * 0.25)]),
                'q3': float(eval_results['returns'][int(len(eval_results['returns']) * 0.75)]),
            }
        }
        
        logger.update_metadata({'summary': summary})
        print(f"results saved to {logger.log_file}")
    return 'success'


def run_pi(env_name: str, config: dict, seed: int):
    """run policy iteration on blackjack (cartpole not supported)"""
    algo_name = 'pi'
    
    # only blackjack supported (cartpole pi skipped due to evaluation bugs)
    if env_name != 'blackjack':
        print(f"warning: {algo_name} on {env_name} not implemented (mdp evaluation issues)")
        return 'skipped'
    
    # extract hyperparameters
    env_config = get_env_config(config, env_name)
    natural = env_config.get('natural', False)
    sab = env_config.get('sab', False)
    
    # pi specific params
    gamma = config.get(algo_name, {}).get('gamma', config.get('hyperparameters', {}).get('gamma', 0.99))
    theta = config.get(algo_name, {}).get('theta', 0.0001)
    max_iterations = config.get(algo_name, {}).get('max_iterations', 1000)
    max_eval_iterations = config.get(algo_name, {}).get('max_eval_iterations', 1000)
    eval_episodes = config.get(algo_name, {}).get('eval_episodes', 100)
    
    display_params = {
        'gamma': gamma,
        'theta': theta,
        'max_iterations': max_iterations,
        'max_eval_iterations': max_eval_iterations,
        'natural': natural,
        'sab': sab
    }
    
    print_experiment_header(algo_name, env_name, seed, display_params)
    set_all_seeds(seed)
    
    # create simplified mdp environment
    env = SimplifiedBlackjackMDP()
    print(f"blackjack state space: {env.num_states} states")
    
    # setup metadata
    metadata = {
        'algorithm': 'policy_iteration',
        'environment': 'Blackjack-v1',
        'seed': seed,
        'gamma': gamma,
        'theta': theta,
        'max_iterations': max_iterations,
        'max_eval_iterations': max_eval_iterations,
        'eval_episodes': eval_episodes,
        'natural': natural,
        'sab': sab,
    }
    
    # run algorithm
    print(f"running policy iteration (gamma={gamma}, theta={theta})...")
    solver = PolicyIteration(
        env=env,
        gamma=gamma,
        theta=theta,
        max_iterations=max_iterations,
        max_eval_iterations=max_eval_iterations,
        seed=seed,
    )
    results = solver.train()
    
    # evaluate learned policy
    print(f"evaluating policy over {eval_episodes} episodes...")
    eval_results = solver.evaluate_policy(num_episodes=eval_episodes)
    
    # log convergence results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'raw' / algo_name / env_name
    with ExperimentLogger(output_dir, f'{algo_name}_{env_name}', seed, metadata) as logger:
        env_info = {
            'state_space_size': results['metadata']['num_states'],
            'action_space_n': results['metadata']['num_actions'],
            'command': ' '.join(sys.argv),
        }
        logger.update_metadata({'env': env_info})
        
        # log convergence metrics per iteration
        for iteration in range(results['iterations']):
            metrics = {
                'iteration': iteration,
                'policy_changes': results['policy_changes'][iteration],
                'eval_iterations': results['eval_iterations'][iteration],
                'wall_time': results['wall_times'][iteration],
            }
            logger.log_episode(iteration, metrics)
        
        # add summary with convergence and evaluation results
        summary = {
            'convergence': {
                'converged': results['converged'],
                'iterations': results['iterations'],
                'total_eval_iterations': results['metadata']['total_eval_iterations'],
                'total_time': results['metadata']['total_time'],
            },
            'policy_evaluation': {
                'mean_return': eval_results['mean_return'],
                'std_return': eval_results['std_return'],
                'mean_length': eval_results['mean_length'],
                'q1': float(eval_results['returns'][int(len(eval_results['returns']) * 0.25)]),
                'q3': float(eval_results['returns'][int(len(eval_results['returns']) * 0.75)]),
            }
        }
        
        logger.update_metadata({'summary': summary})
        print(f"results saved to {logger.log_file}")
    return 'success'


# algorithm dispatch registry - direct mapping to individual runners
ALGORITHM_RUNNERS = {
    ('sarsa', 'blackjack'): run_sarsa,
    ('sarsa', 'cartpole'): run_sarsa,
    ('qlearning', 'blackjack'): run_qlearning,
    ('qlearning', 'cartpole'): run_qlearning,
    ('vi', 'blackjack'): run_vi,
    ('vi', 'cartpole'): run_vi,
    ('pi', 'blackjack'): run_pi,
    ('pi', 'cartpole'): run_pi,
}


def run_experiment(algo: str, env: str, config: Dict[str, Any], seed: int) -> str:
    """dispatch to appropriate algorithm runner for the given environment"""
    key = (algo, env)
    
    if key not in ALGORITHM_RUNNERS:
        raise ValueError(f"unsupported combination: algorithm={algo}, environment={env}")
    
    runner = ALGORITHM_RUNNERS[key]
    try:
        status = runner(env, config, seed)
    except Exception:
        # any unexpected exception is a failed run
        return 'failed'

    if status is None:
        return 'success'
    return status


if __name__ == '__main__':
    # this file is imported by main.py, not run directly
    # for testing: python -c "from experiments.run_experiment import run_experiment; ..."
    print("unified experiment runner loaded successfully")
    print(f"supported combinations: {list(ALGORITHM_RUNNERS.keys())}")

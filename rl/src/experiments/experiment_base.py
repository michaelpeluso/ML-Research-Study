"""base experiment utilities - shared across all experiments"""
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.experiment_logger import ExperimentLogger


def load_config() -> dict:
    """load configuration from default.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_hyperparams(config: dict, algo_name: str, env_name: str = None) -> Dict[str, Any]: # type: ignore
    """extract algorithm hyperparameters from config with environment-specific overrides"""
    # get shared defaults
    shared = config.get('hyperparameters', {})
    # get algorithm-specific overrides
    algo_config = config.get(algo_name, {})
    
    # base parameters with algo overrides taking precedence
    params = {
        'alpha': algo_config.get('alpha', shared.get('alpha', 0.1)),
        'gamma': algo_config.get('gamma', shared.get('gamma', 0.99)),
        'epsilon': algo_config.get('epsilon', shared.get('epsilon', 0.1)),
        'episodes': algo_config.get('episodes', shared.get('episodes', 10000))
    }
    
    # apply environment-specific overrides if provided (e.g., alpha_blackjack)
    if env_name:
        for param in ['alpha', 'gamma', 'epsilon_floor', 'epsilon_decay_episodes']:
            env_key = f'{param}_{env_name}'
            if env_key in algo_config:
                params[param] = algo_config[env_key]
        
        # also check for environment-specific episodes (e.g., episodes_blackjack)
        episodes_key = f'episodes_{env_name}'
        if episodes_key in shared:
            params['episodes'] = shared[episodes_key]
        if episodes_key in algo_config:
            params['episodes'] = algo_config[episodes_key]
    
    return params


def get_env_config(config: dict, env_name: str) -> Dict[str, Any]:
    """extract environment configuration"""
    return config.get(env_name, {})


def print_experiment_header(algo_name: str, env_name: str, seed: int, params: Dict[str, Any]):
    """print formatted experiment header"""
    print(f"\n{'='*60}")
    print(f"{algo_name.upper()} on {env_name.title()} (seed={seed})")
    param_str = ', '.join(f"{k}={v}" for k, v in params.items())
    print(f"config: {param_str}")
    print('='*60)


def log_results(
    logger: ExperimentLogger,
    results: Dict[str, list],
    env_info: Optional[Dict[str, Any]] = None
) -> None:
    """log episode results and compute summary statistics"""
    # attach environment info if provided
    if env_info:
        logger.update_metadata({'env': env_info})

    # extract metric lists from results
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
    
    # log each episode
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

    # compute comprehensive summary statistics for RL report
    returns = np.array(results.get('episode_returns', []), dtype=float)
    steps = np.array(results.get('episode_steps', []), dtype=float)
    td_errors = np.array(results.get('episode_td_errors', []), dtype=float)
    q_changes = np.array(results.get('episode_q_changes', []), dtype=float)
    explorations = np.array(results.get('episode_explorations', []), dtype=float)
    entropies = np.array(results.get('episode_action_entropies', []), dtype=float)
    
    if returns.size > 0:
        # basic return statistics with quartiles and IQR
        q1 = float(np.percentile(returns, 25))
        q3 = float(np.percentile(returns, 75))
        return_stats = {
            'mean': float(np.mean(returns)),
            'median': float(np.median(returns)),
            'std': float(np.std(returns)),
            'q1': q1,
            'q3': q3,
            'iqr': float(q3 - q1),
            'max': float(np.max(returns)),
            'min': float(np.min(returns)),
            'first_10_mean': float(np.mean(returns[:10])) if returns.size >= 10 else float(np.mean(returns)),
            'last_10_mean': float(np.mean(returns[-10:])) if returns.size >= 10 else float(np.mean(returns)),
            'last_100_mean': float(np.mean(returns[-100:])) if returns.size >= 100 else float(np.mean(returns)),
        }
        
        # episode length statistics (balancing performance for cartpole)
        episode_stats = {
            'mean_episode_length': float(np.mean(steps)) if steps.size > 0 else None,
            'max_episode_length': float(np.max(steps)) if steps.size > 0 else None,
            'final_10_mean_length': float(np.mean(steps[-10:])) if steps.size >= 10 else None,
        }
        
        # convergence indicators
        convergence = {}
        if td_errors.size > 0:
            convergence['final_10_mean_td_error'] = float(np.mean(td_errors[-10:]))
            convergence['td_error_trend'] = float(np.mean(td_errors[:10]) - np.mean(td_errors[-10:])) if td_errors.size >= 20 else None
        if q_changes.size > 0:
            convergence['final_10_mean_q_change'] = float(np.mean(q_changes[-10:]))
            convergence['q_change_trend'] = float(np.mean(q_changes[:10]) - np.mean(q_changes[-10:])) if q_changes.size >= 20 else None
        if entropies.size > 0:
            convergence['final_10_mean_entropy'] = float(np.mean(entropies[-10:]))
            convergence['entropy_decrease'] = float(np.mean(entropies[:10]) - np.mean(entropies[-10:])) if entropies.size >= 20 else None
        
        # exploration statistics
        exploration_stats = {}
        if explorations.size > 0:
            exploration_stats['mean_exploration_ratio'] = float(np.mean(explorations))
            exploration_stats['final_10_exploration_ratio'] = float(np.mean(explorations[-10:]))
        
        # sample efficiency (improvement rate)
        sample_efficiency = {}
        if returns.size >= 20:
            first_quarter = returns[:len(returns)//4]
            last_quarter = returns[-len(returns)//4:]
            sample_efficiency['return_improvement'] = float(np.mean(last_quarter) - np.mean(first_quarter))
            sample_efficiency['return_improvement_pct'] = float((np.mean(last_quarter) - np.mean(first_quarter)) / (abs(np.mean(first_quarter)) + 1e-10) * 100)
        
        # stability metrics (variance in later episodes)
        stability = {}
        if returns.size >= 100:
            stability['last_100_std'] = float(np.std(returns[-100:]))
            stability['last_100_coefficient_of_variation'] = float(np.std(returns[-100:]) / (abs(np.mean(returns[-100:])) + 1e-10))
        
        summary = {
            'returns': return_stats,
            'episodes': episode_stats,
            'convergence': convergence,
            'exploration': exploration_stats,
            'sample_efficiency': sample_efficiency,
            'stability': stability,
        }
        logger.update_metadata({'summary': summary})


def save_plots(
    config: dict,
    logger: ExperimentLogger,
    algo_name: str,
    env_name: str,
    seed: int,
    title: str
) -> None:
    """per-seed plots disabled - use generate_all_plots() for aggregated report figures"""
    # per-seed plotting removed; all report plots are aggregated across seeds
    # run: python -c "from utils.generate_report_plots import generate_all_plots; generate_all_plots()"
    pass

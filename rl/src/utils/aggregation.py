"""unified aggregation: raw per-seed results → master summary JSONs for rl report"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """compute mean, std, iqr, quartiles for report aggregation"""
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q1': float(np.percentile(arr, 25)),
        'median': float(np.percentile(arr, 50)),
        'q3': float(np.percentile(arr, 75)),
        'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def aggregate_learning_curves(
    raw_dir: Path,
    pattern: str = "*.csv",
    x_col: str = "episode",
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """aggregate episode-by-episode learning curves across seeds
    
    returns dict with per-metric statistics (mean, q1, q3, iqr) indexed by episode
    """
    raw_dir = Path(raw_dir)
    files: List[Path] = sorted(raw_dir.rglob(pattern))
    
    if not files:
        return {}
    
    # default metrics if not specified
    if metrics is None:
        metrics = [
            'episode_return', 'steps', 'mean_abs_td_error', 'mean_abs_q_change', 
            'exploration_ratio', 'q_table_size', 'q_table_nonzero', 'max_q_value', 
            'mean_q_value', 'action_entropy', 'td_error_std', 'q_change_std', 
            'unique_states_visited'
        ]
    
    dfs_by_metric = {metric: [] for metric in metrics}
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if x_col not in df.columns:
                continue
            
            for metric in metrics:
                if metric in df.columns:
                    dfs_by_metric[metric].append(df[[x_col, metric]].set_index(x_col))
        except Exception:
            continue
    
    # aggregate each metric: compute mean + quartiles across seeds
    aggregated = {}
    for metric, dfs in dfs_by_metric.items():
        if not dfs:
            continue
        
        combined = pd.concat(dfs, axis=1)
        numeric = combined.select_dtypes(include='number')
        
        # bellman aggregation: compute E[X] and IQR across seed distribution
        mean = numeric.mean(axis=1)
        q1 = numeric.quantile(0.25, axis=1)
        q3 = numeric.quantile(0.75, axis=1)
        iqr = q3 - q1
        
        aggregated[metric] = {
            'episodes': mean.index.tolist(),
            'mean': mean.values.tolist(),
            'q1': q1.values.tolist(),
            'q3': q3.values.tolist(),
            'iqr': iqr.values.tolist(),
        }
    
    return aggregated


def aggregate_sarsa_qlearning(algo_name: str, results_dir: Path) -> Dict[str, Any]:
    """aggregate sarsa/qlearning results across seeds and environments"""
    algo_dir = results_dir / algo_name
    if not algo_dir.exists():
        return {}
    
    aggregated = {
        'algorithm': algo_name,
        'environments': {},
        'metadata': {
            'seeds': [],
            'total_runs': 0,
        }
    }
    
    # process each environment
    for env_dir in algo_dir.iterdir():
        if not env_dir.is_dir():
            continue
        
        env_name = env_dir.name
        json_files = list(env_dir.glob(f'{algo_name}_{env_name}_seed*.json'))
        
        if not json_files:
            continue
        
        # collect metrics across all seeds
        mean_returns = []
        last_10_returns = []
        seeds_used = []
        hyperparams = None
        wall_times = []
        sample_efficiency = []
        stability_metrics = []
        
        for json_file in json_files:
            try:
                seed = int(json_file.stem.split('seed')[-1])
                seeds_used.append(seed)
                
                with open(json_file, 'r') as f:
                    meta = json.load(f)
                    
                    results = meta.get('results', {})
                    
                    # returns
                    returns_data = results.get('returns', {})
                    mean_returns.append(returns_data.get('mean', 0))
                    last_10_returns.append(returns_data.get('last_10_mean', 0))
                    
                    # sample efficiency
                    efficiency = results.get('sample_efficiency', {})
                    sample_efficiency.append(efficiency.get('return_improvement_pct', 0))
                    
                    # stability
                    stab = results.get('stability', {})
                    stability_metrics.append(stab.get('last_100_coefficient_of_variation', 0))
                    
                    # hyperparameters
                    if hyperparams is None:
                        hp = meta.get('hyperparameters', {})
                        hyperparams = {
                            'alpha': hp.get('alpha'),
                            'gamma': hp.get('gamma'),
                            'epsilon': hp.get('epsilon'),
                            'episodes': hp.get('episodes'),
                        }
                    
                    # wall time
                    exp = meta.get('experiment', {})
                    wall_times.append(exp.get('duration_sec', 0))
            
            except Exception as e:
                print(f"warning: failed to process {json_file}: {e}")
                continue
        
        if not mean_returns:
            continue
        
        # aggregate learning curves from csv files
        learning_curves = aggregate_learning_curves(env_dir, pattern=f'{algo_name}_{env_name}_seed*.csv')
        
        # report requirements: mean + IQR aggregation
        env_summary = {
            'mean_return': compute_statistics(mean_returns),
            'last_10_episodes_return': compute_statistics(last_10_returns),
            'sample_efficiency_pct': compute_statistics(sample_efficiency),
            'stability_cv': compute_statistics(stability_metrics),
            'wall_time_seconds': compute_statistics(wall_times),
            'learning_curves': learning_curves,
            'seeds': sorted(seeds_used),
            'num_seeds': len(seeds_used),
            'hyperparameters': hyperparams,
        }
        
        aggregated['environments'][env_name] = env_summary
        aggregated['metadata']['seeds'].extend(seeds_used)
        aggregated['metadata']['total_runs'] += len(seeds_used)
    
    aggregated['metadata']['seeds'] = sorted(list(set(aggregated['metadata']['seeds'])))
    return aggregated


def aggregate_vi_pi(algo_name: str, results_dir: Path) -> Dict[str, Any]:
    """aggregate vi/pi results across seeds and environments"""
    algo_dir = results_dir / algo_name
    if not algo_dir.exists():
        return {}
    
    aggregated = {
        'algorithm': 'value_iteration' if algo_name == 'vi' else 'policy_iteration',
        'environments': {},
        'metadata': {
            'seeds': [],
            'total_runs': 0,
        }
    }
    
    # process each environment
    for env_dir in algo_dir.iterdir():
        if not env_dir.is_dir():
            continue
        
        env_name = env_dir.name
        json_files = list(env_dir.glob(f'{algo_name}_{env_name}_seed*.json'))
        
        if not json_files:
            continue
        
        # collect convergence metrics across all seeds
        iterations_to_converge = []
        total_times = []
        mean_returns = []
        seeds_used = []
        hyperparams = None
        
        # vi-specific
        final_deltas = []
        
        # pi-specific
        total_eval_iterations = []
        
        for json_file in json_files:
            try:
                seed = int(json_file.stem.split('seed')[-1])
                seeds_used.append(seed)
                
                with open(json_file, 'r') as f:
                    meta = json.load(f)
                    
                    # convergence metrics
                    results = meta.get('results', {})
                    conv = results.get('convergence', meta.get('summary', {}).get('convergence', {}))
                    iterations_to_converge.append(conv.get('iterations', 0))
                    total_times.append(conv.get('total_time', 0))
                    
                    # policy evaluation
                    policy_eval = results.get('policy_evaluation', meta.get('summary', {}).get('policy_evaluation', {}))
                    mean_returns.append(policy_eval.get('mean_return', 0))
                    
                    # algorithm-specific
                    if algo_name == 'vi':
                        final_deltas.append(conv.get('final_delta', 0))
                    else:  # pi
                        total_eval_iterations.append(conv.get('total_eval_iterations', 0))
                    
                    # hyperparams
                    if hyperparams is None:
                        hp = meta.get('hyperparameters', {})
                        extra = meta.get('extra', {})
                        hyperparams = {
                            'gamma': hp.get('gamma') or extra.get('gamma'),
                            'theta': extra.get('theta'),
                            'max_iterations': extra.get('max_iterations'),
                        }
                        if algo_name == 'pi':
                            hyperparams['max_eval_iterations'] = extra.get('max_eval_iterations')
            
            except Exception as e:
                print(f"warning: failed to process {json_file}: {e}")
                continue
        
        if not iterations_to_converge:
            continue
        
        # aggregate statistics
        env_summary = {
            'convergence': {
                'iterations': compute_statistics(iterations_to_converge),
                'wall_time_seconds': compute_statistics(total_times),
            },
            'policy_performance': {
                'mean_return': compute_statistics(mean_returns),
            },
            'seeds': sorted(seeds_used),
            'num_seeds': len(seeds_used),
            'hyperparameters': hyperparams,
        }
        
        # add algorithm-specific metrics
        if algo_name == 'vi' and final_deltas:
            env_summary['convergence']['final_delta'] = compute_statistics(final_deltas)
        elif algo_name == 'pi' and total_eval_iterations:
            env_summary['convergence']['total_eval_iterations'] = compute_statistics(total_eval_iterations)
        
        aggregated['environments'][env_name] = env_summary
        aggregated['metadata']['seeds'].extend(seeds_used)
        aggregated['metadata']['total_runs'] += len(seeds_used)
    
    aggregated['metadata']['seeds'] = sorted(list(set(aggregated['metadata']['seeds'])))
    return aggregated


def create_master_summaries(results_dir: Optional[Path] = None) -> None:
    """create master summary json files for all algorithms (single source of truth for report)"""
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'results' / 'raw'
    
    algorithms = {
        'sarsa': aggregate_sarsa_qlearning,
        'qlearning': aggregate_sarsa_qlearning,
        'vi': aggregate_vi_pi,
        'pi': aggregate_vi_pi,
    }
    
    for algo_name, aggregate_func in algorithms.items():
        algo_dir = results_dir / algo_name
        if not algo_dir.exists():
            print(f"skipping {algo_name}: directory not found")
            continue
        
        print(f"\naggregating {algo_name}...")
        aggregated = aggregate_func(algo_name, results_dir)
        
        if not aggregated or not aggregated.get('environments'):
            print(f"  no data found for {algo_name}")
            continue
        
        # save json (single source of truth)
        json_path = algo_dir / 'master_summary.json'
        with open(json_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"  saved: {json_path}")
        
        # print summary
        print(f"  summary:")
        for env_name, env_data in aggregated['environments'].items():
            print(f"    {env_name} ({env_data['num_seeds']} seeds):")
            if algo_name in ['sarsa', 'qlearning']:
                print(f"      mean_return: {env_data['mean_return']['mean']:.2f} ± {env_data['mean_return']['std']:.2f}")
                print(f"      sample_efficiency: {env_data['sample_efficiency_pct']['mean']:.2f}%")
                print(f"      stability_cv: {env_data['stability_cv']['mean']:.2f}")
            else:
                print(f"      iterations: {env_data['convergence']['iterations']['mean']:.1f}")
                print(f"      wall_time: {env_data['convergence']['wall_time_seconds']['mean']:.3f}s")
                print(f"      policy_return: {env_data['policy_performance']['mean_return']['mean']:.2f}")
    
    print(f"\n{'='*60}")
    print("MASTER SUMMARIES CREATED")
    print('='*60)
    print(f"location: {results_dir}")
    print("format: master_summary.json (single source of truth)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='aggregate results for rl report')
    parser.add_argument('--results-dir', type=str, default='../../results/raw',
                       help='path to results/raw directory')
    args = parser.parse_args()
    
    results_path = Path(__file__).parent.parent.parent / 'results' / 'raw'
    if args.results_dir:
        results_path = Path(args.results_dir)
    
    create_master_summaries(results_path)

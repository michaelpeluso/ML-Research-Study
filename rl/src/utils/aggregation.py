"""unified aggregation: raw per-seed results → master summary JSONs + plots for rl report"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict


def generate_learning_curve_plot(
    raw_dir: Path,
    env_name: str,
    algo_name: str,
    output_dir: Path,
    pattern: str = "*.csv",
) -> None:
    """generate individual learning curve: mean + iqr bands across seeds
    
    uses episode binning to aggregate discrete rewards into meaningful intervals.
    this is standard practice for noisy RL data (especially blackjack with -1/0/+1 rewards).
    aligns with report requirement: mean ± variability (IQR) over 30-50 seeds
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob(pattern))
    
    if not files:
        return
    
    # aggregate returns across all seeds
    all_returns = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'episode_return' in df.columns:
                all_returns.append(df['episode_return'].values)
        except Exception:
            continue
    
    if not all_returns:
        return
    
    # align by minimum length
    min_len = min(len(r) for r in all_returns)
    returns_array = np.array([r[:min_len] for r in all_returns])
    
    # bin size: larger for blackjack (discrete -1/0/+1), smaller for cartpole
    bin_size = 100 if env_name == 'blackjack' else 50
    
    # bin episodes: compute mean return per bin for each seed
    n_bins = min_len // bin_size
    binned_returns = []
    for seed_returns in returns_array:
        seed_binned = []
        for i in range(n_bins):
            bin_mean = seed_returns[i*bin_size:(i+1)*bin_size].mean()
            seed_binned.append(bin_mean)
        binned_returns.append(seed_binned)
    
    binned_array = np.array(binned_returns)
    
    # compute mean ± IQR across seeds
    mean = binned_array.mean(axis=0)
    q1 = np.percentile(binned_array, 25, axis=0)
    q3 = np.percentile(binned_array, 75, axis=0)
    
    bin_centers = np.arange(n_bins) * bin_size + bin_size // 2
    
    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_centers, mean, label='mean return', linewidth=2, color='C0', alpha=0.9)
    ax.fill_between(bin_centers, q1, q3, alpha=0.25, label='IQR', color='C0')
    ax.set_xlabel('episode')
    ax.set_ylabel(f'mean return (per {bin_size} episodes)')
    ax.set_title(f'{algo_name.upper()} on {env_name.title()} (Mean ± IQR)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # add reference lines for cartpole
    if env_name == 'cartpole':
        ax.axhline(y=195, color='green', linestyle=':', alpha=0.7, label='solved (195)')
        ax.axhline(y=500, color='red', linestyle=':', alpha=0.5, label='max (500)')
        ax.legend()
    
    # save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{algo_name}_{env_name}_learning_curve.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved plot: {output_path}")


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


def compute_episodes_to_threshold(returns: Any, threshold: float) -> int:
    """compute sample efficiency: episodes needed to reach return threshold
    
    report requirement: compare sarsa vs q-learning on sample efficiency
    returns first episode where moving average (window=100) exceeds threshold
    """
    returns = np.asarray(returns)  # ensure numpy array
    if len(returns) < 100:
        return len(returns)  # not enough data
    
    # smooth with 100-episode moving average
    smoothed = pd.Series(returns).rolling(100, min_periods=1).mean()
    
    # find first episode where smoothed return >= threshold
    mask = smoothed >= threshold
    if mask.any():
        return int(mask.idxmax()) + 1  # +1 for 1-indexed episodes
    else:
        return len(returns)  # never reached threshold


def compute_learning_variability(returns: Any, window: int = 100) -> float:
    """compute learning stability: coefficient of variation of last episodes
    
    report requirement: assess stability differences between sarsa/q-learning
    lower cv = more stable learning (less bouncy)
    """
    returns = np.asarray(returns)  # ensure numpy array
    if len(returns) < window:
        window = len(returns)
    
    last_window = returns[-window:]
    mean_return = np.mean(last_window)
    std_return = np.std(last_window)
    
    # coefficient of variation: cv = std / |mean|
    cv = std_return / (abs(mean_return) + 1e-10)
    return float(cv)


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
        episodes_to_threshold_list = []  # sample efficiency: episodes to reach performance
        learning_variability_list = []  # stability: cv of last 100 episodes
        run_timestamps = []  # track when experiments were run
        
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
                    
                    # report-required metrics: sample efficiency and stability
                    # compute from raw CSV data for accurate threshold tracking
                    csv_file = json_file.with_suffix('.csv')
                    if csv_file.exists():
                        try:
                            df = pd.read_csv(csv_file)
                            if 'episode_return' in df.columns:
                                returns = df['episode_return'].values
                                
                                # sample efficiency: episodes to threshold
                                # threshold = 0.0 for blackjack (win rate), 195 for cartpole (solve threshold)
                                threshold = 195 if env_name == 'cartpole' else 0.0
                                eps_to_thresh = compute_episodes_to_threshold(returns, threshold)
                                episodes_to_threshold_list.append(eps_to_thresh)
                                
                                # stability: learning variability (cv)
                                cv = compute_learning_variability(returns, window=100)
                                learning_variability_list.append(cv)
                        except Exception:
                            pass
                    
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
                    
                    # track timestamps for reproducibility reporting
                    created = exp.get('created_at_utc', '')
                    if created:
                        run_timestamps.append(created)
            
            except Exception as e:
                print(f"warning: failed to process {json_file}: {e}")
                continue
        
        if not mean_returns:
            continue
        
        # generate learning curve plot (don't store data)
        figures_dir = results_dir.parent / 'figures'
        generate_learning_curve_plot(env_dir, env_name, algo_name, figures_dir, pattern=f'{algo_name}_{env_name}_seed*.csv')
        
        # report requirements: mean + IQR aggregation (summary stats only)
        env_summary = {
            'mean_return': compute_statistics(mean_returns),
            'last_10_episodes_return': compute_statistics(last_10_returns),
            'episodes_to_threshold': compute_statistics(episodes_to_threshold_list) if episodes_to_threshold_list else {},
            'learning_variability_cv': compute_statistics(learning_variability_list) if learning_variability_list else {},
            'wall_time_seconds': compute_statistics(wall_times),
            'seeds': sorted(seeds_used),
            'num_seeds': len(seeds_used),
            'hyperparameters': hyperparams,
            'run_timestamps': {
                'first': min(run_timestamps) if run_timestamps else None,
                'last': max(run_timestamps) if run_timestamps else None,
            },
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
        run_timestamps = []  # track when experiments were run
        
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
                    
                    # track timestamps for reproducibility reporting
                    exp = meta.get('experiment', {})
                    created = exp.get('created_at_utc', '')
                    if created:
                        run_timestamps.append(created)
            
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
            'run_timestamps': {
                'first': min(run_timestamps) if run_timestamps else None,
                'last': max(run_timestamps) if run_timestamps else None,
            },
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
                if env_data.get('episodes_to_threshold'):
                    print(f"      episodes_to_threshold: {env_data['episodes_to_threshold']['mean']:.0f}")
                if env_data.get('learning_variability_cv'):
                    print(f"      learning_variability (cv): {env_data['learning_variability_cv']['mean']:.3f}")
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

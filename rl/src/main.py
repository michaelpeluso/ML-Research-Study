#!/usr/bin/env python3
"""
single command entrypoint for all rl experiments
usage: python src/main.py

all parameters configured in src/config/default.yaml
no command-line arguments needed!
"""
import os, sys
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

import sys
from pathlib import Path
import yaml
from typing import Dict, Any, Optional, List, Tuple
import traceback
from multiprocessing import Pool, cpu_count
 
# unified experiment runner
from experiments.run_experiment import run_experiment as run_unified_experiment
# aggregate master summaries after experiments
from utils.aggregation import create_master_summaries
# generate comprehensive report plots
from utils.generate_report_plots import generate_all_plots


def clean_previous_results(
    results_dir: Path,
    experiments: List[Tuple[str, str]],
    seeds: List[int],
    seeds_vi_pi: List[int]
) -> None:
    """remove previous experiment results to prevent data accumulation bug
    
    CRITICAL: old results from different seed ranges would otherwise accumulate
    and pollute the master_summary.json aggregation.
    
    only cleans data for experiments that will be run (algo/env combinations).
    """
    raw_dir = results_dir / 'raw'
    if not raw_dir.exists():
        return
    
    print("\n--- Cleaning previous results ---")
    cleaned_count = 0
    
    for algo, env in experiments:
        algo_env_dir = raw_dir / algo / env
        if not algo_env_dir.exists():
            continue
        
        # determine which seeds this experiment uses
        exp_seeds = seeds_vi_pi if algo in ['vi', 'pi'] else seeds
        
        # find ALL existing result files for this algo/env
        existing_csvs = list(algo_env_dir.glob(f'{algo}_{env}_seed*.csv'))
        existing_jsons = list(algo_env_dir.glob(f'{algo}_{env}_seed*.json'))
        
        # delete files that won't be overwritten by current run
        # (i.e., seeds not in current seed list)
        for f in existing_csvs + existing_jsons:
            try:
                seed_str = f.stem.split('seed')[-1]
                file_seed = int(seed_str)
                if file_seed not in exp_seeds:
                    f.unlink()
                    cleaned_count += 1
            except (ValueError, IndexError):
                # can't parse seed, delete to be safe
                f.unlink()
                cleaned_count += 1
        
        # also delete master_summary.json (will be regenerated)
        master_summary = raw_dir / algo / 'master_summary.json'
        if master_summary.exists():
            master_summary.unlink()
    
    if cleaned_count > 0:
        print(f"  removed {cleaned_count} stale result files")
    else:
        print("  no stale files found")
    print()


def load_config(config_path: Optional[Path] = None) -> dict:
    """load yaml configuration file"""
    if config_path is None:
        config_path = Path(__file__).parent / 'config' / 'default.yaml'

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(algo: str, env: str, config: Dict[str, Any], seed: int) -> int:
    """run experiment by directly calling the training function"""
    print(f"\n{'='*60}")
    print(f"running: {algo} on {env} (seed={seed})")
    print('='*60)
    
    try:
        # dispatch to unified experiment runner which returns a status
        status = run_unified_experiment(algo, env, config, seed)
        if status is None or status == 'success':
            return 0
        if status == 'skipped':
            print(f"skipped: {algo} on {env} (seed={seed})")
            return 2
        if status == 'failed':
            return 1
        # unknown status: treat as failure
        print(f"unknown status returned: {status}")
        return 1
    except ValueError as e:
        print(f"error: {e}")
        return 1
    except Exception as e:
        print(f"error running experiment: {e}")
        traceback.print_exc()
        return 1


def run_experiment_wrapper(args):
    """wrapper for parallel execution - unpacks tuple and runs experiment"""
    algo, env, config, seed = args
    return (algo, env, seed, run_experiment(algo, env, config, seed))


def main():
    # load config (no argparse needed!)
    config_path = Path(__file__).parent / 'config' / 'default.yaml'
    
    if not config_path.exists():
        print(f"error: config file not found: {config_path}")
        print("please create src/config/default.yaml")
        return 1
    
    config = load_config(config_path)
    
    # apply smoke test mode if enabled
    smoke_test = config.get('smoke_test', False)
    if smoke_test:
        config['seeds'] = config.get('seeds_smoke', [0, 1])
        if 'hyperparameters' in config:
            prod_episodes = config['hyperparameters'].get('episodes', 10000)
            smoke_episodes = config['hyperparameters'].get('episodes_smoke', 2000)
            config['hyperparameters']['episodes'] = smoke_episodes
            
            # scale epsilon_decay_episodes proportionally for smoke test
            # otherwise epsilon barely decays during short smoke test runs
            scale_factor = smoke_episodes / prod_episodes
            
            for algo in ['sarsa', 'qlearning', 'vi', 'pi']:
                if algo in config:
                    for key in list(config[algo].keys()):
                        if 'epsilon_decay_episodes' in key:
                            original = config[algo][key]
                            scaled = int(original * scale_factor)
                            config[algo][key] = scaled
                            
        print("\n[SMOKE TEST] using reduced seeds and episodes")
        print(f"    epsilon decay scaled to {scale_factor:.1%} of production") # type: ignore
        print("    set smoke_test: false in config/default.yaml for production runs\n")
    
    # extract experiment settings from config
    exp_config = config.get('experiments', {})
    run_all = exp_config.get('run_all', True)
    
    # determine which experiments to run
    experiments = []
    
    if run_all:
        # run all combinations specified in config
        algorithms = exp_config.get('algorithms', ['sarsa', 'qlearning'])
        environments = exp_config.get('environments', ['cartpole', 'blackjack'])
        for algo in algorithms:
            for env in environments:
                experiments.append((algo, env))
    else:
        # run only specified combinations
        algorithms = exp_config.get('algorithms', [])
        environments = exp_config.get('environments', [])
        if not algorithms or not environments:
            print("error: when run_all=false, must specify algorithms and environments in config")
            return 1
        for algo in algorithms:
            for env in environments:
                experiments.append((algo, env))
    
    # get seeds from config (already adjusted for smoke test above)
    seeds = config.get('seeds', [0, 1, 2, 3, 4])
    seeds_vi_pi = config.get('seeds_vi_pi', [0, 1, 2, 3, 4])
    
    # clean previous results to prevent data accumulation bug
    # set clean_previous_results: false in config to disable
    results_dir = Path(__file__).parent.parent / 'results'
    if config.get('clean_previous_results', True):
        clean_previous_results(results_dir, experiments, seeds, seeds_vi_pi)
    
    # parallelization settings
    n_jobs = config.get('n_jobs', 1)  # default: sequential
    if n_jobs == -1:
        n_jobs = cpu_count()  # use all cores
    elif n_jobs < -1:
        n_jobs = max(1, cpu_count() + n_jobs + 1)  # e.g., -2 = all but 1
    
    parallel_mode = n_jobs > 1
    
    print(f"\n{'='*60}")
    print("RL EXPERIMENT RUNNER")
    print('='*60)
    print(f"mode: {'SMOKE TEST' if smoke_test else 'PRODUCTION'}")
    print(f"config file: {config_path}")
    print(f"experiments: {len(experiments)}")
    print(f"SARSA/Q-Learning seeds: {len(seeds)} (required: 30-50)")
    print(f"VI/PI seeds: {len(seeds_vi_pi)} (optional for robustness)")
    print(f"total runs: {sum(len(seeds) if algo in ['sarsa', 'qlearning'] else len(seeds_vi_pi) for algo, _ in experiments)}")
    print(f"parallelization: {'YES (' + str(n_jobs) + ' workers)' if parallel_mode else 'NO (sequential)'}")
    print('='*60)
    
    # prepare experiment tasks
    tasks = []
    for algo, env in experiments:
        # use appropriate seed list based on algorithm type
        algo_seeds = seeds_vi_pi if algo in ['vi', 'pi'] else seeds
        for seed in algo_seeds:
            tasks.append((algo, env, config, seed))
    
    # run experiments (parallel or sequential)
    failed_runs = []
    successful_runs = []
    skipped_runs = []
    
    if parallel_mode:
        print(f"\n[PARALLEL] running {len(tasks)} experiments with {n_jobs} workers...")
        with Pool(processes=n_jobs) as pool:
            results = pool.map(run_experiment_wrapper, tasks)
        
        # process results
        for algo, env, seed, exit_code in results:
            if exit_code == 0:
                successful_runs.append((algo, env, seed))
            elif exit_code == 2:
                skipped_runs.append((algo, env, seed))
            else:
                failed_runs.append((algo, env, seed))
    else:
        print(f"\n[SEQUENTIAL] running {len(tasks)} experiments...")
        for algo, env, config_task, seed in tasks:
            exit_code = run_experiment(algo, env, config_task, seed)
            if exit_code == 0:
                successful_runs.append((algo, env, seed))
            elif exit_code == 2:
                skipped_runs.append((algo, env, seed))
            else:
                failed_runs.append((algo, env, seed))
    
    # count runs per algorithm (successful/skipped/failed)
    algo_counts = {}
    for algo, env, seed in successful_runs:
        algo_counts.setdefault(algo, {'success': 0, 'skipped': 0, 'failed': 0})
        algo_counts[algo]['success'] += 1
    for algo, env, seed in skipped_runs:
        algo_counts.setdefault(algo, {'success': 0, 'skipped': 0, 'failed': 0})
        algo_counts[algo]['skipped'] += 1
    for algo, env, seed in failed_runs:
        algo_counts.setdefault(algo, {'success': 0, 'skipped': 0, 'failed': 0})
        algo_counts[algo]['failed'] += 1
    
    # summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print('='*60)
    total_expected = sum(len(seeds) if algo in ['sarsa', 'qlearning'] else len(seeds_vi_pi) for algo, _ in experiments)
    print(f"total runs: {total_expected}")
    print(f"successful: {len(successful_runs)}")
    print(f"failed: {len(failed_runs)}")
    
    if algo_counts:
        print(f"\nruns per algorithm:")
        for algo in sorted(algo_counts.keys()):
            counts = algo_counts[algo]
            print(f"  {algo}: {counts['success']} success, {counts['skipped']} skipped, {counts['failed']} failed")
    
    if failed_runs:
        print("\nfailed runs:")
        for algo, env, seed in failed_runs:
            print(f"  - {algo} on {env} (seed={seed})")
        return 1
    
    print("\n[SUCCESS] all experiments completed successfully!")
    print(f"results saved to: results/raw/")
    print(f"figures saved to: results/figures/")
    # create master summary JSONs (single source of truth for report)
    try:
        print("\ncreating master summaries (this may take a moment)...")
        # by default looks in results/raw
        create_master_summaries()
        print("master summaries created: results/raw/*/master_summary.json")
    except Exception as e:
        print(f"warning: failed to create master summaries: {e}")
    
    # generate comprehensive report plots
    try:
        print("\ngenerating comprehensive report plots...")
        generate_all_plots()
        print("all report plots generated successfully!")
    except Exception as e:
        print(f"warning: failed to generate report plots: {e}")
        traceback.print_exc()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

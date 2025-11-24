#!/usr/bin/env python3
"""
single command entrypoint for all rl experiments
usage: python src/main.py

all parameters configured in src/config/default.yaml
no command-line arguments needed!
"""
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

import sys
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
import traceback
 
# unified experiment runner
from experiments.run_experiment import run_experiment as run_unified_experiment
# aggregate master summaries after experiments
from utils.aggregation import create_master_summaries


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
            config['hyperparameters']['episodes'] = config['hyperparameters'].get('episodes_smoke', 100)
        print("\n⚠️  SMOKE TEST MODE ENABLED - using reduced seeds and episodes")
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
    
    print(f"\n{'='*60}")
    print("RL EXPERIMENT RUNNER")
    print('='*60)
    print(f"mode: {'SMOKE TEST' if smoke_test else 'PRODUCTION'}")
    print(f"config file: {config_path}")
    print(f"experiments: {len(experiments)}")
    print(f"seeds per experiment: {len(seeds)}")
    print(f"total runs: {len(experiments) * len(seeds)}")
    print('='*60)
    
    # run experiments
    failed_runs = []
    successful_runs = []
    skipped_runs = []

    for algo, env in experiments:
        for seed in seeds:
            exit_code = run_experiment(algo, env, config, seed)
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
    print(f"total runs: {len(experiments) * len(seeds)}")
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
    return 0


if __name__ == '__main__':
    sys.exit(main())

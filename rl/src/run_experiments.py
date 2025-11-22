#!/usr/bin/env python3
"""
single command entrypoint for all rl experiments
usage: python src/run_experiments.py --all
"""
import sys
from pathlib import Path
import argparse
import yaml
import subprocess
from typing import List

# add src to path
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: Path) -> dict:
    """load yaml configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(script_path: Path, args: List[str]) -> int:
    """run experiment script and return exit code"""
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*60}")
    print(f"running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='run rl experiments')
    parser.add_argument('--all', action='store_true', help='run all experiments')
    parser.add_argument('--algorithm', type=str, choices=['sarsa', 'qlearning', 'vi', 'pi'], help='specific algorithm')
    parser.add_argument('--environment', type=str, choices=['blackjack', 'cartpole'], help='specific environment')
    parser.add_argument('--seeds', type=int, default=5, help='number of seeds to run')
    parser.add_argument('--config', type=str, default='src/config/default.yaml', help='config file path')
    args = parser.parse_args()
    
    # load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config file not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    
    # determine which experiments to run
    experiments = []
    
    if args.all:
        # run all combinations
        algorithms = ['sarsa', 'qlearning']
        environments = ['blackjack', 'cartpole']
        for algo in algorithms:
            for env in environments:
                experiments.append((algo, env))
    elif args.algorithm and args.environment:
        experiments.append((args.algorithm, args.environment))
    else:
        print("error: specify --all or both --algorithm and --environment")
        return 1
    
    # get seeds from config or command line
    if 'seeds' in config:
        seeds = config['seeds'][:args.seeds]
    else:
        seeds = list(range(args.seeds))
    
    print(f"\nrunning {len(experiments)} experiment(s) with {len(seeds)} seed(s) each")
    print(f"total runs: {len(experiments) * len(seeds)}")
    
    # run experiments
    experiments_dir = Path(__file__).parent / 'experiments'
    failed_runs = []
    
    for algo, env in experiments:
        script_name = f"{algo}_{env}.py"
        script_path = experiments_dir / script_name
        
        if not script_path.exists():
            print(f"\nwarning: script not found: {script_path}")
            print("skipping this experiment...")
            continue
        
        for seed in seeds:
            exit_code = run_experiment(script_path, ['--seed', str(seed)])
            if exit_code != 0:
                failed_runs.append((algo, env, seed))
    
    # summary
    print(f"\n{'='*60}")
    print("experiment summary")
    print('='*60)
    print(f"total runs: {len(experiments) * len(seeds)}")
    print(f"successful: {len(experiments) * len(seeds) - len(failed_runs)}")
    print(f"failed: {len(failed_runs)}")
    
    if failed_runs:
        print("\nfailed runs:")
        for algo, env, seed in failed_runs:
            print(f"  - {algo} on {env} (seed={seed})")
        return 1
    
    print("\nall experiments completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

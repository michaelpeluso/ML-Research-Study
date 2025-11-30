"""staged hyperparameter search: coarse random search with early stopping

run separately before main.py to find optimal hyperparameters:
  python hyperparam_search.py sarsa blackjack
  python hyperparam_search.py --ablation --seeds 3 --n_jobs -2

then update config/default.yaml with best settings
"""
import os
import sys
from pathlib import Path

# add src to path
script_dir = Path(__file__).parent.parent
os.chdir(script_dir)
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
import time
import yaml
from multiprocessing import Pool, cpu_count


def _resolve_n_jobs(n_jobs: int) -> int:
    """resolve n_jobs to actual worker count (same logic as main.py)"""
    if n_jobs == -1:
        return cpu_count()
    elif n_jobs < -1:
        return max(1, cpu_count() + n_jobs + 1)  # e.g., -2 = all but 1
    return n_jobs


def _evaluate_candidate_wrapper(args: Tuple) -> Dict:
    """wrapper for parallel execution - unpacks tuple"""
    i, config, algo, env, n_pilot_episodes, base_seed, base_config = args
    return _evaluate_candidate(i, config, algo, env, n_pilot_episodes, base_seed, base_config)


def _evaluate_candidate(i: int, config: Dict[str, Any], algo: str, env: str, 
                         n_pilot_episodes: int, base_seed: int, base_config: Dict) -> Dict:
    """worker function for parallel candidate evaluation"""
    from experiments.run_experiment import run_experiment
    from utils.seeding import set_all_seeds
    
    candidate_seed = base_seed + i
    
    # merge hyperparams into base config
    pilot_config = base_config.copy()
    pilot_config['hyperparameters'] = pilot_config.get('hyperparameters', {}).copy()
    pilot_config['hyperparameters'].update({
        'alpha': config['alpha'],
        'gamma': config['gamma'],
        'epsilon': config.get('epsilon_start', 1.0),  # default to 1.0 if not provided
        'episodes': n_pilot_episodes,
    })
    
    # algorithm-specific settings - must set BOTH base and env-specific keys
    # because get_hyperparams applies env overrides (e.g., alpha_blackjack)
    if algo == 'sarsa':
        pilot_config['sarsa'] = pilot_config.get('sarsa', {}).copy()
        pilot_config['sarsa']['alpha'] = config['alpha']
        pilot_config['sarsa']['gamma'] = config['gamma']
        pilot_config['sarsa']['epsilon_floor'] = config['epsilon_floor']
        pilot_config['sarsa']['epsilon_decay_episodes'] = config['epsilon_decay_episodes']
        # env-specific overrides (e.g., alpha_blackjack, alpha_cartpole)
        pilot_config['sarsa'][f'alpha_{env}'] = config['alpha']
        pilot_config['sarsa'][f'gamma_{env}'] = config['gamma']
        pilot_config['sarsa'][f'epsilon_floor_{env}'] = config['epsilon_floor']
        pilot_config['sarsa'][f'epsilon_decay_episodes_{env}'] = config['epsilon_decay_episodes']
    elif algo == 'qlearning':
        pilot_config['qlearning'] = pilot_config.get('qlearning', {}).copy()
        pilot_config['qlearning']['alpha'] = config['alpha']
        pilot_config['qlearning']['gamma'] = config['gamma']
        pilot_config['qlearning']['epsilon_floor'] = config['epsilon_floor']
        pilot_config['qlearning']['epsilon_decay_episodes'] = config['epsilon_decay_episodes']
        # env-specific overrides
        pilot_config['qlearning'][f'alpha_{env}'] = config['alpha']
        pilot_config['qlearning'][f'gamma_{env}'] = config['gamma']
        pilot_config['qlearning'][f'epsilon_floor_{env}'] = config['epsilon_floor']
        pilot_config['qlearning'][f'epsilon_decay_episodes_{env}'] = config['epsilon_decay_episodes']
    
    set_all_seeds(candidate_seed)
    status = run_experiment(algo, env, pilot_config, candidate_seed)
    
    if status != 'success':
        return {**config, 'mean_return': -np.inf, 'final_return': -np.inf, 'rank': -1}
    
    # read results
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / 'results' / 'raw' / algo / env / f'{algo}_{env}_seed{candidate_seed}.csv'
    
    if not csv_path.exists():
        return {**config, 'mean_return': -np.inf, 'final_return': -np.inf, 'rank': -1}
    
    df = pd.read_csv(csv_path)
    if 'episode_return' not in df.columns or len(df) == 0:
        return {**config, 'mean_return': -np.inf, 'final_return': -np.inf, 'rank': -1}
    
    mean_return = df['episode_return'].mean()
    final_return = df['episode_return'].iloc[-1] if len(df) > 0 else -np.inf
    
    return {**config, 'mean_return': mean_return, 'final_return': final_return, 'rank': -1}


def sample_hyperparams(n_candidates: int, env: str = 'blackjack', seed: int = 42) -> List[Dict[str, Any]]:
    """sample n hyperparameter configurations from log-scaled ranges
    
    ranges based on report recommendations:
    - alpha: 10^U(-3, 0) = [0.001, 1.0]
    - gamma: [0.95, 0.999]
    - epsilon_floor: [0.005, 0.05]
    - epsilon_decay_episodes: env-specific (50% of production episodes)
      - blackjack: [10000, 40000] (production: 50k)
      - cartpole: [2000, 8000] (production: 10k)
    """
    np.random.seed(seed)
    
    candidates = []
    for i in range(n_candidates):
        # epsilon_decay_episodes scales with production episode count
        # explore range of 20-80% of production episodes
        if env == 'blackjack':
            decay_min, decay_max = 10000, 40000  # 20-80% of 50k
        else:  # cartpole
            decay_min, decay_max = 2000, 8000   # 20-80% of 10k
            
        config = {
            'id': i,
            'alpha': 10 ** np.random.uniform(-3, 0),  # log-scaled: 0.001 to 1.0
            'gamma': np.random.uniform(0.95, 0.999),
            'epsilon_start': 1.0,  # always start at 1.0
            'epsilon_floor': np.random.uniform(0.005, 0.05),
            'epsilon_decay_episodes': int(np.random.uniform(decay_min, decay_max)),
            # optional: q-value initialization
            'q_init': np.random.choice(['zeros', 'optimistic']),
            'optimistic_value': np.random.uniform(0.1, 1.0) if np.random.rand() > 0.5 else 0.5,
        }
        candidates.append(config)
    
    return candidates


def run_pilot_episode(algo: str, env: str, config: Dict[str, Any], 
                      n_pilot_episodes: int, seed: int, base_config: Dict) -> Tuple[float, float]:
    """run pilot episodes for early evaluation
    
    returns: (mean_return, final_return) over pilot episodes
    """
    from experiments.run_experiment import run_experiment
    from utils.seeding import set_all_seeds
    
    # merge hyperparams into base config
    pilot_config = base_config.copy()
    pilot_config['hyperparameters'] = pilot_config.get('hyperparameters', {}).copy()
    pilot_config['hyperparameters'].update({
        'alpha': config['alpha'],
        'gamma': config['gamma'],
        'epsilon': config.get('epsilon_start', 1.0),  # default to 1.0 if not provided
        'episodes': n_pilot_episodes,
    })
    
    # algorithm-specific settings - must set BOTH base and env-specific keys
    # because get_hyperparams applies env overrides (e.g., alpha_blackjack)
    if algo == 'sarsa':
        pilot_config['sarsa'] = pilot_config.get('sarsa', {}).copy()
        pilot_config['sarsa']['alpha'] = config['alpha']
        pilot_config['sarsa']['gamma'] = config['gamma']
        pilot_config['sarsa']['epsilon_floor'] = config['epsilon_floor']
        pilot_config['sarsa']['epsilon_decay_episodes'] = config['epsilon_decay_episodes']
        # env-specific overrides (e.g., alpha_blackjack, alpha_cartpole)
        pilot_config['sarsa'][f'alpha_{env}'] = config['alpha']
        pilot_config['sarsa'][f'gamma_{env}'] = config['gamma']
        pilot_config['sarsa'][f'epsilon_floor_{env}'] = config['epsilon_floor']
        pilot_config['sarsa'][f'epsilon_decay_episodes_{env}'] = config['epsilon_decay_episodes']
    elif algo == 'qlearning':
        pilot_config['qlearning'] = pilot_config.get('qlearning', {}).copy()
        pilot_config['qlearning']['alpha'] = config['alpha']
        pilot_config['qlearning']['gamma'] = config['gamma']
        pilot_config['qlearning']['epsilon_floor'] = config['epsilon_floor']
        pilot_config['qlearning']['epsilon_decay_episodes'] = config['epsilon_decay_episodes']
        # env-specific overrides
        pilot_config['qlearning'][f'alpha_{env}'] = config['alpha']
        pilot_config['qlearning'][f'gamma_{env}'] = config['gamma']
        pilot_config['qlearning'][f'epsilon_floor_{env}'] = config['epsilon_floor']
        pilot_config['qlearning'][f'epsilon_decay_episodes_{env}'] = config['epsilon_decay_episodes']
    
    # run experiment
    set_all_seeds(seed)
    status = run_experiment(algo, env, pilot_config, seed)
    
    if status != 'success':
        return -np.inf, -np.inf
    
    # read results from expected location (go up to project root)
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / 'results' / 'raw' / algo / env / f'{algo}_{env}_seed{seed}.csv'
    if not csv_path.exists():
        return -np.inf, -np.inf
    
    df = pd.read_csv(csv_path)
    if 'episode_return' not in df.columns or len(df) == 0:
        return -np.inf, -np.inf
    
    mean_return = df['episode_return'].mean()
    final_return = df['episode_return'].iloc[-1] if len(df) > 0 else -np.inf
    
    return mean_return, final_return


def stage1_coarse_search(
    algo: str,
    env: str,
    n_candidates: int = 20,
    n_pilot_episodes: int = 100,
    seed: int = 42,
    output_dir: Path = Path('results/hyperparam_search'),
    n_jobs: int = 1
) -> List[Dict[str, Any]]:
    """stage 1: coarse random search with early stopping
    
    args:
        algo: algorithm name ('sarsa' or 'qlearning')
        env: environment name ('blackjack' or 'cartpole')
        n_candidates: number of random configurations to sample
        n_pilot_episodes: episodes per candidate (early stopping budget)
        seed: random seed for reproducibility
        output_dir: where to save search results
        n_jobs: number of parallel workers (1=sequential, -1=all cores, -2=all-1)
    
    returns:
        top 50% candidates ranked by interim return
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: COARSE RANDOM SEARCH")
    print('='*60)
    print(f"algorithm: {algo}")
    print(f"environment: {env}")
    print(f"candidates: {n_candidates}")
    print(f"pilot episodes: {n_pilot_episodes}")
    print(f"seed: {seed}")
    print(f"parallel workers: {n_jobs}")
    print()
    
    # output directory for results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'hyperparam_search'
    
    # load base config
    config_path = Path('config/default.yaml')
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    # sample candidates (env-specific ranges for epsilon_decay_episodes)
    print("sampling hyperparameter configurations...")
    candidates = sample_hyperparams(n_candidates, env, seed)
    
    # evaluate candidates (parallel or sequential)
    start_time = time.time()
    actual_workers = _resolve_n_jobs(n_jobs)
    
    if actual_workers == 1:
        # sequential execution with progress output
        results = []
        for i, config in enumerate(candidates):
            print(f"[{i+1}/{n_candidates}] evaluating config {config['id']}...")
            print(f"  alpha={config['alpha']:.4f}, gamma={config['gamma']:.4f}, "
                  f"eps_floor={config['epsilon_floor']:.4f}")
            
            result = _evaluate_candidate(i, config, algo, env, n_pilot_episodes, seed, base_config)
            results.append(result)
            print(f"  -> mean return: {result['mean_return']:.3f}, final return: {result['final_return']:.3f}")
    else:
        # parallel execution using multiprocessing.Pool (same as main.py)
        print(f"running {n_candidates} candidates in parallel with {actual_workers} workers...")
        tasks = [(i, config, algo, env, n_pilot_episodes, seed, base_config) 
                 for i, config in enumerate(candidates)]
        with Pool(processes=actual_workers) as pool:
            results = pool.map(_evaluate_candidate_wrapper, tasks)
    
    elapsed = time.time() - start_time
    
    # sort by mean return (primary) and final return (tiebreaker)
    results_sorted = sorted(
        results, 
        key=lambda x: (x['mean_return'], x['final_return']), 
        reverse=True
    )
    
    # assign ranks
    for rank, r in enumerate(results_sorted):
        r['rank'] = rank + 1
    
    # keep top 50% (successive halving)
    n_keep = max(1, n_candidates // 2)
    top_candidates = results_sorted[:n_keep]
    
    print()
    print("="*60)
    print(f"STAGE 1 COMPLETE ({elapsed:.1f}s)")
    print("="*60)
    print(f"evaluated: {n_candidates} candidates")
    print(f"kept: {n_keep} (top 50%)")
    print()
    print("top 3 configurations:")
    for i, c in enumerate(top_candidates[:3]):
        print(f"  {i+1}. alpha={c['alpha']:.4f}, gamma={c['gamma']:.4f}, "
              f"eps_floor={c['epsilon_floor']:.4f} -> return={c['mean_return']:.3f}")
    
    # save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f'stage1_{algo}_{env}_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'algorithm': algo,
            'environment': env,
            'n_candidates': n_candidates,
            'n_pilot_episodes': n_pilot_episodes,
            'seed': seed,
            'elapsed_seconds': elapsed,
            'all_results': results_sorted,
            'top_candidates': top_candidates,
        }, f, indent=2)
    
    print(f"\nresults saved: {results_path}")
    
    return top_candidates


def stage2_successive_halving(algo: str, env: str, stage1_results_path: Path, 
                               n_extended_episodes: int, seed: int = 42) -> list:
    """stage 2: successive halving - evaluate top candidates with more episodes
    
    takes top 10 from stage 1, runs with 4x more episodes, keeps top 5
    this refines the search by giving candidates more time to prove themselves
    """
    
    print("\n" + "="*60)
    print("STAGE 2: SUCCESSIVE HALVING")
    print("="*60)
    print(f"algorithm: {algo}")
    print(f"environment: {env}")
    print(f"extended episodes: {n_extended_episodes}")
    print(f"seed: {seed}")
    print()
    
    # load stage 1 results
    with open(stage1_results_path, 'r') as f:
        stage1_data = json.load(f)
    
    top10_configs = stage1_data['top_candidates']  # already top 10
    print(f"evaluating {len(top10_configs)} candidates from stage 1...")
    
    # load base config
    config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # evaluate each candidate with extended episodes
    results = []
    start_time = time.time()
    
    for i, config in enumerate(top10_configs):
        print(f"[{i+1}/{len(top10_configs)}] evaluating config {i}...")
        print(f"  alpha={config['alpha']:.4f}, gamma={config['gamma']:.4f}, "
              f"eps_floor={config['epsilon_floor']:.4f}")
        
        # unique seed for this candidate
        candidate_seed = seed + 100 + i  # offset by 100 to avoid stage 1 collisions
        
        mean_return, final_return = run_pilot_episode(
            algo, env, config, n_extended_episodes, candidate_seed, base_config
        )
        
        results.append({
            'config_id': i,
            'alpha': config['alpha'],
            'gamma': config['gamma'],
            'epsilon_floor': config['epsilon_floor'],
            'epsilon_decay_episodes': config['epsilon_decay_episodes'],
            'mean_return': mean_return,
            'final_return': final_return,
            'stage1_mean_return': config['mean_return'],  # track improvement
        })
        
        print(f"  -> mean return: {mean_return:.3f}, final return: {final_return:.3f}")
        print(f"     (stage 1 baseline: {config['mean_return']:.3f})")
    
    elapsed = time.time() - start_time
    
    # sort by mean return and keep top 5 (50% retention)
    results_sorted = sorted(results, key=lambda x: x['mean_return'], reverse=True)
    top5_candidates = results_sorted[:5]
    
    print(f"\n{'='*60}")
    print(f"STAGE 2 COMPLETE ({elapsed:.1f}s)")
    print("="*60)
    print(f"evaluated: {len(results)} candidates")
    print(f"kept: {len(top5_candidates)} (top 50%)")
    print()
    print("top 3 configurations:")
    for i, c in enumerate(top5_candidates[:3]):
        improvement = c['mean_return'] - c['stage1_mean_return']
        print(f"  {i+1}. alpha={c['alpha']:.4f}, gamma={c['gamma']:.4f}, "
              f"eps_floor={c['epsilon_floor']:.4f} -> return={c['mean_return']:.3f} "
              f"(+{improvement:.3f})")
    
    # save results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'hyperparam_search'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f'stage2_{algo}_{env}_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'algorithm': algo,
            'environment': env,
            'n_extended_episodes': n_extended_episodes,
            'seed': seed,
            'elapsed_seconds': elapsed,
            'stage1_results_path': str(stage1_results_path),
            'all_results': results_sorted,
            'top_candidates': top5_candidates,
        }, f, indent=2)
    
    print(f"\nresults saved: {results_path}")
    
    return top5_candidates


def stage3_local_refinement(algo: str, env: str, stage2_results_path: Path,
                            n_refinement_episodes: int, n_refinement_samples: int = 12,
                            seed: int = 42) -> list:
    """stage 3: local refinement - fine-tune top 3 configs with sensitivity analysis
    
    takes top 3 from stage 2, creates perturbations around each:
    - alpha: ±2× (e.g., if alpha=0.1, test [0.05, 0.1, 0.2])
    - epsilon_decay_episodes: ±25%
    - one-at-a-time sensitivity to understand what matters vs noise
    """
    
    print("\n" + "="*60)
    print("STAGE 3: LOCAL REFINEMENT")
    print("="*60)
    print(f"algorithm: {algo}")
    print(f"environment: {env}")
    print(f"refinement episodes: {n_refinement_episodes}")
    print(f"samples per champion: {n_refinement_samples // 3} (alpha+decay variations)")
    print(f"seed: {seed}")
    print()
    
    # load stage 2 results
    with open(stage2_results_path, 'r') as f:
        stage2_data = json.load(f)
    
    top3_configs = stage2_data['top_candidates'][:3]  # top 3 only
    print(f"refining top {len(top3_configs)} champions from stage 2...")
    
    # load base config
    config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # generate refinement candidates
    refinement_candidates = []
    candidate_id = 0
    
    for champion_idx, champion in enumerate(top3_configs):
        print(f"\nchampion {champion_idx+1}: alpha={champion['alpha']:.4f}, "
              f"gamma={champion['gamma']:.4f}, eps_floor={champion['epsilon_floor']:.4f}")
        
        # baseline (champion itself)
        refinement_candidates.append({
            'id': candidate_id,
            'champion_id': champion_idx,
            'variation': 'baseline',
            'alpha': champion['alpha'],
            'gamma': champion['gamma'],
            'epsilon_start': 1.0,  # always start at 1.0
            'epsilon_floor': champion['epsilon_floor'],
            'epsilon_decay_episodes': champion['epsilon_decay_episodes'],
        })
        candidate_id += 1
        
        # alpha variations: ±2×
        for alpha_mult in [0.5, 2.0]:
            refinement_candidates.append({
                'id': candidate_id,
                'champion_id': champion_idx,
                'variation': f'alpha_{alpha_mult}x',
                'alpha': champion['alpha'] * alpha_mult,
                'gamma': champion['gamma'],
                'epsilon_start': 1.0,
                'epsilon_floor': champion['epsilon_floor'],
                'epsilon_decay_episodes': champion['epsilon_decay_episodes'],
            })
            candidate_id += 1
        
        # decay horizon variations: ±25%
        for decay_mult in [0.75, 1.25]:
            refinement_candidates.append({
                'id': candidate_id,
                'champion_id': champion_idx,
                'variation': f'decay_{int(decay_mult*100)}pct',
                'alpha': champion['alpha'],
                'gamma': champion['gamma'],
                'epsilon_start': 1.0,
                'epsilon_floor': champion['epsilon_floor'],
                'epsilon_decay_episodes': int(champion['epsilon_decay_episodes'] * decay_mult),
            })
            candidate_id += 1
    
    print(f"\ngenerated {len(refinement_candidates)} refinement candidates")
    
    # evaluate each refinement candidate
    results = []
    start_time = time.time()
    
    for i, config in enumerate(refinement_candidates):
        print(f"\n[{i+1}/{len(refinement_candidates)}] evaluating {config['variation']} "
              f"(champion {config['champion_id']+1})...")
        print(f"  alpha={config['alpha']:.4f}, gamma={config['gamma']:.4f}, "
              f"eps_floor={config['epsilon_floor']:.4f}, decay_eps={config['epsilon_decay_episodes']}")
        
        # unique seed for this candidate
        candidate_seed = seed + 200 + i  # offset by 200 to avoid stage 1/2 collisions
        
        mean_return, final_return = run_pilot_episode(
            algo, env, config, n_refinement_episodes, candidate_seed, base_config
        )
        
        results.append({
            'config_id': config['id'],
            'champion_id': config['champion_id'],
            'variation': config['variation'],
            'alpha': config['alpha'],
            'gamma': config['gamma'],
            'epsilon_floor': config['epsilon_floor'],
            'epsilon_decay_episodes': config['epsilon_decay_episodes'],
            'mean_return': mean_return,
            'final_return': final_return,
        })
        
        print(f"  -> mean return: {mean_return:.3f}, final return: {final_return:.3f}")
    
    elapsed = time.time() - start_time
    
    # sort by mean return and identify best overall
    results_sorted = sorted(results, key=lambda x: x['mean_return'], reverse=True)
    best_config = results_sorted[0]
    
    # analyze sensitivity per champion
    sensitivity_analysis = {}
    for champion_idx in range(len(top3_configs)):
        champion_results = [r for r in results if r['champion_id'] == champion_idx]
        baseline = next(r for r in champion_results if r['variation'] == 'baseline')
        
        alpha_improvements = [
            r['mean_return'] - baseline['mean_return']
            for r in champion_results if 'alpha_' in r['variation']
        ]
        decay_improvements = [
            r['mean_return'] - baseline['mean_return']
            for r in champion_results if 'decay_' in r['variation']
        ]
        
        sensitivity_analysis[champion_idx] = {
            'baseline_return': baseline['mean_return'],
            'alpha_sensitivity': max(abs(x) for x in alpha_improvements) if alpha_improvements else 0,
            'decay_sensitivity': max(abs(x) for x in decay_improvements) if decay_improvements else 0,
        }
    
    print(f"\n{'='*60}")
    print(f"STAGE 3 COMPLETE ({elapsed:.1f}s)")
    print("="*60)
    print(f"evaluated: {len(results)} refinement candidates")
    print()
    print("sensitivity analysis (max deviation from baseline):")
    for champ_idx, sens in sensitivity_analysis.items():
        print(f"  champion {champ_idx+1}: baseline={sens['baseline_return']:.3f}, "
              f"alpha_sens={sens['alpha_sensitivity']:.3f}, "
              f"decay_sens={sens['decay_sensitivity']:.3f}")
    print()
    print("best refined configuration:")
    print(f"  champion {best_config['champion_id']+1}, variation: {best_config['variation']}")
    print(f"  alpha={best_config['alpha']:.4f}, gamma={best_config['gamma']:.4f}, "
          f"eps_floor={best_config['epsilon_floor']:.4f}")
    print(f"  mean_return={best_config['mean_return']:.3f}")
    
    # save results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'hyperparam_search'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f'stage3_{algo}_{env}_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'algorithm': algo,
            'environment': env,
            'n_refinement_episodes': n_refinement_episodes,
            'seed': seed,
            'elapsed_seconds': elapsed,
            'stage2_results_path': str(stage2_results_path),
            'all_results': results_sorted,
            'best_config': best_config,
            'sensitivity_analysis': sensitivity_analysis,
        }, f, indent=2)
    
    print(f"\nresults saved: {results_path}")
    
    return best_config


def _run_ablation_task_wrapper(args: Tuple) -> Dict:
    """wrapper for parallel execution - unpacks tuple"""
    algo, bins, seed, samples_per_sa, base_config = args
    return _run_ablation_task(algo, bins, seed, samples_per_sa, base_config)


def _run_ablation_task(algo: str, bins: List[int], seed: int, samples_per_sa: int, 
                        base_config: Dict) -> Dict:
    """worker function for parallel ablation study"""
    from experiments.run_experiment import run_vi, run_pi
    from utils.seeding import set_all_seeds
    
    # create config with this bin configuration
    test_config = base_config.copy()
    test_config['cartpole'] = test_config.get('cartpole', {}).copy()
    test_config['cartpole']['vi_bins'] = bins
    test_config['cartpole']['pi_bins'] = bins
    test_config['cartpole']['samples_per_sa'] = samples_per_sa
    
    set_all_seeds(seed)
    
    result = {
        'algo': algo,
        'bins': bins,
        'seed': seed,
        'status': 'failed',
        'iterations': -1,
        'wall_time': -1,
        'mean_return': -1,
        'mean_length': -1,
        'converged': False,
    }
    
    try:
        if algo == 'vi':
            status = run_vi('cartpole', test_config, seed)
        elif algo == 'pi':
            status = run_pi('cartpole', test_config, seed)
        else:
            return result
        
        if status != 'success':
            return result
        
        # read results
        project_root = Path(__file__).parent.parent.parent
        json_path = project_root / 'results' / 'raw' / algo / 'cartpole' / f'{algo}_cartpole_seed{seed}.json'
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                run_data = json.load(f)
            
            conv = run_data.get('results', {}).get('convergence', {})
            eval_data = run_data.get('results', {}).get('policy_evaluation', {})
            
            result['status'] = 'success'
            result['iterations'] = conv.get('iterations', -1)
            result['wall_time'] = conv.get('total_time', -1)
            result['converged'] = conv.get('converged', False)
            result['mean_return'] = eval_data.get('mean_return', -1)
            result['mean_length'] = eval_data.get('mean_length', -1)
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def discretization_ablation_study(
    algorithms: List[str] = ['vi', 'pi'],
    bin_configs: List[List[int]] = None, # type: ignore
    n_seeds: int = 3,
    samples_per_sa: int = 50,
    base_seed: int = 42,
    output_dir: Path = None, # type: ignore
    n_jobs: int = 1
) -> Dict[str, Any]:
    """ablation study: test different discretization bin configurations for cartpole
    
    per rl report: "ablate the grid: run coarse→fine and document policy quality,
    stability, wall-clock, and convergence"
    
    args:
        algorithms: list of algorithms to test (default: vi, pi)
        bin_configs: list of [x, x_dot, theta, theta_dot] bin configurations
        n_seeds: seeds per configuration for robustness
        samples_per_sa: monte carlo samples for transition model
        base_seed: starting seed for reproducibility
        output_dir: where to save results
        n_jobs: number of parallel workers (1=sequential, -1=all cores, -2=all-1)
    
    returns:
        dict with results per algorithm and bin configuration
    """
    import yaml
    
    print("\n" + "="*60)
    print("DISCRETIZATION ABLATION STUDY")
    print("="*60)
    print("testing cartpole bin configurations for vi/pi")
    print()
    
    # default bin configurations: coarse → fine (prioritizing theta, theta_dot)
    if bin_configs is None:
        bin_configs = [
            [1, 1, 6, 8],     # minimal: ignore position/velocity, focus on angle
            [2, 2, 6, 8],     # coarse position, focus on angle
            [3, 3, 6, 8],     # current default
            [3, 3, 8, 10],    # finer angle resolution
            [3, 3, 10, 12],   # even finer angle
            [4, 4, 10, 12],   # balanced fine
            [3, 3, 12, 16],   # very fine angle (focus)
            [5, 5, 12, 16],   # fine all dimensions
        ]
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'results' / 'discretization_ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load base config
    config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    print(f"algorithms: {algorithms}")
    print(f"bin configurations: {len(bin_configs)}")
    print(f"seeds per config: {n_seeds}")
    print(f"samples per (s,a): {samples_per_sa}")
    print(f"parallel workers: {n_jobs}")
    print()
    
    for i, bins in enumerate(bin_configs):
        num_states = bins[0] * bins[1] * bins[2] * bins[3]
        print(f"  {i+1}. {bins} = {num_states} states")
    print()
    
    # build task list: (algo, bins, seed)
    tasks = []
    for algo in algorithms:
        for bin_idx, bins in enumerate(bin_configs):
            for seed_idx in range(n_seeds):
                seed = base_seed + bin_idx * 100 + seed_idx
                tasks.append((algo, bins, seed))
    
    total_tasks = len(tasks)
    print(f"total tasks: {total_tasks} ({len(algorithms)} algos × {len(bin_configs)} configs × {n_seeds} seeds)")
    print()
    
    start_time = time.time()
    actual_workers = _resolve_n_jobs(n_jobs)
    
    if actual_workers == 1:
        # sequential with progress
        all_results = []
        for i, (algo, bins, seed) in enumerate(tasks):
            print(f"[{i+1}/{total_tasks}] {algo} bins={bins} seed={seed}...", end=" ", flush=True)
            result = _run_ablation_task(algo, bins, seed, samples_per_sa, base_config)
            all_results.append(result)
            if result['status'] == 'success':
                print(f"return={result['mean_return']:.1f}, iters={result['iterations']}")
            else:
                print(f"FAILED")
    else:
        # parallel execution using multiprocessing.Pool (same as main.py)
        print(f"running {total_tasks} tasks in parallel with {actual_workers} workers...")
        task_args = [(algo, bins, seed, samples_per_sa, base_config) for algo, bins, seed in tasks]
        with Pool(processes=actual_workers) as pool:
            all_results = pool.map(_run_ablation_task_wrapper, task_args)
    
    elapsed = time.time() - start_time
    
    # aggregate results by algorithm and bin configuration
    results = {
        'metadata': {
            'algorithms': algorithms,
            'bin_configs': bin_configs,
            'n_seeds': n_seeds,
            'samples_per_sa': samples_per_sa,
            'base_seed': base_seed,
            'n_jobs': n_jobs,
            'total_elapsed_seconds': elapsed,
        },
        'results': {}
    }
    
    # aggregate results by algorithm and bin configuration
    for algo in algorithms:
        results['results'][algo] = {}
        
        for bin_idx, bins in enumerate(bin_configs):
            num_states = bins[0] * bins[1] * bins[2] * bins[3]
            
            # filter results for this algo and bin config
            bin_key = str(bins)
            matching = [r for r in all_results if r['algo'] == algo and r['bins'] == bins]
            
            bin_results = {
                'bins': bins,
                'num_states': num_states,
                'seeds': [r['seed'] for r in matching if r['status'] == 'success'],
                'iterations': [r['iterations'] for r in matching if r['status'] == 'success'],
                'wall_times': [r['wall_time'] for r in matching if r['status'] == 'success'],
                'mean_returns': [r['mean_return'] for r in matching if r['status'] == 'success'],
                'mean_lengths': [r['mean_length'] for r in matching if r['status'] == 'success'],
                'converged': [r['converged'] for r in matching if r['status'] == 'success'],
            }
            
            # compute aggregate statistics
            if bin_results['mean_returns']:
                bin_results['summary'] = {
                    'mean_iterations': float(np.mean(bin_results['iterations'])),
                    'std_iterations': float(np.std(bin_results['iterations'])),
                    'mean_wall_time': float(np.mean(bin_results['wall_times'])),
                    'std_wall_time': float(np.std(bin_results['wall_times'])),
                    'mean_return': float(np.mean(bin_results['mean_returns'])),
                    'std_return': float(np.std(bin_results['mean_returns'])),
                    'mean_length': float(np.mean(bin_results['mean_lengths'])),
                    'converge_rate': sum(bin_results['converged']) / len(bin_results['converged']),
                }
            
            results['results'][algo][bin_key] = bin_results
    
    # save results
    results_path = output_dir / 'discretization_ablation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # print summary table
    print("\n" + "="*60)
    print("DISCRETIZATION ABLATION SUMMARY")
    print("="*60)
    print(f"total time: {elapsed:.1f}s")
    print()
    
    print(f"{'bins':<20} {'states':>8} | {'VI return':>10} {'VI iters':>10} | {'PI return':>10} {'PI iters':>10}")
    print("-" * 80)
    
    for bins in bin_configs:
        num_states = bins[0] * bins[1] * bins[2] * bins[3]
        row = f"{str(bins):<20} {num_states:>8} |"
        
        for algo in algorithms:
            bin_key = str(bins)
            if bin_key in results['results'].get(algo, {}):
                data = results['results'][algo][bin_key]
                if 'summary' in data:
                    row += f" {data['summary']['mean_return']:>10.2f} {data['summary']['mean_iterations']:>10.0f} |"
                else:
                    row += f" {'N/A':>10} {'N/A':>10} |"
            else:
                row += f" {'N/A':>10} {'N/A':>10} |"
        
        print(row)
    
    print()
    print(f"results saved to: {results_path}")
    print("="*60)
    
    return results


if __name__ == '__main__':
    """run staged hyperparameter search for all algorithm × environment combinations
    
    Stage 1: Coarse random search (20 candidates, pilot episodes)
    Stage 2: Successive halving (top 10 → top 5, extended episodes)
    Stage 3: Local refinement (top 3 with ±2× alpha, ±25% decay variations)
    
    Additional: Discretization ablation study for VI/PI on CartPole
    
    Usage:
        python hyperparam_search.py                              # run all hyperparameter search (sequential)
        python hyperparam_search.py --n_jobs -2                  # run all search with parallelism
        python hyperparam_search.py --ablation                   # run discretization ablation only
        python hyperparam_search.py --ablation --n_jobs -2       # ablation with parallelism
        python hyperparam_search.py --all --n_jobs -2            # run both search and ablation
        python hyperparam_search.py sarsa blackjack              # run specific algorithm+env
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter search and discretization ablation')
    parser.add_argument('--ablation', action='store_true', help='run discretization ablation study only')
    parser.add_argument('--all', action='store_true', help='run both hyperparam search and ablation')
    parser.add_argument('--seeds', type=int, default=3, help='seeds per bin config for ablation (default: 3)')
    parser.add_argument('--n_jobs', type=int, default=1, help='parallel workers: 1=sequential, -1=all cores, -2=all-1 (default: 1)')
    parser.add_argument('algorithm', nargs='?', help='algorithm: sarsa, qlearning')
    parser.add_argument('environment', nargs='?', help='environment: blackjack, cartpole')
    
    args = parser.parse_args()
    
    # run discretization ablation if requested
    if args.ablation or args.all:
        print("\n" + "#"*60)
        print("# DISCRETIZATION ABLATION STUDY")
        print("#"*60)
        ablation_results = discretization_ablation_study(
            algorithms=['vi', 'pi'],
            n_seeds=args.seeds,
            n_jobs=args.n_jobs
        )
        
        if args.ablation and not args.all:
            # only ablation was requested, exit
            print("\nAblation study complete. Exiting.")
            sys.exit(0)
    
    # if specific algo/env provided, run only that
    if args.algorithm and args.environment:
        algo = args.algorithm.lower()
        env = args.environment.lower()
        
        print(f"\nRunning hyperparameter search for {algo} on {env}...")
        
        results_dir = Path(__file__).parent.parent.parent / 'results' / 'hyperparam_search'
        n_pilot = 500 if env == 'blackjack' else 200
        
        stage1_configs = stage1_coarse_search(algo=algo, env=env, n_candidates=20, n_pilot_episodes=n_pilot, seed=42, n_jobs=args.n_jobs)
        
        stage1_path = results_dir / f'stage1_{algo}_{env}_results.json'
        if stage1_path.exists():
            stage2_configs = stage2_successive_halving(algo=algo, env=env, stage1_results_path=stage1_path, n_extended_episodes=n_pilot*4, seed=42)
            
            stage2_path = results_dir / f'stage2_{algo}_{env}_results.json'
            if stage2_path.exists():
                stage3_best = stage3_local_refinement(algo=algo, env=env, stage2_results_path=stage2_path, n_refinement_episodes=int(n_pilot*6), n_refinement_samples=15, seed=42)
        
        print(f"\nSearch complete for {algo} on {env}.")
        sys.exit(0)
    
    # all combinations to search
    combinations = [
        ('sarsa', 'blackjack'),
        ('sarsa', 'cartpole'),
        ('qlearning', 'blackjack'),
        ('qlearning', 'cartpole'),
    ]
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH: STAGED STRATEGY")
    print("="*60)
    print(f"total combinations: {len(combinations)}")
    print(f"parallel workers: {args.n_jobs}")
    print()
    print("episode counts (scaled to production: 50k blackjack, 10k cartpole):")
    print("  stage 1: 20 candidates, pilot episodes (5000 blackjack / 1000 cartpole)")
    print("  stage 2: top 10 candidates, extended episodes (15000 / 3000)")
    print("  stage 3: top 3 refinement, full production episodes (50000 / 10000)")
    print("="*60 + "\n")
    
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'hyperparam_search'
    all_results = {}
    
    for algo, env in combinations:
        # stage 1: coarse search (10% of production episodes for quick filtering)
        # blackjack: 50k production -> 5k pilot
        # cartpole: 10k production -> 1k pilot
        n_pilot = 5000 if env == 'blackjack' else 1000
        
        print(f"\n{'#'*60}")
        print(f"# STAGE 1: {algo.upper()} on {env.upper()}")
        print(f"{'#'*60}")
        
        stage1_configs = stage1_coarse_search(
            algo=algo,
            env=env,
            n_candidates=20,
            n_pilot_episodes=n_pilot,
            seed=42,
            n_jobs=args.n_jobs
        )
        
        # stage 2: successive halving (30% of production episodes)
        # blackjack: 15k, cartpole: 3k
        n_extended = 15000 if env == 'blackjack' else 3000
        stage1_path = results_dir / f'stage1_{algo}_{env}_results.json'
        
        # verify file exists before proceeding
        if not stage1_path.exists():
            print(f"\nERROR: Stage 1 results not found at {stage1_path}")
            print("Skipping Stage 2 for this combination.")
            all_results[f'{algo}_{env}'] = {
                'stage1_top10': stage1_configs,
                'stage2_top5': None,
                'best_config': stage1_configs[0] if stage1_configs else None
            }
            continue
        
        print(f"\n{'#'*60}")
        print(f"# STAGE 2: {algo.upper()} on {env.upper()}")
        print(f"{'#'*60}")
        
        stage2_configs = stage2_successive_halving(
            algo=algo,
            env=env,
            stage1_results_path=stage1_path,
            n_extended_episodes=n_extended,
            seed=42
        )
        
        # stage 3: local refinement (full production episodes)
        # blackjack: 50k, cartpole: 10k
        n_refinement = 50000 if env == 'blackjack' else 10000
        stage2_path = results_dir / f'stage2_{algo}_{env}_results.json'
        
        # verify file exists before proceeding
        if not stage2_path.exists():
            print(f"\nERROR: Stage 2 results not found at {stage2_path}")
            print("Skipping Stage 3 for this combination.")
            all_results[f'{algo}_{env}'] = {
                'stage1_top10': stage1_configs,
                'stage2_top5': stage2_configs,
                'stage3_best': None,
                'best_config': stage2_configs[0] if stage2_configs else None
            }
            continue
        
        print(f"\n{'#'*60}")
        print(f"# STAGE 3: {algo.upper()} on {env.upper()}")
        print(f"{'#'*60}")
        
        stage3_best = stage3_local_refinement(
            algo=algo,
            env=env,
            stage2_results_path=stage2_path,
            n_refinement_episodes=n_refinement,
            n_refinement_samples=15,  # 3 champions × 5 variations each
            seed=42
        )
        
        all_results[f'{algo}_{env}'] = {
            'stage1_top10': stage1_configs,
            'stage2_top5': stage2_configs,
            'stage3_best': stage3_best,
            'best_config': stage3_best if stage3_best else (stage2_configs[0] if stage2_configs else None)
        }
    
    # summary
    print("\n" + "="*60)
    print("STAGED SEARCH COMPLETE - BEST CONFIGURATIONS")
    print("="*60)
    print("\nStage 3 Winners (locally refined with sensitivity analysis):")
    print("-" * 60)
    
    for key, result in all_results.items():
        if result['best_config']:
            c = result['best_config']
            print(f"\n{key}:")
            print(f"  alpha: {c['alpha']:.4f}")
            print(f"  gamma: {c['gamma']:.4f}")
            print(f"  epsilon_floor: {c['epsilon_floor']:.4f}")
            print(f"  epsilon_decay_episodes: {c['epsilon_decay_episodes']}")
            print(f"  mean_return: {c['mean_return']:.3f}")
            if 'variation' in c:
                print(f"  refinement: {c['variation']} (champion {c.get('champion_id', 0)+1})")
    
    print(f"\n{'='*60}")
    print("Results saved to: results/hyperparam_search/")
    print("  - stage1_*.json: Initial coarse search results")
    print("  - stage2_*.json: Refined successive halving results")
    print("  - stage3_*.json: Local refinement with sensitivity analysis")
    print("\nNext steps:")
    print("1. Review stage3 results (most reliable, includes sensitivity)")
    print("2. Update config/default.yaml with stage3 best hyperparameters")
    print("3. Run production experiments: python main.py")
    print("="*60 + "\n")


"""save training artifacts (q-tables, policies, value functions) for visualization"""
import numpy as np
from pathlib import Path
from typing import Dict, Any


def save_td_artifacts(output_dir: Path, algo_name: str, env_name: str, seed: int, results: Dict[str, Any]):
    """save q-table and derived policy from sarsa/q-learning
    
    saves:
        - q_table.npy: dictionary of (state, action) -> q_value
        - policy.npy: extracted greedy policy from q-table
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save q-table
    q_table = results.get('q_table', {})
    q_table_path = output_dir / f'{algo_name}_{env_name}_seed{seed}_qtable.npy'
    np.save(q_table_path, q_table, allow_pickle=True)
    
    # extract and save greedy policy
    policy = {}
    states = set(state for state, _ in q_table.keys())
    for state in states:
        # find best action: argmax_a Q(s,a)
        actions = [action for (s, action) in q_table.keys() if s == state]
        if actions:
            best_action = max(actions, key=lambda a: q_table[(state, a)])
            policy[state] = best_action
    
    policy_path = output_dir / f'{algo_name}_{env_name}_seed{seed}_policy.npy'
    np.save(policy_path, policy, allow_pickle=True) # type: ignore


def save_dp_artifacts(output_dir: Path, algo_name: str, env_name: str, seed: int, 
                     value_function: np.ndarray, policy: np.ndarray):
    """save value function and policy from vi/pi
    
    saves:
        - value_function.npy: V(s) for all states
        - policy.npy: π(s) for all states
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save value function
    value_path = output_dir / f'{algo_name}_{env_name}_seed{seed}_value.npy'
    np.save(value_path, value_function)
    
    # save policy
    policy_path = output_dir / f'{algo_name}_{env_name}_seed{seed}_policy.npy'
    np.save(policy_path, policy)

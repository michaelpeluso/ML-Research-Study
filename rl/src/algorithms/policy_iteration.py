"""policy iteration for mdps

implements policy iteration with convergence tracking for rl report analysis
"""
from typing import Dict
import numpy as np
import time
from typing import Dict, Any, Optional


class PolicyIteration:
    """policy iteration algorithm with detailed convergence tracking
    alternates between policy evaluation and policy improvement
    tracks iterations, policy changes, and timing for rl report metrics
    """
    
    def __init__(
        self,
        env,
        gamma: float = 0.99,
        theta: float = 1e-6,
        max_iterations: int = 10000,
        max_eval_iterations: int = 1000,
        seed: Optional[int] = None
    ):
        """initialize policy iteration solver
        
        args:
            env: environment with discrete state/action spaces
            gamma: discount factor
            theta: convergence threshold for policy evaluation
            max_iterations: maximum policy improvement iterations
            max_eval_iterations: max iterations per policy evaluation
            seed: random seed for initial policy
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.max_eval_iterations = max_eval_iterations
        self.seed = seed
        
        # get state and action space sizes
        num_states = getattr(env.observation_space, 'n', None)
        if num_states is None:
            num_states = getattr(env, 'num_states', None)
        if num_states is None or not isinstance(num_states, (int, np.integer)):
            raise ValueError("environment must have discrete state space or num_states attribute (int)")
        num_actions = getattr(env.action_space, 'n', None)
        if num_actions is None or not isinstance(num_actions, (int, np.integer)):
            raise ValueError("environment must have discrete action space or action_space.n attribute (int)")
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        
        # initialize policy and value function
        if seed is not None:
            np.random.seed(seed)
        
        self.policy = np.random.randint(0, self.num_actions, size=self.num_states)
        self.V = np.zeros(self.num_states)
        
    def train(self) -> Dict[str, Any]:
        """run policy iteration until convergence
        
        returns dictionary with:
            - V: final value function
            - policy: optimal policy
            - iterations: number of policy improvement iterations
            - policy_changes: list of number of states changed per iteration
            - eval_iterations: list of evaluation iterations per improvement step
            - wall_times: cumulative wall time at each iteration
            - converged: whether policy became stable
            - metadata: additional info (gamma, theta, num_states, num_actions)
        """
        start_time = time.time()
        
        policy_changes = []
        eval_iterations_list = []
        wall_times = []
        
        for iteration in range(self.max_iterations):
            old_policy = self.policy.copy()
            
            # policy evaluation: compute v^π
            eval_iters = self._evaluate_policy()
            eval_iterations_list.append(eval_iters)
            
            # policy improvement: π' = greedy(v^π)
            self._improve_policy()
            
            # count policy changes
            num_changes = np.sum(self.policy != old_policy)
            policy_changes.append(int(num_changes))
            wall_times.append(time.time() - start_time)
            
            # check convergence: policy is stable
            if num_changes == 0:
                return {
                    'V': self.V,
                    'policy': self.policy,
                    'iterations': iteration + 1,
                    'policy_changes': policy_changes,
                    'eval_iterations': eval_iterations_list,
                    'wall_times': wall_times,
                    'converged': True,
                    'metadata': {
                        'gamma': self.gamma,
                        'theta': self.theta,
                        'num_states': self.num_states,
                        'num_actions': self.num_actions,
                        'total_time': wall_times[-1],
                        'total_eval_iterations': sum(eval_iterations_list),
                    }
                }
        
        # max iterations reached without stable policy
        return {
            'V': self.V,
            'policy': self.policy,
            'iterations': self.max_iterations,
            'policy_changes': policy_changes,
            'eval_iterations': eval_iterations_list,
            'wall_times': wall_times,
            'converged': False,
            'metadata': {
                'gamma': self.gamma,
                'theta': self.theta,
                'num_states': self.num_states,
                'num_actions': self.num_actions,
                'total_time': wall_times[-1] if wall_times else 0.0,
                'total_eval_iterations': sum(eval_iterations_list),
            }
        }
    
    def _evaluate_policy(self) -> int:
        """evaluate current policy: compute v^π(s) for all states
        
        iteratively update: v(s) = sum_s' p(s'|s,π(s))[r + γv(s')]
        
        returns number of iterations needed for convergence
        """
        for iteration in range(self.max_eval_iterations):
            delta = 0.0
            V_old = self.V.copy()
            
            for s in range(self.num_states):
                action = self.policy[s]
                
                # compute expected value under current policy
                v_new = 0.0
                if hasattr(self.env, 'P'):
                    transitions = self.env.P[s][action]
                    for prob, next_state, reward, done in transitions:
                        v_new += prob * (reward + self.gamma * V_old[next_state] * (1 - done))
                else:
                    raise NotImplementedError(
                        "environment must provide transition model via P attribute. "
                        "for model-free environments, use sarsa or q-learning instead."
                    )
                
                self.V[s] = v_new
                delta = max(delta, abs(self.V[s] - V_old[s]))
            
            # check convergence
            if delta < self.theta:
                return iteration + 1
        
        return self.max_eval_iterations
    
    def _improve_policy(self) -> None:
        """improve policy: π'(s) = argmax_a q^π(s,a)
        
        makes policy greedy with respect to current value function
        """
        for s in range(self.num_states):
            q_values = self._compute_q_values(s)
            self.policy[s] = int(np.argmax(q_values))
    
    def _compute_q_values(self, state: int) -> np.ndarray:
        """compute q-values for all actions in given state
        
        q(s,a) = sum_s' p(s'|s,a)[r(s,a,s') + gamma * v(s')]
        """
        q_values = np.zeros(self.num_actions)
        
        for action in range(self.num_actions):
            if hasattr(self.env, 'P'):
                transitions = self.env.P[state][action]
                for prob, next_state, reward, done in transitions:
                    q_values[action] += prob * (reward + self.gamma * self.V[next_state] * (1 - done))
            else:
                raise NotImplementedError(
                    "environment must provide transition model via P attribute"
                )
        
        return q_values
    
    def evaluate_policy(self, num_episodes: int = 100) -> Dict[str, Any]:
        """evaluate learned policy by running episodes
        
        returns mean return and episode lengths
        """
        returns = []
        lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            if hasattr(state, '__len__'):
                state = self._state_to_index(state)
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < 1000:
                action = int(self.policy[state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if hasattr(next_state, '__len__'):
                    next_state = self._state_to_index(next_state)
                
                episode_return += reward
                episode_length += 1
                state = next_state
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'mean_length': float(np.mean(lengths)),
            'returns': returns,
            'lengths': lengths,
        }
    
    def _state_to_index(self, state) -> int:
        """convert multi-dimensional state to single index"""
        # for environments with state_to_idx mapping
        if hasattr(self.env, 'state_to_idx') and state in self.env.state_to_idx:
            return self.env.state_to_idx[state]
        # for discretized cartpole or similar
        if hasattr(self.env, 'discretizer'):
            return self.env.discretizer.discretize(state)
        # assume state is already an index
        if isinstance(state, int):
            return state
        # if tuple of ints, assume it's already a discrete state index
        if isinstance(state, tuple):
            # try to find in state_to_idx
            if hasattr(self.env, 'state_to_idx'):
                return self.env.state_to_idx.get(state, 0)
            return 0
        return int(state)

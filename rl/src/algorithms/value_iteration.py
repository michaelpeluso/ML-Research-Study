"""value iteration for mdps

implements value iteration with convergence tracking for rl report analysis
"""
import numpy as np
import time
from typing import Dict, Any, Optional


class ValueIteration:
    """value iteration algorithm with detailed convergence tracking
    
    tracks iterations, value function deltas, and timing for rl report metrics
    """
    
    def __init__(
        self,
        env,
        gamma: float = 0.99,
        theta: float = 1e-6,
        max_iterations: int = 10000,
        seed: Optional[int] = None
    ):
        """initialize value iteration solver
        
        args:
            env: environment with discrete state/action spaces
            gamma: discount factor
            theta: convergence threshold (max delta)
            max_iterations: maximum iterations before stopping
            seed: random seed (for consistency, though vi is deterministic)
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.seed = seed
        
        # get state and action space sizes
        num_states = getattr(env.observation_space, 'n', None)
        if num_states is None:
            num_states = getattr(env, 'num_states', None)
        if num_states is None or not isinstance(num_states, (int, np.integer)):
            raise ValueError("environment must have discrete state space or num_states attribute (int)")
        self.num_states = int(num_states)
        
        num_actions = getattr(env.action_space, 'n', None)
        if num_actions is None or not isinstance(num_actions, (int, np.integer)):
            raise ValueError("environment must have discrete action space or action_space.n attribute (int)")
        self.num_actions = int(num_actions)
        
        # initialize value function
        if seed is not None:
            np.random.seed(seed)
        self.V = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)
        
    def train(self) -> Dict[str, Any]:
        """run value iteration until convergence
        
        returns dictionary with:
            - V: final value function
            - policy: extracted optimal policy
            - iterations: number of iterations to converge
            - deltas: list of max value changes per iteration
            - wall_times: cumulative wall time at each iteration
            - converged: whether algorithm converged
            - metadata: additional info (gamma, theta, num_states, num_actions)
        """
        start_time = time.time()
        
        deltas = []
        wall_times = []
        
        for iteration in range(self.max_iterations):
            delta = 0.0
            V_old = self.V.copy()
            
            # update value function for each state
            if self.num_states is None or not isinstance(self.num_states, int):
                raise ValueError("num_states must be an integer for value iteration loop")
            for s in range(self.num_states):
                # compute q-values for all actions
                q_values = self._compute_q_values(s, V_old)
                
                # bellman optimality update: v(s) = max_a q(s,a)
                self.V[s] = np.max(q_values)
                
                # track maximum change
                delta = max(delta, abs(self.V[s] - V_old[s]))
            
            deltas.append(float(delta))
            wall_times.append(time.time() - start_time)
            
            # check convergence
            if delta < self.theta:
                # extract optimal policy
                self._extract_policy()
                
                return {
                    'V': self.V,
                    'policy': self.policy,
                    'iterations': iteration + 1,
                    'deltas': deltas,
                    'wall_times': wall_times,
                    'converged': True,
                    'metadata': {
                        'gamma': self.gamma,
                        'theta': self.theta,
                        'num_states': self.num_states,
                        'num_actions': self.num_actions,
                        'final_delta': float(delta),
                        'total_time': wall_times[-1],
                    }
                }
        
        # max iterations reached without convergence
        self._extract_policy()
        
        return {
            'V': self.V,
            'policy': self.policy,
            'iterations': self.max_iterations,
            'deltas': deltas,
            'wall_times': wall_times,
            'converged': False,
            'metadata': {
                'gamma': self.gamma,
                'theta': self.theta,
                'num_states': self.num_states,
                'num_actions': self.num_actions,
                'final_delta': deltas[-1] if deltas else None,
                'total_time': wall_times[-1] if wall_times else 0.0,
            }
        }
    
    def _compute_q_values(self, state: int, V: np.ndarray) -> np.ndarray:
        """compute q-values for all actions in given state
        
        q(s,a) = sum_s' p(s'|s,a)[r(s,a,s') + gamma * v(s')]
        """
        q_values = np.zeros(self.num_actions)
        
        for action in range(self.num_actions):
            # get transition dynamics from environment
            # format: [(prob, next_state, reward, done), ...]
            if hasattr(self.env, 'P'):
                # standard gym format (blackjack-like)
                transitions = self.env.P[state][action]
                for prob, next_state, reward, done in transitions:
                    q_values[action] += prob * (reward + self.gamma * V[next_state] * (1 - done))
            else:
                # for discretized environments without explicit transition model
                # use model-free approach or require explicit transition function
                raise NotImplementedError(
                    "environment must provide transition model via P attribute. "
                    "for model-free environments, use sarsa or q-learning instead."
                )
        
        return q_values
    
    def _extract_policy(self) -> None:
        """extract greedy policy from current value function
        
        policy(s) = argmax_a q(s,a)
        """
        if self.num_states is None or not isinstance(self.num_states, int):
            raise ValueError("num_states must be an integer for policy extraction")
        for s in range(self.num_states):
            q_values = self._compute_q_values(s, self.V)
            self.policy[s] = np.argmax(q_values)
    
    def evaluate_policy(self, num_episodes: int = 100) -> Dict[str, Any]:
        """evaluate learned policy by running episodes
        
        returns mean return and episode lengths
        """
        returns = []
        lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            # convert state to index immediately
            state_idx = self._state_to_index(state)
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < 1000:
                action = int(self.policy[state_idx])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # convert next state to index
                state_idx = self._state_to_index(next_state)
                
                episode_return += reward
                episode_length += 1
            
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
        # assume state is already an index (most common case)
        if isinstance(state, (int, np.integer)):
            return int(state)
        # for environments with state_to_idx mapping
        if hasattr(self.env, 'state_to_idx') and state in self.env.state_to_idx:
            return self.env.state_to_idx[state]
        # for discretized cartpole or similar (continuous state → discrete index)
        if hasattr(self.env, 'discretizer'):
            return self.env.discretizer.discretize_to_index(state)
        # if tuple of ints, try to find in state_to_idx
        if isinstance(state, tuple):
            if hasattr(self.env, 'state_to_idx'):
                return self.env.state_to_idx.get(state, 0)
            return 0
        return int(state)

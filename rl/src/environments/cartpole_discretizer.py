"""CartPole discretizer utilities"""
"""state space discretizer for continuous environments"""
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Optional


class StateDiscretizer:
    """discretize continuous state space into discrete bins"""
    
    def __init__(self, env: gym.Env, bins: List[int], bounds: Optional[List[Tuple[float, float]]] = None):
        """
        initialize discretizer with environment and bin counts
        
        args:
            env: gymnasium environment
            bins: list of bin counts per state dimension
            bounds: optional custom bounds per dimension as [(low, high), ...]
                   if None, uses environment bounds with default clamps for infinites
                   cartpole recommendation (rl report):
                     x: [-2.4, 2.4], x_dot: [-3.0, 3.0]
                     theta: [-0.209, 0.209], theta_dot: [-3.5, 3.5]
        """
        self.env = env
        self.bins = bins
        self.n_dims = len(bins)
        
        # extract bounds from environment
        env_low = np.array(env.observation_space.low)  # type: ignore
        env_high = np.array(env.observation_space.high)  # type: ignore
        
        if bounds is not None:
            # use custom bounds (recommended for cartpole per rl report)
            self.low = np.array([b[0] for b in bounds])
            self.high = np.array([b[1] for b in bounds])
        else:
            # default: use env bounds, clip infinite values to reasonable defaults
            # cartpole-v1 defaults: x=[-4.8,4.8], x_dot=inf, theta=[-0.418,0.418], theta_dot=inf
            # we use tighter clamps per rl report recommendations
            default_clamps = [
                (-2.4, 2.4),    # x: termination boundary
                (-3.0, 3.0),    # x_dot: reasonable velocity range
                (-0.209, 0.209), # theta: ~12 degrees (termination boundary)
                (-3.5, 3.5)     # theta_dot: reasonable angular velocity
            ]
            self.low = np.array([
                env_low[i] if np.isfinite(env_low[i]) else default_clamps[i][0]
                for i in range(self.n_dims)
            ])
            self.high = np.array([
                env_high[i] if np.isfinite(env_high[i]) else default_clamps[i][1]
                for i in range(self.n_dims)
            ])
            # further clamp to rl report recommendations if env bounds are wider
            for i, (lo, hi) in enumerate(default_clamps):
                self.low[i] = max(self.low[i], lo)
                self.high[i] = min(self.high[i], hi)
        
        # create bin edges for each dimension
        self.bin_edges = [
            np.linspace(self.low[i], self.high[i], bins[i] + 1)
            for i in range(self.n_dims)
        ]
    
    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """convert continuous state to discrete state tuple"""
        # clip state to bounds
        state_clipped = np.clip(state, self.low, self.high)
        
        # map continuous state value to discrete bin index per dimension
        discrete_state = tuple(
            np.digitize(state_clipped[i], self.bin_edges[i]) - 1
            for i in range(self.n_dims)
        )
        
        # ensure indices are within valid range
        discrete_state = tuple(
            max(0, min(discrete_state[i], self.bins[i] - 1))
            for i in range(self.n_dims)
        )
        
        return discrete_state
    
    def tuple_to_index(self, discrete_tuple: Tuple[int, ...]) -> int:
        """convert discrete state tuple to flat integer index"""
        index = 0
        for i, val in enumerate(discrete_tuple):
            # compute flat index using row-major ordering
            multiplier = int(np.prod(self.bins[i+1:]))  # product of remaining dimensions
            index += val * multiplier
        return int(index)
    
    def discretize_to_index(self, state: np.ndarray) -> int:
        """convert continuous state directly to flat integer index"""
        discrete_tuple = self.discretize(state)
        return self.tuple_to_index(discrete_tuple)
    
    def get_num_states(self) -> int:
        """return total number of discrete states"""
        return int(np.prod(self.bins))


class DiscretizedEnv:
    """wrapper that discretizes continuous state space for rl algorithms"""
    
    def __init__(self, env: gym.Env, discretizer: StateDiscretizer):
        """
        wrap gymnasium environment with discretization
        
        args:
            env: gymnasium environment with continuous states
            discretizer: statediscretizer instance for state conversion
        """
        self.env = env
        self.discretizer = discretizer
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def reset(self, seed=None):
        """reset environment and discretize initial state"""
        state, info = self.env.reset(seed=seed)
        return self.discretizer.discretize(state), info
    
    def step(self, action):
        """take action and discretize resulting state"""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return self.discretizer.discretize(next_state), reward, terminated, truncated, info

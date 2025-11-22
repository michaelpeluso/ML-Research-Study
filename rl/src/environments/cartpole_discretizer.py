# AI Use Statement: CartPole discretizer created with GitHub Copilot assistance
"""state space discretizer for continuous environments"""
import numpy as np
import gymnasium as gym
from typing import List, Tuple


class StateDiscretizer:
    """discretize continuous state space into discrete bins"""
    
    def __init__(self, env: gym.Env, bins: List[int]):
        """
        initialize discretizer with environment and bin counts
        
        args:
            env: gymnasium environment
            bins: list of bin counts per state dimension
        """
        self.env = env
        self.bins = bins
        self.n_dims = len(bins)
        
        # extract bounds from environment
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        
        # clip infinite bounds
        self.low = np.where(np.isfinite(self.low), self.low, -10.0)
        self.high = np.where(np.isfinite(self.high), self.high, 10.0)
        
        # create bin edges for each dimension
        self.bin_edges = [
            np.linspace(self.low[i], self.high[i], bins[i] + 1)
            for i in range(self.n_dims)
        ]
    
    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """convert continuous state to discrete state tuple"""
        # clip state to bounds
        state_clipped = np.clip(state, self.low, self.high)
        
        # digitize each dimension
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
    
    def get_num_states(self) -> int:
        """return total number of discrete states"""
        return int(np.prod(self.bins))

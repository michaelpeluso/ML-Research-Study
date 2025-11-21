# ai use statement: github copilot assisted with discretization logic
"""state space discretization for continuous environments."""

from typing import List
import numpy as np


class StateDiscretizer:
    """discretize continuous state spaces into bins."""
    
    def __init__(self, env, bins: List[int]):
        """
        initialize discretizer.
        
        env: gymnasium environment
        bins: number of bins for each state dimension
        """
        self.env = env
        self.bins = bins
        
        # get observation space bounds
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        
        # handle infinite bounds
        self.low = np.where(np.isinf(self.low), -10.0, self.low)
        self.high = np.where(np.isinf(self.high), 10.0, self.high)
        
        # create bin edges for each dimension
        self.bin_edges = [
            np.linspace(self.low[i], self.high[i], bins[i] + 1)
            for i in range(len(bins))
        ]
    
    def discretize(self, state: np.ndarray) -> tuple:
        """
        convert continuous state to discrete tuple.
        
        state: continuous state vector
        returns: tuple of bin indices
        """
        discrete_state = []
        for i, value in enumerate(state):
            # clip to bounds
            value = np.clip(value, self.low[i], self.high[i])
            # find bin index
            bin_idx = np.digitize(value, self.bin_edges[i]) - 1
            # handle edge case
            bin_idx = min(bin_idx, self.bins[i] - 1)
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def get_num_states(self) -> int:
        """return total number of discrete states."""
        return np.prod(self.bins)

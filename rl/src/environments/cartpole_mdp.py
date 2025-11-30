"""cartpole mdp with explicit transition model for value/policy iteration"""
import numpy as np
from typing import Dict, Tuple, List, Optional
import gymnasium as gym
from environments.cartpole_discretizer import StateDiscretizer


class CartPoleMDP:
    """cartpole with discretized state space and empirical transition model"""
    
    def __init__(self, bins: List[int] = [3, 3, 8, 12], samples_per_state_action: int = 100):
        """
        initialize cartpole mdp with transition model
        
        args:
            bins: discretization bins per dimension
            samples_per_state_action: samples to estimate transitions
        """
        self.env = gym.make('CartPole-v1')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # discretize state space
        self.discretizer = StateDiscretizer(self.env, bins)
        self.num_states = self.discretizer.get_num_states()
        self.bins = bins
        
        print(f"building transition model for {self.num_states} discrete states...")
        print(f"  this may take several minutes (sampling {samples_per_state_action} transitions per state-action)...")
        
        # build transition model via sampling
        self.P = self._build_transition_model(samples_per_state_action)
        print(f"transition model complete!")
    
    def _continuous_from_discrete(self, discrete_state: Tuple[int, ...]) -> np.ndarray:
        """estimate continuous state from discrete bin indices"""
        continuous = np.zeros(4)
        for i, bin_idx in enumerate(discrete_state):
            # use bin center as representative continuous value
            bin_edges = self.discretizer.bin_edges[i]
            if bin_idx >= len(bin_edges) - 1:
                bin_idx = len(bin_edges) - 2
            continuous[i] = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2.0
        return continuous
    
    def _build_transition_model(self, samples_per_state_action: int) -> Dict:
        """
        empirically estimate transition probabilities via monte carlo sampling
        
        P[s][a] = [(prob, next_state_idx, reward, done), ...]
        """
        P = {}
        total_states = self.num_states
        
        # enumerate all discrete states
        discrete_states = []
        for i0 in range(self.bins[0]):
            for i1 in range(self.bins[1]):
                for i2 in range(self.bins[2]):
                    for i3 in range(self.bins[3]):
                        discrete_states.append((i0, i1, i2, i3))
        
        for state_idx, discrete_state in enumerate(discrete_states):
            if (state_idx + 1) % 100 == 0 or state_idx == 0:
                print(f"  processed {state_idx + 1}/{total_states} states...")
            
            P[state_idx] = {}
            num_actions = getattr(self.action_space, 'n', None)
            if num_actions is None:
                raise TypeError("action_space.n is not defined for this environment")
            for action in range(num_actions):
                transitions = {}
                
                # sample transitions from this state-action pair
                for _ in range(samples_per_state_action):
                    # reset and attempt to reach this discrete state                    
                    # approximate: use any initial state and take action
                    obs, _ = self.env.reset()
                    
                    # take the action
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_discrete = self.discretizer.discretize(next_obs)
                    next_idx = (next_discrete[0] * self.bins[1] * self.bins[2] * self.bins[3] +
                               next_discrete[1] * self.bins[2] * self.bins[3] +
                               next_discrete[2] * self.bins[3] +
                               next_discrete[3])
                    key = (next_idx, reward, done)
                    transitions[key] = transitions.get(key, 0) + 1
                total = sum(transitions.values()) if transitions else 1
                P[state_idx][action] = [
                    (count / total, next_s, r, d)
                    for (next_s, r, d), count in transitions.items()
                ]
                if not P[state_idx][action]:
                    P[state_idx][action] = [(1.0, state_idx, 1.0, False)]
        
        return P
    
    def reset(self, seed: Optional[int] = None):
        """reset environment and return discretized state"""
        obs, info = self.env.reset(seed=seed)
        discrete_obs = self.discretizer.discretize(obs)
        return discrete_obs, info
    
    def step(self, action: int):
        """take action and return discretized next state"""
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        discrete_next = self.discretizer.discretize(next_obs)
        return discrete_next, reward, terminated, truncated, info
    
    def close(self):
        """close environment"""
        self.env.close()


class SimplifiedCartPoleMDP:
    """cartpole with empirical transition model for vi/pi
    
    uses monte carlo sampling to estimate transition probabilities
    from actual cartpole physics (accurate but slower to build)
    """
    
    def __init__(self, bins: List[int] = [3, 3, 6, 8], samples_per_sa: int = 50, seed: int = 0):
        """initialize cartpole mdp with empirical transitions
        
        args:
            bins: discretization resolution [x, x_dot, theta, theta_dot]
            samples_per_sa: monte carlo samples per state-action pair
            seed: random seed for reproducibility
        """
        self.env = gym.make('CartPole-v1')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.seed = seed
        
        self.discretizer = StateDiscretizer(self.env, bins)
        self.num_states = self.discretizer.get_num_states()
        self.bins = bins
        self.samples_per_sa = samples_per_sa
        
        print(f"building empirical transition model:")
        print(f"  discrete states: {self.num_states}")
        print(f"  bins: {bins}")
        print(f"  samples per (s,a): {samples_per_sa}")
        print(f"  estimated build time: ~{self.num_states * 2 * samples_per_sa / 10000:.1f}s")
        
        # build transition model via rollout sampling
        self.P = self._build_empirical_transitions()
        print(f"transition model complete!")
    
    def reset(self, seed=None):
        """reset environment and return discrete state index"""
        state, info = self.env.reset(seed=seed)
        discrete_idx = self.discretizer.discretize_to_index(state)
        return discrete_idx, info
    
    def step(self, action):
        """take action and return discrete next state index"""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_discrete_idx = self.discretizer.discretize_to_index(next_state)
        return next_discrete_idx, reward, terminated, truncated, info
    
    def close(self):
        """close environment"""
        self.env.close()
    
    def _continuous_from_discrete(self, discrete_idx: int) -> np.ndarray:
        """reconstruct continuous state from discrete index (use bin centers)"""
        # decode index to bin indices
        total = discrete_idx
        bins_rev = list(reversed(self.bins))
        bin_indices = []
        for b in bins_rev:
            bin_indices.append(total % b)
            total //= b
        bin_indices.reverse()
        
        # map to continuous values (use bin centers)
        continuous = np.zeros(4)
        for i, bin_idx in enumerate(bin_indices):
            edges = self.discretizer.bin_edges[i]
            if bin_idx >= len(edges) - 1:
                bin_idx = len(edges) - 2
            continuous[i] = (edges[bin_idx] + edges[bin_idx + 1]) / 2.0
        
        return continuous
    
    def _build_empirical_transitions(self) -> Dict:
        """estimate P(s'|s,a) via monte carlo rollouts from each state
        
        for each discrete state:
          1. reconstruct approximate continuous state (bin center)
          2. manually set env state to that continuous state
          3. sample transitions by taking each action multiple times
          4. aggregate into transition probabilities
        """
        np.random.seed(self.seed)
        P = {}
        
        num_actions = self.action_space.n # type: ignore
        total_sa_pairs = self.num_states * num_actions
        processed = 0
        
        for state_idx in range(self.num_states):
            P[state_idx] = {}
            
            # get approximate continuous state for this discrete state
            continuous_state = self._continuous_from_discrete(state_idx)
            
            for action in range(num_actions):
                # collect samples from this (s, a) pair
                transitions = []
                
                for _ in range(self.samples_per_sa):
                    # reset and set to target continuous state
                    self.env.reset(seed=self.seed + processed)
                    
                    # manually inject state (cartpole allows this via env.state)
                    self.env.unwrapped.state = continuous_state.copy() # type: ignore
                    
                    # take action and observe outcome
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    # discretize next state to integer index
                    next_discrete_idx = self.discretizer.discretize_to_index(next_obs)
                    
                    transitions.append((next_discrete_idx, reward, done))
                
                # aggregate into transition probabilities
                # P[s][a] = [(prob, s', r, done), ...]
                transition_counts = {}
                for next_s, r, done in transitions:
                    key = (next_s, r, done)
                    transition_counts[key] = transition_counts.get(key, 0) + 1
                
                P[state_idx][action] = [
                    (count / self.samples_per_sa, next_s, r, done)
                    for (next_s, r, done), count in transition_counts.items()
                ]
                
                processed += 1
                if processed % 100 == 0:
                    print(f"    processed {processed}/{total_sa_pairs} (s,a) pairs...")
        
        return P

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
    """simplified cartpole with coarse discretization for faster vi/pi"""
    
    def __init__(self, bins: List[int] = [2, 2, 4, 4]):
        """initialize simplified cartpole (coarse bins for speed)"""
        self.env = gym.make('CartPole-v1')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.discretizer = StateDiscretizer(self.env, bins)
        self.num_states = self.discretizer.get_num_states()
        self.bins = bins
        
        print(f"building simplified transition model for {self.num_states} discrete states...")
        self.P = self._build_simplified_transitions()
        print(f"transition model complete!")
    
    def _build_simplified_transitions(self) -> Dict:
        """build simplified transition model with physics approximation"""
        P = {}
        
        # enumerate all discrete states
        discrete_states = []
        for i0 in range(self.bins[0]):
            for i1 in range(self.bins[1]):
                for i2 in range(self.bins[2]):
                    for i3 in range(self.bins[3]):
                        discrete_states.append((i0, i1, i2, i3))
        
        for state_idx, discrete_state in enumerate(discrete_states):
            P[state_idx] = {}
            
            # simplified physics: pole angle dominates termination
            pole_angle_bin = discrete_state[2]
            pole_velocity_bin = discrete_state[3]
            num_actions = getattr(self.action_space, 'n', None)
            if num_actions is None:
                raise TypeError("action_space.n is not defined for this environment")
            for action in range(num_actions):
                # deterministic simplified transitions
                # if pole angle extreme → terminal
                if pole_angle_bin == 0 or pole_angle_bin == self.bins[2] - 1:
                    P[state_idx][action] = [(1.0, -1, 0.0, True)]
                else:
                    # stay in similar state, get reward
                    # simplified: assume action affects pole velocity
                    if action == 0:  # push left
                        new_pole_vel = max(0, pole_velocity_bin - 1)
                    else:  # push right
                        new_pole_vel = min(self.bins[3] - 1, pole_velocity_bin + 1)
                    
                    # pole angle affected by velocity
                    if new_pole_vel < self.bins[3] // 2:
                        new_pole_angle = max(1, pole_angle_bin - 1)
                    else:
                        new_pole_angle = min(self.bins[2] - 2, pole_angle_bin + 1)
                    
                    # approximate next state
                    next_discrete = (
                        discrete_state[0],  # cart position unchanged
                        discrete_state[1],  # cart velocity approximated
                        new_pole_angle,
                        new_pole_vel
                    )
                    
                    # convert to index
                    next_idx = (next_discrete[0] * self.bins[1] * self.bins[2] * self.bins[3] +
                               next_discrete[1] * self.bins[2] * self.bins[3] +
                               next_discrete[2] * self.bins[3] +
                               next_discrete[3])
                    
                    P[state_idx][action] = [(1.0, next_idx, 1.0, False)]
        
        return P
    
    def reset(self, seed: Optional[int] = None):
        """reset environment"""
        obs, info = self.env.reset(seed=seed)
        discrete_obs = self.discretizer.discretize(obs)
        return discrete_obs, info
    
    def step(self, action: int):
        """take action"""
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        discrete_next = self.discretizer.discretize(next_obs)
        return discrete_next, reward, terminated, truncated, info
    
    def close(self):
        """close environment"""
        self.env.close()

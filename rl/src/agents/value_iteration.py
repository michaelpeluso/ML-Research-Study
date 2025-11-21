# ai use statement: github copilot assisted with value iteration algorithm structure
"""value iteration algorithm for mdp solving."""

from typing import Optional, Tuple, Dict
import numpy as np


class ValueIteration:
    """value iteration for discrete mdps."""
    
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6, seed: Optional[int] = None):
        """
        initialize value iteration solver.
        
        env: gymnasium environment
        gamma: discount factor
        theta: convergence threshold
        seed: random seed for reproducibility
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.seed = seed
        
        # get state and action spaces
        if hasattr(env.observation_space, 'n'):
            self.n_states = env.observation_space.n
        else:
            raise ValueError("value iteration requires discrete observation space")
        
        if hasattr(env.action_space, 'n'):
            self.n_actions = env.action_space.n
        else:
            raise ValueError("value iteration requires discrete action space")
    
    def train(self) -> Dict:
        """
        run value iteration algorithm.
        
        returns dict with policy, value function, and metadata
        """
        # initialize value function
        V = np.zeros(self.n_states)
        iterations = 0
        
        while True:
            delta = 0
            iterations += 1
            
            # update each state
            for s in range(self.n_states):
                v = V[s]
                
                # compute max over actions
                action_values = []
                for a in range(self.n_actions):
                    # for environments with transition dynamics
                    if hasattr(self.env, 'P'):
                        # access transition probabilities
                        transitions = self.env.P[s][a]
                        action_value = sum(prob * (reward + self.gamma * V[next_state])
                                         for prob, next_state, reward, done in transitions)
                    else:
                        # simple estimate for environments without explicit dynamics
                        action_value = 0
                    
                    action_values.append(action_value)
                
                V[s] = max(action_values) if action_values else 0
                delta = max(delta, abs(v - V[s]))
            
            # check convergence
            if delta < self.theta:
                break
        
        # extract policy
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            action_values = []
            for a in range(self.n_actions):
                if hasattr(self.env, 'P'):
                    transitions = self.env.P[s][a]
                    action_value = sum(prob * (reward + self.gamma * V[next_state])
                                     for prob, next_state, reward, done in transitions)
                else:
                    action_value = 0
                action_values.append(action_value)
            
            policy[s] = np.argmax(action_values)
        
        return {
            'policy': policy,
            'V': V,
            'iterations': iterations,
            'converged': True,
            'algorithm': 'value_iteration',
        }

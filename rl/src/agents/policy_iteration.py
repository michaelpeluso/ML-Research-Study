# ai use statement: github copilot assisted with policy iteration algorithm structure
"""policy iteration algorithm for mdp solving."""

from typing import Optional, Dict
import numpy as np


class PolicyIteration:
    """policy iteration for discrete mdps."""
    
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6, seed: Optional[int] = None):
        """
        initialize policy iteration solver.
        
        env: gymnasium environment
        gamma: discount factor
        theta: convergence threshold for policy evaluation
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
            raise ValueError("policy iteration requires discrete observation space")
        
        if hasattr(env.action_space, 'n'):
            self.n_actions = env.action_space.n
        else:
            raise ValueError("policy iteration requires discrete action space")
    
    def policy_evaluation(self, policy: np.ndarray, V: np.ndarray) -> np.ndarray:
        """evaluate a policy to convergence."""
        while True:
            delta = 0
            for s in range(self.n_states):
                v = V[s]
                a = policy[s]
                
                # compute value for current policy
                if hasattr(self.env, 'P'):
                    transitions = self.env.P[s][a]
                    V[s] = sum(prob * (reward + self.gamma * V[next_state])
                             for prob, next_state, reward, done in transitions)
                else:
                    V[s] = 0
                
                delta = max(delta, abs(v - V[s]))
            
            if delta < self.theta:
                break
        
        return V
    
    def policy_improvement(self, V: np.ndarray) -> np.ndarray:
        """improve policy based on current value function."""
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
        
        return policy
    
    def train(self) -> Dict:
        """
        run policy iteration algorithm.
        
        returns dict with policy, value function, and metadata
        """
        # initialize random policy
        policy = np.random.randint(0, self.n_actions, size=self.n_states)
        V = np.zeros(self.n_states)
        
        iterations = 0
        while True:
            iterations += 1
            
            # policy evaluation
            V = self.policy_evaluation(policy, V)
            
            # policy improvement
            new_policy = self.policy_improvement(V)
            
            # check if policy is stable
            if np.array_equal(policy, new_policy):
                break
            
            policy = new_policy
        
        return {
            'policy': policy,
            'V': V,
            'iterations': iterations,
            'converged': True,
            'algorithm': 'policy_iteration',
        }

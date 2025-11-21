# ai use statement: github copilot assisted with sarsa algorithm structure
"""sarsa on-policy td control algorithm."""

from typing import Optional, Dict, Tuple
import numpy as np


class SARSA:
    """sarsa temporal difference learning (on-policy)."""
    
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99, 
                 epsilon: float = 0.1, num_episodes: int = 10000, seed: Optional[int] = None):
        """
        initialize sarsa agent.
        
        env: gymnasium environment
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate for epsilon-greedy
        num_episodes: number of training episodes
        seed: random seed for reproducibility
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # determine state space size
        if hasattr(env.observation_space, 'n'):
            self.n_states = env.observation_space.n
            self.discrete_states = True
        else:
            # for continuous spaces, will need discretization
            self.discrete_states = False
        
        # action space
        if hasattr(env.action_space, 'n'):
            self.n_actions = env.action_space.n
        else:
            raise ValueError("sarsa requires discrete action space")
        
        # initialize q-table
        if self.discrete_states:
            self.Q = np.zeros((self.n_states, self.n_actions))
        else:
            self.Q = {}  # use dict for discretized continuous states
    
    def get_state_key(self, state) -> Tuple:
        """convert state to hashable key for q-table."""
        if self.discrete_states:
            return state
        else:
            # for tuple states (discretized)
            return tuple(state) if isinstance(state, (list, np.ndarray)) else state
    
    def epsilon_greedy(self, state) -> int:
        """select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_key = self.get_state_key(state)
            if self.discrete_states:
                return np.argmax(self.Q[state_key])
            else:
                if state_key not in self.Q:
                    self.Q[state_key] = np.zeros(self.n_actions)
                return np.argmax(self.Q[state_key])
    
    def train(self) -> Dict:
        """
        run sarsa training.
        
        returns dict with q-table, rewards, and metadata
        """
        episode_rewards = []
        
        for episode in range(self.num_episodes):
            state, _ = self.env.reset(seed=self.seed + episode if self.seed else None)
            action = self.epsilon_greedy(state)
            
            episode_reward = 0
            done = False
            
            while not done:
                # take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # select next action
                next_action = self.epsilon_greedy(next_state)
                
                # sarsa update
                state_key = self.get_state_key(state)
                next_state_key = self.get_state_key(next_state)
                
                if self.discrete_states:
                    td_target = reward + self.gamma * self.Q[next_state_key, next_action] * (1 - done)
                    td_error = td_target - self.Q[state_key, action]
                    self.Q[state_key, action] += self.alpha * td_error
                else:
                    if state_key not in self.Q:
                        self.Q[state_key] = np.zeros(self.n_actions)
                    if next_state_key not in self.Q:
                        self.Q[next_state_key] = np.zeros(self.n_actions)
                    
                    td_target = reward + self.gamma * self.Q[next_state_key][next_action] * (1 - done)
                    td_error = td_target - self.Q[state_key][action]
                    self.Q[state_key][action] += self.alpha * td_error
                
                state = next_state
                action = next_action
            
            episode_rewards.append(episode_reward)
        
        return {
            'Q': self.Q,
            'episode_rewards': episode_rewards,
            'num_episodes': self.num_episodes,
            'algorithm': 'sarsa',
        }

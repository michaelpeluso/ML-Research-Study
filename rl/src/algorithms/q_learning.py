# AI Use Statement: Q-Learning algorithm created with GitHub Copilot assistance
"""q-learning off-policy td learning"""
import numpy as np
from typing import Optional, Dict, Any
import gymnasium as gym


class QLearning:
    """q-learning off-policy temporal difference learning"""
    
    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        num_episodes: int = 10000,
        seed: Optional[int] = None
    ):
        """
        initialize q-learning agent
        
        args:
            env: gymnasium environment
            alpha: learning rate
            gamma: discount factor
            epsilon: exploration rate
            num_episodes: number of training episodes
            seed: random seed for reproducibility
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.seed = seed
        
        # initialize q-table
        self.q_table = {}
        
        if seed is not None:
            np.random.seed(seed)
    
    def get_q_value(self, state, action) -> float:
        """get q value for state-action pair"""
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state) -> int:
        """epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # greedy action
        q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]
        return int(np.argmax(q_values))
    
    def train(self) -> Dict[str, Any]:
        """train q-learning agent and return results"""
        episode_returns = []
        
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            if isinstance(state, np.ndarray):
                state = tuple(state)
            
            episode_return = 0.0
            done = False
            
            while not done:
                # choose and take action
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if isinstance(next_state, np.ndarray):
                    next_state = tuple(next_state)
                
                episode_return += reward
                
                # q-learning update: q(s,a) <- q(s,a) + alpha * [r + gamma * max_a' q(s',a') - q(s,a)]
                max_next_q = max([self.get_q_value(next_state, a) for a in range(self.env.action_space.n)])
                td_target = reward + self.gamma * max_next_q * (not done)
                td_error = td_target - self.get_q_value(state, action)
                self.q_table[(state, action)] = self.get_q_value(state, action) + self.alpha * td_error
                
                state = next_state
            
            episode_returns.append(episode_return)
        
        return {
            'q_table': self.q_table,
            'episode_returns': episode_returns,
            'metadata': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'num_episodes': self.num_episodes,
                'seed': self.seed
            }
        }

# ai use statement: github copilot assisted with dqn network and replay buffer structure
"""deep q-network (dqn) for continuous state spaces - optional extra credit."""

from typing import Optional, Dict, Tuple
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """experience replay buffer for dqn."""
    
    def __init__(self, capacity: int):
        """initialize replay buffer with fixed capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """sample random batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQN:
    """
    deep q-network implementation.
    
    note: requires pytorch. install with: pip install torch
    """
    
    def __init__(self, env, network_config: dict, replay_config: dict, 
                 training_config: dict, seed: Optional[int] = None):
        """
        initialize dqn agent.
        
        env: gymnasium environment
        network_config: dict with hidden_layers, activation
        replay_config: dict with buffer_size, batch_size
        training_config: dict with gamma, learning_rate, target_update_freq, num_episodes
        seed: random seed for reproducibility
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("pytorch required for dqn. install with: pip install torch")
        
        self.env = env
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # config
        self.gamma = training_config['gamma']
        self.learning_rate = training_config['learning_rate']
        self.target_update_freq = training_config['target_update_freq']
        self.num_episodes = training_config['num_episodes']
        self.epsilon_start = training_config.get('epsilon_start', 1.0)
        self.epsilon_end = training_config.get('epsilon_end', 0.01)
        self.epsilon_decay = training_config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start
        
        # state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # replay buffer
        self.replay_buffer = ReplayBuffer(replay_config['buffer_size'])
        self.batch_size = replay_config['batch_size']
        
        # placeholder for network (implement in experiment if needed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device: {self.device}")
    
    def select_action(self, state):
        """epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # implement network forward pass here
            return self.env.action_space.sample()  # placeholder
    
    def train(self) -> Dict:
        """
        run dqn training.
        
        returns dict with rewards and metadata
        """
        episode_rewards = []
        
        for episode in range(self.num_episodes):
            state, _ = self.env.reset(seed=self.seed + episode if self.seed else None)
            episode_reward = 0
            done = False
            
            while not done:
                # select action
                action = self.select_action(state)
                
                # take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # store in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # train network (implement in experiment if needed)
                if len(self.replay_buffer) > self.batch_size:
                    pass  # implement training step
                
                state = next_state
            
            # decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            episode_rewards.append(episode_reward)
        
        return {
            'episode_rewards': episode_rewards,
            'num_episodes': self.num_episodes,
            'algorithm': 'dqn',
        }

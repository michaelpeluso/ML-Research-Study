"""Q-Learning algorithm implementation"""
"""q-learning off-policy td learning"""
import numpy as np
from typing import Optional, Dict, Any, Set
import gymnasium as gym


def compute_entropy(probs: np.ndarray) -> float:
    """compute shannon entropy of probability distribution"""
    probs = probs[probs > 0]  # filter zeros
    if len(probs) == 0:
        return 0.0

    # amount of information needed to describe the outcome (measures randomness)
    # shannon entropy: H(p) = - sum_i p_i * log(p_i)
    return float(np.sum(-probs * np.log(probs)))


class QLearning:
    """q-learning off-policy temporal difference learning"""
    
    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_floor: float = 0.01,
        epsilon_decay_episodes: int = 5000,
        num_episodes: int = 10000,
        seed: Optional[int] = None
    ):
        """
        initialize q-learning agent
        
        args:
            env: gymnasium environment
            alpha: learning rate
            gamma: discount factor
            epsilon: initial exploration rate (start high, decay over time)
            epsilon_floor: minimum exploration rate (never go below this)
            epsilon_decay_episodes: episodes over which to decay epsilon to floor
            num_episodes: number of training episodes
            seed: random seed for reproducibility
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon_floor = epsilon_floor
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon  # current epsilon (will decay)
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
        q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]  # type: ignore
        return int(np.argmax(q_values))
    
    def train(self) -> Dict[str, Any]:
        """train q-learning agent and return results"""
        episode_returns = []
        episode_steps = []
        episode_td_errors = []
        episode_q_changes = []
        episode_explorations = []
        episode_q_table_sizes = []
        episode_q_table_nonzeros = []
        episode_max_q_values = []
        episode_mean_q_values = []
        episode_action_entropies = []
        episode_td_error_stds = []
        episode_q_change_stds = []
        episode_unique_states = []
        
        for episode in range(self.num_episodes):
            # linear epsilon decay: ε(t) = max(ε_floor, ε_start - t * (ε_start - ε_floor) / decay_episodes)
            if episode < self.epsilon_decay_episodes:
                decay_progress = episode / self.epsilon_decay_episodes
                self.epsilon = self.epsilon_start - decay_progress * (self.epsilon_start - self.epsilon_floor)
            else:
                self.epsilon = self.epsilon_floor
            
            state, _ = self.env.reset()
            if isinstance(state, np.ndarray):
                state = tuple(state)
            
            episode_return = 0.0
            steps = 0
            td_errors_sum = 0.0
            q_changes_sum = 0.0
            explorations = 0
            td_errors_list = []
            q_changes_list = []
            unique_states: Set = set()
            action_counts = np.zeros(self.env.action_space.n)  # type: ignore
            
            done = False
            while not done:
                # track unique states and actions
                unique_states.add(state)
                
                # choose and take action
                action = self.choose_action(state)
                action_counts[action] += 1
                
                # track exploration
                if np.random.random() < self.epsilon:
                    explorations += 1
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if isinstance(next_state, np.ndarray):
                    next_state = tuple(next_state)
                
                episode_return += float(reward)
                steps += 1
                
                # off-policy TD bootstrap toward the greedy next-state value
                # q-learning update: q(s,a) <- q(s,a) + α [ r + γ max_{a'} Q(s',a') - Q(s,a) ]
                old_q = self.get_q_value(state, action)
                # max_next_q = max_{a'} Q(s',a')
                max_next_q = max([self.get_q_value(next_state, a) for a in range(self.env.action_space.n)])  # type: ignore
                # td_target = r + γ * max_next_q  (bootstrap; zero at terminal)
                td_target = float(reward) + self.gamma * max_next_q * (not done)
                # δ = td_target - Q(s,a)
                td_error = td_target - old_q
                # Q(s,a) ← Q(s,a) + α * δ  (gradient-free incremental update)
                new_q = old_q + self.alpha * td_error
                self.q_table[(state, action)] = new_q
                
                # track metrics
                td_errors_sum += abs(td_error)
                q_changes_sum += abs(new_q - old_q)
                td_errors_list.append(abs(td_error))
                q_changes_list.append(abs(new_q - old_q))
                
                state = next_state
            
            episode_returns.append(episode_return)
            episode_steps.append(steps)
            episode_td_errors.append(td_errors_sum / max(steps, 1))
            episode_q_changes.append(q_changes_sum / max(steps, 1))
            episode_explorations.append(explorations / max(steps, 1))
            
            # q-table statistics
            episode_q_table_sizes.append(len(self.q_table))
            nonzero_q = sum(1 for v in self.q_table.values() if abs(v) > 1e-10)
            episode_q_table_nonzeros.append(nonzero_q)
            
            # value function statistics
            q_values = list(self.q_table.values())
            if q_values:
                episode_max_q_values.append(max(q_values))
                episode_mean_q_values.append(np.mean(q_values))
            else:
                episode_max_q_values.append(0.0)
                episode_mean_q_values.append(0.0)
            
            # action entropy (policy convergence)
            action_probs = action_counts / max(action_counts.sum(), 1)
            action_probs = action_probs[action_probs > 0]
            episode_action_entropies.append(compute_entropy(action_probs))
            
            # variance metrics (stability)
            episode_td_error_stds.append(np.std(td_errors_list) if len(td_errors_list) > 1 else 0.0)
            episode_q_change_stds.append(np.std(q_changes_list) if len(q_changes_list) > 1 else 0.0)
            
            # exploration breadth
            episode_unique_states.append(len(unique_states))
        
        return {
            'q_table': self.q_table,
            'episode_returns': episode_returns,
            'episode_steps': episode_steps,
            'episode_td_errors': episode_td_errors,
            'episode_q_changes': episode_q_changes,
            'episode_explorations': episode_explorations,
            'episode_q_table_sizes': episode_q_table_sizes,
            'episode_q_table_nonzeros': episode_q_table_nonzeros,
            'episode_max_q_values': episode_max_q_values,
            'episode_mean_q_values': episode_mean_q_values,
            'episode_action_entropies': episode_action_entropies,
            'episode_td_error_stds': episode_td_error_stds,
            'episode_q_change_stds': episode_q_change_stds,
            'episode_unique_states': episode_unique_states,
            'metadata': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_floor': self.epsilon_floor,
                'epsilon_decay_episodes': self.epsilon_decay_episodes,
                'num_episodes': self.num_episodes,
                'seed': self.seed
            }
        }

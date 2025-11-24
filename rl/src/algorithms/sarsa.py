"""SARSA algorithm implementation"""
"""sarsa on-policy td learning"""
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


class SARSA:
    """sarsa on-policy temporal difference learning"""
    
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
        initialize sarsa agent
        
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
        # sample from environment action space
        # exploratory action: P(explore) = ε
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # greedy action: argmax_a Q(s,a)
        q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]  # type: ignore
        return int(np.argmax(q_values))
    
    def train(self) -> Dict[str, Any]:
        """train sarsa agent and return results"""
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
            state, _ = self.env.reset()
            if isinstance(state, np.ndarray):
                state = tuple(state)
            
            action = self.choose_action(state)
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
                action_counts[action] += 1
                
                # track exploration indicator for this step
                # bernoulli trial: I_explore = 1 if U(0,1) < ε
                if np.random.random() < self.epsilon:
                    explorations += 1
                
                # take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if isinstance(next_state, np.ndarray):
                    next_state = tuple(next_state)
                
                episode_return += float(reward)
                steps += 1
                
                # choose next action according to current policy π (epsilon-greedy)
                # π(s') -> a' (depends on ε and Q)
                next_action = self.choose_action(next_state)
                
                # sarsa update: q(s,a) <- q(s,a) + alpha * [r + gamma * q(s',a') - q(s,a)]

                # old_q = Q(s,a)
                old_q = self.get_q_value(state, action)
                # td_target = r + γ * Q(s',a') * (1 - I_terminal)
                td_target = float(reward) + self.gamma * self.get_q_value(next_state, next_action) * (not done)
                # temporal difference error: δ = td_target - Q(s,a)
                td_error = td_target - old_q
                # Q update: Q(s,a) ← Q(s,a) + α * δ
                new_q = old_q + self.alpha * td_error
                self.q_table[(state, action)] = new_q
                
                # track metrics
                td_errors_sum += abs(td_error)
                q_changes_sum += abs(new_q - old_q)
                td_errors_list.append(abs(td_error))
                q_changes_list.append(abs(new_q - old_q))
                
                state = next_state
                action = next_action
            
            episode_returns.append(episode_return)
            episode_steps.append(steps)
            episode_td_errors.append(td_errors_sum / max(steps, 1))
            episode_q_changes.append(q_changes_sum / max(steps, 1))
            episode_explorations.append(explorations / max(steps, 1))
            
            # q-table statistics
            # size = |Q| (number of stored state-action entries)
            episode_q_table_sizes.append(len(self.q_table))
            # nonzero_q = count_{(s,a)} [ |Q(s,a)| > threshold ]
            nonzero_q = sum(1 for v in self.q_table.values() if abs(v) > 1e-10)
            episode_q_table_nonzeros.append(nonzero_q)
            
            # value function statistics
            # max Q = max_{(s,a)} Q(s,a)
            # mean Q = (1/n) * sum_{(s,a)} Q(s,a)
            q_values = list(self.q_table.values())
            if q_values:
                episode_max_q_values.append(max(q_values))
                episode_mean_q_values.append(np.mean(q_values))
            else:
                episode_max_q_values.append(0.0)
                episode_mean_q_values.append(0.0)
            
            # action entropy (policy convergence)
            # action_probs = p_i = n_i / Σ n_j
            action_probs = action_counts / max(action_counts.sum(), 1)
            action_probs = action_probs[action_probs > 0]  # filter zeros for entropy
            # H = - sum_i p_i log p_i
            episode_action_entropies.append(compute_entropy(action_probs))
            
            # variance metrics (stability)
            # std = sqrt( (1/n) * sum_i (x_i - mean)^2 )
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
                'epsilon': self.epsilon,
                'num_episodes': self.num_episodes,
                'seed': self.seed
            }
        }

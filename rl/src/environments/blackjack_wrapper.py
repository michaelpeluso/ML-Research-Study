"""Blackjack environment wrapper"""
"""gymnasium blackjack wrapper with seed control"""
import gymnasium as gym
from typing import Tuple, Any


class BlackjackWrapper:
    """wrapper for gymnasium blackjack environment"""
    
    def __init__(self, natural: bool = False, sab: bool = False):
        """
        initialize blackjack environment
        
        args:
            natural: whether natural blackjack pays 1.5x
            sab: whether to use stick-and-bust variant
        """
        self.env = gym.make('Blackjack-v1', natural=natural, sab=sab)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # expose transition model for dynamic programming algorithms
        if hasattr(self.env.unwrapped, 'P'):
            self.P = self.env.unwrapped.P
        
        # compute state space size for VI/PI
        self.num_states = len(self.get_state_space())
    
    def reset(self, seed: int = None) -> Tuple[Any, dict]:
        """reset environment with optional seed"""
        if seed is not None:
            return self.env.reset(seed=seed)
        return self.env.reset()
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        """take action and return (state, reward, terminated, truncated, info)"""
        return self.env.step(action)
    
    def close(self) -> None:
        """close environment"""
        self.env.close()
    
    def get_state_space(self) -> list:
        """return list of all possible states"""
        # player sum: 4-21, dealer showing: 1-10, usable ace: True/False
        states = []
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                for usable_ace in [False, True]:
                    states.append((player_sum, dealer_card, usable_ace))
        return states

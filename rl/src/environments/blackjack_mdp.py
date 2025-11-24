"""blackjack mdp with explicit transition model for value/policy iteration"""
import numpy as np
from typing import Dict, Tuple, List, Optional
import gymnasium as gym


class BlackjackMDP:
    """blackjack environment with explicit transition model P for dynamic programming"""
    
    def __init__(self, natural: bool = False, sab: bool = False):
        """
        initialize blackjack mdp with transition model
        
        args:
            natural: whether natural blackjack pays 1.5x
            sab: whether to use stick-and-bust variant
        """
        self.env = gym.make('Blackjack-v1', natural=natural, sab=sab)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.natural = natural
        self.sab = sab
        
        # build state space
        self.states = self._build_state_space()
        self.num_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        # build transition model via sampling
        print(f"building transition model for {self.num_states} states...")
        self.P = self._build_transition_model(samples_per_state_action=1000)
        print(f"transition model complete: {sum(len(self.P[s][a]) for s in self.P for a in self.P[s])} transitions")
    
    def _build_state_space(self) -> List[Tuple[int, int, bool]]:
        """enumerate all possible states"""
        states = []
        # player sum: 4-21 (below 4 is impossible without bust)
        for player_sum in range(4, 22):
            # dealer showing: ace(1) through 10
            for dealer_card in range(1, 11):
                # usable ace: true/false
                for usable_ace in [False, True]:
                    # skip invalid states (usable ace with sum < 12)
                    if usable_ace and player_sum < 12:
                        continue
                    states.append((player_sum, dealer_card, usable_ace))
        return states
    
    def _build_transition_model(self, samples_per_state_action: int = 1000) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        """
        empirically estimate transition probabilities via monte carlo sampling
        
        P[s][a] = [(prob, next_state, reward, done), ...]
        """
        num_actions = getattr(self.action_space, 'n', None)
        if num_actions is None:
            raise TypeError("action_space.n is not defined for this environment")
        P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        for state_idx, state in enumerate(self.states):
            if (state_idx + 1) % 50 == 0:
                print(f"  processed {state_idx + 1}/{self.num_states} states...")
            P[state_idx] = {}
            for action in range(num_actions):
                transitions = {}
                for _ in range(samples_per_state_action):
                    self.env.reset()
                    obs, _ = self.env.reset()
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    if next_obs in self.state_to_idx:
                        next_state_idx = self.state_to_idx[next_obs]
                    else:
                        next_state_idx = -1
                    key = (next_state_idx, reward, done)
                    transitions[key] = transitions.get(key, 0) + 1
                total = sum(transitions.values())
                P[state_idx][action] = [
                    (count / total, next_s, r, d)
                    for (next_s, r, d), count in transitions.items()
                ]
        return P
    
    def reset(self, seed: Optional[int] = None):
        # gymnasium expects int or None for seed
        return self.env.reset(seed=seed)
        """reset environment"""
        return self.env.reset(seed=seed)
    
    def step(self, action: int):
        """take action"""
        return self.env.step(action)
    
    def close(self):
        """close environment"""
        self.env.close()


class SimplifiedBlackjackMDP:
    """simplified blackjack mdp with analytical transition model"""
    
    def __init__(self):
        """initialize simplified blackjack for vi/pi demonstration"""
        self.env = gym.make('Blackjack-v1', natural=False, sab=False)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # simplified state space: focus on key decision states
        # state = (player_sum, dealer_showing, usable_ace)
        # player_sum: 12-21 (below 12 always hit, above 21 is bust)
        # dealer_showing: 2-11 (ace=11 for simplicity)
        # usable_ace: 0 or 1
        
        self.player_sums = list(range(12, 22))  # 12-21
        self.dealer_cards = list(range(2, 12))  # 2-11 (ace as 11)
        self.usable_aces = [0, 1]
        
        self.states = []
        for ps in self.player_sums:
            for dc in self.dealer_cards:
                for ua in self.usable_aces:
                    self.states.append((ps, dc, ua))
        
        self.num_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        # build simplified transition model
        print(f"building simplified transition model for {self.num_states} states...")
        self.P = self._build_simplified_transitions()
        print(f"transition model complete!")
    
    def _build_simplified_transitions(self) -> Dict:
        """build simplified analytical transition model"""
        P = {}
        
        for state_idx, state in enumerate(self.states):
            player_sum, dealer_card, usable_ace = state
            P[state_idx] = {}
            
            # action 0: stick
            # when sticking, game resolves immediately
            # simplified: estimate dealer bust probability and win/loss
            if dealer_card <= 6:
                # dealer likely to bust
                P[state_idx][0] = [(0.6, -1, 1.0, True), (0.4, -1, -1.0, True)]
            elif dealer_card <= 9:
                # dealer moderate
                if player_sum >= 17:
                    P[state_idx][0] = [(0.5, -1, 1.0, True), (0.5, -1, -1.0, True)]
                else:
                    P[state_idx][0] = [(0.3, -1, 1.0, True), (0.7, -1, -1.0, True)]
            else:
                # dealer strong
                if player_sum >= 19:
                    P[state_idx][0] = [(0.4, -1, 1.0, True), (0.6, -1, -1.0, True)]
                else:
                    P[state_idx][0] = [(0.2, -1, 1.0, True), (0.8, -1, -1.0, True)]
            
            # action 1: hit
            # when hitting, draw a card (simplified uniform distribution)
            P[state_idx][1] = []
            card_probs = 1.0 / 10.0  # simplified: 10 possible cards (2-11)
            
            for card in range(2, 12):  # cards 2-11
                new_sum = player_sum + card
                
                if new_sum > 21:
                    # bust
                    if usable_ace:
                        # convert ace from 11 to 1
                        new_sum -= 10
                        new_usable_ace = 0
                        if new_sum <= 21:
                            # didn't bust, continue
                            next_state = (new_sum, dealer_card, new_usable_ace)
                            if next_state in self.state_to_idx:
                                next_idx = self.state_to_idx[next_state]
                                P[state_idx][1].append((card_probs, next_idx, 0.0, False))
                            else:
                                P[state_idx][1].append((card_probs, -1, -1.0, True))
                        else:
                            # still bust
                            P[state_idx][1].append((card_probs, -1, -1.0, True))
                    else:
                        # bust without usable ace
                        P[state_idx][1].append((card_probs, -1, -1.0, True))
                else:
                    # didn't bust
                    new_usable_ace = 1 if (card == 11 or usable_ace) else 0
                    next_state = (new_sum, dealer_card, new_usable_ace)
                    if next_state in self.state_to_idx:
                        next_idx = self.state_to_idx[next_state]
                        P[state_idx][1].append((card_probs, next_idx, 0.0, False))
                    else:
                        # terminal or invalid
                        P[state_idx][1].append((card_probs, -1, 0.0, True))
        
        return P
    
    def reset(self, seed: Optional[int] = None):
        # gymnasium expects int or None for seed
        return self.env.reset(seed=seed)
        """reset environment"""
        return self.env.reset(seed=seed)
    
    def step(self, action: int):
        """take action"""
        return self.env.step(action)
    
    def close(self):
        """close environment"""
        self.env.close()

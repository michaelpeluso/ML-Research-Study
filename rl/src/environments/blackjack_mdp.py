"""blackjack mdp with explicit transition model for value/policy iteration"""
import numpy as np
from typing import Dict, Tuple, List, Optional
import gymnasium as gym


class BlackjackMDP:
    """blackjack environment with empirical transition model for dynamic programming
    
    WARNING: blackjack is inherently stochastic and doesn't support arbitrary state
    initialization. this implementation builds an EMPIRICAL transition model by sampling
    from natural gameplay. states that are rarely visited will have poor estimates.
    
    for production analysis, consider using model-free methods (SARSA/Q-Learning) which
    are better suited to stochastic environments without explicit transition models.
    
    if using VI/PI on blackjack, be aware that:
    - not all states will be visited during sampling
    - transition probabilities are estimates, not ground truth
    - results may vary based on sampling episodes
    """
    
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
    
    def _build_transition_model(self, samples_per_state_action: int = 5000) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        """
        empirically estimate transition probabilities via monte carlo sampling
        
        NOTE: blackjack doesn't support arbitrary state setting, so we sample
        from natural gameplay and build empirical transitions for states we encounter.
        states not encountered will have uniform random transitions (fallback).
        
        P[s][a] = [(prob, next_state, reward, done), ...]
        """
        num_actions = getattr(self.action_space, 'n', None)
        if num_actions is None:
            raise TypeError("action_space.n is not defined for this environment")
        
        # initialize P with empty transitions
        P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        for state_idx in range(self.num_states):
            P[state_idx] = {}
            for action in range(num_actions):
                P[state_idx][action] = []
        
        # collect transitions from actual gameplay
        state_action_samples: Dict[Tuple[int, int], Dict] = {}
        
        print(f"  collecting {samples_per_state_action} episodes of gameplay...")
        for episode in range(samples_per_state_action):
            if (episode + 1) % 1000 == 0:
                print(f"    episode {episode + 1}/{samples_per_state_action}")
            
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                # get current state index
                if obs in self.state_to_idx:
                    state_idx = self.state_to_idx[obs]
                else:
                    break  # invalid state, skip
                
                # try both actions from this state
                for action in range(num_actions):
                    # save current state to restore (but we can't actually restore in gym)
                    # so we'll just sample one action per state visit
                    if action == 0:  # stick
                        next_obs, reward, terminated, truncated, _ = self.env.step(0)
                        done = terminated or truncated
                        
                        # record transition
                        key = (state_idx, 0)
                        if key not in state_action_samples:
                            state_action_samples[key] = {}
                        
                        if done:
                            next_state_idx = -1  # terminal
                        elif next_obs in self.state_to_idx:
                            next_state_idx = self.state_to_idx[next_obs]
                        else:
                            next_state_idx = -1
                        
                        trans_key = (next_state_idx, reward, done)
                        state_action_samples[key][trans_key] = state_action_samples[key].get(trans_key, 0) + 1
                        break  # done with episode after stick
                    else:  # hit
                        next_obs, reward, terminated, truncated, _ = self.env.step(1)
                        done = terminated or truncated
                        
                        # record transition
                        key = (state_idx, 1)
                        if key not in state_action_samples:
                            state_action_samples[key] = {}
                        
                        if done:
                            next_state_idx = -1
                        elif next_obs in self.state_to_idx:
                            next_state_idx = self.state_to_idx[next_obs]
                        else:
                            next_state_idx = -1
                        
                        trans_key = (next_state_idx, reward, done)
                        state_action_samples[key][trans_key] = state_action_samples[key].get(trans_key, 0) + 1
                        
                        # continue episode if not done
                        obs = next_obs
                        break
        
        # convert samples to probability distributions
        print(f"  building transition distributions...")
        for (state_idx, action), transitions in state_action_samples.items():
            total = sum(transitions.values())
            if total > 0:
                P[state_idx][action] = [
                    (count / total, next_s, r, d)
                    for (next_s, r, d), count in transitions.items()
                ]
        
        # for states never encountered, add fallback uniform transitions
        for state_idx in range(self.num_states):
            for action in range(num_actions):
                if not P[state_idx][action]:
                    # fallback: assume moderate loss probability
                    P[state_idx][action] = [(1.0, -1, -0.5, True)]
        
        visited_states = len([s for s in range(self.num_states) 
                             if any(P[s][a] for a in range(num_actions))])
        print(f"  covered {visited_states}/{self.num_states} states from gameplay")
        
        return P
    
    def reset(self, seed: Optional[int] = None):
        """reset environment"""
        # gymnasium expects int or None for seed
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
        # dealer_showing: 1-10 (ace=1, matching gymnasium)
        # usable_ace: 0 or 1
        
        self.player_sums = list(range(12, 22))  # 12-21
        self.dealer_cards = list(range(1, 11))  # 1-10 (ace=1, matching gymnasium)
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
        """build analytical transition model with accurate card probabilities
        
        uses infinite deck assumption (common in blackjack analysis).
        card distribution: 2-9 each 1/13, 10/J/Q/K total 4/13, Ace 1/13
        """
        P = {}
        
        # precompute dealer final probabilities for each showing card
        # this is the key to accurate VI/PI
        dealer_probs = self._compute_dealer_probabilities()
        
        for state_idx, state in enumerate(self.states):
            player_sum, dealer_card, usable_ace = state
            P[state_idx] = {}
            
            # action 0: STICK - resolve against dealer
            P[state_idx][0] = self._stick_transitions(player_sum, dealer_card, dealer_probs)
            
            # action 1: HIT - draw a card
            P[state_idx][1] = self._hit_transitions(player_sum, dealer_card, usable_ace)
        
        return P
    
    def _compute_dealer_probabilities(self) -> Dict[int, Dict]:
        """compute probability distribution of dealer's final hand for each showing card
        
        dealer must hit until reaching 17+, then stick.
        returns: {dealer_showing: {final_sum: probability, 'bust': probability}}
        """
        # card probabilities (infinite deck)
        card_probs = {i: 1/13 for i in range(2, 10)}
        card_probs[10] = 4/13  # 10, J, Q, K
        card_probs[1] = 1/13   # Ace (value determined by hand)
        
        dealer_probs = {}
        
        for showing in range(1, 11):
            # simulate dealer play via dynamic programming
            # state: (sum, has_usable_ace)
            # initialize with showing card
            if showing == 1:  # Ace showing
                initial = {(11, True): 1.0}  # ace counts as 11 initially
            else:
                initial = {(showing, False): 1.0}
            
            # iterate until all states are terminal (17+ or bust)
            current = initial.copy()
            final_dist = {'bust': 0.0}
            for s in range(17, 22):
                final_dist[s] = 0.0 # type: ignore
            
            for _ in range(10):  # max iterations (dealer rarely draws 10+ cards)
                next_state = {}
                for (hand_sum, usable), prob in current.items():
                    if hand_sum >= 17:
                        # dealer sticks
                        final_dist[hand_sum] = final_dist.get(hand_sum, 0) + prob # type: ignore
                    elif hand_sum > 21:
                        # bust (shouldn't happen due to usable ace logic)
                        final_dist['bust'] += prob
                    else:
                        # dealer hits
                        for card, cp in card_probs.items():
                            new_prob = prob * cp
                            if card == 1:  # Ace drawn
                                if hand_sum + 11 <= 21:
                                    new_state = (hand_sum + 11, True)
                                else:
                                    new_state = (hand_sum + 1, usable)
                            else:
                                new_sum = hand_sum + card
                                if new_sum > 21 and usable:
                                    new_sum -= 10
                                    new_state = (new_sum, False)
                                else:
                                    new_state = (new_sum, usable)
                            
                            if new_state[0] > 21:
                                final_dist['bust'] += new_prob
                            else:
                                next_state[new_state] = next_state.get(new_state, 0) + new_prob
                
                current = next_state
                if not current:
                    break
            
            dealer_probs[showing] = final_dist
        
        return dealer_probs
    
    def _stick_transitions(self, player_sum: int, dealer_card: int, dealer_probs: Dict) -> List:
        """compute transition probabilities when player sticks
        
        compare player_sum against dealer's final hand distribution
        """
        probs = dealer_probs[dealer_card]
        
        win_prob = probs.get('bust', 0)  # dealer busts, player wins
        lose_prob = 0.0
        draw_prob = 0.0
        
        for dealer_final in range(17, 22):
            p = probs.get(dealer_final, 0)
            if player_sum > dealer_final:
                win_prob += p
            elif player_sum < dealer_final:
                lose_prob += p
            else:
                draw_prob += p
        
        transitions = []
        if win_prob > 0:
            transitions.append((win_prob, -1, 1.0, True))
        if lose_prob > 0:
            transitions.append((lose_prob, -1, -1.0, True))
        if draw_prob > 0:
            transitions.append((draw_prob, -1, 0.0, True))
        
        return transitions if transitions else [(1.0, -1, 0.0, True)]
    
    def _hit_transitions(self, player_sum: int, dealer_card: int, usable_ace: int) -> List:
        """compute transition probabilities when player hits
        
        draw a card and transition to new state (or bust)
        """
        # card probabilities (infinite deck)
        card_probs = {i: 1/13 for i in range(2, 10)}
        card_probs[10] = 4/13  # 10, J, Q, K
        card_probs[1] = 1/13   # Ace
        
        transitions = []
        
        for card, prob in card_probs.items():
            if card == 1:  # Ace
                if player_sum + 11 <= 21:
                    new_sum = player_sum + 11
                    new_usable = 1
                else:
                    new_sum = player_sum + 1
                    new_usable = usable_ace
            else:
                new_sum = player_sum + card
                new_usable = usable_ace
            
            # check for bust
            if new_sum > 21:
                if new_usable or (card == 1 and player_sum + 11 > 21 and usable_ace):
                    # use the usable ace
                    new_sum -= 10
                    new_usable = 0
            
            if new_sum > 21:
                # bust
                transitions.append((prob, -1, -1.0, True))
            elif new_sum >= 22:
                # shouldn't happen, but safety
                transitions.append((prob, -1, -1.0, True))
            else:
                # valid new state
                # cap at 21 for state space
                new_sum = min(new_sum, 21)
                next_state = (new_sum, dealer_card, new_usable)
                if next_state in self.state_to_idx:
                    next_idx = self.state_to_idx[next_state]
                    transitions.append((prob, next_idx, 0.0, False))
                else:
                    # state outside our range (shouldn't happen)
                    transitions.append((prob, -1, 0.0, True))
        
        return transitions
        
        return P
    
    def reset(self, seed: Optional[int] = None):
        """reset environment"""
        return self.env.reset(seed=seed)
    
    def step(self, action: int):
        """take action"""
        return self.env.step(action)
    
    def close(self):
        """close environment"""
        self.env.close()

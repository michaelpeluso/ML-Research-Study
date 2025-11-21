"""agent module exports."""

from .value_iteration import ValueIteration
from .policy_iteration import PolicyIteration
from .sarsa import SARSA
from .q_learning import QLearning
from .discretization import StateDiscretizer
from .utils import set_seeds, get_git_sha, make_filename, save_metadata, ensure_figures_dir

try:
    from .dqn import DQN, ReplayBuffer
except ImportError:
    # pytorch not installed
    DQN = None
    ReplayBuffer = None

__all__ = [
    'ValueIteration',
    'PolicyIteration',
    'SARSA',
    'QLearning',
    'StateDiscretizer',
    'DQN',
    'ReplayBuffer',
    'set_seeds',
    'get_git_sha',
    'make_filename',
    'save_metadata',
    'ensure_figures_dir',
]

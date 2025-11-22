# AI Use Statement: seeding utility functions created with GitHub Copilot assistance
"""unified seed control for reproducibility"""
import random
import numpy as np


def set_all_seeds(seed: int) -> None:
    """set seeds for random, numpy, and torch (if available)"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seed_env(env, seed: int):
    """seed a gymnasium environment"""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

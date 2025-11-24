"""Seeding utilities: set RNG seeds for reproducibility"""
"""unified seed control for reproducibility"""
import random
import numpy as np
#import torch


def set_all_seeds(seed: int) -> None:
    """set seeds for random, numpy, and torch (if available)"""
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(seed)


def seed_env(env, seed: int):
    """seed a gymnasium environment"""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

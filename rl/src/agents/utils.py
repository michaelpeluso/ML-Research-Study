# ai use statement: github copilot assisted with boilerplate structure and docstring formatting
"""utility functions for reproducibility, seeding, and metadata tracking."""

import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


def set_seeds(seed: int) -> None:
    """set all random seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_sha(short: bool = True) -> str:
    """return current git commit sha or 'dev' if not in git repo."""
    try:
        cmd = ['git', 'rev-parse', '--short' if short else '', 'HEAD']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'dev'


def make_filename(task: str, sha: str, ext: str, timestamp: Optional[str] = None, suffix: str = '') -> str:
    """generate filename with format: {task}_{sha}_{timestamp}_{suffix}.{ext}"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    
    parts = [task, sha, timestamp]
    if suffix:
        parts.append(suffix)
    
    base = '_'.join(parts)
    return f"{base}.{ext}"


def save_metadata(filepath: str, metadata: dict) -> None:
    """save experiment metadata as json sidecar."""
    metadata_path = Path(filepath).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def ensure_figures_dir() -> Path:
    """ensure figures directory exists and return path."""
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    return figures_dir

"""Experiment logging utilities

robust experiment logging and metadata utilities

features:
- write episode-level CSVs with consistent headers
- write JSON metadata sidecar containing timestamp, seed, hyperparams, and command
- filenames are stable (no timestamps) so repeated runs replace previous outputs
- context-manager support and safe append
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def _convert_to_json_serializable(obj: Any) -> Any:
    """recursively convert numpy arrays and other non-serializable types to json-safe types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def make_filename(base: str, ext: str = 'csv') -> str:
    """generate stable filename (no timestamp) so outputs overwrite previous runs"""
    return f"{base}.{ext}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class ExperimentLogger:
    """CSV + JSON metadata logger for experiments.

    Usage:
        with ExperimentLogger(output_dir, 'sarsa_cartpole', seed=0, metadata=meta) as logger:
            logger.log_episode(0, {'episode_return': 1.0})
    """

    def __init__(
        self,
        output_dir: Path,
        experiment_name: str,
        seed: int,
        metadata: Optional[Dict[str, Any]] = None,
        filename_base: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        _ensure_dir(self.output_dir)

        self.experiment_name = experiment_name
        self.seed = seed
        self.metadata = metadata or {}

        base = filename_base or f"{experiment_name}_seed{seed}"
        self.filename = make_filename(base, ext='csv')
        self.log_file = self.output_dir / self.filename

        # json metadata sidecar path
        self.meta_file = self.log_file.with_suffix('.json')

        self._start_time = time.time()
        self._file_handle = None
        self._writer = None
        self._fieldnames = None

        # prepare metadata
        self.metadata.setdefault('experiment_name', experiment_name)
        self.metadata.setdefault('seed', seed)
        self.metadata.setdefault('created_at_utc', datetime.utcnow().isoformat() + 'Z')

        # write metadata sidecar upfront
        self._write_metadata()

    def _write_metadata(self) -> None:
        """write metadata to json sidecar file with safe serialization and organized structure"""
        try:
            # convert any non-serializable objects (numpy arrays, etc.)
            serializable_metadata = _convert_to_json_serializable(self.metadata)
            
            # reorganize into logical sections for better readability
            organized = {
                'experiment': {
                    'name': serializable_metadata.get('experiment_name', ''),
                    'algorithm': serializable_metadata.get('algorithm', ''),
                    'environment': serializable_metadata.get('environment', ''),
                    'seed': serializable_metadata.get('seed', None),
                    'created_at_utc': serializable_metadata.get('created_at_utc', ''),
                    'completed_at_utc': serializable_metadata.get('completed_at_utc', ''),
                    'duration_sec': serializable_metadata.get('duration_sec', None),
                },
                'hyperparameters': {
                    'alpha': serializable_metadata.get('alpha', None),
                    'gamma': serializable_metadata.get('gamma', None),
                    'epsilon': serializable_metadata.get('epsilon', None),
                    'episodes': serializable_metadata.get('episodes', None),
                },
                'environment_info': serializable_metadata.get('env', {}),
                'results': serializable_metadata.get('summary', {}),
            }
            
            # add any extra fields not captured above
            standard_keys = {'experiment_name', 'algorithm', 'environment', 'seed', 'created_at_utc', 
                           'duration_sec', 'alpha', 'gamma', 'epsilon', 'episodes', 'env', 'summary'}
            extra = {k: v for k, v in serializable_metadata.items() if k not in standard_keys}
            if extra:
                organized['extra'] = extra
            
            with open(self.meta_file, 'w', encoding='utf-8') as mf:
                json.dump(organized, mf, indent=2)
        except Exception as e:
            # write error info to a companion .err file for debugging
            err_file = self.meta_file.with_suffix('.json.err')
            try:
                with open(err_file, 'w') as ef:
                    ef.write(f"JSON serialization error: {e}\n")
                    ef.write(f"Metadata keys: {list(self.metadata.keys())}\n")
            except Exception:
                pass

    def __enter__(self) -> 'ExperimentLogger':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def log_episode(self, episode: int, metrics: Dict[str, Any]) -> None:
        """log metrics for a single episode; handle header initialization and safe append"""
        if self._writer is None:
            # initialize CSV writer
            self._fieldnames = ['episode', 'wall_clock_sec'] + sorted(metrics.keys())
            mode = 'w'
            self._file_handle = open(self.log_file, mode, newline='', encoding='utf-8')
            self._writer = csv.DictWriter(self._file_handle, fieldnames=self._fieldnames)
            self._writer.writeheader()

        # ensure _fieldnames is initialized
        if self._fieldnames is None:
            self._fieldnames = ['episode', 'wall_clock_sec'] + sorted(metrics.keys())

        # ensure row follows fieldnames order
        row = {'episode': episode, 'wall_clock_sec': time.time() - self._start_time}
        for key in self._fieldnames:
            if key in row:
                continue
            if key in metrics:
                row[key] = metrics[key]
            else:
                row[key] = ''

        self._writer.writerow(row)
        try:
            self._file_handle.flush() # type: ignore
        except Exception:
            pass

    def close(self) -> None:
        """close internal file handle"""
        # attach run duration and completion timestamp to metadata
        try:
            self.metadata['duration_sec'] = time.time() - self._start_time
            self.metadata['completed_at_utc'] = datetime.utcnow().isoformat() + 'Z'
        except Exception:
            pass

        # write updated metadata
        self._write_metadata()

        try:
            if self._file_handle:
                self._file_handle.close()
        finally:
            self._file_handle = None

    def update_metadata(self, new: Dict[str, Any], write: bool = True) -> None:
        """merge new metadata fields and optionally rewrite sidecar"""
        try:
            self.metadata.update(new)
            if write:
                self._write_metadata()
        except Exception:
            pass
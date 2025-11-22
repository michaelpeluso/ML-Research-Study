# AI Use Statement: experiment logging utilities created with GitHub Copilot assistance
"""robust experiment logging and metadata utilities

features:
- write episode-level CSVs with consistent headers
- write JSON metadata sidecar containing commit sha, timestamp, seed, hyperparams, and command
- filenames include git short sha and timestamp
- context-manager support and safe append
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import sys
import platform
from importlib import metadata as importlib_metadata


def get_git_sha(short: bool = True) -> str:
    """return current git commit short sha or 'dev' if unavailable"""
    try:
        args = ['git', 'rev-parse', '--short', 'HEAD'] if short else ['git', 'rev-parse', 'HEAD']
        sha = subprocess.check_output(args, cwd=Path('.')).decode().strip()
        return sha
    except Exception:
        return 'dev'


def make_filename(base: str, sha: Optional[str] = None, ext: str = 'csv') -> str:
    """generate filename with optional git sha and timestamp"""
    sha = sha or get_git_sha()
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    return f"{base}_{sha}_{timestamp}.{ext}"


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
        self.sha = get_git_sha()
        self.filename = make_filename(base, sha=self.sha, ext='csv')
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
        self.metadata.setdefault('commit_sha', self.sha)
        self.metadata.setdefault('created_at_utc', datetime.utcnow().isoformat() + 'Z')

        # write metadata sidecar upfront
        self._write_metadata()

    def _write_metadata(self) -> None:
        try:
            with open(self.meta_file, 'w', encoding='utf-8') as mf:
                json.dump(self.metadata, mf, indent=2, sort_keys=True)
        except Exception:
            # do not crash experiments for metadata write failure
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
            self._file_handle.flush()
        except Exception:
            pass

    def close(self) -> None:
        """close internal file handle"""
        # attach run duration to metadata and rewrite sidecar
        try:
            self.metadata['duration_sec'] = time.time() - self._start_time
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

    def attach_system_info(self, packages: Optional[list] = None, write: bool = True) -> None:
        """collect python/platform and package versions and attach to metadata"""
        try:
            sys_info = {
                'python_version': sys.version.replace('\n', ' '),
                'platform': platform.platform(),
            }
            pkg_versions = get_package_versions(packages)
            self.metadata.setdefault('system', {})
            self.metadata['system'].update(sys_info)
            self.metadata['system']['packages'] = pkg_versions
            if write:
                self._write_metadata()
        except Exception:
            pass

    def get_log_path(self) -> Path:
        return self.log_file

    def get_meta_path(self) -> Path:
        return self.meta_file


def get_package_versions(packages: Optional[list] = None) -> Dict[str, str]:
    """return a dict of package -> version for requested packages (best-effort)"""
    packages = packages or ['numpy', 'pandas', 'gymnasium', 'matplotlib', 'seaborn', 'torch', 'pyyaml']
    out: Dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = 'not-installed'
        except Exception:
            out[pkg] = 'unknown'
    return out


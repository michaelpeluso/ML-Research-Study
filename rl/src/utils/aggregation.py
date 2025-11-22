"""aggregate raw per-seed CSVs into mean + IQR summary CSVs"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import sys
import pandas as pd

# ensure src is on path when executed as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging import get_git_sha, make_filename


def aggregate_returns(
    raw_dir: Path,
    aggregated_dir: Path,
    pattern: str = "*.csv",
    x_col: str = "episode",
    metrics: Optional[List[str]] = None,
    filename_base: Optional[str] = None,
) -> Path:
    """read raw csvs from raw_dir (including subdirs), compute mean and iqr per x_col for all metrics, save to aggregated_dir

    returns path to saved aggregated csv
    """
    raw_dir = Path(raw_dir)
    aggregated_dir = Path(aggregated_dir)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    # recursively find all csv files
    files: List[Path] = sorted(raw_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no files found in {raw_dir} matching {pattern}")

    # default metrics if not specified
    if metrics is None:
        metrics = [
            'episode_return', 'steps', 'mean_abs_td_error', 'mean_abs_q_change', 'exploration_ratio',
            'q_table_size', 'q_table_nonzero', 'max_q_value', 'mean_q_value', 'action_entropy',
            'td_error_std', 'q_change_std', 'unique_states_visited'
        ]

    dfs_by_metric = {metric: [] for metric in metrics}
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if x_col not in df.columns:
                continue
            
            # collect each metric
            for metric in metrics:
                if metric in df.columns:
                    dfs_by_metric[metric].append(df[[x_col, metric]].set_index(x_col))
        except Exception:
            continue

    # aggregate each metric
    out_dfs = []
    for metric, dfs in dfs_by_metric.items():
        if not dfs:
            continue
        
        combined = pd.concat(dfs, axis=1)
        numeric = combined.select_dtypes(include='number')
        
        mean = numeric.mean(axis=1)
        q1 = numeric.quantile(0.25, axis=1)
        q3 = numeric.quantile(0.75, axis=1)
        iqr = q3 - q1
        
        metric_df = pd.DataFrame({
            x_col: mean.index,
            f"{metric}_mean": mean.values,
            f"{metric}_q1": q1.values,
            f"{metric}_q3": q3.values,
            f"{metric}_iqr": iqr.values,
        })
        out_dfs.append(metric_df)
    
    if not out_dfs:
        raise ValueError("no suitable CSVs found to aggregate")
    
    # merge all metrics by episode
    result = out_dfs[0]
    for df in out_dfs[1:]:
        result = result.merge(df, on=x_col, how='outer')
    
    base = filename_base or "aggregated_metrics"
    sha = get_git_sha()
    out_name = make_filename(base, sha=sha, ext='csv')
    out_path = aggregated_dir / out_name
    result.to_csv(out_path, index=False)
    return out_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='aggregate raw results into mean+iqr CSV')
    parser.add_argument('--raw-dir', type=str, default='results/raw')
    parser.add_argument('--out-dir', type=str, default='results/aggregated')
    parser.add_argument('--pattern', type=str, default='*.csv')
    parser.add_argument('--algorithm', type=str, help='filter by algorithm subdirectory')
    parser.add_argument('--environment', type=str, help='filter by environment subdirectory')
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    
    # if algorithm/environment specified, narrow down search
    if args.algorithm:
        raw_path = raw_path / args.algorithm
    if args.environment:
        raw_path = raw_path / args.environment
    
    out = aggregate_returns(raw_path, Path(args.out_dir), pattern=args.pattern)
    print(f'aggregated results saved to {out}')

# Logging Summary — RL Project

## Overview

All experiments now log comprehensive metrics **by default** with no flags required. Logging is organized hierarchically by algorithm and environment for better navigation with multiple experiments.

## Directory Structure

```
results/
├── raw/
│   └── <algorithm>/
│       └── <environment>/
│           ├── <experiment>_seed<N>_<sha>_<timestamp>.csv
│           └── <experiment>_seed<N>_<sha>_<timestamp>.json
├── aggregated/
│   └── aggregated_metrics_<sha>_<timestamp>.csv
└── figures/
    └── <algorithm>/
        └── <environment>/
            └── <algorithm>_<environment>_seed<N>.png
```

## Per-Episode CSV Metrics (Logged by Default)

Each episode logs the following columns in CSV format:

| Column              | Description                           | Use in Report                    |
| ------------------- | ------------------------------------- | -------------------------------- |
| `episode`           | Episode number                        | X-axis for learning curves       |
| `wall_clock_sec`    | Elapsed time since training start     | Runtime analysis                 |
| `episode_return`    | Total reward for episode              | Primary learning metric          |
| `steps`             | Number of steps in episode            | Episode length analysis          |
| `mean_abs_td_error` | Mean absolute TD error per step       | Convergence analysis             |
| `mean_abs_q_change` | Mean absolute Q-value change per step | Learning stability               |
| `exploration_ratio` | Fraction of exploratory actions       | Exploration/exploitation balance |

## JSON Metadata Sidecar (Logged by Default)

Each experiment produces a JSON file with complete reproducibility information:

### Core Experiment Info

-   `algorithm`: Algorithm name (e.g., "sarsa")
-   `environment`: Environment name (e.g., "CartPole-v1")
-   `seed`: Random seed used
-   `episodes`: Number of training episodes
-   `commit_sha`: Git commit short SHA (or 'dev')
-   `created_at_utc`: ISO timestamp of run start
-   `duration_sec`: Total wall-clock time

### Hyperparameters

-   `alpha`: Learning rate
-   `gamma`: Discount factor
-   `epsilon`: Exploration rate

### Environment Details (`env` key)

-   `action_space_n`: Number of discrete actions
-   `observation_space_shape`: Shape of observation vector
-   `observation_space_low`: Lower bounds per dimension
-   `observation_space_high`: Upper bounds per dimension
-   `discretizer_bins`: Bin counts per dimension (for continuous→discrete)
-   `num_discrete_states`: Total discrete state count
-   `command`: Exact command line used to run experiment

### System Information (`system` key)

-   `python_version`: Full Python version string
-   `platform`: OS platform details
-   `packages`: Version of key packages (numpy, pandas, gymnasium, matplotlib, seaborn, torch, pyyaml)

### Summary Statistics (`summary` key)

Computed after training completes:

-   `episode_return_mean`: Mean return across all episodes
-   `episode_return_median`: Median return
-   `episode_return_std`: Standard deviation of returns
-   `episode_return_min`: Minimum return
-   `episode_return_max`: Maximum return
-   `last_100_mean`: Mean return over last 100 episodes (if ≥100 episodes)

## Aggregated Metrics CSV

The aggregation utility (`src/utils/aggregation.py`) recursively scans `results/raw/` and produces:

| Metric Group   | Columns Generated                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------- |
| Episode Return | `episode_return_mean`, `episode_return_q1`, `episode_return_q3`, `episode_return_iqr`             |
| Steps          | `steps_mean`, `steps_q1`, `steps_q3`, `steps_iqr`                                                 |
| TD Error       | `mean_abs_td_error_mean`, `mean_abs_td_error_q1`, `mean_abs_td_error_q3`, `mean_abs_td_error_iqr` |
| Q-Value Change | `mean_abs_q_change_mean`, `mean_abs_q_change_q1`, `mean_abs_q_change_q3`, `mean_abs_q_change_iqr` |
| Exploration    | `exploration_ratio_mean`, `exploration_ratio_q1`, `exploration_ratio_q3`, `exploration_ratio_iqr` |

**Usage:**

```bash
# Aggregate all experiments
.venv/Scripts/python src/utils/aggregation.py --raw-dir results/raw --out-dir results/aggregated

# Aggregate specific algorithm/environment
.venv/Scripts/python src/utils/aggregation.py --algorithm sarsa --environment cartpole
```

## Example Output Files

### Raw CSV (first 5 rows):

```csv
episode,wall_clock_sec,episode_return,exploration_ratio,mean_abs_q_change,mean_abs_td_error,steps
0,0.11,9.0,0.111,0.092,0.921,9
1,0.11,9.0,0.111,0.093,0.929,9
2,0.11,9.0,0.111,0.091,0.908,9
3,0.11,11.0,0.0,0.107,1.067,11
4,0.11,11.0,0.273,0.088,0.880,11
```

### JSON Metadata (abbreviated):

```json
{
  "algorithm": "sarsa",
  "seed": 200,
  "commit_sha": "fc8dedf",
  "env": {
    "discretizer_bins": [3, 3, 8, 12],
    "num_discrete_states": 864
  },
  "summary": {
    "episode_return_mean": 9.4,
    "episode_return_std": 0.49
  },
  "system": {
    "python_version": "3.12.4",
    "packages": {"numpy": "1.26.4", ...}
  }
}
```

## Report Usage

All logged values are designed for the final report:

-   **Learning curves**: Plot `episode_return` vs `episode` with IQR shading from aggregated data
-   **Convergence analysis**: Plot `mean_abs_td_error` and `mean_abs_q_change` over time
-   **Exploration analysis**: Plot `exploration_ratio` to show exploration decay
-   **Episode length**: Plot `steps` to show learning efficiency
-   **Reproducibility**: Include `commit_sha`, `seed`, full command, and package versions in appendix
-   **Runtime**: Report `duration_sec` and `wall_clock_sec` for performance analysis
-   **Hyperparameter details**: All α, γ, ε values logged for comparison across experiments

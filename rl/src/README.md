# Source Code

Core implementation of RL algorithms and experimental framework.

## Module Overview

### `algorithms/`
RL algorithm implementations:
- `sarsa.py` - On-policy TD learning
- `q_learning.py` - Off-policy TD learning  
- `value_iteration.py` - Model-based DP
- `policy_iteration.py` - Model-based DP

### `environments/`
Environment wrappers and MDPs:
- `blackjack_wrapper.py` - Model-free Blackjack environment
- `blackjack_mdp.py` - Explicit MDP for VI/PI
- `cartpole_discretizer.py` - Continuous→discrete state mapping
- `cartpole_mdp.py` - Empirical MDP for VI/PI

### `experiments/`
Experiment execution framework:
- `run_experiment.py` - Single algorithm/environment/seed runner
- `experiment_base.py` - Shared utilities
- `hyperparam_search.py` - Hyperparameter optimization

### `utils/`
Helper modules:
- `aggregation.py` - Multi-seed statistics
- `experiment_logger.py` - CSV/JSON logging
- `save_artifacts.py` - Save Q-tables, policies, values
- `generate_report_plots.py` - **Unified plotting** (all visualization functionality)
- `seeding.py` - Reproducibility utilities

### `config/`
Configuration files:
- `default.yaml` - Experiment settings (seeds, episodes, hyperparameters)
- `hyperparam_ranges.yaml` - Search space for optimization

## Entry Point

**`main.py`** - Single entry point for all experiments

```bash
# Run all experiments (uses config/default.yaml)
python main.py

# Configure via default.yaml:
# - smoke_test: true/false (5 vs 50 seeds)
# - algorithms: [sarsa, qlearning, vi, pi]
# - environments: [blackjack, cartpole]
```

## Development Guidelines

1. **Algorithms return dict**: Include `q_table` (TD) or `V`/`policy` (DP)
2. **Logging via ExperimentLogger**: CSV for metrics, JSON for metadata
3. **Save artifacts**: Use `save_artifacts.py` for Q-tables/policies
4. **Reproducibility**: Call `set_all_seeds()` before env/agent creation
5. **Configuration-driven**: No hardcoded hyperparameters in code

See `.github/copilot-instructions.md` for detailed coding guidelines.

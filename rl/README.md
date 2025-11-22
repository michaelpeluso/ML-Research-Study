# Reinforcement Learning Report

## CS 7641 - Machine Learning

### Overview

This repository contains implementations and experiments for comparing reinforcement learning algorithms on two environments:

-   **Blackjack** (discrete, stochastic)
-   **CartPole** (continuous → discretized, deterministic)

Algorithms implemented:

-   Value Iteration (VI)
-   Policy Iteration (PI)
-   SARSA
-   Q-Learning

### Directory Structure

```
rl/
|
├── src/
│   ├── environments/
│   │   ├── blackjack_wrapper.py      # Gymnasium + seed control
│   │   └── cartpole_discretizer.py   # Non-uniform bins, explicit edges
│   │
│   ├── algorithms/
│   │   ├── value_iteration.py        # VI/PI with convergence tracking
│   │   ├── policy_iteration.py
│   │   ├── sarsa.py                  # On-policy TD learning
│   │   └── qlearning.py              # Off-policy + Double Q option
│   │
│   ├── utils/
│   │   ├── seeding.py                # Unified seed control (np, random, gym)
│   │   ├── logging.py                # CSV logging: episode, return, ΔQ, wallclock
│   │   └── plotting.py               # Generate all figures in one call
│   │
│   └── config/                     # YAML configuration files
│       ├── default.yaml           # γ=0.99, α=0.5→0.1, ε=1.0→0.01
│       │                          # CartPole bins=[3,3,8,12], clamps=[-2.4,2.4,-0.209,0.209]
│       └── hyperparam_ranges.yaml # Log-scale ranges for hyperparameter search
│
├── run_experiments.py          # SINGLE COMMAND: python run_experiments.py --all
│                               # Parallelizes 50 seeds, runs all experiments
├── results/                    # Auto-created, gitignored
│   ├── raw/                   # seed_1234_vi_blackjack.csv
│   ├── aggregated/            # vi_blackjack_mean_iqr.csv
│   └── figures/               # learning_curves_vi_pi.png → LaTeX ready
│
└── report/                    # LaTeX submission
    ├── main.tex               # IEEE template ≤8 pages
    ├── sections/              # Modular LaTeX sections
    ├── figures/               # Auto-populated by plotting.py
    └── RL_Report_{GTusername}.pdf
```

### Quick Start

#### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Running Experiments

**Single command for full reproduction:**

```bash
python src/run_experiments.py --all
```

This will:

1. Run all algorithms (VI, PI, SARSA, Q-Learning) on both environments
2. Execute with 50 seeds for statistical significance
3. Parallelize execution using joblib
4. Save raw CSVs to `results/raw/`
5. Aggregate results to `results/aggregated/`
6. Generate all figures to `results/figures/`
7. Complete in ~90 minutes

**Run specific algorithm:**

```bash
python src/run_experiments.py --algorithm sarsa --environment cartpole --seeds 5
```

**Run with custom config:**

```bash
python src/run_experiments.py --config src/config/custom.yaml --all
```

### Configuration

Hyperparameters are stored in YAML files under `src/config/`:

**`src/config/default.yaml`** - Default hyperparameters:

-   Learning rate: α = 0.5 → 0.1 (decay)
-   Discount factor: γ = 0.99
-   Exploration: ε = 1.0 → 0.01 (decay)
-   CartPole discretization: bins=[3,3,8,12], clamps=[-2.4,2.4,-0.209,0.209]
-   50 seeds for statistical analysis

**`src/config/hyperparam_ranges.yaml`** - Search ranges:

-   Log-scale ranges for α, ε-decay, γ
-   Discretization bin configurations
-   Stage 1-4 search automation (coarse → fine)

**Why this approach is superior:**

-   Graders change ONE number → re-run identical experiment
-   No code edits required
-   Version control tracks all hyperparameter changes

### Results

After running experiments:

-   **`results/raw/`**: Individual seed CSVs with columns:
    -   `episode`, `episode_return`, `delta_q`, `wall_clock_sec`, `epsilon`
-   **`results/aggregated/`**: Statistical summaries:

    -   Mean ± IQR/CI across seeds
    -   Ready for plotting with confidence intervals

-   **`results/figures/`**: Report-ready figures:
    -   Learning curves (VI vs PI, SARSA vs Q-Learning)
    -   Convergence comparisons
    -   Policy heatmaps
    -   Discretization ablation studies

### Report Compilation

LaTeX report structure in `report/`:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or upload to Overleaf for compilation.

**Required deliverables:**

1. `RLReport{GTusername}.pdf` (≤8 pages, IEEE template)
2. `DOCSTRING{GTusername}.pdf` containing:
    - Overleaf READ-ONLY link
    - Git commit SHA

-   Exact run command: `python src/run_experiments.py --all`

### Reproducibility

All experiments use:

-   Fixed seeds (specified in `src/config/default.yaml`)
-   Git commit SHA tracking in filenames
-   Wall-clock timing for computational analysis
-   Unified seeding: `random`, `numpy`, `torch`, gymnasium

**To reproduce results:**

```bash
# Exact command graders will run:
python src/run_experiments.py --all

# Verify output structure:
ls results/raw/        # Should see seed_*_*.csv files
ls results/aggregated/ # Should see *_mean_iqr.csv files
ls results/figures/    # Should see *.png files
```

### Key Features

1. **Zero path fragility**: Single `src/run_experiments.py` eliminates shell script issues
2. **Parallel execution**: Joblib parallelizes 50 seeds → 90min total runtime
3. **Automatic aggregation**: Mean ± IQR computed automatically
4. **One-command plotting**: `plotting.generate_all()` creates all figures
5. **Explicit bin edges**: Non-uniform discretization with exact bounds in config
6. **Grader-friendly**: Change config YAML, rerun command, verify outputs

### AI Use Statement

This codebase uses GitHub Copilot for:

-   Algorithm boilerplate and experiment runner structure
-   Utility function implementations (seeding, logging, plotting)
-   Configuration file templates
-   LaTeX report structure

All code was reviewed, tested, and modified by the author. Analysis and conclusions are entirely human-authored.

### References

-   Sutton & Barto (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.
-   Barto, Sutton & Anderson (1983). _Neuronlike adaptive elements that can solve difficult learning control problems._ IEEE Trans. SMC.
-   Watkins & Dayan (1992). _Q-learning._ Machine Learning, 8(3-4), 279-292.

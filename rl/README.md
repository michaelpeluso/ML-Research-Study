# Reinforcement Learning Report — CS 7641

## Overview

This repository contains implementations and experiments for the CS 7641 Reinforcement Learning assignment (Fall 2025).

**Assignment Requirements:**
- Solve two MDPs: **Blackjack** (discrete/stochastic) and **CartPole** (continuous/deterministic)
- Implement four algorithms: **Value Iteration**, **Policy Iteration**, **SARSA**, **Q-Learning**
- Optional Extra Credit: **DQN** variant (e.g., Double Q-Learning, Dueling Networks)
- Generate reproducible figures for report with git commit SHA tracking

---

## Project Structure

```
rl/
├── agents/              # algorithm implementations (VI, PI, SARSA, Q-Learning, DQN)
├── experiments/         # experiment runners (CLI scripts)
├── figures/             # all generated plots, CSVs, JSONs
├── resources/           # assignment spec (RL_Report.md)
├── .github/prompts/     # agent-based development templates
├── ARCHITECTURE.md      # detailed system design
├── FILE_CREATION_CHECKLIST.md  # step-by-step file creation guide
└── DOCSTRING_TEMPLATE.md       # submission artifact template
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for complete system design, component responsibilities, and interfaces.

---

## Quick Start

### 1. Create File Structure

```bash
# Option A: Use the setup script
bash setup.sh

# Option B: Follow the manual checklist
# See FILE_CREATION_CHECKLIST.md for step-by-step instructions
```

### 2. Install Dependencies

```bash
# create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# install packages
pip install -r requirements.txt
```

### 3. Implement Core Utilities

Start with `agents/utils.py` (all experiments depend on this):
- `set_seeds(seed)` — set random, numpy, torch seeds
- `get_git_sha()` — return current commit SHA
- `make_filename(task, sha, ext, timestamp)` — generate output filenames

### 4. Implement First Algorithm

Example: `agents/value_iteration.py`
```python
class ValueIteration:
    def __init__(self, env, gamma=0.99, theta=1e-6, seed=None):
        # implementation
        pass
    
    def train(self) -> dict:
        # returns: {'policy': ..., 'V': ..., 'iterations': int, 'metadata': {...}}
        pass
```

### 5. Create First Experiment

Example: `experiments/blackjack_vi_pi.py`
```bash
python experiments/blackjack_vi_pi.py --seed 42
# outputs: figures/blackjack_vi_pi_<sha>_<timestamp>_*.png
```

### 6. Verify Output

```bash
ls -lh figures/
# should see: *_<sha>_<timestamp>.png and *_results.json
```

---

## Running Experiments

All experiment runners accept `--seed` for reproducibility.

### Blackjack Experiments
```bash
python experiments/blackjack_vi_pi.py --seed 42
python experiments/blackjack_sarsa.py --seed 42 --episodes 50000
python experiments/blackjack_qlearning.py --seed 42 --episodes 50000
```

### CartPole Experiments
```bash
python experiments/cartpole_vi_pi.py --seed 42
python experiments/cartpole_sarsa.py --seed 42 --episodes 10000
python experiments/cartpole_qlearning.py --seed 42 --episodes 10000
```

### Optional: DQN (Extra Credit)
```bash
python experiments/cartpole_dqn.py --seed 42 --episodes 1000
```

---

## Output Files

All outputs saved to `figures/` with naming convention:
```
{task}_{gitsha}_{timestamp}_{suffix}.{ext}
```

Example:
- `blackjack_sarsa_a3f2c1d_20251121T143055_learning_curve.png`
- `cartpole_vi_a3f2c1d_20251121T143100_convergence.png`
- `blackjack_qlearning_a3f2c1d_20251121T143200_results.json`

Each experiment generates:
1. Figures (PNG) — learning curves, convergence plots, policy heatmaps
2. Metadata (JSON) — commit SHA, seed, hyperparameters, command

---

## Reproducibility

### Seeding
Every experiment uses deterministic seeding:
```python
from agents.utils import set_seeds
set_seeds(42)  # sets random, numpy, torch
```

### Git Commit Tracking
Outputs automatically include git commit SHA:
```python
from agents.utils import get_git_sha
sha = get_git_sha()  # embedded in filenames
```

### Verification
```bash
# check current commit
git rev-parse --short HEAD

# verify reproducibility
python experiments/blackjack_sarsa.py --seed 42
# re-run should produce identical results (different timestamp, same SHA)
```

---

## Development Workflow

This project uses an **agent-based development system** with GitHub Copilot. See `.github/prompts/` for templates.

### Agent Roles:
- **Code Writer** — implement algorithms and runners
- **Debug Agent** — fix errors and ensure reproducibility
- **Documentation Agent** — write docstrings and README updates
- **Spec Validator** — verify assignment requirements

Example prompt:
```
Acting as the Code Writer agent, implement Value Iteration in agents/value_iteration.py 
with a seed parameter and train() method that returns results dict with metadata.
```

See `.github/prompts/AGENT_WORKFLOW.md` for complete guide.

---

## Assignment Deliverables

### 1. Report (`RLReport{GTusername}.pdf`)
- Written in LaTeX on Overleaf using IEEE Conference template
- Maximum 8 pages including citations
- Include AI Use Statement before References

### 2. DOCSTRING (`DOCSTRING{GTusername}.pdf`)
Use `DOCSTRING_TEMPLATE.md` as starting point. Must include:
- READ-ONLY Overleaf link
- GitHub commit hash (single SHA)
- Exact Linux run instructions with seeds

### 3. Code Repository
- GT Enterprise GitHub repository
- Final commit must be reproducible
- Include `requirements.txt` and run instructions

---

## Key Requirements

- ✓ Reproducible with fixed seeds
- ✓ Git commit SHA embedded in output filenames
- ✓ Metadata JSON sidecar for each experiment
- ✓ AI Use Statement in modified source files
- ✓ No unit tests (per project constraints)
- ✓ Figures legible at 100% zoom

---

## Resources

- **Assignment Spec:** `resources/RL_Report.md`
- **Architecture:** `ARCHITECTURE.md`
- **File Creation Guide:** `FILE_CREATION_CHECKLIST.md`
- **Submission Template:** `DOCSTRING_TEMPLATE.md`
- **Agent Prompts:** `.github/prompts/`

---

## AI Use Statement

<!-- Update this section as you work -->

I used [GitHub Copilot / ChatGPT / other] to:
- [List what AI tools helped with]
- [Example: Generate boilerplate code for agent classes]
- [Example: Debug numpy indexing issues]
- [Example: Refactor plotting code]

I reviewed, verified, and understand all AI-assisted content. All analysis and conclusions are my own original work.

---

## Getting Help

1. Review `ARCHITECTURE.md` for system design
2. Check `.github/prompts/` for development templates
3. See `resources/RL_Report.md` for assignment requirements
4. Use agent-based prompts with GitHub Copilot for implementation

---

## License

Academic project for CS 7641 — Georgia Institute of Technology
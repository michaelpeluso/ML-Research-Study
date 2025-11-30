# Reinforcement Learning Algorithm Comparison

**Georgia Institute of Technology: CS 7641 Machine Learning — Fall 2025**

Experimental analysis of Value Iteration, Policy Iteration, SARSA, and Q-Learning on Blackjack and CartPole MDPs.

---

## Overview

This project implements and compares four reinforcement learning algorithms across two environments with distinct characteristics:

| Environment | State Space | Dynamics | Algorithms |
|-------------|-------------|----------|------------|
| **Blackjack** | Discrete (200 states) | Stochastic | VI, PI, SARSA, Q-Learning |
| **CartPole** | Continuous → Discretized | Deterministic | VI, PI, SARSA, Q-Learning |

**Key Comparisons:**
- Model-based (VI, PI) vs. Model-free (SARSA, Q-Learning)
- On-policy (SARSA) vs. Off-policy (Q-Learning)
- Convergence rates, sample efficiency, stability, and final return

---

## Quick Start

```bash
# install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run all experiments 
cd src && python main.py

# smoke test: 5 seeds, ~10 min
# edit src/config/default.yaml: smoke_test: true
cd src && python main.py
```

**Output:**
- `results/raw/{algo}/{env}/` — per-seed CSV metrics + JSON metadata + .npy artifacts
- `results/figures/` — visual plots

---

## Project Structure

```
rl/
├── src/
│   ├── main.py                    # single entry point
│   ├── algorithms/                # VI, PI, SARSA, Q-Learning
│   ├── environments/              # wrappers, discretizers, MDPs
│   ├── experiments/               # runners, hyperparameter search
│   ├── utils/                     # logging, plotting, aggregation
│   └── config/
│       ├── default.yaml           # hyperparameters, seeds, episodes
│       └── hyperparam_ranges.yaml # staged search configuration
└── results/
    ├── raw/                       # per-seed data
    ├── figures/                   # generated plots
    └── hyperparam_search/         # tuning results
```

---

## Algorithms

### Model-Based Dynamic Programming

These algorithms require complete knowledge of the MDP (transition probabilities P(s'|s,a) and reward function). They compute optimal policies through iterative sweeps over the entire state space.

**Value Iteration (VI)**

Iteratively applies the Bellman optimality equation until the value function converges:

```
V(s) ← max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
```

- **Convergence criterion:** `max_s |V_{k+1}(s) - V_k(s)| < θ` (default θ = 1e-6)
- **Implementation:** `src/algorithms/value_iteration.py`
- **Tracked metrics:** Iterations to converge, max delta per sweep, wall-clock time
- **Complexity:** `O(|S|²|A|)` per iteration

**Policy Iteration (PI)**

Alternates between policy evaluation (computing V^π) and policy improvement (making policy greedy w.r.t. V^π):

```
1. Policy Evaluation:  V^π(s) ← Σ_s' P(s'|s,π(s))[R + γV^π(s')]  (until convergence)
2. Policy Improvement: π(s) ← argmax_a Σ_s' P(s'|s,a)[R + γV^π(s')]
3. Repeat until π is stable
```

- **Convergence criterion:** Policy unchanged between iterations
- **Implementation:** `src/algorithms/policy_iteration.py`
- **Tracked metrics:** Policy improvement iterations, evaluation iterations per step, states changed per iteration
- **Typically faster:** Fewer outer iterations than VI (often 3-7 vs 20-50)

### Model-Free Temporal Difference Learning

These algorithms learn directly from experience without requiring a model. They update Q-values incrementally after each transition using bootstrapped targets.

**SARSA (On-Policy TD)**

Updates Q-values using the action *actually taken* in the next state (follows the behavior policy):

```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                    └────────────┘
                       TD target (uses a' from policy)
```

- **On-policy:** Learns the value of the policy being followed (including exploration)
- **Exploration:** ε-greedy with linear decay (ε: 1.0 → 0.01 over configurable episodes)
- **Implementation:** `src/algorithms/sarsa.py`
- **Tracked metrics:** Episode returns, TD errors, Q-table changes, exploration ratio, policy entropy

**Q-Learning (Off-Policy TD)**

Updates Q-values using the *best possible* action in the next state (learns optimal policy regardless of behavior):

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                    └───────────────────┘
                       TD target (uses max over all a')
```

- **Off-policy:** Learns optimal Q* while following exploratory policy
- **Key difference:** Target uses max (greedy) rather than actual next action
- **Implementation:** `src/algorithms/q_learning.py`
- **Tracked metrics:** Same as SARSA for fair comparison

### Algorithm Comparison

| Property | VI | PI | SARSA | Q-Learning |
|----------|----|----|-------|------------|
| **Type** | Model-based | Model-based | Model-free | Model-free |
| **Policy** | — | — | On-policy | Off-policy |
| **Requires P(s'\|s,a)** | Yes | Yes | No | No |
| **Update** | Full sweep | Full sweep | Per transition | Per transition |
| **Convergence** | Guaranteed | Guaranteed | Asymptotic | Asymptotic |

### Exploration Strategy

Both SARSA and Q-Learning use **ε-greedy exploration** with linear decay:

```
ε(t) = max(ε_floor, ε_start - (ε_start - ε_floor) * t / decay_episodes)
```

- **ε_start:** 1.0 (fully random initially)
- **ε_floor:** 0.01 (maintain minimal exploration)
- **decay_episodes:** 5000 (configurable)

This ensures sufficient exploration early in training while converging to near-greedy behavior.

---

## Environments

### Blackjack (Gymnasium: Blackjack-v1)

A discrete, stochastic MDP based on the classic card game.

**State space (200 states):**
- Player's current sum (12–21, values below 12 always hit)
- Dealer's showing card (1–10, where 1 = Ace)
- Usable ace (boolean: can count ace as 11 without busting)

**Action space:** Hit (draw card) or Stand (stop drawing)

**Rewards:** +1 (win), -1 (lose), 0 (draw)

**Stochasticity:** Card draws from infinite deck, dealer follows fixed policy (hit on ≤16)

**Why interesting:** Tests algorithm behavior under inherent uncertainty where optimal play still loses due to house edge.

### CartPole (Gymnasium: CartPole-v1)

A continuous-state, deterministic physics simulation requiring state discretization for tabular methods.

**Original state space (continuous):**
- Cart position x ∈ [-4.8, 4.8]
- Cart velocity ẋ ∈ (-∞, ∞)
- Pole angle θ ∈ [-0.418, 0.418] rad
- Angular velocity θ̇ ∈ (-∞, ∞)

**Discretization (1,920 states):**

| Feature | Clamp Range | Bins | Rationale |
|---------|-------------|------|-----------|
| Cart position (x) | [-2.4, 2.4] | 4 | Less critical for balance |
| Cart velocity (ẋ) | [-3.0, 3.0] | 4 | Less critical for balance |
| Pole angle (θ) | [-0.209, 0.209] rad | 10 | **Critical:** small angles matter |
| Angular velocity (θ̇) | [-3.5, 3.5] | 12 | **Critical:** predicts future angle |

**Action space:** Push left (0) or Push right (1)

**Reward:** +1 per timestep pole remains upright (max 500 per episode)

**Why interesting:** Tests discretization effects—coarse bins cause state aliasing where distinct physical states map to the same discrete state, degrading policy quality.

### Model Construction for VI/PI

Since VI and PI require explicit transition probabilities, we construct empirical MDPs:

**Blackjack:** Transition model built analytically from game rules (infinite deck assumption allows exact P(s'|s,a) computation).

**CartPole:** Empirical transition model via Monte Carlo sampling:
1. For each discrete state s and action a, sample `N` transitions (default N=50)
2. Count resulting discrete states s' to estimate P(s'|s,a)
3. Average rewards to estimate R(s,a)

This approach enables VI/PI on CartPole but introduces approximation error from discretization.

---

## Hyperparameter Validation

Following the report requirement of **≥3 hyperparameters validated per algorithm**, tuning uses a staged search protocol:

**Stage 1 — Coarse Random Search:**
- Sample N=24 candidates from log-scaled ranges
- Run pilot episodes (200), discard bottom 50% by interim return
- Ranges: α ∈ [10⁻³, 1], γ ∈ [0.9, 0.999], ε_decay ∈ [1000, 20000]

**Stage 2 — Successive Halving:**
- Promote top 8 candidates, extend training to 600 episodes
- Prune to top 3 by mean return

**Stage 3 — Local Refinement:**
- Narrow search around winners (±2× on α, ±25% on decay)
- Final evaluation over 30-50 seeds

**Validated hyperparameters per algorithm:**

| Algorithm | Hyperparameters Validated |
|-----------|---------------------------|
| **VI** | γ (discount), θ (convergence threshold), discretization bins |
| **PI** | γ (discount), θ (eval threshold), max_eval_iterations |
| **SARSA** | α (learning rate), γ (discount), ε schedule (start, floor, decay) |
| **Q-Learning** | α (learning rate), γ (discount), ε schedule (start, floor, decay) |

Results stored in `results/hyperparam_search/stage{1,2,3}_{algo}_{env}_results.json`.

---

## Statistical Rigor

All experiments use **30–50 independent random seeds** with unified seeding across `random`, `numpy`, and Gymnasium.

**Seeding implementation** (`src/utils/seeding.py`):
```python
def set_all_seeds(seed: int, env=None):
    random.seed(seed)
    np.random.seed(seed)
    if env:
        env.reset(seed=seed)
        env.action_space.seed(seed)
```

**Aggregation metrics:**
- **Central tendency:** Mean, median
- **Variability:** Standard deviation, IQR (Q3 - Q1), coefficient of variation
- **Visualization:** Mean ± IQR bands on learning curves

**Per-seed tracking (15+ metrics per episode):**
- Episode return, episode length
- TD error (mean, max), Q-table change magnitude
- Exploration ratio (random vs greedy actions)
- Policy entropy (action distribution uncertainty)
- Cumulative reward, wall-clock time

**Reproducibility guarantee:** Same seed → identical results across runs.

---

## Key Outputs

### Raw Data (per seed)

```
results/raw/{algo}/{env}/
├── {algo}_{env}_seed{N}.csv      # episode metrics
├── {algo}_{env}_seed{N}.json     # metadata + summary stats
├── {algo}_{env}_seed{N}_qtable.npy   # Q-table (SARSA/Q-Learning)
├── {algo}_{env}_seed{N}_policy.npy   # learned policy
└── {algo}_{env}_seed{N}_value.npy    # value function (VI/PI)
```

### Aggregated Results

```
results/raw/{algo}/master_summary_{algo}.json
```

Contains mean, std, median, Q1, Q3, IQR, convergence stats across all seeds.

### Figures

| Plot | Purpose |
|------|---------|
| `blackjack_heatmap.png` | Policy visualization (hit/stand by state) |
| `cartpole_learned_strategy.png` | Action probabilities by angle/angular velocity |
| `learning_curves_*.png` | Return vs. episode with IQR bands |
| `convergence_*.png` | ΔV or ΔQ vs. iteration |

---

## Configuration

Edit `src/config/default.yaml` to customize experiments:

```yaml
smoke_test: false          # true = 5 seeds
seeds: [0, 1, ..., 49]     # exact seed list

experiments:
  algorithms: [sarsa, qlearning, vi, pi]
  environments: [blackjack, cartpole]

sarsa:
  blackjack: {alpha: 0.0011, gamma: 0.9511, epsilon: 1.0, episodes: 2000}
  cartpole: {alpha: 0.8123, gamma: 0.9908, epsilon: 1.0, episodes: 2000}

cartpole:
  bins: [4, 4, 10, 12]     # discretization
  samples_per_sa: 50       # for VI/PI transition model
```

---

## References

**References**
- Sutton & Barto (2018), *Reinforcement Learning*: An Introduction (2nd ed.), Example 5.3 “Blackjack,” [online book](http://incompleteideas.net/book/RLbook2018.pdf). Gym Environment: [Blackjack-v1](https://gymnasium.farama.org/environments/toy_text/blackjack/).

- Barto, Sutton & Anderson (1983), "Neuronlike adaptive elements that can solve difficult learning control problems." *IEEE Transactions on Systems, Man, and Cybernetics* 13(5):834–846, [doi:10.1109/TSMC.1983.6313077](https://doi.org/10.1109/TSMC.1983.6313077). Gym Environment: [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

---

## Tool Use Statement

This project uses artificial intelligence editor and tooling features to accelerate development (e.g., code suggestions, template generators, linters).

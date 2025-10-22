# Genetic Algorithm (GA) Documentation for OL Report

## Implementation Details

This document provides the GA configuration details required for the OL Report Section 4.

### GA Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Population Size** | 50 | Number of individuals in each generation |
| **Max Evaluations** | 10,000 | Budget for function evaluations |
| **Mutation Rate** | 0.1 (10%) | Probability of applying Gaussian mutation to an offspring |
| **Mutation Std** | 0.01 | Standard deviation of Gaussian noise added during mutation |
| **Crossover Rate** | 0.7 (70%) | Probability of performing crossover between parents |

### Selection Method
- **Type**: Truncation Selection
- **Description**: The top 50% of the population (25 individuals) with the best (lowest) fitness are selected as parents for reproduction
- **Selection Size**: `pop_size // 2 = 25` parents per generation
- **Deterministic**: Yes - always selects the fittest individuals

### Crossover Method
- **Type**: Single-Point Crossover
- **Description**: A random crossover point is selected, and parent genomes are swapped at that point to create two offspring
- **Mechanism**: 
  - Randomly select crossover point in range [0, genome_length]
  - Offspring 1: First half from Parent 1, second half from Parent 2
  - Offspring 2: First half from Parent 2, second half from Parent 1
- **Fallback**: If crossover not applied (30% chance), offspring are clones of parents

### Mutation Method
- **Type**: Gaussian Mutation
- **Description**: Add random noise sampled from N(0, σ²) to offspring parameters
- **Mechanism**: Each offspring has 10% chance of mutation, applied independently
- **Noise Distribution**: N(0, 0.01²) added to all parameters if mutation triggered

### Elitism Strategy
- **Enabled**: Yes
- **Elite Count**: 1
- **Description**: The single best individual from the current generation is always preserved unchanged in the next generation
- **Guarantee**: Monotonic improvement in best-so-far fitness (elite never mutated or crossed over)

### Additional Notes
- **Initialization**: Population initialized with Gaussian noise around initial model parameters
- **Evaluation**: Only new offspring are evaluated (elite fitness already known), saving function evaluations
- **Termination**: Stops when `max_evals` reached or plateau detected (no improvement for `plateau_threshold` evaluations)

### Summary for Report

**For OL Report Section 4, document GA as:**

"Genetic Algorithm (GA) with population size 50, truncation selection (top 50%), single-point crossover (70% rate), Gaussian mutation (10% rate, σ=0.01), and elitism (1 elite preserved per generation)."

**Logged Metrics Available:**
- `pop_size`: 50
- `selection_method`: 'Truncation Selection (top 50%)'
- `selection_size`: 25
- `crossover_method`: 'Single-Point Crossover'
- `crossover_rate`: 0.7
- `mutation_method`: 'Gaussian Mutation'
- `mutation_rate`: 0.1
- `mutation_std`: 0.01
- `elitism`: True
- `elitism_count`: 1

These metrics are logged in `experiment_logs.json` for easy reference when writing the report.

# Dropout Placement Documentation for OL Report

## Implementation Details

This document provides explicit dropout placement documentation as required by OL Report Section 4.

### Dropout Architecture Pattern

**Pattern:** Linear → Activation → Dropout → Linear → Activation → Dropout → ... → Linear (output)

### Placement Details

- **Location**: Dropout is applied **AFTER each hidden layer activation function** (ReLU or Tanh)
- **Position**: Dropout is applied **BEFORE the next linear transformation**
- **Final Layer**: **NO dropout** is applied after the final output layer
- **Application**: Dropout randomly zeroes out neurons during training with probability `p`

### Example Architecture

For a network with 2 hidden layers [256, 128] and dropout probability p=0.3:

```
Input (in_dim features)
    ↓
Linear(in_dim → 256)
    ↓
ReLU()
    ↓
Dropout(p=0.3)  ← Applied here (after activation, before next layer)
    ↓
Linear(256 → 128)
    ↓
ReLU()
    ↓
Dropout(p=0.3)  ← Applied here (after activation, before next layer)
    ↓
Linear(128 → out_dim)
    ↓
Output (out_dim predictions, NO dropout here)
```

### Code Implementation

See `src/core/models.py` - MLP class, lines 38-55:

```python
if dropout_p > 0:
    # DROPOUT PLACEMENT: Insert dropout layer immediately after each activation function
    # This applies dropout to the activated outputs before feeding to the next linear layer
    new_layers = []
    for layer in self.net:
        new_layers.append(layer)
        if isinstance(layer, (nn.ReLU, nn.Tanh)):  # After activation functions only
            new_layers.append(nn.Dropout(p=dropout_p))  # Dropout applied here
    self.net = nn.Sequential(*new_layers)
```

### Rationale

This placement strategy:
1. **Regularizes activations**: Drops out activated neuron outputs, forcing redundancy
2. **Standard practice**: Follows PyTorch/TensorFlow convention for dense networks
3. **Training mode**: Automatically disabled during evaluation (model.eval())
4. **Preserves output**: No dropout on final predictions ensures deterministic inference

### Experiment Configuration (Part 3)

- **Dropout Grid Tested**: [0.1, 0.2, 0.3, 0.4, 0.5]
- **Baseline**: p=0.0 (no dropout)
- **Best dropout value**: Determined via validation loss sweep
- **Applied uniformly**: Same dropout rate after all hidden layer activations

### Logged Metrics

The following metrics are logged in `experiment_logs.json`:
- `dropout_placement`: "After each hidden layer activation, before next linear transformation"
- `dropout_architecture_pattern`: "Linear → Activation → Dropout → Linear → ... → Linear (output)"
- `dropout_grid`: [0.1, 0.2, 0.3, 0.4, 0.5]
- `best_dropout`: Optimal dropout probability from validation sweep

### Summary for Report

**For OL Report Section 3 (Regularization), document dropout as:**

"Dropout is applied after each hidden layer activation function (ReLU or Tanh) and before the subsequent linear transformation, following the pattern: Linear → Activation → Dropout → Linear → ... → Linear (output). No dropout is applied after the final output layer. We evaluated dropout probabilities p ∈ {0.1, 0.2, 0.3, 0.4, 0.5} and selected the optimal value via validation loss."

**Architecture-specific example:**

"For the nn_2 architecture with hidden layers [256, 128], dropout is applied twice: once after the first ReLU (256-dimensional activations) and once after the second ReLU (128-dimensional activations), before the final output layer."

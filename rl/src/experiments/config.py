# ai use statement: github copilot assisted with boilerplate imports
"""shared hyperparameter configurations for experiments."""

# blackjack configuration
BLACKJACK_CONFIG = {
    'gamma': 0.99,
    'theta': 1e-6,  # vi/pi convergence threshold
    'alpha': 0.1,   # sarsa/q-learning learning rate
    'epsilon': 0.1, # exploration rate
    'num_episodes': 50000,
}

# cartpole configuration
CARTPOLE_CONFIG = {
    'gamma': 0.99,
    'theta': 1e-6,
    'alpha': 0.1,
    'epsilon': 0.1,
    'num_episodes': 10000,
    'discretization_bins': [6, 6, 12, 12],  # [x, x_dot, theta, theta_dot]
}

# dqn configuration (optional extra credit)
DQN_CONFIG = {
    'network': {
        'hidden_layers': [128, 128],
        'activation': 'relu',
    },
    'replay': {
        'buffer_size': 100000,
        'batch_size': 64,
    },
    'training': {
        'gamma': 0.99,
        'learning_rate': 0.001,
        'target_update_freq': 100,
        'num_episodes': 1000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
    }
}

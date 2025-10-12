DEFAULT_HYPERPARAMETERS = {
    ("nn", "c"): {
        "hidden_layer_sizes": (512, 512), # (256, 256, 128, 128),
        "batch_size": 1024,               # Fixed: [512, 1024, 2048]
        "activation": "relu",             # "relu", "tanh" # not listed on report
        "learning_rate": "adaptive",       # [0.001, 0.01, 0.1] # not listed on report
        "early_stopping": False,          # Fixed: False
        "n_iter_no_change": 2,            # Fixed: Patience 2–3
        "solver": "sgd",                  # Fixed: SGD only
    },
    ("nn", "r"): {
        "hidden_layer_sizes": (512, 512), # (256, 256, 128, 128),
        "batch_size": 1024,               # Fixed: [512, 1024, 2048]
        "activation": "relu",             # "relu", "tanh" # not listed on report
        "learning_rate": "adaptive",       # [0.001, 0.01, 0.1] # not listed on report
        "early_stopping": False,          # Fixed: False
        "n_iter_no_change": 2,            # Fixed: Patience 2–3
        "solver": "sgd",                  # Fixed: SGD only
    },
}
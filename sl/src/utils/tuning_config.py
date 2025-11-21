# default parameters of each model
DEFAULT_HYPERPARAMETERS = {
    # Decision Trees
    ("dt", "c"): {
        "criterion": "entropy",         # "gini", "entropy"
        "max_depth": 18,                # [6, 10, 14, 18]
        "min_samples_leaf": 50,         # [50, 100, 200]
        "min_samples_split": 100,       # [100, 200, 400]
        "max_features": 0.5,            # "sqrt", "log2", 0.5
        "ccp_alpha": 1e-6,               # [0.0, 1e-4, 5e-4, 1e-3]
        #"class_weight": "balanced",     # "balanced" if imbalanced
    },
    ("dt", "r"): {
        "max_depth": 18,              # [6, 10, 14, 18]
        "min_samples_leaf": 50,      # [50, 100, 200]
        "min_samples_split": 100,     # [100, 200, 400]
        "max_features": 0.5,       # "sqrt", "log2", 0.5
        "ccp_alpha": 1e-6,          # [0.0, 1e-4, 5e-4, 1e-3]
    },

    # Linear SVM
    ("lsvm", "c"): {
        "alpha": 1e-4,                # [1e-5, 1e-4, 1e-3]
        "max_iter": 5000,             # [5000, 10000, 20000]
        #"class_weight": "balanced",   # "balanced" if imbalanced
        "n_jobs": -1,                 # only for classifier
        "tol" : 0.01,                 # tolerance for stopping early
        "eta0": 0.01
    },
    ("lsvm", "r"): {
        "alpha": 1e-4,                # [1e-5, 1e-4, 1e-3]
        "max_iter": 5000,             # [5000, 10000, 20000]
        "tol" : 0.01,                 
        "eta0": 0.01
    },

    # Kernel SVM
    ("svm", "c"): {
        "C": 1,                       # [0.5, 2, 8]
        "kernel": "rbf",              # "linear", "rbf"
        "gamma": "scale",             # "scale", 1/d, 2/d
        "max_iter": 20000,
        "tol" : 0.1,                 # tolerance for stopping early
        "cache_size" : 500,            # reduce memory overload            
        "shrinking" : True 
    },
    ("svm", "r"): {
        "C": 1,                       # [0.5, 2, 8]
        "kernel": "rbf",              # "linear", "rbf"
        "gamma": "scale",             # "scale", 1/d, 2/d
        "max_iter": 20000,
        "tol" : 0.1,                 
        "cache_size" : 500,            
        "shrinking" : True            
    },

    # k-NN
    ("knn", "c"): {
        "n_neighbors": 5,             # [3, 5, 11, 21]
        "metric": "euclidean",        # Fixed
        "algorithm": "brute",         # Fixed: "brute"
        "n_jobs": -1,                 # Fixed: -1
    },
    ("knn", "r"): {
        "n_neighbors": 5,             # [3, 5, 11, 21]
        "metric": "euclidean",        # Fixed
        "algorithm": "brute",         # Fixed: "brute"
        "n_jobs": -1,                 # Fixed: -1
    },

    # Neural Networks (MLP with SGD)
    ("nn", "c"): {
        # change together..
        "hidden_layer_sizes": (512, 512), # (256, 256, 128, 128),
        "batch_size": 1024,               # Fixed: [512, 1024, 2048]
        # ....
        "activation": "relu",             # "relu", "tanh" # not listed on report
        "learning_rate": "adaptive",       # [0.001, 0.01, 0.1] # not listed on report
        "early_stopping": False,          # Fixed: False
        "n_iter_no_change": 2,            # Fixed: Patience 2–3
        "solver": "sgd",                  # Fixed: SGD only
    },
    ("nn", "r"): {
        # change together..
        "hidden_layer_sizes": (512, 512), # (256, 256, 128, 128),
        "batch_size": 1024,               # Fixed: [512, 1024, 2048]
        # ....
        "activation": "relu",             # "relu", "tanh" # not listed on report
        "learning_rate": "adaptive",       # [0.001, 0.01, 0.1] # not listed on report
        "early_stopping": False,          # Fixed: False
        "n_iter_no_change": 2,            # Fixed: Patience 2–3
        "solver": "sgd",                  # Fixed: SGD only
    },
}

# specified parameter ranges for each algorithm
# 'estimator__' added for sklearn parameter com
TUNING_GRIDS = {
    "dt": {
        "classification": {
            "estimator__max_depth": [6, 10, 14, 18, 22, 28, None], 
            "estimator__min_samples_leaf": [50, 100, 200],
            "estimator__min_samples_split": [100, 200, 400],
            "estimator__max_features": ["sqrt", "log2", 0.5],
            "estimator__ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3],
        },
        "regression": {
            "estimator__max_depth": [6, 10, 14, 18, 22],
            "estimator__min_samples_leaf": [50, 100, 200],
            "estimator__min_samples_split": [100, 200, 400],
            "estimator__max_features": ["sqrt", "log2", 0.5],
            "estimator__ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3],
        }
    },
    "lsvm": {
        "classification": {
            "estimator__alpha": [1e-5, 1e-4, 1e-3],
            "estimator__penalty" : ['l2', 'l1', 'elasticnet', None],
            "estimator__max_iter": [5000, 10000, 15000, 20000],
            "estimator__tol": [0.001, 0.0001],
            "estimator__learning_rate": ['optimal', 'constant', 'invscaling', 'adaptive']
        },
        "regression": {
            "estimator__alpha": [1e-5, 1e-4, 1e-3],
            "estimator__penalty" : ['l2', 'l1', 'elasticnet', None],
            "estimator__max_iter": [5000, 10000, 15000, 20000],
            "estimator__tol": [0.001, 0.0001],
            "estimator__learning_rate": ['optimal', 'constant', 'invscaling', 'adaptive']
        }
    },
    
    "svm": {
        "classification": {
            "estimator__C": [0.5, 2, 4, 8],
            "estimator__gamma": ["scale", "auto"],  # For RBF kernel
        },
        "regression": {
            "estimator__C": [0.5, 2, 4, 8],
            "estimator__gamma": ["scale", "auto"],
        }
    },
    
    "knn": {
        "classification": {
            "estimator__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 18, 21],
        },
        "regression": {
            "estimator__n_neighbors": [3, 5, 11, 21],
        }
    },
    
    "nn": {
        "classification": {
            "estimator__learning_rate_init": [0.001, 0.005, 0.01],
            "estimator__activation": ["relu", "tanh", "logistic"],
            #"estimator__max_iter": [100, 200, 300],
            "estimator__max_iter": [10, 15, 20],
            "estimator__alpha": [0.0001, 0.001, 0.01]
        },
        "regression": {
            "estimator__learning_rate_init": [0.001, 0.005, 0.1],
            "estimator__activation": ["relu", "tanh", "logistic"],
            #"estimator__max_iter": [100, 200, 300],
            "estimator__max_iter": [10, 15, 20],
            "estimator__alpha": [0.0001, 0.001, 0.01]
        }
    }
}

# Primary parameters for model-complexity curves
PRIMARY_PARAMS = {
    "dt": "estimator__max_depth",
    "lsvm": "estimator__alpha", 
    "svm": "estimator__C",
    "knn": "estimator__n_neighbors",
    "nn": "estimator__learning_rate_init"
}


def get_tuning_grid(method: str, model: str):
    method_key = "classification" if method.lower().startswith("c") else "regression"
    return TUNING_GRIDS.get(model, {}).get(method_key, {})

def get_primary_params(model: str):
    return PRIMARY_PARAMS.get(model)

def calculate_grid_size(param_grid: dict):
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total
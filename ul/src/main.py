
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

from experiment import Experiment


def main():
    
    # L2 testing
    testing = True
    subsample = 0.1 if testing else None
    max_evals = 500 if testing else 3000
    seed = 42
    
    experiments = [
        {
            'dataset': "hotels",
            'target': "is_canceled",
            'method': "classification",
            'subsample': subsample or 1.0,
            'batch_size': 64,
            'backbone_configs': [
                {'name': 'nn_2', 'hidden_layer_sizes': (256, 128), 'learning_threshold': 0.48, 'alpha': 1e-05, 'activation': 'tanh'},
                {'name': 'nn_4', 'hidden_layer_sizes': (256, 256, 128, 128), 'learning_threshold': 0.48, 'alpha': 1e-05, 'activation': 'tanh'}
            ],
            'k_range': (2, 5)
        },
        {
            'dataset': "accidents",
            'target': "Duration_Seconds",
            'method': "regression",
            'subsample': subsample or 0.8,  # > 80%
            'batch_size': 1024,
            'backbone_configs': [
                {'name': 'nn_2', 'hidden_layer_sizes': (256, 128), 'learning_threshold': 0.43, 'alpha': 0.001, 'activation': 'relu'},
                {'name': 'nn_4', 'hidden_layer_sizes': (256, 256, 128, 128), 'learning_threshold': 0.43, 'alpha': 0.001, 'activation': 'relu'}
            ]
        }
    ]

    config = experiments[0]
    exp = Experiment(
        dataset=config['dataset'],
        target=config['target'],
        method=config['method']
    )
    X_train, X_val, X_test, y_train, y_val, y_test = exp.get_data(subsample=config['subsample'])
    exp.run_kmeans((4, 8), 10)
    #exp.run_em((2, 10), X_train, seed)



if __name__ == "__main__":
    main()
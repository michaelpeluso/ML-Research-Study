import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

from experiment import Experiment


def main():

    # L2 testing
    testing = False
    subsample = 0.1 if testing else None
    seeds = [42] if testing else [42, 4242, 424242]
    max_evals = 500 if testing else 10000

    experiments = [
        {
            'dataset': "hotels",
            'target': "is_canceled",
            'method': "classification",
            'subsample': subsample or 1.0,
            'batch_size': 64,
            'backbone_configs': [
                {'name': 'nn_2', 'max_iter': 15, 'learning_rate_init': 0.05, 'hidden_layer_sizes': (256, 128), 'alpha': 1e-05, 'activation': 'tanh'},  # shallow: 2 layers
                {'name': 'nn_4', 'max_iter': 15, 'learning_rate_init': 0.05, 'hidden_layer_sizes': (256, 256, 128, 128), 'alpha': 1e-05, 'activation': 'tanh'}  # deep: 4 layers
            ]
        },
        {
            'dataset': "accidents",
            'target': "Duration_Seconds",
            'method': "regression",
            'subsample': subsample or 0.8,  # > 80%
            'batch_size': 1024,
            'backbone_configs': [
                {'name': 'nn_2', 'max_iter': 5, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (256, 128), 'alpha': 0.001, 'activation': 'relu'},  # shallow: 2 layers
                {'name': 'nn_4', 'max_iter': 5, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (256, 256, 128, 128), 'alpha': 0.001, 'activation': 'relu'}  # deep: 4 layers
            ]
        }
    ]

    for exp_config in experiments:
        exp = Experiment(
            dataset=exp_config['dataset'],
            target=exp_config['target'],
            method=exp_config['method'],
            subsample=exp_config['subsample'],
            batch_size=exp_config['batch_size']
        )

        for backbone in exp_config['backbone_configs']:

            exp.best_params.update(backbone)
            exp.update_logs()
            exp.ml_logger.current_logs = []
            exp.save_path = os.path.join(os.environ['ROOT'], f"figures/{exp.dataset}/{backbone['name']}")
            os.makedirs(exp.save_path, exist_ok=True)
            
            exp.run_random_optimization(
                max_param=50000, 
                max_evals=max_evals, 
                plateau_threshold=250
            )
            
            params = exp.run_adam_ablations(
                max_updates=max_evals, 
                learning_threshold=0.5, 
                learning_rate=backbone['alpha'], 
                seeds=seeds
            )
            adam_alpha = params['adam'][0]
            adam_betas = params['adam'][1], params['adam'][2]

            exp.run_targeted_regularization(
                max_updates=max_evals, 
                learning_rate=adam_alpha, 
                betas=adam_betas,
                seeds=seeds
            )


if __name__ == "__main__":
    main()
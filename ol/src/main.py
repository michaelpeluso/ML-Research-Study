import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

from experiment import Experiment
from experiments.compare_performance import generate_comparison_report


def main():

    # L2 testing
    testing = False
    subsample = 0.1 if testing else None
    seeds = [42] if testing else [42, 4242, 424242]
    max_evals = 500 if testing else 3000

    # experiment flags
    run_random_optimization = True
    run_adam_ablations = True
    run_targeted_regularization = True


    experiments = [
        {
            'dataset': "hotels",
            'target': "is_canceled",
            'method': "classification",
            'subsample': subsample or 1.0,
            'batch_size': 64,
            'learning_threshold': 0.48,
            'backbone_configs': [
                {'name': 'nn_2', 'max_iter': 15, 'learning_rate_init': 0.05, 'hidden_layer_sizes': (256, 128), 'alpha': 1e-05, 'activation': 'tanh'},
                {'name': 'nn_4', 'max_iter': 15, 'learning_rate_init': 0.05, 'hidden_layer_sizes': (256, 256, 128, 128), 'alpha': 1e-05, 'activation': 'tanh'}
            ]
        },
        {
            'dataset': "accidents",
            'target': "Duration_Seconds",
            'method': "regression",
            'subsample': subsample or 0.8,  # > 80%
            'batch_size': 1024,
            'learning_threshold': 0.43,
            'backbone_configs': [
                {'name': 'nn_2', 'max_iter': 5, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (256, 128), 'alpha': 0.001, 'activation': 'relu'},
                {'name': 'nn_4', 'max_iter': 5, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (256, 256, 128, 128), 'alpha': 0.001, 'activation': 'relu'}
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
            
            # Reset data loaders
            exp.train_loader = None
            exp.val_loader = None
            exp.test_loader = None
            
            # Store results
            ro_results = None
            adam_results = None
            reg_results = None
            
            if run_random_optimization:
                ro_results = exp.run_random_optimization(
                    max_param=50000, 
                    max_evals=max_evals, 
                    plateau_threshold=500,
                    seeds=seeds
                )
            
            if run_adam_ablations:
                adam_hypers, adam_results = exp.run_adam_ablations(
                    max_updates=max_evals, 
                    learning_threshold=exp_config['learning_threshold'], 
                    learning_rate=backbone['alpha'], 
                    seeds=seeds
                )
                adam_alpha = adam_hypers['adam'][0] if 'adam' in adam_hypers else backbone['alpha']
                adam_betas = adam_hypers['adam'][1], adam_hypers['adam'][2]
            else:
                adam_alpha = backbone['alpha']
                adam_betas = (0.9, 0.999)

            if run_targeted_regularization:
                reg_results = exp.run_targeted_regularization(
                    max_updates=max_evals, 
                    learning_rate=adam_alpha, 
                    betas=adam_betas,
                    seeds=seeds
                )
            
            if run_random_optimization and run_adam_ablations and run_targeted_regularization:
                generate_comparison_report(
                    exp=exp,
                    architecture=backbone['name'],
                    seeds=seeds,
                    ro_results=ro_results,
                    adam_results=adam_results,
                    reg_results=reg_results
                )


if __name__ == "__main__":
    main()
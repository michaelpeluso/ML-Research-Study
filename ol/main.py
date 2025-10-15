import os, random, numpy as np, torch

from src.experiment import Experiment

def main():

    subsample=False # L2 testing
    seeds=[42, 4242, 424242]

    # ------------------
    # Hotels (~120k × 32)
    # ------------------
    hotels_exp = Experiment(
        id=5,
        dataset="hotels",
        target="is_canceled",
        method="classification",
        subsample=subsample or 1.0,
        batch_size=64
    )

    backbone_configs = [
        {'name': 'nn_2', 'max_iter': 15, 'learning_rate_init': 0.05, 'hidden_layer_sizes': (256, 128), 'alpha': 1e-05, 'activation': 'tanh'},  # shallow: 2 layers
        {'name': 'nn_4', 'max_iter': 15, 'learning_rate_init': 0.05, 'hidden_layer_sizes': (256, 256, 128, 128), 'alpha': 1e-05, 'activation': 'tanh'}  # deep: 4 layers
    ]

    for config in backbone_configs:
        hotels_exp.best_params.update(config)
        hotels_exp.update_logs()
        hotels_exp.ml_logger.current_logs = []
        hotels_exp.save_path = f"figures/{hotels_exp.dataset}/{config['name']}"
        os.makedirs(hotels_exp.save_path, exist_ok=True)
        
        hotels_exp.run_part1_ro(
            max_param=50000, 
            max_evals=10000, 
           plateau_threshold=250
        )
        hotels_exp.run_part2_ablations(
            max_updates=10000, 
            learning_threshold=0.5, 
            learning_rate=0.01, 
            seeds=seeds
            )

    # ------------------
    # US Accidents (~8M × 46)
    # ------------------
    accidents_exp = Experiment(
        id=10,
        dataset="accidents",
        target="Duration_Seconds",
        method="regression",
        subsample=subsample or 0.8, # > 80%
        batch_size=1024
    )

    backbone_configs = [
        {'name': 'nn_2', 'max_iter': 5, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (256, 128), 'alpha': 0.001, 'activation': 'relu'},  # shallow: 2 layers
        {'name': 'nn_4', 'max_iter': 5, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (256, 256, 128, 128), 'alpha': 0.001, 'activation': 'relu'}  # deep: 4 layers
    ]

    for config in backbone_configs:
        accidents_exp.best_params.update(config)
        accidents_exp.update_logs()

        accidents_exp.ml_logger.current_logs = []
        accidents_exp.save_path = f"figures/{accidents_exp.dataset}/{config['name']}"
        os.makedirs(accidents_exp.save_path, exist_ok=True)
        
        accidents_exp.run_part1_ro(
            max_param=50000, 
            max_evals=10000, 
            plateau_threshold=250
        )
        accidents_exp.run_part2_ablations(
            max_updates=10000, 
            learning_threshold=0.5, 
            learning_rate=0.01, 
            seeds=seeds
        )   
    

if __name__ == "__main__":
    main()
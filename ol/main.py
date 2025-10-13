import os, random, numpy as np, torch

from src.experiment import Experiment

def set_seed(s=4242):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    
set_seed(4242)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    subsample=0.1 # L2 testing

    # ------------------
    # Hotels (~120k × 32)
    # ------------------
    hotels_nn_experiment = Experiment(
        id=5,
        dataset="hotels",
        target="is_canceled",
        method="classification",
        subsample=subsample or 1.0,
    )

    backbone_configs = [
        {'name': 'nn_2', 'max_iter': 15, 'learning_rate_init': 0.05, 'hidden_layer_sizes': (512, 512), 'alpha': 1e-05, 'activation': 'tanh'},  # shallow: 2 layers
        {'name': 'nn_4', 'max_iter': 15, 'learning_rate_init': 0.05, 'hidden_layer_sizes': (256, 256, 128, 128), 'alpha': 1e-05, 'activation': 'tanh'}  # deep: 4 layers
    ]

    for config in backbone_configs:
        hotels_nn_experiment.best_params.update(config)
        hotels_nn_experiment.update_logs()
        hotels_nn_experiment.ml_logger.current_logs = []
        hotels_nn_experiment.save_path = f"figures/{hotels_nn_experiment.dataset}/{config['name']}"
        os.makedirs(hotels_nn_experiment.save_path, exist_ok=True)
        hotels_nn_experiment.run_part2_ablations()
        #hotels_nn_experiment.run_part1_ro()

    # ------------------
    # US Accidents (~8M × 46)
    # ------------------
    accidents_nn_experiment = Experiment(
        id=10,
        dataset="accidents",
        target="Duration_Seconds",
        method="regression",
        subsample=subsample or 0.8  # > 80%
    )

    backbone_configs = [
        {'name': 'nn_2', 'max_iter': 5, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (512, 512), 'alpha': 0.001, 'activation': 'relu'},  # shallow: 2 layers
        {'name': 'nn_4', 'max_iter': 5, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (256, 256, 128, 128), 'alpha': 0.001, 'activation': 'relu'}  # deep: 4 layers
    ]

    for config in backbone_configs:
        accidents_nn_experiment.best_params.update(config)
        accidents_nn_experiment.update_logs()
        accidents_nn_experiment.ml_logger.current_logs = []
        accidents_nn_experiment.save_path = f"figures/{accidents_nn_experiment.dataset}/{config['name']}"
        os.makedirs(accidents_nn_experiment.save_path, exist_ok=True)
        accidents_nn_experiment.run_part1_ro()
    

if __name__ == "__main__":
    main()
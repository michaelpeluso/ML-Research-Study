import os, random, numpy as np, torch

from src.ModelingExperiment import ModelingExperiment

def set_seed(s=4242):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    
set_seed(4242)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    seed=1
    tuning=True
    subsample=0.1 # L2 testing

    # ------------------
    # Hotels (~120k × 32)
    # ------------------

    # Neural Network
    hotels_nn_experiment = ModelingExperiment(
        id=5,
        dataset="hotels",
        target="is_canceled",
        method="classification",
        model="nn",
        subsample=subsample or .10,
        seed=seed,
        tuning=tuning,
        cv_splits=5,
        combination_cap=2,
        best_params={'estimator__max_iter': 15, 'estimator__learning_rate_init': 0.01, 'estimator__hidden_layer_sizes': 512, 'estimator__alpha': 0.001, 'estimator__activation': 'relu'}
    )

    # ------------------
    # US Accidents (~8M × 46)
    # ------------------

    # Neural Network
    accidents_nn_experiment = ModelingExperiment(
        id=10,
        dataset="accidents",
        target="Duration_Seconds",
        method="regression",
        model="nn",
        subsample=subsample or 0.8,  # > 80%
        seed=seed,
        tuning=tuning,
        cv_splits=3,
        combination_cap=10,
        best_params={'estimator__max_iter': 10, 'estimator__learning_rate_init': 0.005, 'estimator__alpha': 0.0001, 'estimator__activation': 'tanh'}
    )

    
    # ------------------
    # Execute runners
    # ------------------
    hotels_nn_experiment.best_params.update({'hidden_layer_sizes': (512, 512)})
    hotels_nn_experiment.run()
    hotels_nn_experiment.best_params.update({'hidden_layer_sizes': (256, 256, 128, 128)})
    hotels_nn_experiment.run()
    
    accidents_nn_experiment.best_params.update({'hidden_layer_sizes': (512, 512)})
    accidents_nn_experiment.run()
    accidents_nn_experiment.best_params.update({'hidden_layer_sizes': (256, 256, 128, 128)})
    accidents_nn_experiment.run()
    

if __name__ == "__main__":
    main()
import os, random, numpy as np, torch

from src.modeling_experiment import ModelingExperiment

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

    # Decision Tree
    hotels_dt_experiment = ModelingExperiment(
        id=1,
        dataset="hotels",
        target="is_canceled",
        method="classification",
        model="dt",
        subsample=subsample if subsample else 1.0, #1.0,  # full dataset
        seed=seed,
        tuning=tuning,
        cv_splits=10,
        combination_cap=40,
        best_params = {'estimator__min_samples_split': 400, 'estimator__min_samples_leaf': 50, 'estimator__max_features': 0.5, 'estimator__max_depth': 18, 'estimator__ccp_alpha': 0.0}
    )

    # KNN
    hotels_knn_experiment = ModelingExperiment(
        id=2,
        dataset="hotels",
        target="is_canceled",
        method="classification",
        model="knn",
        subsample=subsample if subsample else 1.0,
        seed=seed,
        tuning=tuning,
        cv_splits=5,
        combination_cap=50,
        best_params={'estimator__n_neighbors': 5}
    )

    # Linear SVM
    hotels_lsvm_experiment = ModelingExperiment(
        id=3,
        dataset="hotels",
        target="is_canceled",
        method="classification",
        model="lsvm",
        subsample=subsample or 1.0,
        seed=seed,
        tuning=tuning,
        cv_splits=10,
        combination_cap=30,
        best_params={'estimator__tol': 0.001, 'estimator__penalty': None, 'estimator__max_iter': 5000, 'estimator__learning_rate': 'optimal', 'estimator__alpha': 1e-05}
    )

    # Kernel SVM
    hotels_svm_experiment = ModelingExperiment(
        id=4,
        dataset="hotels",
        target="is_canceled",
        method="classification",
        model="svm",
        subsample=subsample or 1.0,
        seed=seed,
        tuning=tuning,
        cv_splits=3,
        combination_cap=10,
        best_params={'estimator__C': 8, 'estimator__gamma': 'auto'}
    )

    # Neural Network
    hotels_nn_experiment = ModelingExperiment(
        id=5,
        dataset="hotels",
        target="is_canceled",
        method="classification",
        model="nn",
        subsample=subsample or 1.0,
        seed=seed,
        tuning=tuning,
        cv_splits=5,
        combination_cap=20,
        best_params={'estimator__max_iter': 15, 'estimator__learning_rate_init': 0.01, 'estimator__hidden_layer_sizes': 512, 'estimator__alpha': 0.001, 'estimator__activation': 'relu'}
    )

    # ------------------
    # US Accidents (~8M × 46)
    # ------------------

    # Decision Tree
    accidents_dt_experiment = ModelingExperiment(
        id=6,
        dataset="accidents",
        target="Duration_Seconds",
        method="regression",
        model="dt",
        subsample=subsample or 1.0,  # 1M rows
        seed=seed,
        tuning=tuning,
        cv_splits=5,
        combination_cap=30,
        best_params={'estimator__min_samples_split': 400, 'estimator__min_samples_leaf': 50, 'estimator__max_features': 0.5, 'estimator__max_depth': 18, 'estimator__ccp_alpha': 0.0}
    )

    # kNN
    accidents_knn_experiment = ModelingExperiment(
        id=7,
        dataset="accidents",
        target="Duration_Seconds",
        method="regression",
        model="knn",
        subsample=subsample or 0.3,  # <250K rows, but runs fast
        seed=seed,
        tuning=tuning,
        cv_splits=5,
        combination_cap=10,
        best_params={'estimator__n_neighbors': 21}
    )

    # Linear SVM
    accidents_lsvm_experiment = ModelingExperiment(
        id=8,
        dataset="accidents",
        target="Duration_Seconds",
        method="regression",
        model="lsvm",
        subsample=subsample or 1.0,  # 1M rows
        seed=seed,
        tuning=tuning,
        cv_splits=5,
        combination_cap=25,
        best_params={'estimator__tol': 0.0001, 'estimator__penalty': None, 'estimator__max_iter': 20000, 'estimator__learning_rate': 'adaptive', 'estimator__alpha': 1e-05}
    )

    # Kernel SVM
    accidents_svm_experiment = ModelingExperiment(
        id=9,
        dataset="accidents",
        target="Duration_Seconds",
        method="regression",
        model="svm",
        subsample=subsample or 0.1,  # <100K rows
        seed=seed,
        tuning=tuning,
        cv_splits=3,
        combination_cap=10,
        best_params={'estimator__C': 0.5, 'estimator__gamma': 'scale'}
    )

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
    hotels_dt_experiment.run()
    hotels_knn_experiment.run()
    hotels_lsvm_experiment.run() 
    hotels_svm_experiment.run() 
    hotels_nn_experiment.best_params.update({'hidden_layer_sizes': (512, 512)})
    hotels_nn_experiment.run()
    hotels_nn_experiment.best_params.update({'hidden_layer_sizes': (256, 256, 128, 128)})
    hotels_nn_experiment.run()
    
    accidents_dt_experiment.run() 
    accidents_knn_experiment.run()
    accidents_lsvm_experiment.run() 
    accidents_svm_experiment.run() 
    accidents_nn_experiment.best_params.update({'hidden_layer_sizes': (512, 512)})
    accidents_nn_experiment.run()
    accidents_nn_experiment.best_params.update({'hidden_layer_sizes': (256, 256, 128, 128)})
    accidents_nn_experiment.run()
    

if __name__ == "__main__":
    main()
import os
import warnings
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")  # Suppress NN batch_size/convergence warnings

from src.utils.logger import MLLogger
from src.data_processing import  load_or_process_data
from src.tuning import save_results, tune_parameter_grid, train_nn_with_curves
from src.utils.diagnostics import get_hardware_info, get_model_info, measure_runtime
from src.evaluation import compute_complexity_curve, compute_learning_curve, evaluate_test_set
from src.pipelines import build_column_transformer, build_pipeline
from src.utils.plotter import plot_complexity_curve, plot_feature_importance, plot_model_evaluation, plot_pruning_path, plot_epoch_curve
from src.utils.tuning_config import get_primary_params, get_tuning_grid

class ModelingExperiment:
    def __init__(self, id, dataset, target, method, model, subsample, seed, tuning=True, cv_splits=5, combination_cap=50, best_params={}):
        self.experiment_id = id
        self.dataset = dataset
        self.target = target
        self.method = method
        self.model = model
        self.subsample = subsample
        self.seed = seed
        self.tuning = tuning
        self.cv_splits = cv_splits
        self.combination_cap = combination_cap
        self.best_params = best_params
        self.save_path = f"figures/{self.dataset}/{self.model}"

        self.ml_logger = MLLogger()
        self.ml_logger.current_logs = []
        self.update_logs()

    # Load or process data
    def get_data(self):
        with self.ml_logger.log_step("Load Data") as step_info:
            X_train, X_test, y_train, y_test, data_info = load_or_process_data(self.dataset, self.target, self.method, self.subsample, self.seed)
            step_info.update(data_info) # type: ignore
        return X_train, X_test, y_train, y_test


    def run(self):
        print(f"\n\nExecuting  {self.model}  {self.method}  on  {self.dataset}...".upper())

        X_train, X_test, y_train, y_test = self.get_data()
       
            
    def update_logs(self):
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method,
            model=self.model,
            subsample=self.subsample,
            seed=self.seed,
            tuning=self.tuning,
            cv_splits=self.cv_splits,
            combination_cap=self.combination_cap
        )
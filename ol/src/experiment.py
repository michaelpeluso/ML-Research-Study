import os
from typing import Any, Dict, List, Optional, Tuple
import warnings
import torch

from experiments.random_optimization import random_optimization
from experiments.adam_ablations import adam_ablations
from experiments.targeted_regularization import targeted_regularization

from utils.logger import MLLogger
from utils.data_processing import load_or_process_data, wrap_into_loaders
from core.models import set_seed


# Suppress warnings
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class Experiment:
    def __init__(self, dataset: str, target: str, method: str, subsample: float, batch_size: int, best_params: Dict[str, Any]|None=None):
        """Initialize experiment with dataset config and setup logging/system info."""
        self.dataset = dataset
        self.target = target
        self.method = method
        self.subsample = subsample
        self.batch_size = batch_size
        self.seed = set_seed()
        self.best_params = best_params or {}
        self.save_path = os.path.join(os.environ['ROOT'], f"figures/{self.dataset}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ml_logger = MLLogger()
        self.ml_logger.current_logs = []
        self.update_logs()


    def get_data(self) -> Tuple:
        """Load/process data and wrap into PyTorch DataLoaders with logging."""
        with self.ml_logger.log_step("Load Data") as step_info:
            X_train, X_val, X_test, y_train, y_val, y_test, data_info = load_or_process_data(
                self.dataset, self.target, self.method, self.subsample, self.seed
            )
            step_info.update(data_info)
            train_loader, val_loader, test_loader = wrap_into_loaders(
                self.method, X_train, X_val, X_test, y_train, y_val, y_test, self.batch_size
            )
        return train_loader, val_loader, test_loader

    def update_logs(self):
        """Update experiment context in logger for traceability."""
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method,
            subsample=self.subsample,
            seed=self.seed
        )

    # part 1: Random Optimization
    def run_random_optimization(self, max_param: int = 50000, max_evals: int = 11000, plateau_threshold: int = 250):
        random_optimization(self, max_param, max_evals, plateau_threshold)
    
    # part 2: Adam Ablations
    def run_adam_ablations(self, max_updates: int = 10000, learning_threshold: float = 0.5, learning_rate: float = 0.01, seeds: List[int] = [42, 4242, 424242]):
        return adam_ablations(self, max_updates, learning_threshold, learning_rate, seeds)

    # part 3: Adam Regularization
    def run_targeted_regularization(self, max_updates: int = 10000, learning_rate: Optional[float] = None, betas: Tuple[float, float] = (0.9, 0.999), seeds: List[int] = [42, 4242, 424242]):
        targeted_regularization(self, max_updates, learning_rate, betas, seeds)
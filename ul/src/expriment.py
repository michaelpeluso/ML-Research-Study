import os
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.utils.data_processing import load_or_process_data, wrap_into_loaders
from src.utils.logger import MLLogger

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class Experiment:
    def __init__(self, dataset: str, target: str, method: str, subsample: float, batch_size: int, best_params: Dict[str, Any]|None=None):
        """Initialize experiment with dataset config and setup logging/system info."""
        self.dataset = dataset
        self.target = target
        self.method = method
        self.subsample = subsample
        self.batch_size = batch_size
        self.seed = self.set_seed()
        self.best_params = best_params or {}
        self.save_path = os.path.join(os.environ['ROOT'], f"figures/{self.dataset}")

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ml_logger = MLLogger()
        self.ml_logger.current_logs = []
        self.update_logs()

    def get_data(self) -> Tuple:
        """Load/process data and wrap into PyTorch DataLoaders with caching."""
        if self.train_loader is None:
            with self.ml_logger.log_step("Load Data") as step_info:
                X_train, X_val, X_test, y_train, y_val, y_test, data_info = load_or_process_data(
                    self.dataset, self.target, self.method, self.subsample, self.seed
                )
                step_info.update(data_info)
                self.train_loader, self.val_loader, self.test_loader = wrap_into_loaders(
                    self.method, X_train, X_val, X_test, y_train, y_val, y_test, self.batch_size
                )
        return self.train_loader, self.val_loader, self.test_loader

    def set_seed(self, seed=4242) -> int:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.seed = seed
        return seed
    
    def update_logs(self):
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method,
            subsample=self.subsample,
            seed=self.seed
        )

    # Clustering on raw data
    def run_step1(self, k_range=(2, 10)):
        return None

    # Dimensionality reduction on raw data
    def run_step2(self, target_dim=None):
        return None

    # Clustering on dimensionality reduced data
    def run_step3(self, dr_method, cluster_method, k):
        return None

    # Neural network dimensionality reduction
    def run_step4(self, dr_method, max_updates=1500):
       return None

    # Neural network with clustering
    def run_step5(self, cluster_method, feature_mode="append", max_updates=1500):
        return None
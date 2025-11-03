import os
from typing import Tuple, Literal
import torch

from utils.data_processing import load_or_process_data, wrap_into_loaders
from utils.logger import MLLogger
from clustering.kmeans import KMeansClustering
from clustering.em import EMClustering

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class Experiment:
    def __init__(self, dataset: str, target: str, method: str, seed: int = 42):
        """Initialize experiment with dataset config and setup logging/system info."""
        self.dataset = dataset
        self.target = target
        self.method = method
        self.seed = seed
        self.save_path = os.path.join(os.environ['ROOT'], f"figures/{self.dataset}")
        os.makedirs(self.save_path, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ml_logger = MLLogger()
        self.ml_logger.current_logs = []
        self.update_logs()

    def update_logs(self):
        self.ml_logger.set_experiment_context(
            dataset=self.dataset,
            target=self.target,
            method=self.method
        )

    def get_data(self, subsample=1.0) -> Tuple:
        """Load/process data"""
        with self.ml_logger.log_step("Load Data") as step_info:
            X_train, X_val, X_test, y_train, y_val, y_test, data_info = \
                load_or_process_data(self.dataset, self.target, self.method, subsample, seed=42)
            step_info.update(data_info)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_loaders(self, xtr=None, xv=None, xt=None, ytr=None, yv=None, yt=None, subsample=1.0, batch_size=None) -> Tuple:
        """wrap data into PyTorch DataLoaders with caching."""
        if xtr is None: xtr, xv, xt, ytr, yv, yt, data_info = self.get_data(subsample=subsample)
        train_loader, val_loader, test_loader = wrap_into_loaders(self.method, xtr, xv, xt, ytr, yv, yt, batch_size)
        return train_loader, val_loader, test_loader


    def run_kmeans(self, k_range=(2, 10), stability_runs=10, data_subsample=1.0, plot_subsample_size=5000, n_init: int = 10, metric_weights={'silhouette_score': 1.0}, n_jobs:int=1):
        """Run K-Means clustering on raw data."""
        X_train, _, _, _, _, _ = self.get_data(subsample=data_subsample)
        clustering = KMeansClustering(self.dataset, f"{self.save_path}/kmeans", self.ml_logger, plot_subsample_size=plot_subsample_size, seed=self.seed, n_jobs=n_jobs)
        model = clustering.run_kmeans(X_train, k_range, stability_runs, n_init=n_init, metric_weights=metric_weights)
        return model

    def run_em(self, n_components_range=(2, 15), stability_runs=10, data_subsample=1.0, plot_subsample_size=5000, covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = 'full', metric_weights={'silhouette_score': 1.0}, n_jobs:int=1):
        """Run Expectation-Maximization (GMM) clustering on raw data."""
        X_train, _, _, _, _, _ = self.get_data(subsample=data_subsample)
        clustering = EMClustering(self.dataset, f"{self.save_path}/em", self.ml_logger, plot_subsample_size=plot_subsample_size, seed=self.seed, n_jobs=n_jobs)
        model = clustering.run_gmm(X_train, n_components_range, stability_runs, covariance_type=covariance_type, metric_weights=metric_weights)
        return model

    # Clustering on raw data - EM
    def run_em_step1(self, k_range=(2, 15), chosen_components=5, stability_runs=10):
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
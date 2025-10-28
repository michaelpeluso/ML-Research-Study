import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from clustering.kmeans import run_em, run_kmeans
from utils.data_processing import load_or_process_data, wrap_into_loaders
from utils.logger import MLLogger

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class Experiment:
    def __init__(self, dataset: str, target: str, method: str):
        """Initialize experiment with dataset config and setup logging/system info."""
        self.dataset = dataset
        self.target = target
        self.method = method
        self.save_path = os.path.join(os.environ['ROOT'], f"figures/{self.dataset}")

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


    # Clustering on raw data
    def run_kmeans(self, k_range=(2, 10), X_train=None, seeds=[42]):
        if X_train is None: X_train, _, _, _, _, _ = self.get_data()
        
        for seed in seeds:
            for k in range(k_range[0], k_range[1] + 1):
                print(run_kmeans(X_train, n_clusters=k, random_state=seed))


    # Dimensionality reduction on raw data
    def run_em(self, k_range=(2, 10), X_train=None, seeds=[42]):
        if X_train is None: X_train, _, _, _, _, _ = self.get_data()
        
        for seed in seeds:
            for k in range(k_range[0], k_range[1] + 1):
                print(run_em(X_train, n_components=k, random_state=seed))


    # Clustering on dimensionality reduced data
    def run_step3(self, dr_method, cluster_method, k):
        return None

    # Neural network dimensionality reduction
    def run_step4(self, dr_method, max_updates=1500):
       return None

    # Neural network with clustering
    def run_step5(self, cluster_method, feature_mode="append", max_updates=1500):
        return None
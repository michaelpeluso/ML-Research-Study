import os
import time
import numpy as np
from typing import Literal, Any

from sklearn.mixture import GaussianMixture
from utils.plotter import plot_curve, plot_multiple_y_axes
from utils.logger import print_t as print
from clustering.base import BaseClustering


class EMClustering(BaseClustering):
    """Expectation-Maximization (Gaussian Mixture Model) clustering for unsupervised learning experiments."""
    
    # Configuration - override base class attributes
    param_name = 'n_components'
    algorithm_name = 'GMM'
    centers_attr = 'means_'
    
    def __init__(self, dataset: str, save_path: str, ml_logger, silhouette_subsample: int = 10000, seed: int = 42, 
                 covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = 'full', tol: float = 1e-3, max_iter: int = 100):
        """Initialize EM clustering with algorithm-specific parameters."""
        super().__init__(dataset, save_path, ml_logger, silhouette_subsample, seed)
        self.covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = covariance_type
        self.tol = tol
        self.max_iter = max_iter
    
    def fit_model(self, X, n_clusters, seed, n_init) -> Any:
        """Fit GaussianMixture and return the fitted model."""
        print(f"Fitting GMM model with n_components={n_clusters}, covariance_type={self.covariance_type}, n_init={n_init}, tol={self.tol}, max_iter={self.max_iter}")
        gmm = GaussianMixture(
            n_components=n_clusters, 
            covariance_type=self.covariance_type,
            random_state=seed if seed is not None else self.seed, 
            n_init=n_init,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=0
        )
        gmm.fit(X)
        return gmm
    
    def get_additional_metrics(self, model, X_train):
        """Get GMM-specific metrics: BIC, AIC, log-likelihood."""
        return {
            "log_likelihood": float(model.score(X_train) * len(X_train)),
            "bic": float(model.bic(X_train)),
            "aic": float(model.aic(X_train)),
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_)
        }
    
    def plot_metrics(self, selection_results, chosen_n: int | None = None):
        """Plot all GMM metrics individually and together with multiple y-axes."""
        n_values = [r["n_components"] for r in selection_results]

        shared_metrics_config = [
            ('silhouette_score', 'Silhouette Score', 'Silhouette', 'higher', 'blue'),
            ('calinski_harabasz_score', 'Calinski-Harabasz Index', 'CHI', 'higher', 'green'),
            ('davies_bouldin_score', 'Davies-Bouldin Index', 'DBI', 'lower', 'orange'),
            ('dunn_index', 'Dunn Index', 'Dunn', 'higher', 'red'),
        ]
        all_metrics_config = [
            *shared_metrics_config,
            ('bic', 'BIC', 'BIC', 'lower', 'purple'),
            ('aic', 'AIC', 'AIC', 'lower', 'brown'),
            ('log_likelihood', 'Log-Likelihood', 'Log-Likelihood', 'higher', 'lightblue'),
        ]
        individuals_path = os.path.join(self.save_path, "individuals")
        os.makedirs(individuals_path, exist_ok=True)
        metric_data = []
        for key, full_name, short_name, direction, color in all_metrics_config:
            values = [r[key] for r in selection_results]
            metric_data.append((values, short_name, color))
            plot_curve(
                x=n_values,
                y_list=values,
                labels=[full_name],
                xlabel="Number of Components",
                ylabel=f"{full_name} ({direction} is better)",
                title=f"GMM: {full_name} vs n on {self.dataset}",
                save_path=os.path.join(individuals_path, f"{key}.png"),
                colors=[color],
                marker='o'
            )
        # Multi-axis plot for shared metrics
        plot_multiple_y_axes(
            x=n_values,
            y_series=[data[0] for data in metric_data[:4]],
            labels=[data[1] for data in metric_data[:4]],
            xlabel="Number of Components",
            title=f"GMM: Shared Metrics on {self.dataset}",
            save_path=os.path.join(self.save_path, "all_metrics.png"),
            colors=[data[2] for data in metric_data[:4]],
            vline_x=chosen_n,
            vline_label=f"Optimal n={chosen_n}" if chosen_n else None
        )

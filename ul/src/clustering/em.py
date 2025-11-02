import os
import time
import numpy as np
from typing import Literal

from sklearn.mixture import GaussianMixture
from utils.plotter import plot_dual_axis
from utils.logger import print_t as print
from clustering.base import BaseClustering


class EMClustering(BaseClustering):
    """Expectation-Maximization (Gaussian Mixture Model) clustering for unsupervised learning experiments."""
    
    # Configuration - override base class attributes
    param_name = 'n_components'
    dir_prefix = 'n'
    cluster_label = 'Component'
    cluster_color = 'green'
    algorithm_name = 'GMM'
    centers_attr = 'means_'
    
    def fit_model(self, X, n_clusters, seed, n_init=10, covariance_type: Literal['full', 'tied', 'diag', 'spherical']='full', tol=1e-3, max_iter=100, **kwargs):
        """Fit GaussianMixture and return the fitted model."""
        print(f"Fitting GMM model")
        gmm = GaussianMixture(
            n_components=n_clusters, 
            covariance_type=covariance_type,
            random_state=seed, 
            n_init=n_init,
            tol=tol,
            max_iter=max_iter,
            verbose=1
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
        """Plot BIC, AIC, silhouette, and Dunn vs n_components."""
        n_values = [r["n_components"] for r in selection_results]
        bic_values = [r["bic"] for r in selection_results]
        aic_values = [r["aic"] for r in selection_results]
        sil_scores = [r["silhouette_score"] for r in selection_results]
        dunn_scores = [r["dunn_index"] for r in selection_results]
        
        plot_dual_axis(
            x=n_values,
            y_left=[bic_values, aic_values],
            y_right=[sil_scores, dunn_scores],
            left_labels=["BIC", "AIC"],
            right_labels=["Silhouette Score", "Dunn Index"],
            left_ylabel="BIC / AIC",
            right_ylabel="Silhouette Score / Dunn Index",
            xlabel="Number of Components",
            title=f"GMM Metrics vs mixture_components on {self.dataset}",
            save_path=os.path.join(self.save_path, "combined_curve.png"),
            vline_x=chosen_n,
            vline_label=f"Optimal n={chosen_n}" if chosen_n else None,
        )

    def run_gmm(self, 
                X_train, 
                mixture_components: int | tuple = (2, 10), 
                stability_runs=10, 
                seed: int | None = None, 
                n_init: int = 10, 
                covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = 'full', 
                bic_weight: float = 0.4, 
                silhouette_weight: float = 0.4, 
                dunn_weight: float = 0.2
                ):
        """Run GMM clustering. Wrapper around base run_clustering with GMM-specific defaults."""
        metric_weights = {
            'bic': bic_weight,
            'silhouette_score': silhouette_weight,
            'dunn_index': dunn_weight
        }
        return self.run_clustering(X_train, mixture_components, stability_runs, seed, metric_weights, 
                                   n_init=n_init, covariance_type=covariance_type)

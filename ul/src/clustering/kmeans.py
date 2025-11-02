import os
import time
import numpy as np

from sklearn.cluster import KMeans
from utils.plotter import plot_dual_axis
from utils.logger import print_t as print
from clustering.base import BaseClustering


class KMeansClustering(BaseClustering):
    """KMeans clustering implementation extending BaseClustering."""
    
    # Configuration - override base class attributes
    param_name = 'k'
    dir_prefix = 'k'
    cluster_label = 'Cluster'
    cluster_color = 'orange'
    algorithm_name = 'K-Means'
    centers_attr = 'cluster_centers_'
    
    def fit_model(self, X, n_clusters, seed, n_init=10, **kwargs):
        """Fit KMeans and return the fitted model."""
        print(f"Fitting KMeans model")
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
        kmeans.fit(X)
        return kmeans
    
    def get_additional_metrics(self, model, X_train):
        """Get KMeans-specific metric: inertia."""
        return {
            "inertia": float(model.inertia_)
        }
    
    def plot_metrics(self, selection_results, chosen_n: int | None = None):
        """Plot inertia, silhouette, and Dunn vs k with dual axes."""
        k_values = [r["k"] for r in selection_results]
        inertias = [r["inertia"] for r in selection_results]
        sil_scores = [r["silhouette_score"] for r in selection_results]
        dunn_scores = [r["dunn_index"] for r in selection_results]
        plot_dual_axis(
            x=k_values,
            y_left=inertias,
            y_right=[sil_scores, dunn_scores],
            left_labels=["Inertia"],
            right_labels=["Silhouette Score", "Dunn Index"],
            left_ylabel="Inertia",
            right_ylabel="Silhouette Score / Dunn Index",
            xlabel="Number of Clusters k",
            title=f"Inertia, Silhouette, and Dunn vs k (K-Means) on {self.dataset}",
            save_path=os.path.join(self.save_path, "combined_curve.png"),
            vline_x=chosen_n,
            vline_label=f"Optimal k={chosen_n}" if chosen_n else None
        )

    def run_kmeans(self, X_train, k: int | tuple = (2, 10), stability_runs=10, seed: int|None=None, n_init: int = 10, silhouette_dunn_weight: tuple = (0.25, 0.75)):
        """Run K-Means clustering. Wrapper around base run_clustering with KMeans-specific defaults."""
        metric_weights = {
            'silhouette_score': silhouette_dunn_weight[0],
            'dunn_index': silhouette_dunn_weight[1]
        }
        return self.run_clustering(X_train, k, stability_runs, seed, metric_weights, n_init=n_init)

                       
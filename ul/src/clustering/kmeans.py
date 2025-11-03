import os
import time
import numpy as np

from sklearn.cluster import KMeans
from utils.plotter import plot_curve, plot_multiple_y_axes
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
    
    def fit_model(self, X, n_clusters, n_init=10, **kwargs):
        """Fit KMeans and return the fitted model."""
        print(f"Fitting KMeans model")
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=n_init)
        kmeans.fit(X)
        return kmeans
    
    def get_additional_metrics(self, model, X_train):
        """Get KMeans-specific metric: inertia."""
        return {
            "inertia": float(model.inertia_)
        }
    
    def plot_metrics(self, selection_results, chosen_n: int | None = None):
        """Plot all K-Means metrics individually and together with multiple y-axes."""
        k_values = [r["k"] for r in selection_results]
        
        # Define metrics with their properties
        metrics_config = [
            ('silhouette_score', 'Silhouette Score', 'Silhouette', 'higher', 'blue'),
            ('calinski_harabasz_score', 'Calinski-Harabasz Index', 'CHI', 'higher', 'green'),
            ('davies_bouldin_score', 'Davies-Bouldin Index', 'DBI', 'lower', 'orange'),
            ('dunn_index', 'Dunn Index', 'Dunn', 'higher', 'purple'),
            ('inertia', 'Inertia', 'Inertia', 'lower', 'red'),
        ]
        
        # Extract metric values and generate individual plots
        individuals_path = os.path.join(self.save_path, "individuals")
        os.makedirs(individuals_path, exist_ok=True)
        
        metric_data = []
        for key, full_name, short_name, direction, color in metrics_config:
            values = [r[key] for r in selection_results]
            metric_data.append((values, short_name))
            
            plot_curve(
                x=k_values,
                y_list=values,
                labels=[full_name],
                xlabel="Number of Clusters k",
                ylabel=f"{full_name} ({direction} is better)",
                title=f"K-Means: {full_name} vs k on {self.dataset}",
                save_path=os.path.join(individuals_path, f"{key}.png"),
                colors=[color],
                marker='o'
            )
        
        # Comprehensive plot with all metrics using multiple y-axes
        plot_multiple_y_axes(
            x=k_values,
            y_series=[data[0] for data in metric_data],
            labels=[data[1] for data in metric_data],
            xlabel="Number of Clusters k",
            title=f"K-Means: All Metrics on {self.dataset}",
            save_path=os.path.join(self.save_path, "all_metrics_multi_axis.png"),
            vline_x=chosen_n,
            vline_label=f"Optimal k={chosen_n}" if chosen_n else None
        )
        
    def run_kmeans(self, X_train, k_range=(2, 15), stability_runs=10, n_init: int = 10, metric_weights={'silhouette_score': 1.0}, n_jobs:int=1):
        """Run K-Means clustering with configurable metric weights."""
        return self.run_clustering(X_train, k_range, stability_runs, metric_weights, n_init=n_init)

                       
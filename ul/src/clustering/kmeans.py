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
    
    def fit_model(self, X, n_clusters, seed=None, n_init=10, **kwargs):
        """Fit KMeans and return the fitted model."""
        print(f"Fitting KMeans model")
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed or self.seed, n_init=n_init)
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
        # Shared metrics and colors
        shared_metrics_config = [
            ('silhouette_score', 'Silhouette Score', 'Silhouette', 'higher', 'blue'),
            ('calinski_harabasz_score', 'Calinski-Harabasz Index', 'CHI', 'higher', 'green'),
            ('davies_bouldin_score', 'Davies-Bouldin Index', 'DBI', 'lower', 'orange'),
            ('dunn_index', 'Dunn Index', 'Dunn', 'higher', 'red'),
        ]
        # All metrics (including KMeans-specific)
        all_metrics_config = [
            *shared_metrics_config,
            ('inertia', 'Inertia', 'Inertia', 'lower', 'purple'),
        ]
        individuals_path = os.path.join(self.save_path, "individuals")
        os.makedirs(individuals_path, exist_ok=True)
        metric_data = []
        for key, full_name, short_name, direction, color in all_metrics_config:
            values = [r[key] for r in selection_results]
            metric_data.append((values, short_name, color))
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
        # Multi-axis plot for shared metrics
        plot_multiple_y_axes(
            x=k_values,
            y_series=[data[0] for data in metric_data[:4]],
            labels=[data[1] for data in metric_data[:4]],
            xlabel="Number of Clusters k",
            title=f"K-Means: Shared Metrics on {self.dataset}",
            save_path=os.path.join(self.save_path, "shared_metrics_multi_axis.png"),
            colors=[data[2] for data in metric_data[:4]],
            vline_x=chosen_n,
            vline_label=f"Optimal k={chosen_n}" if chosen_n else None
        )
        

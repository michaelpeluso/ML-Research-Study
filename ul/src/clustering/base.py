import os
import shutil
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from joblib import Parallel, delayed

from utils.plotter import plot_silhouette
from utils.logger import MLLogger, print_t as print
from utils.data_processing import sample_fit_labels


class BaseClustering(ABC):
    """Abstract base class for clustering algorithms."""
    
    # Configuration - subclasses should override these class attributes
    param_name = 'n_clusters'       # 'k' or 'n_components'
    dir_prefix = 'n'                # 'k' or 'n'
    cluster_label = 'Cluster'       # 'Cluster' or 'Component'
    cluster_color = 'blue'          # Color for plots
    algorithm_name = 'Clustering'   # Algorithm name for titles
    centers_attr = 'cluster_centers_'  # Attribute name for cluster centers
    
    def __init__(self, dataset: str, save_path: str, ml_logger: MLLogger, plot_subsample_size: int = 10000, seed: int = 42):
        self.dataset = dataset
        self.save_path = save_path
        self.ml_logger = ml_logger
        self.plot_subsample_size = plot_subsample_size
        self.seed = seed
        os.makedirs(self.save_path, exist_ok=True)
    
    @abstractmethod
    def fit_model(self, X, n_clusters, **kwargs) -> Any:
        """Fit and return the clustering model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def plot_metrics(self, selection_results, chosen_n: int | None = None):
        """Plot metrics vs number of clusters. Must be implemented by subclasses."""
        pass
    
    def extract_results(self, model, X):
        """Extract labels and centers from a fitted model."""
        labels = model.predict(X)
        centers = getattr(model, self.centers_attr)
        return labels, centers
    
    def get_additional_metrics(self, model, X_train):
        """Get algorithm-specific metrics (e.g., inertia for KMeans, BIC/AIC for GMM). """
        return {}
    
    def measure_clustering(self, X_train, n_clusters, **kwargs):
        """Measure clustering quality for a specific number of clusters."""
        print(f"Measuring {self.algorithm_name} ({self.param_name}={n_clusters})")
        with self.ml_logger.log_step(f"{self.algorithm_name} Hyperparameter Selection ({self.param_name}={n_clusters})") as step_info:  
            start_time = time.perf_counter()
            
            # Run clustering multiple times with different seeds
            all_metrics = []
            models = []
                        
            # Fit model and extract results
            model = self.fit_model(X_train, n_clusters=n_clusters, seed=self.seed, **kwargs)
            labels, centers = self.extract_results(model, X_train)
            models.append((model, labels, centers))
            
            # Compute metrics for this run
            sil_score = -1
            chi_score = -1
            dbi_score = -1
            
            if n_clusters > 1:
                Xs, ys, sample_idx = sample_fit_labels(X_train, labels, sample_size=self.plot_subsample_size, seed=self.seed)
                sil_score, _ = self.compute_silhouette(Xs, ys)
                chi_score = float(calinski_harabasz_score(X_train, labels))
                dbi_score = float(davies_bouldin_score(X_train, labels))
            
            dunn_idx = self.dunn_index(X_train, labels, centers)
            
            # Get additional algorithm-specific metrics
            additional_metrics = self.get_additional_metrics(model, X_train)
            
            all_metrics.append({
                'silhouette_score': sil_score,
                'calinski_harabasz_score': chi_score,
                'davies_bouldin_score': dbi_score,
                'dunn_index': dunn_idx,
                **additional_metrics
            })
            
            # Average metrics across runs
            avg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
                if values:
                    avg_metrics[key] = float(np.mean(values))
                else:
                    avg_metrics[key] = all_metrics[0][key]
            
            # Use first run for plotting
            model, labels, centers = models[0]
            sil_score = avg_metrics['silhouette_score']
            chi_score = avg_metrics['calinski_harabasz_score']
            dbi_score = avg_metrics['davies_bouldin_score']
            dunn_idx = avg_metrics['dunn_index']
            
            sample_sil_vals = None
            sample_idx = None
            
            if n_clusters > 1:
                Xs, ys, sample_idx = sample_fit_labels(X_train, labels, sample_size=self.plot_subsample_size, seed=self.seed)
                _, sample_sil_vals = self.compute_silhouette(Xs, ys)

            # Generate plots
            self.generate_plots(X_train, labels, n_clusters, sil_score, sample_sil_vals, sample_idx, centers)

            # Log results
            time_taken = time.perf_counter() - start_time
            metrics_str = f"silhouette: {sil_score:.3f}, dunn: {dunn_idx:.3f}, chi: {chi_score:.3f}, dbi: {dbi_score:.3f}"
            
            for key, val in avg_metrics.items():
                if key not in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'dunn_index'] and not key.endswith('_std'):
                    if isinstance(val, (int, float)):
                        metrics_str += f", {key}: {val:.3f}"
                    else:
                        metrics_str += f", {key}: {val}"
            
            print(f"{self.algorithm_name} ({self.param_name}={n_clusters}) - {metrics_str}")
            
            selection_results = {
                self.param_name: n_clusters,
                "cluster_centers": centers.tolist(),
                "silhouette_score": sil_score,
                "calinski_harabasz_score": chi_score,
                "davies_bouldin_score": dbi_score,
                "dunn_index": dunn_idx,
                "time": time_taken,
                **{k: v for k, v in avg_metrics.items() if k not in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'dunn_index']}
            }
            step_info.update(selection_results)
            
            return model, selection_results
    
    
    def generate_plots(self, X_train, labels, n_clusters, sil_score, sample_sil_vals=None, sample_idx=None, centers=None):
        """Generate silhouette and cluster size plots for a specific number of clusters."""
        cluster_dir = os.path.join(self.save_path, "clusters", f"{self.dir_prefix}{n_clusters}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Silhouette plot
        sil_path = os.path.join(cluster_dir, "silhouette.png")
        if sample_sil_vals is None and n_clusters > 1:
            Xs, ys, sample_idx = sample_fit_labels(X_train, labels, sample_size=self.plot_subsample_size, seed=self.seed)
            sil_score, sample_sil_vals = self.compute_silhouette(Xs, ys)

        if sample_sil_vals is not None:
            print(f"Generating silhouette plot for {self.dir_prefix}={n_clusters}")
            plot_silhouette(X_train, labels,
                           title=f"Silhouette Plot ({self.dir_prefix}={n_clusters}) - Score: {sil_score:.3f}",
                           save_path=sil_path,
                           silhouette_avg=sil_score,
                           sample_silhouette_values=sample_sil_vals,
                           sample_indices=sample_idx)
        else:
            print(f"No silhouette values for {self.dir_prefix}={n_clusters}; skipping silhouette plot.")

    def compute_silhouette(self, X: np.ndarray, labels: np.ndarray):
        """ Compute silhouette average and per-sample silhouette values for provided X and labels. Measures cohesion and separation for each point. """
        print(f"Computing silhouette score")
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            return -1.0, None
        sil_avg = float(silhouette_score(X, labels))
        sil_vals = np.asarray(silhouette_samples(X, labels))
        return sil_avg, sil_vals


    def dunn_index(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray):
        """ Compute Dunn index: min inter-cluster distance / max intra-cluster distance. Measures separation and compactness for the entire clustering. """
        print(f"Computing Dunn index")
        n_clusters = len(centers)
        if n_clusters < 2:
            return -1  
        
        # Intra-cluster distances: max distance within each cluster
        intra_distances = []
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                distances = np.linalg.norm(cluster_points - centers[i], axis=1)
                intra_distances.append(np.max(distances))
        max_intra = np.max(intra_distances) if intra_distances else 0
        
        # Inter-cluster distances: min distance between any two centers
        inter_distances = []
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                dist = np.linalg.norm(centers[i] - centers[j])
                inter_distances.append(dist)
        min_inter = np.min(inter_distances) if inter_distances else 0

        return min_inter / max_intra if max_intra > 0 else -1

    
    def stability_analysis(self, X_train, chosen_n, stability_runs, **kwargs):
        """Run stability analysis by clustering multiple times and computing pairwise ARI."""
        print(f"Running stability analysis with {stability_runs} runs for {self.param_name}={chosen_n}")
        
        with self.ml_logger.log_step(f"{self.algorithm_name} Stability Analysis ({self.param_name}={chosen_n}, runs={stability_runs})") as step_info:
            # Fit multiple times with different seeds
            labels_list = []
            for i in range(stability_runs):
                print(f"Stability run {i + 1}/{stability_runs} for {self.param_name}={chosen_n}")
                # Use more diverse seeds for better stability testing
                seed_i = self.seed + (i * 1000)
                model = self.fit_model(X_train, n_clusters=chosen_n, seed=seed_i, **kwargs)
                labels_i, centers = self.extract_results(model, X_train)
                labels_list.append(labels_i)

            # Compute stability metrics using pairwise ARI
            n = len(labels_list)
            pairwise = np.ones((n, n), dtype=float)
            
            for i in range(n):
                for j in range(i + 1, n):
                    ari = adjusted_rand_score(labels_list[i], labels_list[j])
                    pairwise[i, j] = ari
                    pairwise[j, i] = ari

            # Compute off-diagonal statistics
            if n > 1:
                offdiag = pairwise[np.triu_indices(n, k=1)]
                pair_mean = float(np.mean(offdiag))
                pair_std = float(np.std(offdiag))
                pair_min = float(np.min(offdiag))
                pair_max = float(np.max(offdiag))
            else:
                pair_mean = pair_std = pair_min = pair_max = float('nan')

            print(f"Stability analysis complete - Mean pairwise ARI: {pair_mean:.3f} ± {pair_std:.3f}")
            
            results = {
                self.param_name: chosen_n,
                "total_runs": stability_runs,
                "stability_score": pair_mean,
                "stability_std": pair_std,
                "stability_min": pair_min,
                "stability_max": pair_max,
                "pairwise_matrix": pairwise.tolist(),
            }

            # Save pairwise matrix
            pair_path = os.path.join(self.save_path, "stability_pairwise.csv")
            np.savetxt(pair_path, pairwise, delimiter=',', fmt='%.6f')
            
            step_info.update(results)
    
    def run(self, X_train, n_clusters: int | tuple = (2, 10), stability_runs=10, n_jobs: int = 1, **kwargs):
        """Generic clustering runner that works for all algorithms."""
        start_log_index = len(self.ml_logger.current_logs)
        with self.ml_logger.log_step(f"{self.algorithm_name} Clustering ({self.param_name}={n_clusters})") as step_info:
            function_start = time.perf_counter()

            if isinstance(n_clusters, int):
                chosen_n = n_clusters
                model, selection_results = self.measure_clustering(X_train, chosen_n, stability_runs=1, **kwargs)
                step_info.update(selection_results)
                
            elif isinstance(n_clusters, tuple) and len(n_clusters) == 2:
                print(f"Evaluating {self.param_name} values from {n_clusters[0]} to {n_clusters[1]} (parallel jobs={n_jobs})")

                n_values = list(range(n_clusters[0], n_clusters[1] + 1))
                
                # Parallel execution - single run per k (n_init handles stability)
                results_with_models = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
                    delayed(self.measure_clustering)(X_train, n_val, stability_runs=1, **kwargs)
                    for n_val in n_values
                )
                
                # Unpack results
                selection_results = []
                models = {}
                for n_val, (model, result) in zip(n_values, results_with_models): # type: ignore
                    selection_results.append(result)
                    models[n_val] = model
                
                # Compute composite scores using Rank Aggregation
                chosen_n = self._compute_composite_scores(selection_results)
                best_result = next(r for r in selection_results if r[self.param_name] == chosen_n)
                
                # Plot metrics
                self.plot_metrics(selection_results, chosen_n=chosen_n)

                metrics_str = f"silhouette={best_result['silhouette_score']:.3f}, dunn={best_result['dunn_index']:.3f}"
                for key, val in best_result.items():
                    if key not in [self.param_name, 'cluster_centers', 'silhouette_score', 'dunn_index', 'time', 'composite_score']:
                        metrics_str += f", {key}={val:.3f}"
                print(f"Selected {self.param_name}={chosen_n} with {metrics_str}, composite={best_result['composite_score']:.3f}")
                
                step_info.update({
                    "total_time": sum(r["time"] for r in selection_results),
                    f"{self.param_name}_range": n_clusters,
                    f"{self.param_name}_values_tested": n_values,
                    "stability_runs": stability_runs,
                    "seed": self.seed,
                    "total_samples": len(X_train),
                    "total_features": X_train.shape[1],
                    "total_evaluations": len(n_values),
                    f"chosen_{self.param_name}": chosen_n,
                    **{f"chosen_{self.param_name}_{k}": v for k, v in best_result.items() if k != self.param_name},
                    "selection_results": selection_results,
                })

                model = models[chosen_n]
            else: 
                raise ValueError(f"{self.param_name} must be either an int or a tuple of (start, end)")

            self.stability_analysis(X_train, chosen_n, stability_runs, **kwargs)
            model = self.fit_model(X_train, chosen_n, **kwargs)
            labels, centers = self.extract_results(model, X_train)
            self.generate_evaluation_plots(X_train, labels, chosen_n, centers=centers)

            function_elapsed = time.perf_counter() - function_start
            self.ml_logger.log_metric('total_duration', function_elapsed)
            print(f"Total execution time: {function_elapsed:.2f}s")

        self.ml_logger.generate_log_report(output_file=f"{self.save_path}/execution_report.txt", start_index=start_log_index)
        return model
    
    def _compute_composite_scores(self, selection_results):
        """Compute composite scores using Rank Aggregation (HALVING principle).
        Uses Borda count variant: ranks all k values for each metric and selects the k with minimum sum of ranks."""
        
        # Define metric directions
        # higher is better = 'maximize', lower is better = 'minimize'
        METRIC_DIRECTIONS = {
            'silhouette_score': 'maximize',
            'calinski_harabasz_score': 'maximize',
            'dunn_index': 'maximize',
            'davies_bouldin_score': 'minimize',
            'bic': 'minimize',
            'aic': 'minimize',
            'inertia': 'minimize',
            'log_likelihood': 'maximize',
        }
        
        # Always use all 4 shared metrics equally
        metrics_to_use = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'dunn_index']
        
        print(f"Using Rank Aggregation with metrics: {metrics_to_use}")
        
        rank_sums = {r[self.param_name]: 0 for r in selection_results}
        
        for metric_name in metrics_to_use:
            metric_vals = [r[metric_name] for r in selection_results]
            direction = METRIC_DIRECTIONS.get(metric_name, 'maximize')
            
            if direction == 'maximize':
                ranks = np.argsort(np.argsort(metric_vals)[::-1]) + 1
            else:
                ranks = np.argsort(np.argsort(metric_vals)) + 1
            
            # Find best k for this metric for logging
            best_idx = np.argmin(ranks)
            best_k = selection_results[best_idx][self.param_name]
            print(f"  {metric_name}: best {self.param_name}={best_k}")
            
            for i, r in enumerate(selection_results):
                rank_sums[r[self.param_name]] += ranks[i]
        
        # Find k with minimum rank sum (best overall ranking)
        chosen_k = min(rank_sums.keys(), key=lambda k: rank_sums[k])
        print(f"Rank Aggregation winner: {self.param_name}={chosen_k} (rank sum={rank_sums[chosen_k]})")
        
        # Store results
        for r in selection_results:
            r['rank_sum'] = rank_sums[r[self.param_name]]
            r['composite_score'] = -rank_sums[r[self.param_name]]  # Negative for sorting (higher is better)
        
        return chosen_k

    
    def generate_evaluation_plots(self, X_train, labels, chosen_n, centers=None):
        """Copy or generate final evaluation plots (silhouette and scatter) for the chosen cluster number."""
        print(f"Generating evaluation plots for {self.dir_prefix}={chosen_n}...")
        
        cluster_dir = os.path.join(self.save_path, "clusters", f"{self.dir_prefix}{chosen_n}")
        
        # Silhouette plot
        sil_src = os.path.join(cluster_dir, "silhouette.png")
        sil_dst = os.path.join(self.save_path, "silhouette.png")
        if os.path.exists(sil_src):
            shutil.copy(sil_src, sil_dst)
        else:
            Xs, ys, sample_idx = sample_fit_labels(X_train, labels, self.plot_subsample_size, self.seed)
            sil_score, sample_sil_vals = self.compute_silhouette(Xs, ys)
            if sample_sil_vals is not None:
                plot_silhouette(X_train, labels,
                              title=f"Silhouette Plot for {self.algorithm_name} ({self.dir_prefix}={chosen_n}) on {self.dataset}",
                              save_path=sil_dst,
                              silhouette_avg=sil_score,
                              sample_silhouette_values=sample_sil_vals,
                              sample_indices=sample_idx)
            else:
                print(f"Skipping silhouette plot for evaluation: silhouette undefined for {self.dir_prefix}={chosen_n}")

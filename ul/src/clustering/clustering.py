import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from utils.plotter import plot_cluster_scatter, plot_curve, plot_silhouette
from utils.logger import MLLogger


class Clustering:
    """Clustering algorithms for unsupervised learning experiments."""

    def __init__(self, ml_logger: MLLogger, dataset: str, save_path: str):
        self.ml_logger = ml_logger
        self.dataset = dataset
        self.save_path = save_path
        print(f"Initialized Clustering module for dataset: {dataset}")

    # KMeans clustering
    def kmeans(self, X, n_clusters=8, seed=42):
        """Run K-Means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        kmeans.fit(X)
        return kmeans

    # Estimation maximization
    def run_em(self, X, n_components=1, random_state=42):
        """Run EM/GMM clustering."""
        print(f"Running EM/GMM clustering with {n_components} components on {self.dataset}")
        
        with self.ml_logger.log_step(f"EM/GMM Clustering (components={n_components})") as step_info:
            gmm = GaussianMixture(n_components=n_components, random_state=random_state)
            gmm.fit(X)
            preds = gmm.predict(X)

            step_info.update({
                "n_components": n_components,
                "means": np.array(gmm.means_).tolist(),
                "covariances": np.array(gmm.covariances_).tolist() if hasattr(gmm, "covariances_") else None,
                "labels": preds.tolist(),
            })

        print(f"EM/GMM clustering complete with {n_components} components")
        return gmm, preds, gmm.means_


    def run_kmeans(self, X_train, k: int | tuple = (2, 10), stability_runs=10, seed=42):
        """Run K-Means clustering with automatic or manual k selection."""
        print(f"Starting K-Means clustering on {self.dataset} with k={k}")
        
        # Determine chosen_k based on input type
        if isinstance(k, int):
            chosen_k = k
            selection_results = None
            print(f"Using fixed k={chosen_k}")
        elif isinstance(k, tuple) and len(k) == 2:
            chosen_k, selection_results = self.find_optimal_k(X_train, k, seed)
            print(f"Auto-selected optimal k={chosen_k} from range {k}")

            # Elbow plot
            k_values = list(range(k[0], k[1] + 1))
            inertias = [result["inertia"] for result in selection_results]
            plot_curve(x=k_values, y_list=[inertias], xlabel="Number of Clusters (k)", ylabel="Inertia",
                      title=f"Elbow Plot for K-Means on {self.dataset}",
                      save_path=os.path.join(self.save_path, "kmeans_elbow.png"))
        else:
            raise ValueError("k must be either an int or a tuple of (start_k, end_k)")

        # Run clustering with chosen k
        with self.ml_logger.log_step(f"KMeans Clustering (k={chosen_k})") as step_info:
            model = self.kmeans(X_train, n_clusters=chosen_k, seed=seed)
            labels, centers, inertia = model.labels_, model.cluster_centers_, model.inertia_

            step_info.update({
                "clusters": chosen_k,
                "inertia": inertia,
                "cluster_centers": centers.tolist(),
                "labels": labels.tolist(),
            })

        stability_scores = self.stability_analysis(X_train, chosen_k, labels, stability_runs, seed)
        self.generate_evaluation_plots(X_train, labels, chosen_k, stability_scores)

        return model, labels, centers

    def find_optimal_k(self, X_train, k, seed):
        """Find optimal k using silhouette score."""
        print(f"Evaluating k values from {k[0]} to {k[1]}...")
        
        with self.ml_logger.log_step(f"KMeans Hyperparameter Selection (k_range={k})") as step_info:
            k_values = list(range(k[0], k[1] + 1))
            selection_results = []

            for k_val in k_values:
                print(f"Evaluating k={k_val}")
                model = self.kmeans(X_train, n_clusters=k_val, seed=seed)
                labels, centers, inertia = model.labels_, model.cluster_centers_, model.inertia_

                sil_score = silhouette_score(X_train, labels) if k_val > 1 else -1
                selection_results.append({
                    "k": k_val,
                    "inertia": inertia,
                    "silhouette_score": sil_score,
                    "samples": len(labels),
                    "features": X_train.shape[1]
                })

            # Choose k with highest silhouette score
            valid_results = [r for r in selection_results if r["k"] > 1]
            if valid_results:
                best_result = max(valid_results, key=lambda x: x["silhouette_score"])
                chosen_k = best_result["k"]
                print(f"Selected k={chosen_k} with silhouette score: {best_result['silhouette_score']:.3f}")
            else:
                chosen_k = selection_results[0]["k"]
                print(f"No valid results, using k={chosen_k}")

            step_info.update({
                "k_range": k,
                "k_values_tested": k_values,
                "chosen_k": chosen_k,
                "selection_criterion": "highest_silhouette_score",
                "selection_results": selection_results,
                "total_evaluations": len(k_values)
            })

            return chosen_k, selection_results

    def stability_analysis(self, X_train, chosen_k, base_labels, stability_runs, seed):
        """Run stability analysis by clustering multiple times."""
        print(f"Running stability analysis with {stability_runs} runs for k={chosen_k}...")
        
        with self.ml_logger.log_step(f"KMeans Stability Analysis (k={chosen_k}, runs={stability_runs})") as step_info:
            stability_scores = []
            stability_results = []

            for i in range(stability_runs):
                seed_i = 42 + i
                model = self.kmeans(X_train, n_clusters=chosen_k, seed=seed_i)
                ari = adjusted_rand_score(base_labels, model.labels_)
                stability_scores.append(ari)
                stability_results.append({
                    "run": i+1,
                    "ari_score": ari,
                    "seed": seed_i
                })
                if (i + 1) % 5 == 0 or i == 0:  # Print progress every 5 runs or first run
                    print(f"  Run {i+1}/{stability_runs}: ARI = {ari:.3f}")

            mean_ari = np.mean(stability_scores)
            std_ari = np.std(stability_scores)
            print(f"Stability analysis complete - Mean ARI: {mean_ari:.3f} ± {std_ari:.3f}")
            
            step_info.update({
                "k": chosen_k,
                "total_runs": stability_runs,
                "stability_scores": stability_scores,
                "mean_ari": mean_ari,
                "std_ari": std_ari,
                "stability_results": stability_results
            })

        return stability_scores

    def generate_evaluation_plots(self, X_train, labels, chosen_k, stability_scores):
        """Generate evaluation plots for clustering results."""
        print(f"📈 Generating evaluation plots for k={chosen_k}...")
        
        plot_silhouette(X_train, labels,
                       title=f"Silhouette Plot for K-Means (k={chosen_k}) on {self.dataset}",
                       save_path=os.path.join(self.save_path, "kmeans_silhouette.png"))
        print("  ✓ Silhouette plot saved")
        
        plot_cluster_scatter(X_train, labels, method='pca',
                           title=f"Cluster Scatter for K-Means (k={chosen_k}) on {self.dataset}",
                           save_path=os.path.join(self.save_path, "kmeans_scatter.png"))
        print("  ✓ Cluster scatter plot saved")
        
        plot_curve(x=range(1, len(stability_scores) + 1), y_list=[stability_scores],
                  xlabel='Run Number', ylabel='ARI with Base Run',
                  title=f'K-Means Stability (k={chosen_k}) on {self.dataset}',
                  save_path=os.path.join(self.save_path, "kmeans_stability.png"), marker='o')
        print("  ✓ Stability plot saved")
        print(f"🎉 K-Means clustering complete for {self.dataset}!")


import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

from utils.data_processing import load_or_process_data
from utils.logger import MLLogger
from clustering import KMeansClustering, EMClustering
from dimensionality_reduction import PCAReduction, ICAReduction, RandomProjection

# -------------------------------------
# Clustering Helper
# -------------------------------------
def run_clustering(X_train, dataset, save_path, ml_logger, seed, n_jobs, cluster_range, stability_runs, n_init, plot_subsample_size):
    # K-Means Clustering
    print("\n" + "-"*80 + "\nK-Means Clustering\n" + "-"*80)
    kmeans = KMeansClustering(dataset, f"{save_path}/clustering/kmeans", ml_logger, plot_subsample_size=plot_subsample_size, seed=seed)
    kmeans.run(X_train, n_clusters=cluster_range, stability_runs=stability_runs, n_init=n_init, n_jobs=n_jobs)

    # EM/GMM Clustering
    print("\n" + "-"*80 + "\nEM/GMM Clustering\n" + "-"*80)
    em = EMClustering(dataset, f"{save_path}/clustering/em", ml_logger, plot_subsample_size=plot_subsample_size, seed=seed)
    em.run_gmm(X_train, mixture_components=cluster_range, stability_runs=stability_runs, n_init=n_init, covariance_type='full', n_jobs=n_jobs)

# -------------------------------------
# Dimensionality Reduction Helper
# -------------------------------------
def run_dimensionality_reduction(X_train, y_train, dataset, save_path, ml_logger, seed, method, n_jobs, n_components_range):
    # PCA
    print("\n" + "-"*80 + "\nPCA\n" + "-"*80)
    pca = PCAReduction(dataset, f"{save_path}/dr/pca", ml_logger, seed=seed)
    pca.run_dimensionality_reduction(X_train, y_train, n_components_range, task=method)
    
    # ICA
    print("\n" + "-"*80 + "\nICA\n" + "-"*80)
    ica = ICAReduction(dataset, f"{save_path}/dr/ica", ml_logger, seed=seed)
    ica.run_dimensionality_reduction(X_train, y_train, n_components_range, task=method)
    
    # Random Projection
    print("\n" + "-"*80 + "\nRandom Projection\n" + "-"*80)
    rp = RandomProjection(dataset, f"{save_path}/dr/rp", ml_logger, seed=seed)
    rp.run_dimensionality_reduction(X_train, y_train, n_components_range, task=method)

# -------------------------------------
# Experiment Implementation
# -------------------------------------
def run_experiments(dataset, target, method, subsample=0.1, seed=42):    
    print("\n" + "="*80 + f"\nRUNNING EXPERIMENTS ON: {dataset.upper()}\n" + "="*80)
    
    save_path = os.path.join(os.environ['ROOT'], f"figures/{dataset}")
    os.makedirs(save_path, exist_ok=True)
    
    # Logger
    ml_logger = MLLogger()
    ml_logger.set_experiment_context(dataset=dataset, target=target, method=method)
    
    # Load data
    X_train, _, _, y_train, _, _ = load_or_process_data(dataset, target, method, subsample, seed=seed, ml_logger=ml_logger)

    # Clustering
    run_clustering(X_train, dataset, save_path, ml_logger, seed, n_jobs=-1,
        cluster_range=(2, 10), 
        stability_runs=10, 
        n_init=10, 
        plot_subsample_size=10000
    )

    # Dimensionality Reduction
    run_dimensionality_reduction(X_train, y_train, dataset, save_path, ml_logger, seed, method, n_jobs=-1,
        n_components_range=(2, 15),
    )

    print("\n" + "="*80 + f"\nCOMPLETED: {dataset.upper()}\n" + "="*80)

# -------------------------------------
# Full Experiment Runner
# -------------------------------------
def main():
    """Run experiments on both datasets."""
    
    import random
    seed = int(random.random() * 1000)
    subsample = 0.1  # For testing
    
    # Dataset configurations
    datasets = [
        {
            'dataset': 'hotels',
            'target': 'is_canceled',
            'method': 'classification',
        },
        {
            'dataset': 'accidents',
            'target': 'Duration_Seconds',
            'method': 'regression',
        }
    ]
    
    # Run on all datasets
    for config in datasets:
        run_experiments(
            dataset=config['dataset'],
            target=config['target'],
            method=config['method'],
            subsample=subsample,
            seed=seed
        )

    print("\n" + "="*80 + "\n" + "-"*80 + "\nALL EXPERIMENTS COMPLETE!\n" + "-"*80 + "\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
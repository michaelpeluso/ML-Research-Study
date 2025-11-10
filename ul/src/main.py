
import os, sys
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

from core.neural_networks_on_dr import run_neural_networks_on_dr_data
from core.neural_networks_with_clusters import run_neural_networks_with_clusters
from utils.data_processing import load_or_process_data, split_processed_data
from utils.logger import MLLogger
from clustering import KMeansClustering, EMClustering
from dimensionality_reduction import PCAReduction, ICAReduction, RandomProjection
from core.clustering_on_dr import analyze_results


# -------------------------------------
# Clustering
# -------------------------------------
def run_clustering(X_train, dataset, save_path, ml_logger, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample):
    start = time.perf_counter()

    # K-Means Clustering
    print("\n" + "-"*80 + "\nK-Means Clustering\n" + "-"*80)
    kmeans = KMeansClustering(dataset, save_path, ml_logger, silhouette_subsample=silhouette_subsample, seed=seed)
    kmeans_results = kmeans.run(X_train, n_clusters=n_components_range, stability_runs=stability_runs, n_init=n_init, n_jobs=n_jobs)

    # EM/GMM Clustering
    print("\n" + "-"*80 + "\nEM/GMM Clustering\n" + "-"*80)
    em = EMClustering(dataset, save_path, ml_logger, silhouette_subsample=silhouette_subsample, seed=seed, covariance_type='full')
    em_results = em.run(X_train, n_clusters=n_components_range, stability_runs=stability_runs, n_init=n_init, n_jobs=n_jobs)

    print(f"\nClustering completed on {dataset.upper()} - {time.perf_counter() - start:.2f}s\n")
    return {'kmeans': kmeans_results, 'em': em_results}

# -------------------------------------
# Dimensionality Reduction
# -------------------------------------
def run_dimensionality_reduction(X_train, y_train, dataset, method, save_path, ml_logger, seed, n_components_range):
    start = time.perf_counter()
    
    # PCA
    print("\n" + "-"*80 + "\nPCA\n" + "-"*80)

    # PCA
    pca = PCAReduction(dataset, save_path, ml_logger, seed=seed)
    pca_results = pca.run_dimensionality_reduction(X_train, y_train, n_components_range, task=method)
    
    # ICA
    print("\n" + "-"*80 + "\nICA\n" + "-"*80)
    ica = ICAReduction(dataset, save_path, ml_logger, seed=seed)
    ica_results = ica.run_dimensionality_reduction(X_train, y_train, n_components_range, task=method)
    
    # Random Projection
    print("\n" + "-"*80 + "\nRandom Projection\n" + "-"*80)
    rp = RandomProjection(dataset, save_path, ml_logger, seed=seed)
    rp_results = rp.run_dimensionality_reduction(X_train, y_train, n_components_range, task=method)

    # comparison table
    print("\n" + "-"*80 + "\nGenerating DR Comparison Table\n" + "-"*80)
    super(PCAReduction, pca).generate_dr_comparison_table(save_path)
    
    print(f"\nDimensionality Reduction completed on {dataset.upper()} - {time.perf_counter() - start:.2f}s\n")

    return {'pca': pca_results, 'ica': ica_results, 'rp': rp_results}


# -------------------------------------
# Clustering on DR-transformed Data
# -------------------------------------
def run_clustering_on_dr(dr_results, dataset, save_path, ml_logger, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample):
    start = time.perf_counter()
    clustering_dr_results = {}
    
    for dr_method in ['pca', 'ica', 'rp']:
        print("\n" + "="*80 + f"\nClustering with {dr_method.upper()} on reduced {dataset.upper()}\n" + "="*80)
        X_reduced = dr_results[dr_method]['X_transformed']
        print(f"Data shape after {dr_method.upper()}: {X_reduced.shape} (n_components={dr_results[dr_method]['n_components']})")
        
        # Create separate subdirectory for each DR method to avoid overwriting
        dr_save_path = os.path.join(save_path, dr_method)
        os.makedirs(dr_save_path, exist_ok=True)
        
        dr_clustering_results = run_clustering(X_reduced, dataset, dr_save_path, ml_logger, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample)
        clustering_dr_results[dr_method] = dr_clustering_results        
        print("\n" + "="*80 + f"\nCompleted clustering with {dr_method.upper()} on reduced {dataset.upper()} - {time.perf_counter() - start:.2f}s\n" + "="*80)
        print(f"K-Means k={dr_clustering_results['kmeans']['chosen_n']}, EM/GMM n={dr_clustering_results['em']['chosen_n']}\n" + "="*80 + "\n")
    
    return clustering_dr_results


# -------------------------------------
# Experiment Implementation
# -------------------------------------
def run_experiments(dataset, target, method, dataset_subsample, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample):    
    print("\n" + "="*80 + f"\nRUNNING EXPERIMENTS ON: {dataset.upper()}\n" + "="*80)
    start = time.perf_counter()    
    save_path = os.path.join(os.environ['ROOT'], f"figures/{dataset}")
    os.makedirs(save_path, exist_ok=True)
    
    # Logger
    ml_logger = MLLogger()
    ml_logger.set_experiment_context(dataset=dataset, target=target, method=method)
    
    # Load data
    print("Loading data...")
    X_full, y_full = load_or_process_data(dataset, target, method, dataset_subsample, seed=seed, ml_logger=ml_logger)

    # Step 1: Clustering on full dataset
    print("\n" + "="*80 + "\nStep 1: Clustering on Original Data\n" + "="*80)
    clustering_results = run_clustering(X_full, dataset, f"{save_path}/clustering", ml_logger, seed, n_jobs=n_jobs,
        n_components_range=n_components_range, 
        stability_runs=stability_runs, 
        n_init=n_init, 
        silhouette_subsample=silhouette_subsample
    )

    # Step 2: Dimensionality Reduction
    print("\n" + "="*80 + "\nStep 2: Dimensionality Reduction\n" + "="*80)
    dr_results = run_dimensionality_reduction(X_full, y_full, dataset, method, f"{save_path}/dr", ml_logger, seed,
        n_components_range=n_components_range
    )

    # Step 3: Clustering on DR-transformed data
    print("\n" + "="*80 + "\nStep 3: Clustering on DR Data\n" + "="*80)
    clustering_dr_results = run_clustering_on_dr(dr_results, dataset, f"{save_path}/clustering_dr", ml_logger, seed, n_jobs=n_jobs,
        n_components_range=n_components_range,
        stability_runs=stability_runs,
        n_init=n_init,
        silhouette_subsample=silhouette_subsample
    )
    analyze_results(dataset, f"{save_path}/clustering_dr", ml_logger, clustering_results, clustering_dr_results)

    # Step 4 & 5: Neural Networks (US Accidents only per assignment requirements)
    if dataset == 'accidents':
        # Step 4: Neural Networks on Original + DR Data
        print("\n" + "="*80 + "\nStep 4: Neural Networks on Original + DR Data\n" + "="*80)
        nn_results = run_neural_networks_on_dr_data(
            X_full, y_full, dataset, method, f"{save_path}/nn_dr", ml_logger, seed, dr_results,
            # Optimal parameters from supervised learning logs for Accidents
            # SL used 5 epochs with batch_size=1024 on ~480K train samples = ~2340 updates
            batch_size=1024, 
            optimizer='sgd_momentum',  # SGD + Momentum per OL best recipe for Accidents
            hidden_layers=[512, 512],  # Accidents optimal 2-layer architecture
            max_updates=2340,  # 5 epochs worth of updates
            learning_rate=0.01,  # Matched to SL optimal
            betas=(0.9, 0.999),  # Required param (not used by SGD but needed by signature)
            weight_decay=1e-2,  # L2 regularization per OL recipe (alpha=0.001 in SL -> 1e-2 for stronger reg)
            dropout_p=0.25,  # Mid-range of OL recipe (0.2-0.3)
            l_threshold=0.48,  # Regression loss threshold
            label_smoothing_alpha=0.0,  # Not needed for regression
            activation='relu'  # Matched to SL optimal
        )

        # Step 5: Neural Networks with Cluster Features
        print("\n" + "="*80 + "\nStep 5: Neural Networks with Cluster Features\n" + "="*80)
        nn_cluster_results = run_neural_networks_with_clusters(
            X_full, y_full, dataset, method, f"{save_path}/nn_clusters", ml_logger, seed, clustering_results,
            batch_size=1024,
            feature_setup='additive',  # Append cluster features to original inputs
            max_updates=2340,  # Match Step 4 training budget (5 epochs)
            learning_rate=0.01,
            weight_decay=1e-2,  
            hidden_layers=[512, 512],  
            activation='relu',
            tune_if_needed=False
        )

    print("\n" + "="*80 + f"\nCOMPLETED: {dataset.upper()}\nWall Time: {time.perf_counter() - start:.2f}s\n" + "="*80)

    ml_logger.generate_log_report(output_file=os.path.join(save_path, "execution_report.txt"),start_index=0)


# -------------------------------------
# Full Experiment Runner
# -------------------------------------
def main():
    start = time.perf_counter()
    
    # Dataset configurations
    datasets = [
        {
            'dataset': 'hotels',
            'target': 'is_canceled',
            'method': 'classification',
            'subsample': 1.0
        },
        {
            'dataset': 'accidents',
            'target': 'Duration_Seconds',
            'method': 'regression',
            'subsample': 0.75
        }
    ]
    #datasets.remove(datasets[1])
    
    # Run on all datasets
    for config in datasets:
        run_experiments(
            dataset=config['dataset'],              # Dataset name
            target=config['target'],                # Target variable name
            method=config['method'],                # 'classification' or 'regression'
            dataset_subsample=config['subsample'],  # Fraction of dataset to use
            seed=42,                                # Random seed for reproducibility
            n_jobs=7,                               # Number of parallel workers for joblib
            n_components_range=(2, 15),             # Range of k/n values to evaluate (inclusive)
            stability_runs=5,                       # Number of runs for pairwise ARI stability analysis
            n_init=10,                              # Number of random initializations per clustering run (sklearn)
            silhouette_subsample=10000              # Max samples for silhouette computation (for performance)
        )

    print("\n" + "="*80 + "\n" + "-"*80 + f"\nALL EXPERIMENTS COMPLETE\nWall Time: {time.perf_counter() - start:.2f}s\n" + "-"*80 + "\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
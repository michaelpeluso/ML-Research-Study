
import os, sys
import time
import warnings

from utils.plotter import generate_metric_comparison_plots, generate_clustering_heatmaps, generate_step3_clustering_scatters
from utils.validators import validate_experiment_results
from utils.results_parser import extract_and_save_results

warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

from core.neural_networks_on_dr import (
    run_neural_networks_on_data,
    run_neural_networks_on_original,
    run_neural_networks_on_reduced
)
from core.neural_networks_with_clusters import run_neural_networks_with_clusters
from utils.data_processing import load_or_process_data, split_processed_data
from utils.logger import MLLogger, generate_ul_summary
from clustering import KMeansClustering, EMClustering
from dimensionality_reduction import PCAReduction, ICAReduction, RandomProjection
from core.clustering_on_dr import analyze_results


# -------------------------------------
# Clustering
# -------------------------------------
def run_clustering(X_train, dataset, save_path, ml_logger, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample, covariance_type):
    start = time.perf_counter()

    # K-Means Clustering
    print("\n" + "- "*40 + "\nK-Means Clustering\n" + "- "*40)
    kmeans = KMeansClustering(dataset, save_path, ml_logger, silhouette_subsample=silhouette_subsample, seed=seed)
    kmeans_results = kmeans.run(X_train, n_clusters=n_components_range, stability_runs=stability_runs, n_init=n_init, n_jobs=n_jobs)

    # EM/GMM Clustering
    print("\n" + "- "*40 + "\nEM/GMM Clustering\n" + "- "*40)
    em = EMClustering(dataset, save_path, ml_logger, silhouette_subsample=silhouette_subsample, seed=seed, covariance_type=covariance_type)
    em_results = em.run(X_train, n_clusters=n_components_range, stability_runs=stability_runs, n_init=n_init, n_jobs=n_jobs)

    print(f"Clustering completed on {dataset.upper()} - {time.perf_counter() - start:.2f}s\n")
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
    print("\n" + "- "*40 + "\nGenerating DR Comparison Table\n" + "- "*40)
    dr_results = {'pca': pca_results, 'ica': ica_results, 'rp': rp_results}
    super(PCAReduction, pca).generate_dr_comparison_table(dr_results, save_path)
    
    print(f"Dimensionality Reduction completed on {dataset.upper()} - {time.perf_counter() - start:.2f}s\n")

    return dr_results


# -------------------------------------
# Clustering on DR-transformed Data
# -------------------------------------
def run_clustering_on_dr(dr_results, dataset, save_path, ml_logger, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample, covariance_type):
    start = time.perf_counter()
    clustering_dr_results = {}
    
    for dr_method in ['pca', 'ica', 'rp']:
        print("\n" + "-"*80 + f"\nClustering with {dr_method.upper()} on reduced {dataset.upper()}\n" + "-"*80)
        X_reduced = dr_results[dr_method]['X_transformed']
        print(f"Data shape after {dr_method.upper()}: {X_reduced.shape} (n_components={dr_results[dr_method]['n_components']})")
        
        # Create separate subdirectory for each DR method to avoid overwriting
        dr_save_path = os.path.join(save_path, dr_method)
        os.makedirs(dr_save_path, exist_ok=True)
        
        dr_clustering_results = run_clustering(X_reduced, dataset, dr_save_path, ml_logger, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample, covariance_type)
        clustering_dr_results[dr_method] = dr_clustering_results        
        print("\n" + "-"*80 + f"\nCompleted clustering with {dr_method.upper()} on reduced {dataset.upper()} - {time.perf_counter() - start:.2f}s")
        print(f"K-Means k={dr_clustering_results['kmeans']['chosen_n']}, EM/GMM n={dr_clustering_results['em']['chosen_n']}\n" + "-"*80 + "\n")
    
    return clustering_dr_results


# ------------------------------------
# Neural Network Training Helpers
# ------------------------------------
def run_nn_original_only(X_full, y_full, dataset, method, save_path, ml_logger, seed, **nn_kwargs):
    """Train neural networks on original data only."""
    from core.neural_networks_on_dr import run_neural_networks_on_original
    return run_neural_networks_on_original(
        X_full, y_full, dataset, method, save_path, ml_logger, seed, **nn_kwargs
    )


def run_nn_reduced_only(X_full, y_full, dataset, method, save_path, ml_logger, seed, dr_results, **nn_kwargs):
    """Train neural networks on dimensionality-reduced data only."""
    from core.neural_networks_on_dr import run_neural_networks_on_reduced
    return run_neural_networks_on_reduced(
        X_full, y_full, dataset, method, save_path, ml_logger, seed, dr_results, **nn_kwargs
    )


def run_nn_both(X_full, y_full, dataset, method, save_path, ml_logger, seed, dr_results, **nn_kwargs):
    """Train neural networks on both original and reduced data."""
    from core.neural_networks_on_dr import run_neural_networks_on_data
    return run_neural_networks_on_data(
        X_full, y_full, dataset, method, save_path, ml_logger, seed, dr_results, **nn_kwargs
    )


# ------------------------------------
# Experiment Implementation
# ------------------------------------
def run_experiments(dataset, target, method, dataset_subsample, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample, covariance_type):    
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
        silhouette_subsample=silhouette_subsample,
        covariance_type=covariance_type
    )

    # Step 2: Dimensionality Reduction
    print("\n" + "-"*80 + "\nStep 2: Dimensionality Reduction\n" + "-"*80)
    dr_results = run_dimensionality_reduction(X_full, y_full, dataset, method, f"{save_path}/dr", ml_logger, seed,
        n_components_range=n_components_range
    )

    # Step 3: Clustering on DR-transformed data
    print("\n" + "-"*80 + "\nStep 3: Clustering on DR Data\n" + "-"*80)
    clustering_dr_results = run_clustering_on_dr(dr_results, dataset, f"{save_path}/clustering_dr", ml_logger, seed, n_jobs=n_jobs,
        n_components_range=n_components_range,
        stability_runs=stability_runs,
        n_init=n_init,
        silhouette_subsample=silhouette_subsample,
        covariance_type=covariance_type
    )
    analyze_results(dataset, f"{save_path}/clustering_dr", ml_logger, clustering_results, clustering_dr_results)

    # Step 4 & 5: Neural Networks
    step_4a_results = step_4b_results = step_5_results = None
    if dataset == 'accidents':
        # Common NN hyperparameters
        nn_kwargs = {
            'batch_size': 1024,
            'optimizer': 'sgd_momentum',  # SGD + Momentum per OL best recipe
            'hidden_layers': [512, 512],   # Accidents optimal 2-layer
            'max_updates': 2340,           # 5 epochs worth of updates
            'learning_rate': 0.01,
            'betas': (0.9, 0.999),
            'weight_decay': 1e-2,
            'dropout_p': 0.25,
            'l_threshold': 0.48,
            'label_smoothing_alpha': 0.0,
            'activation': 'relu'
        }
        
        # Step 4a: Train on ORIGINAL data only
        print("\n" + "="*80 + "\nStep 4a: Neural Networks on Original Data\n" + "="*80)
        step_4a_results = run_neural_networks_on_original(
            X_full, y_full, dataset, method, f"{save_path}/nn_original", ml_logger, seed, **nn_kwargs
        )
        
        # Step 4b: Train on DR-REDUCED data only
        print("\n" + "="*80 + "\nStep 4b: Neural Networks on Reduced Data\n" + "="*80)
        step_4b_results = run_neural_networks_on_reduced(
            X_full, y_full, dataset, method, f"{save_path}/nn_reduced", ml_logger, seed, dr_results, **nn_kwargs
        )

        try:
            # Step 5: Neural Networks with Cluster Features
            print("\n" + "="*80 + "\nStep 5: Neural Networks with Cluster Features\n" + "="*80)
            step_5_results = run_neural_networks_with_clusters(
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
        except Exception as e:
            print(f"\nError during Step 5 (Neural Networks with Cluster Features): {e}")

    print("\n" + "="*80 + f"\nCOMPLETED: {dataset.upper()}\nWall Time: {time.perf_counter() - start:.2f}s\n" + "="*80)
    ml_logger.generate_log_report(output_file=os.path.join(save_path, "execution_report.txt"),start_index=0)

    # Master results dictionary with standardized structure
    result = {
        'dataset': dataset,
        'metadata': {
            'target': target,
            'method': method,
            'subsample': dataset_subsample,
            'seed': seed,
            'wall_time': time.perf_counter() - start
        },
        # New step-based keys
        'step_1_clustering': clustering_results,
        'step_2_dr': dr_results,
        'step_3_clustering_on_dr': clustering_dr_results,
        'step_4a_nn_original': step_4a_results,
        'step_4b_nn_reduced': step_4b_results,
        'step_5_nn_with_clusters': step_5_results,
        # Legacy keys for backward compatibility with validation/summary functions
        'clustering_results': clustering_results,
        'dr_results': dr_results,
        'clustering_dr_results': clustering_dr_results,
        'nn_results': step_4a_results if step_4a_results else (step_4b_results if step_4b_results else None),
        'nn_cluster_results': step_5_results,
        'save_path': save_path
    }
    
    return result


# -------------------------------------
# Full Experiment Runner
# -------------------------------------
def main():
    start = time.perf_counter()
    all_results = []
    
    # Dataset configurations
    datasets = [
        {
            'dataset': 'hotels',
            'target': 'is_canceled',
            'method': 'classification',
            'subsample': 0.001,
            'covariance_type': 'full'
        },
        {
            'dataset': 'accidents',
            'target': 'Duration_Seconds',
            'method': 'regression',
            'subsample': 0.0001,
            'covariance_type': 'diag'
        }
    ]
    datasets.remove(datasets[1])
    
    # Run on all datasets
    for config in datasets:
        result = run_experiments(
            dataset=config['dataset'],                  # Dataset name
            target=config['target'],                    # Target variable name
            method=config['method'],                    # 'classification' or 'regression'
            dataset_subsample=config['subsample'],      # Fraction of dataset to use
            seed=42,                                    # Random seed for reproducibility
            n_jobs=5,                                   # Number of parallel workers for joblib
            n_components_range=list(range(2, 11)),      # Range of k/n values to evaluate (inclusive)
            stability_runs=5,                           # Number of runs for pairwise ARI stability analysis
            n_init=10,                                  # Number of random initializations per clustering run (sklearn)
            silhouette_subsample=10000,                 # Max samples for silhouette computation (for performance)
            covariance_type=config['covariance_type']   # Covariance type for EM/GMM clustering
        )
        all_results.append(result)
        
        # Validate that all expected data was captured
        print("\n" + "="*80)
        print(f"VALIDATING RESULTS FOR {config['dataset'].upper()}")
        print("="*80)
        validate_experiment_results(result, dataset=config['dataset'])

    # Generate unified summary and save parsed results
    generate_ul_summary(all_results)
    generate_metric_comparison_plots(all_results, save_path=None)
    generate_clustering_heatmaps(all_results, save_path=None)
    generate_step3_clustering_scatters(all_results, save_path=None)
    
    # Parse and save master results for easy analysis
    print("\n" + "="*80)
    print("PARSING AND ORGANIZING MASTER RESULTS")
    print("="*80)
    parsed_results = extract_and_save_results(all_results, save_dir='figures/master_results')
    
    print("\n" + "="*80 + "\n" + "-"*80 + f"\nALL EXPERIMENTS COMPLETE\nWall Time: {time.perf_counter() - start:.2f}s\n" + "-"*80 + "\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
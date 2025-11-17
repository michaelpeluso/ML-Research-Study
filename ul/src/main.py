
import os, sys
import time
import warnings

from utils.plotter import (
    generate_metric_comparison_plots, 
    generate_clustering_heatmaps, 
    generate_step3_clustering_scatters,
    generate_comprehensive_report_plots
)
from utils.validators import validate_experiment_results
from utils.results_parser import extract_and_save_results

warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path: sys.path.insert(0, script_dir)
os.environ['ROOT'] = os.path.dirname(script_dir)

from core.neural_networks_on_dr import run_neural_networks_on_data
from core.neural_networks_with_clusters import run_neural_networks_with_clusters
from utils.data_processing import load_or_process_data
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
        print("\n" + "- "*40 + f"\nClustering with {dr_method.upper()} on reduced {dataset.upper()}\n" + "- "*40)
        X_reduced = dr_results[dr_method]['X_transformed']
        print(f"Data shape after {dr_method.upper()}: {X_reduced.shape} (n_components={dr_results[dr_method]['n_components']})")
        
        # Create separate subdirectory for each DR method to avoid overwriting
        dr_save_path = os.path.join(save_path, dr_method)
        os.makedirs(dr_save_path, exist_ok=True)
        
        dr_clustering_results = run_clustering(X_reduced, dataset, dr_save_path, ml_logger, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample, covariance_type)
        clustering_dr_results[dr_method] = dr_clustering_results        
        print(f"Completed clustering with {dr_method.upper()} on reduced {dataset.upper()} - {time.perf_counter() - start:.2f}s")
        print(f"K-Means k={dr_clustering_results['kmeans']['chosen_n']}, EM/GMM n={dr_clustering_results['em']['chosen_n']}\n")
    
    return clustering_dr_results


# -------------------------------------
# Neural Networks on Original and Reduced Data
# -------------------------------------
def run_neural_networks_on_dr(X_full, y_full, dataset, method, save_path, ml_logger, seed, dr_results, **nn_kwargs):
    """Run neural networks on both original and dimensionality-reduced data."""
    print("\n" + "="*80 + "\nStep 4: Neural Networks on Original and Reduced Data\n" + "="*80)
    return run_neural_networks_on_data(
        X_full, y_full, dataset, method, f"{save_path}/nn_combined", ml_logger, seed, dr_results, **nn_kwargs
    )


# -------------------------------------
# Neural Networks with Cluster Features
# -------------------------------------
def run_neural_networks_with_clusters_step(X_full, y_full, dataset, method, save_path, ml_logger, seed, dr_results, clustering_dr_results, **nn_kwargs):
    """Run neural networks with cluster-derived features on DR-transformed data."""
    print("\n" + "="*80 + "\nStep 5: Neural Networks with Cluster Features on DR Data\n" + "="*80)
    results = {} 
    
    for dr_method in ['pca', 'ica', 'rp']:
        print("\n" + "-"*80 + f"\nNeural Networks with Cluster Features using {dr_method.upper()} DR Data\n" + "-"*80)

        # Get DR-transformed data
        X_dr = dr_results[dr_method]['X_transformed']
        clustering_results = clustering_dr_results[dr_method]
        
        print(f"Using {dr_method.upper()}-transformed data: {X_dr.shape}")
        print(f"K-Means clusters: {clustering_results['kmeans']['chosen_n']}")
        print(f"EM/GMM clusters: {clustering_results['em']['chosen_n']}")
    
        results[dr_method] = run_neural_networks_with_clusters(
            X_dr, y_full, dataset, method, f"{save_path}/nn_clusters/{dr_method}", ml_logger, seed, clustering_results,
            feature_setup='additive',
            tune_if_needed=False,
            **nn_kwargs
        )
    
    return results
        


# ------------------------------------
# Experiment Implementation
# ------------------------------------
def run_experiments(dataset, target, method, dataset_subsample, seed, n_jobs, n_components_range, stability_runs, n_init, silhouette_subsample, covariance_type):    
    print("\n" + "="*80 + f"\nRUNNING EXPERIMENTS ON: {dataset.upper()}\n" + "="*80)
    print(f"Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Target: {target}")
    print(f"  Method: {method}")
    print(f"  Subsample: {dataset_subsample*100:.1f}%")
    print(f"  Seed: {seed}")
    print(f"  Components range: {n_components_range[0]}-{n_components_range[-1]}")
    print(f"  Stability runs: {stability_runs}")
    print(f"  Covariance type: {covariance_type}")
    print("="*80)
    
    start = time.perf_counter()    
    save_path = os.path.join(os.environ['ROOT'], f"figures/{dataset}")
    os.makedirs(save_path, exist_ok=True)
    
    # Logger
    ml_logger = MLLogger()
    ml_logger.set_experiment_context(dataset=dataset, target=target, method=method)
    
    # Load data
    print("\n" + "="*80)
    print("STEP 0: Loading and Processing Data")
    print("="*80)
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
    step_4_results = step_5_results = None
    if dataset == 'accidents':
        # Common NN hyperparameters
        nn_kwargs = {
            'batch_size': 1024,
            'optimizer': 'sgd_momentum',  # SGD + Momentum per OL best recipe
            'hidden_layers': [256, 128],   # Accidents optimal 2-layer
            'max_updates': 250,
            'eval_interval': 5,
            'learning_rate': 0.01,
            'betas': (0.9, 0.999),
            'momentum': 0.9,
            'weight_decay': 1e-2,
            'dropout_p': 0.25,
            'l_threshold': 0.48,
            'label_smoothing_alpha': 0.0,
            'activation': 'relu'
        }

        # Step 4: Neural Networks on both original and reduced data
        step_4_results = run_neural_networks_on_dr(
            X_full, y_full, dataset, method, save_path, ml_logger, seed, dr_results, **nn_kwargs
        )

        # Step 5: Neural Networks with Cluster Features
        step_5_results = run_neural_networks_with_clusters_step(
            X_full, y_full, dataset, method, save_path, ml_logger, seed, dr_results, clustering_dr_results, **nn_kwargs
        )

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
        'step_4_nn_combined': step_4_results,
        'step_5_nn_with_clusters': step_5_results,
        # Legacy keys for backward compatibility with validation/summary functions
        'clustering_results': clustering_results,
        'dr_results': dr_results,
        'clustering_dr_results': clustering_dr_results,
        'nn_results': step_4_results,
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
            'subsample': 1.0,
            'covariance_type': 'full'
        },
        {
            'dataset': 'accidents',
            'target': 'Duration_Seconds',
            'method': 'regression',
            'subsample': 0.6,
            'covariance_type': 'diag'
        }
    ]
    
    # Run on all datasets
    for config in datasets:
        result = run_experiments(
            dataset=config['dataset'],                  # Dataset name
            target=config['target'],                    # Target variable name
            method=config['method'],                    # 'classification' or 'regression'
            dataset_subsample=config['subsample'],      # Fraction of dataset to use
            seed=42,                                    # Random seed for reproducibility
            n_jobs=7,                                   # Number of parallel workers for joblib
            n_components_range=list(range(2, 9)),       # Range of k/n values to evaluate (inclusive)
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
    generate_comprehensive_report_plots(all_results)
    
    # Parse and save master results for easy analysis
    print("\n" + "="*80)
    print("PARSING AND ORGANIZING MASTER RESULTS")
    print("="*80)
    parsed_results = extract_and_save_results(all_results, save_dir='figures/master_results')
    
    print("\n" + "="*80 + "\n" + "-"*80 + f"\nALL EXPERIMENTS COMPLETE\nWall Time: {time.perf_counter() - start:.2f}s\n" + "-"*80 + "\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()